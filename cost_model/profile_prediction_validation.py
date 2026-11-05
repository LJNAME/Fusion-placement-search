"""CPU label-masked prediction checks; no analytical parameter fitting or GPU."""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import statistics
from types import SimpleNamespace

import numpy as np

from calibration.e2_dataset import build_profiling_dataset, make_dataset_view
from calibration.e2_records import load_environment_registry, load_jsonl_records
from cost_model.e2_residual import NearestShapePredictor, _spearman
from cost_model.ffn_profile_search import ROOT, CONFIG, build_case
from cost_model.fusion_kernel_profile import key_id
from cost_model.graph_analytic import analyze, assume, parameters
from simulator.hardware import load_hardware


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def metrics(rows, prediction_field):
    valid = [r for r in rows if r.get(prediction_field) is not None]
    result = {"test_keys": len(rows), "predicted_keys": len(valid),
              "coverage": len(valid) / len(rows) if rows else None}
    if not valid:
        return {**result, **dict.fromkeys(("mape", "median_ape", "p90_ape", "signed_relative_bias", "spearman"))}
    actual = [r["measured_s"] for r in valid]
    predicted = [r[prediction_field] for r in valid]
    signed = [(p - a) / a for p, a in zip(predicted, actual)]
    errors = [abs(e) for e in signed]
    return {**result, "mape": statistics.fmean(errors), "median_ape": statistics.median(errors),
        "p90_ape": float(np.quantile(errors, .9)), "signed_relative_bias": statistics.fmean(signed),
        "spearman": _spearman(actual, predicted)
            if len(actual) > 1 and len(set(actual)) > 1 and len(set(predicted)) > 1 else None}


def comparison_metrics(rows):
    methods = ("analytic", "hybrid", "nearest")
    matched = [r for r in rows if all(r.get(m + "_s") is not None for m in methods)]
    return {"all": {m: metrics(rows, m + "_s") for m in methods},
        "matched_prediction_subset": {m: metrics(matched, m + "_s") for m in methods},
        "by_family": {family: {m: metrics([r for r in rows if r["family"] == family], m + "_s")
                               for m in methods} for family in sorted({r["family"] for r in rows})}}


def neighbor_scope(key):
    # Same implementation/version, backend, hardware class, dtypes, layout,
    # runtime and compile flags. Only Shape may differ.
    return key_id({k: v for k, v in key.items() if k != "shape"})


def predict_masked(train, queries, protocol):
    """Predict without oracle access. Exact hidden labels cannot be queried."""
    if any("measurement" in q or "target" in q for q in queries):
        raise ValueError("query labels must be hidden before prediction")
    known = {key_id(r["kernel_key"]): r for r in train}
    predictions = {}
    for query in queries:
        if key_id(query["kernel_key"]) in known:
            raise ValueError("held-out full Kernel Key overlaps visible Profile")
        compatible = [r for r in train if neighbor_scope(r["kernel_key"]) == neighbor_scope(query["kernel_key"])]
        nearest_s, neighbor = None, None
        if compatible:
            model = NearestShapePredictor().fit_mean(compatible, protocol)
            selected = compatible[model._nearest_index(query)]
            # Reuse the project's nearest-shape residual baseline, not a new fit.
            nearest_s = query["analytic"]["estimate_s"] * math.exp(selected["target"]["log_residual"])
            neighbor = selected["object_id"]
        predictions[query["object_id"]] = {
            "analytic_s": query["analytic"]["estimate_s"],
            "hybrid_s": query["analytic"]["estimate_s"], "hybrid_exact_hit": False,
            "nearest_s": nearest_s, "nearest_visible_object_id": neighbor,
            "nearest_scope": "same complete non-Shape identity; training-only feature normalization"}
    return predictions


def legacy_prediction_check(records, contract):
    splits = []
    for split in contract["splits"]:
        view = make_dataset_view(records, contract, split_id=split["split_id"], mode="zero_shot", role="predictor")
        train = [r for r in view if r["object_id"] in split["train_object_ids"]]
        queries = [r for r in view if r["object_id"] in split["hidden_object_ids"]]
        predictions = predict_masked(train, queries, contract["prediction_protocol"])
        prediction_id = key_id(predictions)  # Sealed before the evaluator attaches labels.
        oracle = {r["object_id"]: r for r in records}
        rows = [{"object_id": q["object_id"], "family": q["kernel_key"]["operator"],
                 "shape": q["kernel_key"]["shape"], "measured_s": oracle[q["object_id"]]["measurement"]["duration_s"],
                 **predictions[q["object_id"]]} for q in queries]
        splits.append({"split_id": split["split_id"], "train_keys": len(train), "hidden_keys": len(queries),
            "prediction_id_before_evaluation": prediction_id, "hidden_exact_hits": 0,
            "metrics": comparison_metrics(rows), "rows": rows})
    return {"scope": "retrospective E2 eager-event baselines, NOT current Graph model accuracy",
        "previous_label_use": "112 keys already used by earlier predictors and development; not fresh blind data",
        "analytic": "existing hardware-class mean shape-aware Roofline including historical eager launch proxy",
        "nearest": "existing nearest-shape log-residual baseline, filtered to matching non-Shape identity",
        "hybrid": "exact Profile lookup disabled for all hidden keys; same analytic fallback",
        "no_hyperparameter_selection_or_analytical_parameter_refit": True, "splits": splits}


def graph_prediction_check(hardware, measured, identity):
    # Predictions depend only on the frozen assumptions and historical rates.
    cfg = read("configs/graph_analytic_scenarios.json")
    base = parameters(hardware.devices[0].mem_bw, "historical GPU0 copy effective rate")
    unresolved = analyze(base)
    scenario = analyze(assume(base, cfg["assumptions"]))
    predictions = {s["name"]: (s["duration_us"] + scenario["arms"]["native"]["graph_call_overhead_us"]) / 1e6
                   for s in scenario["arms"]["native"]["stages"]}
    predictions["pack_lightop_complete"] = scenario["arms"]["fused"]["duration_us"] / 1e6
    prediction_id = key_id(predictions)
    labels = {r["stage"]: (r["duration_s"], key) for key, r in measured.store.records.items()}
    # The existing local builder has already checked runtime and strict Fused
    # identity. Query its reviewed complete cost through the existing interface.
    nodes, device = measured.base.keys._selected
    fused_group = SimpleNamespace(node_ids=nodes, device=device, pattern_id="silu_mul")
    labels["pack_lightop_complete"] = (measured.base.estimate_group(measured.graph, fused_group).duration,
                                       identity["fused_kernel_key_id"])
    rows = [{"family": family, "profile_identity_id": key, "measured_s": actual,
        "analytic_s": predictions[family], "hybrid_s": predictions[family], "nearest_s": None,
        "hidden_exact_hit": False, "nearest_missing_reason": "no different Shape with matching implementation and Graph contract",
        "role": "previous calibration/development evidence, not independent validation"}
        for family, (actual, key) in labels.items()]
    return {"scope": "current Graph frozen-assumption retrospective diagnostic only",
        "prediction_id_before_evaluation": prediction_id,
        "unresolved_model": {"numeric_prediction_coverage": 0, "test_keys": 3,
            "missing_parameters": sorted(set().union(*(a["missing_parameters"] for a in unresolved["arms"].values())))},
        "explicit_assumed_scenario": cfg["assumptions"], "metrics": comparison_metrics(rows), "rows": rows,
        "independent_test_keys": 0, "new_shape_generalization_validated": False,
        "hybrid_exact_hits_excluded_from_accuracy": 3,
        "no_calibration_or_best_scenario_selection": True}


def build_report():
    records = load_jsonl_records(ROOT / "data/bw1000_e2/profiling_dataset.jsonl")
    manifest = read("data/bw1000_e2/shape_manifest.json")
    contract = read("data/bw1000_e2/split_contract.json")
    raw = load_jsonl_records(ROOT / "data/bw1000_e2/raw_measurements.jsonl")
    environments = load_environment_registry(ROOT / "data/bw1000_e2/environments.json")
    hardware = load_hardware(ROOT / "data/bw1000_2card/bw1000_2card.calibrated.json")
    meta, reconstructed = build_profiling_dataset(manifest, contract, environments, raw, hardware)
    if reconstructed != records or meta != read("data/bw1000_e2/profiling_dataset.meta.json"):
        raise ValueError("legacy labels differ from raw receipt reconstruction")
    _, _, current_hw, measured, identity = build_case(read(CONFIG))
    audit = []
    for record in records:
        key = record["kernel_key"]
        audit.append({"object_id": record["object_id"], "operator": key["operator"],
            "implementation": key["implementation"], "shape": key["shape"],
            "physical_devices": record["measurement"]["physical_device_indices"],
            "current_graph_compatible": False, "independent_current_test_eligible": False,
            "reasons": ["eager_event_loop_not_graph_replay", "HIP_6.3.26093_not_current_6.3.26113",
                        "allocating_reference_recipe_not_current_preallocated_implementation",
                        "previously_used_development_labels"],
            "raw_record_ids": record["measurement"]["raw_record_ids"]})
    old_holdout = read("data/bw1000_gpu_holdout_v1/results_received_20261028/zero_shot_report.json")
    return {"scope": "P0/P1 CPU evidence reuse; no new GPU or model parameters",
        "profile_audit": {"legacy_keys": len(records), "operators": dict(Counter(r["operator"] for r in audit)),
            "raw_reconstruction_passed": True, "current_graph_compatible_legacy_keys": 0,
            "current_graph_independent_test_keys": 0, "legacy_records": audit,
            "reviewed_graph_cost_items": 3, "graph_data_role": "existing calibration/development; exact use is query validation",
            "uuid_changes_are_not_exclusion_reasons": True},
        "legacy_eager_leave_out": legacy_prediction_check(records, contract),
        "archived_unseen_shape_eager_result": {"source": "data/bw1000_gpu_holdout_v1/results_received_20261028/zero_shot_report.json",
            "keys": len(old_holdout["objects"]), "metrics": old_holdout["all"],
            "role": "historical sealed eager result, not current Graph prediction validation or newly blind test",
            "raw_receipts_not_revalidated_in_this_section": True},
        "graph_retrospective": graph_prediction_check(current_hw, measured, identity),
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in (
            "data/bw1000_e2/profiling_dataset.jsonl", "data/bw1000_e2/shape_manifest.json",
            "data/bw1000_e2/split_contract.json", "data/bw1000_2card/bw1000_2card.calibrated.json",
            "configs/graph_analytic_scenarios.json", "cost_model/graph_analytic.py")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"legacy_keys": report["profile_audit"]["legacy_keys"],
        "current_graph_independent_test_keys": 0,
        "graph_retrospective": report["graph_retrospective"]["metrics"]["all"]}, indent=2))


if __name__ == "__main__":
    main()
