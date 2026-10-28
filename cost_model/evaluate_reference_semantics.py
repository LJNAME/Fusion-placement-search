"""Development comparison against existing GPU labels; no parameter fitting."""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from calibration.gpu_holdout import DEFAULT_BUNDLE, load_freeze
from cost_model.e2_residual import _spearman
from cost_model.e2_srop_proxy import DatasetLatencyProfile, E2Workload, object_map_for_workload
from cost_model.reference_semantics import HostSubmissionProfile, ReferenceEagerCostModel
from cost_model.structural import StructuralCostModel
from execution.audit_gpu_holdout import write_json
from simulator.dataset import graph_from_dict, plan_from_dict
from simulator.hardware import hardware_from_dict


def metrics(rows, method):
    ranked = sorted(rows, key=lambda row: (row[method], row["plan_id"]))
    selected = ranked[0]
    best = min(rows, key=lambda row: (row["measured_ms"], row["plan_id"]))
    errors = [abs(row[method] - row["measured_ms"]) / row["measured_ms"] for row in rows]
    return {"selected_plan_id": selected["plan_id"], "measured_subset_best_id": best["plan_id"],
        "selected_regret_pct": (selected["measured_ms"] / best["measured_ms"] - 1) * 100,
        "predicted_top2_margin_ms": ranked[1][method] - selected[method] if len(ranked) > 1 else None,
        "mape_pct": statistics.fmean(errors) * 100,
        "ranking_spearman": _spearman([row[method] for row in rows], [row["measured_ms"] for row in rows])}


def evaluate(bundle=DEFAULT_BUNDLE, host=None):
    freeze, _ = load_freeze(bundle)
    host = host or HostSubmissionProfile()
    received = Path(bundle) / "results_received_20261028"
    original = json.loads((received / "execution_report.json").read_text())
    assert original["freeze_id"] == freeze["freeze_id"]
    hardware = hardware_from_dict(freeze["hardware"])
    estimates, examples = {}, []
    for workload in freeze["workloads"]:
        graph = graph_from_dict(workload["graph"])
        mapping = object_map_for_workload(freeze["query_features"], E2Workload(workload["model_id"],
            "exposed-development", workload["model_id"], workload["model_family"], workload["sequence_length"]))
        profile = DatasetLatencyProfile(hardware, mapping,
            {key: row["nearest_s"] for key, row in freeze["zero_shot_predictions"].items()},
            {key: "frozen_zero_shot" for key in freeze["zero_shot_predictions"]})
        sealed = {row["plan_id"]: row["makespan_s"] for row in workload["predicted_ranking"]}
        for plan_id in workload["execution_plan_ids"]:
            plan = plan_from_dict(workload["candidate_plans"][plan_id])
            dag = StructuralCostModel().predict(graph, plan, profile)
            if not math.isclose(dag.makespan, sealed[plan_id], rel_tol=1e-12, abs_tol=1e-15):
                raise ValueError("original DAG score differs from freeze")
            reference = ReferenceEagerCostModel(host).predict(graph, plan, profile)
            estimates[workload["model_id"], plan_id] = {"plan_id": plan_id,
                "original_dag_ms": dag.makespan * 1000,
                "reference_eager_ms": reference.makespan * 1000,
                "serial_work_sum_diagnostic_ms": (dag.computation_time + dag.communication_time) * 1000,
                "cached_input_service_removed_ms": (dag.computation_time - reference.computation_time) * 1000,
                "ood_reasons": list(reference.ood_reasons), "confidence": reference.confidence}
            if workload["model_id"] == "qwen2.5-3b-s2048-b1":
                examples.append({"plan_id": plan_id, "estimate": reference.to_dict()})
    snapshots = [{"name": "original_gpu", "model_id": workload["model_id"],
        "measurements": {row["plan_id"]: row["median_s"] * 1000 for row in workload["candidates"]}}
        for workload in original["workloads"]]
    for directory, phases in (("diagnostics_3b_s2048", ("before",)), ("diagnostics_3b_s2048_light", ("before", "after"))):
        report = json.loads((received / directory / "diagnostic_report.json").read_text())
        assert report["freeze_id"] == freeze["freeze_id"]
        for phase in phases:
            if phase == "before":
                measured = {row["plan_id"]: row["pooled"]["median_ms"] for row in report["baselines"]}
            else:
                measured = {row["plan_id"]: row["median_ms"] for row in report["post_trace_baselines"]}
            snapshots.append({"name": f"{directory}/{phase}", "model_id": report["model_id"], "measurements": measured})
    comparisons = []
    for snapshot in snapshots:
        rows = [{**estimates[snapshot["model_id"], plan_id], "measured_ms": value} for plan_id, value in snapshot["measurements"].items()]
        comparisons.append({"snapshot": snapshot["name"], "model_id": snapshot["model_id"], "candidates": len(rows),
            "metrics": {method: metrics(rows, method) for method in
                ("original_dag_ms", "reference_eager_ms", "serial_work_sum_diagnostic_ms")}, "rows": rows})
    source_files = [Path(__file__), Path(__file__).with_name("reference_semantics.py")]
    return {"format": "fusion-placement-search.reference-semantics-development", "freeze_id": freeze["freeze_id"],
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
        "host_profile": asdict(host), "unique_original_candidates": len(estimates), "comparisons": comparisons,
        "reference_examples": examples,
        "scope": "Exposed development data: four original workloads plus repeated measurements of one of them, no independent task validation. No new search or fitting and no changes to original predictions.",
        "limitations": ["reference_eager uses current-stream copy barriers based on upstream PyTorch v2.9.0; vendor parity is a hypothesis to validate.",
            "Host-profile environment compatibility with historical GPU-run hosts is not verified; kernel-body/GQA/layout coverage remains unverified.",
            "Cached reference inputs have zero GPU service; the original DAG score retains its old input cost. The common offset is reported separately.",
            "Serial-work summation is a diagnostic only, not an execution model or a latency bound.",
            "This report does not establish milestone-A independent generalization, SROP benefit or training throughput gains."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--host-profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    host = HostSubmissionProfile(**json.loads(args.host_profile.read_text())) if args.host_profile else HostSubmissionProfile()
    result = evaluate(args.bundle, host)
    write_json(args.output, result)
    for row in result["comparisons"]:
        old, new = row["metrics"]["original_dag_ms"], row["metrics"]["reference_eager_ms"]
        print(f"{row['snapshot']} {row['model_id']} candidates={row['candidates']} "
              f"regret_pct={old['selected_regret_pct']:.6f}->{new['selected_regret_pct']:.6f} "
              f"mape_pct={old['mape_pct']:.3f}->{new['mape_pct']:.3f}")
    print(f"report: {args.output}; host_calibrated={host.calibrated}")


if __name__ == "__main__":
    main()
