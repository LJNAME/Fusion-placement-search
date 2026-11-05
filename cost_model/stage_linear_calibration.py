"""Local T(N)=a+bN calibration for GPU0 SiLU/Mul, then holdout validation.

The OpenXLA-style graph_analytic model is unchanged and reported as a
baseline. Parameters are stored per GPU identity and per execution contract;
a lookup never falls back to another GPU, contract or implementation.

Order is enforced: `fit` sees only fit-size labels and writes holdout
predictions; `validate` reads holdout and chain labels afterwards.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import statistics

from calibration.stage_size_calibration import FITTED_STAGES, HIDDEN, SIZES, read
from cost_model.fusion_kernel_profile import key_id

SCHEMA = "stage_linear_params_v1"


def _ols(xs, ys):
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def _device_key(case_identity):
    ident = case_identity["case_identity"]
    rt = ident["runtime"]
    # GPU type plus the software stack that changes kernels; UUID is provenance only.
    return {"device": copy.deepcopy(ident["devices"][0]),
            "runtime": {k: rt[k] for k in ("torch_distribution", "torch_git_revision", "torch_build_sha256",
                                             "torch_extension_sha256", "hip", "driver")}}


def fit_view(evaluation, frozen):
    """The only labels `fit` may read."""
    return {c: copy.deepcopy(q) for c, q in evaluation["cases"].items()
            if frozen["cases"][c].get("role") == "fit"}


def fit(evaluation, frozen):
    protocol = frozen["fit_protocol"]
    visible = fit_view(evaluation, frozen)
    entries, stages = [], {}
    for stage in FITTED_STAGES:
        ids = [f"{stage}/S{s}" for s in protocol["fit_sizes"]]
        failed = [c for c in ids if visible[c]["status"] != "quality_passed"]
        if failed:
            stages[stage] = {"status": "not_fitted", "reason": "fit cases quality_failed", "cases": failed}
            continue
        xs = [frozen["cases"][c]["elements"] for c in ids]
        medians = [visible[c]["statistics"]["median_us"] for c in ids]
        a, b = _ols(xs, medians)
        combos = [_ols(xs, ys) for ys in itertools.product(*(visible[c]["samples_us"] for c in ids))]
        a_range = [min(p[0] for p in combos), max(p[0] for p in combos)]
        b_range = [min(p[1] for p in combos), max(p[1] for p in combos)]
        identities = [evaluation["identity"]["cases"][c] for c in ids]
        device_keys = {key_id(_device_key(i)) for i in identities}
        contracts = {i["contract_id"] for i in identities}
        recipes = {i["case_identity"]["implementation"]["recipe"]["id"] for i in identities}
        if len(device_keys) != 1 or len(contracts) != 1 or len(recipes) != 1:
            raise ValueError("Fit cases mix GPU, contract or implementation identities")
        b_identified = b_range[0] > 0
        a_identified = a_range[0] > 0 or a_range[1] < 0
        holdout = {}
        for size in protocol["holdout_sizes"]:
            n = size * HIDDEN
            holdout[f"{stage}/S{size}"] = {"elements": n, "predicted_us": a + b * n if b_identified else None}
        stages[stage] = {"status": "fitted" if b_identified else "slope_unidentified",
            "fit_cases": ids, "fit_medians_us": medians,
            "residuals_us": [m - (a + b * x) for x, m in zip(xs, medians)],
            "a_round_sensitivity_us": a_range, "b_round_sensitivity_us_per_element": b_range,
            "a_separately_identified": a_identified, "b_identified": b_identified,
            "holdout_predictions": holdout}
        if b_identified:
            entries.append({"stage": stage, "recipe_id": recipes.pop(),
                "gpu_key": _device_key(identities[0]), "gpu_key_id": device_keys.pop(),
                "contract_id": contracts.pop(), "model": protocol["model"],
                "a_us": a if a_identified else None, "a_us_composite_for_prediction": a,
                "b_us_per_element": b, "validated_elements_range": None,
                "fit_elements_range": [min(xs), max(xs)],
                "status": "fitted_pending_holdout"})
    report = {"schema": SCHEMA, "scope": "local GPU0 stage calibration; analytic model unchanged",
        "protocol": copy.deepcopy(protocol), "stages": stages, "entries": entries,
        "labels_read": sorted(visible), "holdout_labels_read": False,
        "plan_sha256": evaluation.get("plan_sha256"), "review_raw_sha256": evaluation.get("raw_sha256")}
    report["prediction_id"] = key_id({s: v.get("holdout_predictions") for s, v in stages.items()})
    return report


class StageLinearCalibration:
    """Exact GPU/contract/recipe lookup; misses and extrapolation are explicit."""
    def __init__(self, report):
        if report["schema"] != SCHEMA:
            raise ValueError("Unknown parameter schema")
        self.entries = {(e["stage"], e["gpu_key_id"], e["contract_id"], e["recipe_id"]): e for e in report["entries"]}

    def predict(self, stage, elements, gpu_key, contract_id, recipe_id):
        entry = self.entries.get((stage, key_id(gpu_key), contract_id, recipe_id))
        if entry is None:
            return None
        low, high = entry["fit_elements_range"]
        return {"duration_us": entry["a_us_composite_for_prediction"] + entry["b_us_per_element"] * elements,
                "source": "stage_linear_calibration", "extrapolated": not low <= elements <= high,
                "validated_range": entry["validated_elements_range"]}


def _errors(rows):
    usable = [r for r in rows if r["predicted_us"] is not None and r["measured_us"] is not None]
    for r in rows:
        ok = r in usable
        r["signed_error"] = (r["predicted_us"] - r["measured_us"]) / r["measured_us"] if ok else None
        r["ape"] = abs(r["signed_error"]) if ok else None
    if not usable:
        return {"coverage": f"0/{len(rows)}", "mape": None, "signed_bias": None, "max_ape": None}
    return {"coverage": f"{len(usable)}/{len(rows)}",
            "mape": statistics.fmean(r["ape"] for r in usable),
            "signed_bias": statistics.fmean(r["signed_error"] for r in usable),
            "max_ape": max(r["ape"] for r in usable)}


def _ordering(rows):
    usable = [r for r in rows if r["predicted_us"] is not None and r["measured_us"] is not None]
    pairs, wrong = 0, []
    for x, y in itertools.combinations(usable, 2):
        pairs += 1
        if (x["predicted_us"] - y["predicted_us"]) * (x["measured_us"] - y["measured_us"]) <= 0:
            wrong.append([x["case"], y["case"]])
    return {"pairs": pairs, "misordered_pairs": wrong}


def analytic_baseline(memory_bandwidth_Bps, size):
    from cost_model.graph_analytic import analyze, assume, parameters
    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs/graph_analytic_scenarios.json")
                     .read_text(encoding="utf-8"))
    result = analyze(assume(parameters(memory_bandwidth_Bps, "historical GPU0 copy effective rate"),
                            cfg["assumptions"]), shape=(1, size, HIDDEN))
    call = result["arms"]["native"]["graph_call_overhead_us"]
    stages = {s["name"]: s["duration_us"] + call for s in result["arms"]["native"]["stages"]}
    stages["chain"] = result["arms"]["native"]["duration_us"]
    return stages


def validate(fit_report, evaluation, frozen, memory_bandwidth_Bps):
    if fit_report["prediction_id"] != key_id({s: v.get("holdout_predictions") for s, v in fit_report["stages"].items()}):
        raise ValueError("Holdout predictions changed after fitting")
    if fit_report["plan_sha256"] != evaluation.get("plan_sha256"):
        raise ValueError("Fit and validation use different collections")
    cases = evaluation["cases"]
    median = lambda c: cases[c]["statistics"]["median_us"] if cases[c]["status"] == "quality_passed" else None
    holdout, analytic = [], []
    for stage in FITTED_STAGES:
        info = fit_report["stages"][stage]
        for size in frozen["fit_protocol"]["holdout_sizes"]:
            case = f"{stage}/S{size}"
            predicted = info.get("holdout_predictions", {}).get(case, {}).get("predicted_us")
            holdout.append({"case": case, "predicted_us": predicted, "measured_us": median(case),
                            "quality": cases[case]["status"]})
            analytic.append({"case": case, "predicted_us": analytic_baseline(memory_bandwidth_Bps, size)[stage],
                             "measured_us": median(case)})
    fitted_metrics, analytic_metrics = _errors(holdout), _errors(analytic)
    lines = {s: fit_report["stages"][s] for s in FITTED_STAGES}
    entries = {e["stage"]: e for e in fit_report["entries"]}
    chain = []
    for size in SIZES:
        n = size * HIDDEN
        silu, mul, measured = median(f"silu/S{size}"), median(f"mul/S{size}"), median(f"chain/S{size}")
        stage_sum = silu + mul if silu is not None and mul is not None else None
        fitted = (sum(entries[s]["a_us_composite_for_prediction"] + entries[s]["b_us_per_element"] * n
                      for s in FITTED_STAGES) if set(FITTED_STAGES) <= set(entries) else None)
        chain.append({"S": size, "role_of_stage_sizes": "fit" if size in frozen["fit_protocol"]["fit_sizes"] else "holdout",
            "chain_measured_us": measured, "silu_plus_mul_measured_us": stage_sum,
            "delta_add_us": stage_sum - measured if stage_sum is not None and measured is not None else None,
            "delta_add_relative": (stage_sum - measured) / measured if stage_sum is not None and measured is not None else None,
            "fitted_sum_us": fitted,
            "fitted_sum_signed_error": (fitted - measured) / measured if fitted is not None and measured is not None else None,
            "analytic_chain_us": analytic_baseline(memory_bandwidth_Bps, size)["chain"]})
    updated = []
    for entry in fit_report["entries"]:
        row = copy.deepcopy(entry)
        own = [r for r in holdout if r["case"].startswith(entry["stage"] + "/")]
        row["holdout"] = [{k: r[k] for k in ("case", "predicted_us", "measured_us", "signed_error")} for r in own]
        row["status"] = "holdout_evaluated"
        row["validated_elements_range"] = None
        updated.append(row)
    copies = {c: {"median_us": median(c), "status": cases[c]["status"],
                  "statistics": cases[c]["statistics"]}
              for c, case in frozen["cases"].items() if case["kind"] == "copy"}
    return {"schema": SCHEMA, "scope": "holdout prediction accuracy first; search effects reported separately",
        "prediction_id_checked": fit_report["prediction_id"],
        "linear_fit": {"lines": lines, "holdout_rows": holdout, "metrics": fitted_metrics,
                       "ordering": _ordering(holdout)},
        "analytic_baseline_frozen_assumptions": {"rows": analytic, "metrics": analytic_metrics,
                                                 "ordering": _ordering(analytic), "refit": False},
        "chain_check": {"rows": chain, "role": "validation only; no chain discount is fitted",
                        "historical_S256_chain_us": 10.092799663543701},
        "copies": copies,
        "stage_medians_us": {c: median(c) for c, case in frozen["cases"].items() if case["kind"] == "stage"},
        "entries_after_holdout": updated,
        "holdout_error_threshold": frozen["fit_protocol"]["holdout_error_threshold"],
        "extrapolation_validated": False}


def _memory_bandwidth():
    from cost_model.local_profile_search import build_case
    _, measured, _ = build_case()
    return measured.base.keys.hardware.devices[0].mem_bw


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "validate"))
    parser.add_argument("--review", type=Path, required=True, help="review directory with evaluation.json")
    parser.add_argument("--plan", type=Path, required=True, help="collected plan.json")
    parser.add_argument("--fit", type=Path, help="fit.json for validate")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    evaluation, frozen = read(args.review / "evaluation.json"), read(args.plan)
    if hashlib.sha256(args.plan.read_bytes()).hexdigest() != evaluation["plan_sha256"]:
        parser.error("plan.json does not belong to this review")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.phase == "fit":
        report = fit(evaluation, frozen)
        name = "fit.json"
    else:
        if args.fit is None:
            parser.error("validate requires --fit")
        report = validate(read(args.fit), evaluation, frozen, _memory_bandwidth())
        name = "validation.json"
    (args.output / name).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"phase": args.phase, "output": str(args.output / name)}))


if __name__ == "__main__":
    main()
