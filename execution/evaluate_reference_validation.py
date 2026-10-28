"""Evaluate preregistered physical decisions and epsilon loss without reranking."""

import argparse
import json
from pathlib import Path

from calibration.e2_records import load_jsonl_records
from calibration.evaluate_gpu_holdout import prediction_metrics
from calibration.reference_validation import DEFAULT_BUNDLE, check
from execution.audit_gpu_holdout import audit_execution, write_json


def evaluate(freeze, records, environment, original, kernel_report=None):
    audited = audit_execution(freeze, records, environment, original)
    by_model = {r["model_id"]: r for r in original["workloads"]}
    workloads = []
    for workload in freeze["workloads"]:
        measured = {row["plan_id"]: row["median_s"] for row in by_model[workload["model_id"]]["candidates"]}
        best = min(measured.values())
        methods = {}
        for method, selected in workload["method_selected_ids"].items():
            ties = workload["method_top1_sets"][method]
            regret = measured[selected] / best - 1
            tie_regrets = [measured[key] / best - 1 for key in ties]
            selected_class = workload["physical_to_class"][selected]
            class_members = [key for key in measured if workload["physical_to_class"][key] == selected_class]
            round_regrets = []
            for repeat in range(freeze["protocol"]["execution"]["repeats"]):
                times = {row["plan_id"]: row["median_s"] for row in records
                         if row["model_id"] == workload["model_id"] and row["repeat"] == repeat and row["status"] == "ok"}
                round_regrets.append(times[selected] / min(times.values()) - 1)
            methods[method] = {"preselected_physical_id": selected, "measured_ms": measured[selected] * 1000,
                "relative_regret": regret, "epsilon_success": {str(e): regret <= e for e in (.001, .005, .01)},
                "relative_regret_by_round": round_regrets,
                "predicted_tie_ids": ties, "tie_regret_range": [min(tie_regrets), max(tie_regrets)],
                "selected_class_best_member_regret_diagnostic": min(measured[key] / best - 1 for key in class_members),
                "class_diagnostic_is_not_deployed_selection": True}
        workloads.append({"model_id": workload["model_id"], "measured_candidates": len(measured),
            "best_measured_ms": best * 1000, "methods": methods})
    kernel_groups = None
    if kernel_report is not None:
        if kernel_report["freeze_id"] != freeze["freeze_id"]:
            raise ValueError("kernel report belongs to a different freeze")
        aliases = {row["object_id"] for row in freeze["feature_alias_audit"]["aliases"]}
        kernel_groups = {"all": prediction_metrics(kernel_report["objects"]),
            "feature_novel": prediction_metrics([row for row in kernel_report["objects"] if row["object_id"] not in aliases]),
            "feature_alias_object_ids": sorted(aliases)}
    return {"format": "fusion-placement-search.reference-epsilon-validation", "freeze_id": freeze["freeze_id"],
        "execution_audit": {"checks": audited["checks"], "records": audited["records"], "timing_samples": audited["timing_samples"],
                            "correctness_pass": audited["correctness_pass"]},
        "primary_method": "reference_host", "primary_epsilon": .01,
        "all_primary_within_1pct": all(w["methods"]["reference_host"]["relative_regret"] <= .01 for w in workloads),
        "workloads": workloads, "kernel_groups": kernel_groups,
        "limitations": ["Best is limited to measured sealed candidates, not global placement optimum.",
            "All physical choices were fixed before target labels; tie best/worst and class-best values are diagnostics, never label-based selection.",
            "No host-profile acceptance promotion, active-calibration or SROP cost-benefit/stopping claim.",
            "Experimental unaccepted host profile and unverified operator-body coverage are frozen limitations."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    freeze, _ = check(args.bundle)
    read = lambda name: json.loads((args.results / name).read_text(encoding="utf-8"))
    report = evaluate(freeze, load_jsonl_records(args.results / "execution_samples.jsonl"),
        read("execution_environment.json"), read("execution_report.json"), read("zero_shot_report.json"))
    write_json(args.results / "epsilon_report.json", report)
    for row in report["workloads"]:
        print(row["model_id"], {name: r["relative_regret"] for name, r in row["methods"].items()})
    print("all_primary_within_1pct:", report["all_primary_within_1pct"])


if __name__ == "__main__":
    main()
