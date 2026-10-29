"""Audit sealed GPU validation results without fitting or changing decisions."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from calibration.e2_records import load_environment_registry, load_jsonl_records
from calibration.evaluate_gpu_holdout import evaluate_measurements
from calibration.reference_validation import DEFAULT_BUNDLE, check
from cost_model.e2_residual import _spearman
from execution.audit_gpu_holdout import audit_execution, write_json
from execution.evaluate_reference_validation import evaluate


def compare_report(actual, stored, path="", roundoff=None):
    """Require identical JSON structure, allowing only float roundoff."""
    if roundoff is None:
        roundoff = []
    if isinstance(actual, dict) and isinstance(stored, dict):
        if actual.keys() != stored.keys():
            raise ValueError(f"report keys differ at {path}")
        for key in actual:
            compare_report(actual[key], stored[key], f"{path}/{key}", roundoff)
    elif isinstance(actual, list) and isinstance(stored, list):
        if len(actual) != len(stored):
            raise ValueError(f"report lengths differ at {path}")
        for index, (left, right) in enumerate(zip(actual, stored)):
            compare_report(left, right, f"{path}/{index}", roundoff)
    elif type(actual) is not type(stored):
        raise ValueError(f"report types differ at {path}")
    elif actual != stored:
        if (not isinstance(actual, float) or
                not math.isclose(actual, stored, rel_tol=1e-12, abs_tol=1e-15)):
            raise ValueError(f"report values differ at {path}: {actual!r} vs {stored!r}")
        roundoff.append({"path": path, "recomputed": actual, "stored": stored,
                         "absolute_delta": abs(actual - stored)})
    return roundoff


def analyze(bundle, results):
    freeze, manifest = check(bundle)
    read = lambda name: json.loads((results / name).read_text(encoding="utf-8"))
    kernel_records = load_jsonl_records(results / "raw_measurements.jsonl")
    environments = load_environment_registry(results / "environments.json")
    kernels = evaluate_measurements(freeze, manifest, kernel_records, environments)
    # JSON converts integer device keys to strings on disk.
    kernels = json.loads(json.dumps(kernels))
    kernel_roundoff = compare_report(kernels, read("zero_shot_report.json"))
    records = load_jsonl_records(results / "execution_samples.jsonl")
    original, environment = read("execution_report.json"), read("execution_environment.json")
    audited = audit_execution(freeze, records, environment, original)
    epsilon = evaluate(freeze, records, environment, original, kernels)
    epsilon_roundoff = compare_report(epsilon, read("epsilon_report.json"))
    expected_samples = freeze["protocol"]["execution"]["iterations"]
    if any(len(row["samples_s"]) != expected_samples for row in records):
        raise ValueError("execution block length differs from sealed protocol")
    actual_by_model = {row["model_id"]: row for row in original["workloads"]}
    audit_by_model = {row["model_id"]: row for row in audited["workloads"]}
    workloads = []
    for workload, decision in zip(freeze["workloads"], epsilon["workloads"]):
        model = workload["model_id"]
        actual = {row["plan_id"]: row["median_s"]
                  for row in actual_by_model[model]["candidates"]}
        best = min(actual.values())
        used_devices = {key: sorted({group["device"] for group in
                       workload["candidate_plans"][key]["groups"]}) for key in actual}
        single_ids = [key for key in actual if len(used_devices[key]) == 1]
        cross_ids = [key for key in actual if len(used_devices[key]) > 1]
        methods = {}
        for method, selection in decision["methods"].items():
            predicted = {row["plan_id"]: row["makespan_s"]
                         for row in workload["method_rankings"][method]}
            key = selection["preselected_physical_id"]
            methods[method] = {**selection, "selected_devices": used_devices[key],
                "predicted_ms": predicted[key] * 1000,
                "selected_absolute_relative_error": abs(predicted[key] / actual[key] - 1),
                "measured_subset_mape": statistics.fmean(
                    abs(predicted[pid] / measured - 1) for pid, measured in actual.items()),
                "measured_subset_spearman": _spearman(
                    list(actual.values()), [predicted[pid] for pid in actual])}
        audit = audit_by_model[model]
        workloads.append({"model_id": model, "measured_candidates": len(actual),
            "best_measured_ms": best * 1000,
            "best_measured_plan_is_single_gpu": any(actual[key] == best for key in single_ids),
            "best_cross_gpu_regret": min(actual[key] / best - 1 for key in cross_ids) if cross_ids else None,
            "single_gpu_controls": [{"plan_id": key, "physical_devices": used_devices[key],
                "measured_ms": actual[key] * 1000, "relative_regret": actual[key] / best - 1,
                "is_frozen_reference": key == workload["reference_plan_id"]}
                for key in sorted(single_ids)],
            "methods": methods,
            "max_absolute_candidate_round_change_pct": audit["max_absolute_round_change_pct"],
            "long_tail_samples_above_twice_candidate_median": len(audit["long_tail_samples"])})
    names = list(workloads[0]["methods"])
    aggregate = {}
    for method in names:
        selected = [row["methods"][method] for row in workloads]
        rounds = [value for row in selected for value in row["relative_regret_by_round"]]
        aggregate[method] = {"pooled_tasks_within_1pct": sum(row["relative_regret"] <= .01 for row in selected),
            "task_count": len(selected), "rounds_within_1pct": sum(value <= .01 for value in rounds),
            "round_count": len(rounds), "max_round_relative_regret": max(rounds),
            "pooled_success_counts": {str(e): sum(row["relative_regret"] <= e for row in selected)
                                      for e in (.001, .005, .01)}}
    reference_regrets = [next(c["relative_regret"] for c in row["single_gpu_controls"]
                        if c["is_frozen_reference"]) for row in workloads]
    inputs = ["environments.json", "execution_samples.jsonl", "raw_measurements.jsonl",
              "zero_shot_report.json", "execution_environment.json", "epsilon_report.json", "execution_report.json"]
    return {"format": "fusion-placement-search.reference-validation-audit",
        "freeze_id": freeze["freeze_id"],
        "input_sha256": {name: hashlib.sha256((results / name).read_bytes()).hexdigest() for name in inputs},
        "checks": {**audited["checks"], "frozen_sources_and_inputs": True,
            "kernel_raw_lineage_and_coverage": True, "kernel_report_reproduced_with_roundoff": True,
            "epsilon_report_reproduced_with_roundoff": True, "execution_block_lengths": True},
        "report_float_tolerance": {"relative": 1e-12, "absolute": 1e-15},
        "kernel_report_float_roundoff": kernel_roundoff, "epsilon_report_float_roundoff": epsilon_roundoff,
        "kernel_records": len(kernel_records), "kernel_quality": kernels["quality"],
        "kernel_quality_target_pass": kernels["quality_target_pass"],
        "kernel_groups": epsilon["kernel_groups"],
        "max_kernel_replica_ratio": max(row["replica_ratio_max_min"] for row in kernels["objects"]),
        "execution_records": len(records), "execution_timing_samples": audited["timing_samples"],
        "correctness_pass": audited["correctness_pass"],
        "max_output_absolute_error": max(row["correctness"]["max_abs_error"] for row in records),
        "execution_samples_above_twice_candidate_median": audited["samples_above_twice_candidate_median"],
        "aggregate": aggregate,
        "fixed_single_gpu_reference": {"physical_device": 0, "relative_regrets": reference_regrets,
            "pooled_tasks_within_1pct": sum(value <= .01 for value in reference_regrets)},
        "workloads": workloads,
        "limitations": epsilon["limitations"] + [
            "New sequence lengths within two familiar model families; not unseen architecture or cross-GPU generalization.",
            "Pooled epsilon success is descriptive, not a per-round or statistical regret bound; only two blocks per candidate.",
            "Fixed single-GPU reference is reported separately; beating structural DAG is not evidence of superiority over that simple baseline.",
            "Common candidates came from structural search and public probes, not a new reference-model-guided search.",
            "These labels are now exposed and cannot be reused as a fresh blind test after tuning.",
            "All samples retained, including long tails; no post-label fitting, candidate selection or threshold changes."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.bundle, args.results)
    write_json(args.output, report)
    print("freeze:", report["freeze_id"])
    print("records:", report["kernel_records"], report["execution_records"],
          "execution_samples:", report["execution_timing_samples"])
    for name, row in report["aggregate"].items():
        print(name, json.dumps(row))
    print("report:", args.output)


if __name__ == "__main__":
    main()
