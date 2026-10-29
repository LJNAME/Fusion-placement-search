"""Reproduce received A3 journals and describe the measured decision space."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from calibration.decoder_competition import check, DEFAULT_BUNDLE
from calibration.e2_records import load_jsonl_records
from cost_model.e2_residual import _spearman
from execution.analyze_reference_validation import compare_report
from execution.audit_gpu_holdout import write_json
from execution.decoder_competition import summarize


def analyze(bundle, results):
    frozen = check(bundle)
    read = lambda name: json.loads((results / name).read_text(encoding="utf-8"))
    original, environment = read("competition_report.json"), read("environment.json")
    records = load_jsonl_records(results / "execution_samples.jsonl")
    reproduced = summarize(frozen, records, environment)
    roundoff = compare_report(reproduced, original)
    if not reproduced["complete"]:
        raise ValueError("incomplete A3 collection; no performance conclusion")
    protocol = frozen["protocol"]
    tasks = {w["task_id"]: w for w in frozen["workloads"]}
    for row in records:
        accuracy = row["correctness"]
        task = tasks[row["task_id"]]
        output = next(n for n in task["graph"]["nodes"] if n["id"] == task["graph"]["extra"]["output_node"])
        if (not accuracy["shape_match"] or accuracy["output_shape"] != list(output["extra"]["output_shape"]) or
                accuracy["rtol"] != protocol["rtol"] or accuracy["atol"] != protocol["atol"]):
            raise ValueError("A3 output shape or tolerance differs from sealed contract")
        if (any(not math.isfinite(accuracy[key]) or accuracy[key] < 0 for key in ("max_abs_error", "mean_abs_error")) or
                accuracy["mean_abs_error"] > accuracy["max_abs_error"]):
            raise ValueError("invalid numerical error summary")
        if not math.isfinite(row["block_wall_s"]) or row["block_wall_s"] < sum(row["samples_s"]):
            raise ValueError("block wall time is shorter than its timed forwards")
        if len(row["peak_memory_bytes"]) != 2 or any(value < 0 for value in row["peak_memory_bytes"]):
            raise ValueError("invalid peak-memory summary")
    workloads = []
    for task_report in reproduced["workloads"]:
        task = tasks[task_report["task_id"]]
        candidates = {c["plan_id"]: c for c in task_report["candidates"]}
        uniform = task["uniform_plan_ids"]
        cross = [c for key, c in candidates.items() if key not in uniform]
        best_cross = min(cross, key=lambda c: (c["median_s"], c["plan_id"]))
        actual = [c["median_s"] for c in candidates.values()]
        predicted = [task["initial_scores_s"][key] for key in candidates]
        workloads.append({"task_id": task["task_id"], "physical_candidates": len(candidates),
            "best_measured_ms": task_report["measured_subset_best_s"] * 1000,
            "fixed_single_gpu_ms": [candidates[key]["median_s"] * 1000 for key in uniform],
            "best_measured_is_single_gpu": any(candidates[k]["relative_regret"] == 0 for k in uniform),
            "best_cross_gpu_plan_id": best_cross["plan_id"], "best_cross_gpu_ms": best_cross["median_s"] * 1000,
            "best_cross_gpu_selection_reasons": task["selection_reasons"][best_cross["plan_id"]],
            "best_cross_gpu_relative_regret": best_cross["relative_regret"],
            "best_cross_gpu_relative_to_window_single": best_cross["relative_to_window_best_single"],
            "initial_selected_id": task["initial_selected_id"],
            "initial_relative_regret": task_report["initial_relative_regret"],
            "initial_relative_regret_by_window": task_report["initial_relative_regret_by_window"],
            "fixed_single_gpu_regrets": task_report["fixed_single_gpu_regrets"],
            "quality_windows": task_report["quality_windows"],
            "max_absolute_anchor_drift": max(abs(v) for w in task_report["windows"] for v in w["anchor_relative_drift"].values()),
            "cross_gpu_candidates": len(cross),
            "repeat_consistent_cross_gpu_advantages": task_report["repeat_consistent_cross_gpu_advantages"],
            "repeat_consistent_slower_cross_gpu_candidates": sum(c["repeat_consistent_difference_beyond_noise"] and
                all(v > 0 for v in c["relative_to_window_best_single"]) for c in cross),
            "initial_model_plan_mape": statistics.fmean(abs(p / a - 1) for p, a in zip(predicted, actual)),
            "initial_model_plan_spearman": _spearman(predicted, actual),
            "kernel_profiling_value": "not assessed"})
    window_regrets = [v for w in workloads for v in w["initial_relative_regret_by_window"]]
    names = ("competition_report.json", "environment.json", "execution_samples.jsonl")
    return {"format": "fusion-placement-search.a3-received-analysis", "freeze_id": frozen["freeze_id"],
        "input_sha256": {name: hashlib.sha256((results / name).read_bytes()).hexdigest() for name in names},
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checks": {"sealed_sources_and_inputs": True, "raw_record_and_environment_hashes": True,
            "exact_expected_slots": True, "formal_sample_counts_and_medians": True,
            "original_report_reproduced": True, "shape_tolerance_and_numerical_summaries": True,
            "block_wall_and_peak_memory": True},
        "report_float_roundoff": roundoff, "report_exactly_reproduced": reproduced == original,
        "environment": environment,
        "summary": {"tasks": len(workloads), "physical_candidates": sum(w["physical_candidates"] for w in workloads),
            "records": len(records), "formal_forward_samples": sum(len(r["samples_s"]) for r in records),
            "correctness_pass_blocks": sum(r["correctness"]["pass"] for r in records),
            "max_output_absolute_error": max(r["correctness"]["max_abs_error"] for r in records),
            "quality_windows": sum(w["quality_windows"] for w in workloads),
            "collection_processes": len({r["collection_run_id"] for r in records}),
            "max_absolute_anchor_drift": max(w["max_absolute_anchor_drift"] for w in workloads),
            "samples_above_twice_block_median": sum(v > 2 * r["median_s"] for r in records for v in r["samples_s"]),
            "initial_epsilon_success_counts": {str(e): sum(w["initial_relative_regret"] <= e for w in workloads)
                                               for e in protocol["epsilon_values"]},
            "initial_windows_within_primary_epsilon": sum(v <= protocol["primary_epsilon"] for v in window_regrets),
            "initial_window_count": len(window_regrets), "max_initial_window_regret": max(window_regrets),
            "fixed_gpu0_tasks_within_primary_epsilon": sum(w["fixed_single_gpu_regrets"]["0"] <= protocol["primary_epsilon"] for w in workloads),
            "fixed_gpu1_tasks_within_primary_epsilon": sum(w["fixed_single_gpu_regrets"]["1"] <= protocol["primary_epsilon"] for w in workloads),
            "tasks_with_repeat_consistent_cross_gpu_advantage": sum(w["repeat_consistent_cross_gpu_advantages"] > 0 for w in workloads)},
        "workloads": workloads,
        "limitations": ["All raw timings retained, including tails. Repeat-consistency and drift flags are descriptive, not confidence intervals.",
            "Only four development tasks and 120 measured A2 members, not all 912 candidates or global placement optimum.",
            "Seq256 operator Keys were already in original112 training; this is not unseen-shape validation.",
            "No candidate latency from this collection was used to adjust the sealed selection, initial scores, epsilon or order.",
            "No new kernel labels, active profiling, refit/re-search or SROP total-cost benefit was evaluated.",
            "Synthetic forward decoder bodies on homogeneous BW cards, without real fused kernels, backward or downstream pipeline scheduling.",
            "Unaccepted host profile and operator-body/layout coverage limitations remain unchanged."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.bundle, args.results)
    write_json(args.results / "audit_report.json", report)
    print(json.dumps(report["summary"]))
    for row in report["workloads"]:
        print(row["task_id"], "initial_regret_pct=", row["initial_relative_regret"] * 100,
              "best_cross_regret_pct=", row["best_cross_gpu_relative_regret"] * 100)


if __name__ == "__main__":
    main()
