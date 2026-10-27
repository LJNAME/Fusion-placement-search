"""Rebuild total-cost curves from raw checkpoints and verify label-completion clocks."""

import argparse
import json
from pathlib import Path
import statistics

from calibration.e2_manifest import content_id
from calibration.e2_records import load_jsonl_records
from .e2_diagnostics import _read_checkpoint, _write_json
from .e2_total_cost import task_summary, summarize_tasks, decision_timeline, cost_summary
from .e2_total_cost_evaluation import semantic_trace, SPECS
from .e2_total_cost import stopped_prefix
from .e2_profiling_cost import time_budget_summary


def main(argv=None):
    directory = Path(__file__).resolve().parents[1]/"data/bw1000_e2"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=directory/"e2c_total_cost.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))
    report = read(args.report)
    suite = read(directory/"e2c_task_suite.json")
    scoring = read(directory/"e2c_scoring_benchmark.json")
    records = load_jsonl_records(directory/"profiling_dataset.jsonl")
    by_id = {row["object_id"]: row for row in records}
    if (report["execution_status"] != "complete" or report["scope"] != "all_frozen_configs" or
            {row["task_id"] for row in report["tasks"]} != {row["task_id"] for row in suite["tasks"]} or
            report["configuration"]["dataset_id"] != content_id("diagnostics-data", records) or
            report["configuration"]["scoring_control_id"] != content_id("total-cost-scoring-control", scoring)):
        raise AssertionError("frozen inputs mismatch or partial run")
    root = directory/"diagnostics_checkpoints"/report["experiment_id"].split(":")[1][:16]
    repeats = report["configuration"]["repeats"]
    replays = observations = trajectory_points = 0
    initial_lists = initial_candidate_scores = 0
    clock_ablation = {mode: {} for mode in ("forced", "stopping")}
    for task in report["tasks"]:
        stored = _read_checkpoint(root/(task["task_id"].split(":")[1]+".json.gz"))
        if stored["compact"] != task:
            raise AssertionError("checkpoint differs from report")
        runs = stored["runs"]
        if len(runs) != repeats*len(SPECS):
            raise AssertionError("missing timing/seed runs")
        for repeat in range(repeats):
            if {(row["policy"], row["seed"]) for row in runs if row["repeat"] == repeat} != set(SPECS):
                raise AssertionError("unbalanced policy/seed coverage")
        for run in runs:
            trace = run["trace"]
            identity = next(row for row in task["timed_runs"] if row["policy"] == run["policy"] and
                            row["seed"] == run["seed"] and row["repeat"] == run["repeat"])
            if identity["semantic_id"] != content_id("total-cost-semantic", semantic_trace(trace)):
                raise AssertionError("trajectory semantics changed")
            for index, point in enumerate(trace):
                if point["budget"] != index or len(set(point["selected_object_ids"])) != index:
                    raise AssertionError("measurement prefix repeats or skips an object")
                measured = sum(by_id[key]["measurement"]["profiling_wall_time_s"] for key in point["selected_object_ids"])
                if point["profiling_wall_time_s"] != measured:
                    raise AssertionError("actual wall-cost lineage mismatch")
                trajectory_points += 1
            observations += trace[-1]["budget"]
            replays += 1
        if task["modes"] != task_summary(runs, task["reference_regret_ms"]):
            raise AssertionError("shared-initialization CPU/profiling cost clock or curve changed")
        for values in task["initial_batch_audit"].values():
            initial_lists += values["ranking_lists"]
            initial_candidate_scores += values["candidate_scores"]
        # Diagnostic axes do not replace the frozen primary four-second clock.
        pool_horizon = max(run["trace"][-1]["profiling_wall_time_s"] for run in runs)
        shared_initial = {run["repeat"]: run["trace"][0]["decision_ready_algorithm_wall_time_s"]
                          for run in runs if run["policy"] == "initial_only"}
        for mode in clock_ablation:
            for policy in sorted({run["policy"] for run in runs}):
                values = []
                for run in runs:
                    if run["policy"] != policy:
                        continue
                    trace = stopped_prefix(run["trace"], stopping=mode == "stopping")
                    regrets = [trace[min(budget, len(trace)-1)]["top1_regret_ms"] for budget in range(17)]
                    count_auc = sum((a+b)*.5 for a, b in zip(regrets, regrets[1:]))/16
                    profiling_auc = time_budget_summary(trace, pool_horizon)["step_regret_time_auc_ms"]
                    # Same primary 4s horizon and same initialization: remove only later CPU.
                    fee = shared_initial[run["repeat"]]
                    free_cpu_trace = [{**point, "decision_ready_algorithm_wall_time_s": fee,
                                       "algorithm_wall_time_s": fee} for point in trace]
                    free_cpu = cost_summary(decision_timeline(free_cpu_trace, run["preparation_s"], task["reference_regret_ms"]))
                    free_cpu_auc = next(row["step_auc_ms"] for row in free_cpu["curve"] if row["budget_s"] == 4.0)
                    values.append((count_auc, profiling_auc, free_cpu_auc))
                clock_ablation[mode].setdefault(policy, []).append(
                    tuple(statistics.fmean(value[index] for value in values) for index in range(3)))
    if summarize_tasks(report["tasks"]) != report["summary"]["all"]:
        raise AssertionError("summary mismatch")
    result = {"configs": len(report["tasks"]), "timed_replays": replays,
        "trajectory_points": trajectory_points, "observed_object_prefixes": observations,
        "initial_baseline_ranking_lists_checked_during_run": initial_lists,
        "initial_baseline_candidate_scores_checked_during_run": initial_candidate_scores,
        "curves_rebuilt_from_raw_clocks": True, "shared_initial_fees_reproduced": True,
        "recorded_profiling_wall_costs_reproduced": True,
        "diagnostic_clock_ablation": {mode: {policy: {
            "mean_count_auc_ms": statistics.fmean(value[0] for value in values),
            "mean_profiling_only_pool_horizon_auc_ms": statistics.fmean(value[1] for value in values),
            "same_init_zero_later_cpu_4s_auc_ms": statistics.fmean(value[2] for value in values)}
            for policy, values in policies.items()} for mode, policies in clock_ablation.items()},
        "clock_ablation_scope": "diagnostic only: pool-horizon profiling-only axis, plus same-initialization 4s axis with zero later CPU; neither replaces primary measured total cost"}
    if args.output:
        _write_json(args.output, result)
    print(json.dumps(result, indent=2))
    print("Fusion E2c total-cost verification: OK")


if __name__ == "__main__":
    main()
