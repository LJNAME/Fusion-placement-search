"""Check frozen decisions, audit counts and paired timing summaries."""

import argparse
import json
from pathlib import Path
import statistics

from calibration.e2_manifest import content_id
from .e2_diagnostics import _read_checkpoint, _write_json
from .e2_scoring_benchmark import semantic_trajectory, summarize


def main(argv=None):
    directory = Path(__file__).resolve().parents[1] / "data/bw1000_e2"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=directory / "e2c_scoring_benchmark.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))
    report = read(args.report)
    baseline = read(directory / "e2c_factorial_evaluation.json")
    suite = read(directory / "e2c_task_suite.json")
    contract = read(directory / "split_contract.json")
    config = report["configuration"]
    if (report["scope"] != "all_frozen_configs" or report["execution_status"] != "complete" or
            {task["task_id"] for task in report["tasks"]} != {task["task_id"] for task in suite["tasks"]} or
            config["baseline_report_id"] != content_id("scoring-baseline", baseline) or
            config["suite_id"] != suite["suite_id"]):
        raise AssertionError("incomplete or mismatched frozen report")
    root = directory / "diagnostics_checkpoints" / baseline["experiment_id"].split(":")[1][:16]
    controls = {row["task_id"]: row for row in baseline["tasks"]}
    baseline_trace_checks = timed_runs = 0
    for task in report["tasks"]:
        trace = _read_checkpoint(root / (task["task_id"].split(":")[1] + ".json.gz"))["trajectory"]
        semantic = semantic_trajectory(trace)
        if (task["semantic_trajectory"] != semantic or
                task["semantic_trajectory_id"] != content_id("scoring-semantics", semantic) or
                task["frozen_count_auc_ms"] != controls[task["task_id"]]["cells"]["distance_refit"]["regret_budget_auc_ms"]):
            raise AssertionError("frozen decisions, metrics or AUC changed")
        pool_size = task["stop_budget"] + controls[task["task_id"]]["cells"]["distance_refit"]["remaining_objects"]
        quantiles = len(contract["active_profiling_protocol"]["srop_proxy_standard_normal_quantiles"])
        expected_lists = sum(1 + (pool_size-point["budget"])*quantiles for point in semantic)
        if (task["ranking_lists_checked"] != expected_lists or
                task["candidate_scores_checked"] != expected_lists*task["candidate_plans"]):
            raise AssertionError("audit did not cover every hypothetical full ranking")
        for repeat in range(config["repeats_per_variant"]):
            runs = [row for row in task["timed_runs"] if row["repeat"] == repeat]
            if len(runs) != 2 or {row["variant"] for row in runs} != {"scalar_scenarios", "scenario_batch"}:
                raise AssertionError("paired CPU timing missing")
            if runs[0]["order"] != runs[1]["order"] or [row["variant"] for row in runs] != runs[0]["order"]:
                raise AssertionError("paired order inconsistent")
            for row in runs:
                components = sum(row[key] for key in ("predictor_fit_wall_time_s", "ranking_wall_time_s", "selection_wall_time_s"))
                if abs(components-row["algorithm_wall_time_s"]) > 1e-12 or components <= 0:
                    raise AssertionError("timing components inconsistent")
            timed_runs += 2
        baseline_trace_checks += len(semantic)
    if report["summary"]["all"] != summarize(report["tasks"]):
        raise AssertionError("timing summary changed")
    result = {"configs": len(report["tasks"]), "frozen_trajectory_points": baseline_trace_checks,
        "full_ranking_lists_checked_during_run": report["summary"]["all"]["ranking_lists_checked"],
        "candidate_scores_checked_during_run": report["summary"]["all"]["candidate_scores_checked"],
        "paired_timed_replays": timed_runs, "all_decisions_scores_and_stops_unchanged": True,
        "mean_preparation_plus_replay_s": {variant: statistics.fmean(statistics.median(
            run["preparation_wall_time_s"]+run["run_wall_time_s"] for run in task["timed_runs"] if run["variant"] == variant)
            for task in report["tasks"]) for variant in ("scalar_scenarios", "scenario_batch")},
        "audit_scope": "full rankings compared exactly inside the untimed worker; here recheck coverage counts and frozen traces"}
    if args.output:
        _write_json(args.output, result)
    print(json.dumps(result, indent=2))
    print("Fusion E2c scenario scoring verification: OK")


if __name__ == "__main__":
    main()
