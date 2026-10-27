"""Verify frozen-pool identity, first-step consistency and full-budget recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibration.e2_manifest import content_id
from .e2_diagnostics import _write_json


def verify(report, parent):
    if report["execution_status"] != "complete" or report["scope"] != "all_frozen_configs":
        raise AssertionError("full completed experiment required")
    if report["configuration"]["parent_report_id"] != content_id("budget-parent-report", parent):
        raise AssertionError("parent report identity differs")
    by_id = {task["task_id"]: task for task in parent["tasks"]}
    if {task["task_id"] for task in report["tasks"]} != set(by_id):
        raise AssertionError("frozen tasks differ")
    trajectories = first_steps = 0
    for task in report["tasks"]:
        previous = by_id[task["task_id"]]
        if (task["candidate_plans"] != previous["zero"]["candidate_plans"] or
                task["measurement_pool_objects"] != previous["zero"]["measurement_pool_objects"]):
            raise AssertionError("candidate or measurement pool changed")
        for name, policy in task["policies"].items():
            for trace in policy["traces"]:
                keys = trace["selected_object_ids"]
                if set(keys) != set(previous["zero"]["measurement_pool_object_ids"]) or len(keys) != len(set(keys)):
                    raise AssertionError("trajectory did not measure each pool object exactly once")
                if abs(trace["full_budget_regret_ms"]) > 1e-9 or trace["full_budget_strict_top_k_recall"] != 1.0:
                    raise AssertionError("full budget did not recover Oracle")
                trajectories += 1
            for mode in ("forced", "stopping"):
                curve = policy["modes"][mode]["curve"]
                if abs(curve[0]["top1_regret_ms"] - previous["zero"]["top1_regret_ms"]) > 1e-9:
                    raise AssertionError("initial ranking changed")
                if any(point["measurements_mean"] > min(point["requested_budget"], task["measurement_pool_objects"]) for point in curve):
                    raise AssertionError("measurement count exceeds budget or pool")
            if name != "random":
                selection = previous["one_step"]["policies"][name]
                for mode in ("forced", "stopping"):
                    expected = (previous["zero"]["top1_regret_ms"] if mode == "stopping" and selection["stop_recommended"]
                                else selection["selected_regret_ms"])
                    point = policy["modes"][mode]["curve"][1]
                    if abs(point["top1_regret_ms"] - expected) > 1e-9:
                        raise AssertionError("first-step result changed")
                    first_steps += 1
    return {"format": "fusion-placement-search.e2-budget-verification", "version": "1.0",
            "experiment_id": report["experiment_id"], "verified_configs": len(report["tasks"]),
            "full_budget_recovery_trajectories": trajectories, "first_step_consistency_checks": first_steps,
            "unchanged_frozen_pool": True, "status": "passed"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--parent", type=Path, default=Path("data/bw1000_e2/e2c_graph_wide_diagnostics.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = verify(json.loads(args.report.read_text(encoding="utf-8")), json.loads(args.parent.read_text(encoding="utf-8")))
    _write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
