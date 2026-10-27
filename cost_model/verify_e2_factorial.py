"""Check all frozen controls and genuine stopping for the new factorial cell."""

import argparse
import json
from pathlib import Path

from calibration.e2_manifest import content_id
from .e2_diagnostics import _read_checkpoint, _write_json
from .e2_factorial_evaluation import frozen_controls, interaction


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    directory = root / "data/bw1000_e2"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    intervals = json.loads((directory / "e2c_uncertainty_stopping.json").read_text(encoding="utf-8"))
    refit = json.loads((directory / "e2c_refit_evaluation.json").read_text(encoding="utf-8"))
    parent = json.loads((directory / "e2c_graph_wide_diagnostics.json").read_text(encoding="utf-8"))
    interval_by_id = {task["task_id"]: task for task in intervals["tasks"]}
    refit_by_id = {task["task_id"]: task for task in refit["tasks"]}
    parent_by_id = {task["task_id"]: task for task in parent["tasks"]}
    if (report["execution_status"] != "complete" or report["scope"] != "all_frozen_configs" or
            report["configuration"]["interval_control_report_id"] != content_id("factorial-interval-control", intervals) or
            report["configuration"]["refit_control_report_id"] != content_id("factorial-refit-control", refit) or
            {task["task_id"] for task in report["tasks"]} != set(interval_by_id)):
        raise ValueError("complete unchanged factorial experiment required")
    folder = directory / "diagnostics_checkpoints" / report["experiment_id"].split(":")[1][:16]
    checks = 0
    for task in report["tasks"]:
        controls = frozen_controls(interval_by_id[task["task_id"]], refit_by_id[task["task_id"]])
        for name, control in controls.items():
            if task["cells"][name] != control:
                raise AssertionError("frozen control was modified")
            checks += 1
        if interaction({name: cell["regret_budget_auc_ms"] for name, cell in task["cells"].items()}) != task["auc_interaction_ms"]:
            raise AssertionError("interaction differs from paired cells")
        stored = _read_checkpoint(folder / (task["task_id"].split(":")[1] + ".json.gz"))
        trajectory = stored["trajectory"]
        if any(point["stop_recommended"] for point in trajectory[:-1]):
            raise AssertionError("labels revealed after stopping")
        selected = trajectory[-1]["selected_object_ids"]
        pool = set(parent_by_id[task["task_id"]]["zero"]["measurement_pool_object_ids"])
        if len(selected) != len(set(selected)) or not set(selected) <= pool:
            raise AssertionError("measurement lineage differs")
        if not trajectory[-1]["stop_recommended"] and set(selected) != pool:
            raise AssertionError("trajectory ended without stop or exhaustion")
        if not task["cells"]["distance_refit"]["full_budget_recovers_all_candidates"]:
            raise AssertionError("separate full-reveal check failed")
    result = {"format": "fusion-placement-search.e2-factorial-verification", "version": "1.0",
        "experiment_id": report["experiment_id"], "verified_configs": len(report["tasks"]),
        "unchanged_control_checks": checks, "actual_stop_trajectory_checks": len(report["tasks"]),
        "separate_full_reveal_profile_checks": len(report["tasks"]), "status": "passed"}
    _write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
