"""Audit wall-cost lineage, score denominators and the common cost-budget axis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibration.e2_dataset import make_dataset_view
from calibration.e2_manifest import content_id
from calibration.e2_records import load_jsonl_records
from .e2_cost_evaluation import evaluate_cost_predictions
from .e2_cost_normalization import CELLS, summarize
from .e2_diagnostics import _read_checkpoint, _write_json, measurement_pool, workload_for_task
from .e2_profiling_cost import ProfilingCostPredictor, time_budget_summary
from .e2_srop_proxy import build_workload_graph, object_map_for_workload
from .e2_task_suite import active_task_contract
from .e2_uncertainty_stopping import summarize_trajectory


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    directory = root / "data/bw1000_e2"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=directory / "e2c_cost_normalization.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))
    report = read(args.report)
    baseline = read(directory / "e2c_factorial_evaluation.json")
    manifest = read(directory / "shape_manifest.json")
    contract = read(directory / "split_contract.json")
    suite = read(directory / "e2c_task_suite.json")
    records = load_jsonl_records(directory / "profiling_dataset.jsonl")
    if (report["scope"] != "all_frozen_configs" or report["execution_status"] != "complete" or
            {row["task_id"] for row in report["tasks"]} != {task["task_id"] for task in suite["tasks"]} or
            report["configuration"]["baseline_report_id"] != content_id("cost-baseline", baseline) or
            report["configuration"]["dataset_id"] != content_id("diagnostics-data", records)):
        raise AssertionError("incomplete or mismatched experiment inputs")
    if report["prediction_evaluation"] != evaluate_cost_predictions(records, manifest, contract, suite):
        raise AssertionError("training-only prediction metrics changed")
    checkpoints = directory / "diagnostics_checkpoints"
    new_root = checkpoints / report["experiment_id"].split(":")[1][:16]
    old_root = checkpoints / baseline["experiment_id"].split(":")[1][:16]
    task_by_id = {task["task_id"]: task for task in suite["tasks"]}
    old_by_id = {task["task_id"]: task for task in baseline["tasks"]}
    records_by_id = {row["object_id"]: row for row in records}
    score_checks = prefix_points = 0
    for row in report["tasks"]:
        task = task_by_id[row["task_id"]]
        runtime = active_task_contract(task, contract, manifest)
        workload = workload_for_task(task, manifest)
        pool = measurement_pool(build_workload_graph(workload, task["scale"]),
            object_map_for_workload(records, workload), task["train_object_ids"])
        horizon = sum(records_by_id[key]["measurement"]["profiling_wall_time_s"] for key in pool)
        filename = row["task_id"].split(":")[1] + ".json.gz"
        old = _read_checkpoint(old_root / filename)["trajectory"]
        new = _read_checkpoint(new_root / filename)["trajectory"]
        for name, trace in zip(CELLS, (old, new)):
            expected = summarize_trajectory(trace, len(pool))
            expected["actual_cost_budget"] = time_budget_summary(trace, horizon)
            if row["cells"][name] != expected:
                raise AssertionError("stopped trajectory or actual-cost curve mismatch")
            for point in trace:
                actual = sum(records_by_id[key]["measurement"]["profiling_wall_time_s"]
                             for key in point["selected_object_ids"])
                if point["profiling_wall_time_s"] != actual:
                    raise AssertionError("evaluation cost is not the sum of observed wall costs")
                prefix_points += 1
        frozen = dict(row["cells"][CELLS[0]])
        frozen.pop("actual_cost_budget")
        if frozen != old_by_id[row["task_id"]]["cells"]["distance_refit"]:
            raise AssertionError("control changed")
        for point in new:
            if point["next_selection"] is None:
                continue
            public = make_dataset_view(records, runtime, split_id=task["split_id"],
                mode="active_calibration", role="sampler", revealed_object_ids=point["selected_object_ids"])
            model = ProfilingCostPredictor().fit(public)
            selected = next(item for item in public if item["object_id"] == point["next_selection"])
            cost = model.predict(selected)
            if point["next_cost_proxy_s"] != cost or point["next_selection_score"] != point["next_expected_decision_regret_ms"]/cost:
                raise AssertionError("acquisition denominator differs from its public-only prediction")
            if set(model.train_object_ids) != set(task["train_object_ids"]) | set(point["selected_object_ids"]):
                raise AssertionError("cost model read a hidden label or missed a visible label")
            score_checks += 1
    if summarize(report["tasks"]) != report["summary"]["all"]:
        raise AssertionError("summary mismatch")
    result = {"configs": len(report["tasks"]), "unchanged_controls": len(report["tasks"]),
        "public_only_score_denominators": score_checks, "actual_cost_prefixes": prefix_points,
        "prediction_evaluation_reproduced": True, "common_time_curves_reproduced": True}
    if args.output:
        _write_json(args.output, result)
    print(json.dumps(result, indent=2))
    print("Fusion E2c profiling-cost verification: OK")


if __name__ == "__main__":
    main()
