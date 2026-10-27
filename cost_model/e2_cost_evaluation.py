"""Evaluate predeclared wall-cost estimates without revealing hidden wall labels."""

from __future__ import annotations

import math
import statistics

from calibration.e2_dataset import make_dataset_view
from calibration.e2_manifest import content_id
from .e2_diagnostics import measurement_pool, workload_for_task
from .e2_profiling_cost import ProfilingCostPredictor, VARIANTS
from .e2_residual import _spearman
from .e2_srop_proxy import build_workload_graph, object_map_for_workload
from .e2_task_suite import active_task_contract


def prediction_metrics(rows):
    result = {}
    for variant in VARIANTS:
        actual = [row["actual_wall_s"] for row in rows]
        predicted = [row["predictions_s"][variant] for row in rows]
        log_errors = [abs(math.log(p/a)) for p, a in zip(predicted, actual)]
        result[variant] = {"objects": len(rows),
            "mape": statistics.fmean(abs(p-a)/a for p, a in zip(predicted, actual)),
            "mean_abs_log_error": statistics.fmean(log_errors),
            "median_error_factor": math.exp(statistics.median(log_errors)),
            "spearman": _spearman(actual, predicted),
            "underestimated_over_2x": sum(p*2 < a for p, a in zip(predicted, actual)),
            "overestimated_over_2x": sum(p > a*2 for p, a in zip(predicted, actual))}
    return result


def evaluate_cost_predictions(records, manifest, contract, suite):
    oracle = {row["object_id"]: row for row in records}
    states = {}
    repetitions = contract["active_profiling_protocol"]["profiling_repetitions"]

    def evaluate(runtime, split_id, mode, object_ids):
        public = make_dataset_view(records, runtime, split_id=split_id, mode=mode, role="sampler")
        predictor = ProfilingCostPredictor().fit(public, repetitions=repetitions)
        state_id = content_id("profiling-cost-state", sorted(predictor.train_object_ids))
        states[state_id] = {"train_object_ids": sorted(predictor.train_object_ids),
            "global_median_s": predictor.global_median_s, "family_medians_s": predictor.family_medians_s}
        by_id = {row["object_id"]: row for row in public}
        rows = []
        for key in sorted(object_ids):
            row = by_id[key]
            if "measurement" in row or "target" in row:
                raise AssertionError("held-out cost labels are visible to the predictor")
            rows.append({"training_state_id": state_id, "object_id": key,
                "family": row["features"]["categorical"]["operator"],
                "predictions_s": {variant: predictor.predict(row, variant) for variant in VARIANTS},
                "actual_wall_s": oracle[key]["measurement"]["profiling_wall_time_s"]})
        return rows

    zero_shot = []
    for split in contract["splits"]:
        rows = evaluate(contract, split["split_id"], "zero_shot", split["hidden_object_ids"])
        zero_shot.append({"split_id": split["split_id"], "metrics": prediction_metrics(rows), "objects": rows})
    unique = {}
    for task in suite["tasks"]:
        runtime = active_task_contract(task, contract, manifest)
        workload = workload_for_task(task, manifest)
        graph = build_workload_graph(workload, task["scale"])
        pool = measurement_pool(graph, object_map_for_workload(records, workload), task["train_object_ids"])
        for row in evaluate(runtime, task["split_id"], "active_calibration", pool):
            identity = (row["training_state_id"], row["object_id"])
            if identity in unique and unique[identity] != row:
                raise AssertionError("duplicate state-object predictions disagree")
            unique[identity] = row
    rows = list(unique.values())
    return {"zero_shot": zero_shot, "training_states": states,
        "active_initial": {"configs": len(suite["tasks"]), "unique_state_objects": len(rows),
            "metrics": prediction_metrics(rows), "objects": rows,
            "by_family": {family: prediction_metrics([row for row in rows if row["family"] == family])
                          for family in sorted({row["family"] for row in rows})}}}
