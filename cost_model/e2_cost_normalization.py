"""Wall-cost normalization ablation on frozen distance-plus-refit development pools."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
from pathlib import Path
import statistics

from calibration.e2_manifest import content_id
from calibration.e2_records import load_jsonl_records
from simulator import default_fusion_registry
from simulator.dataset import plan_from_dict
from simulator.hardware import hardware_to_dict, load_hardware
from .batched_placement_ranker import BatchedPlacementRanker
from .e2_audit import ranking_metrics
from .e2_cost_evaluation import evaluate_cost_predictions
from .e2_diagnostics import _read_checkpoint, _write_checkpoint, _write_json, measurement_pool, workload_for_task
from .e2_profiling_cost import PROTOCOL, time_budget_summary
from .e2_srop_proxy import _active_profile, _oracle_profile, _run_policy_trajectory, build_workload_graph, object_map_for_workload
from .e2_task_suite import active_task_contract, build_task_suite
from .e2_uncertainty_stopping import summarize_trajectory


_CONTEXT = None
CELLS = ("analytic_repetitions", "family_wall_median")


def _initialize(context):
    global _CONTEXT
    _CONTEXT = context


def _run_task(task):
    manifest, contract, records, hardware, candidate_root, baseline_root, controls = _CONTEXT
    filename = task["task_id"].split(":")[1] + ".json.gz"
    stored = _read_checkpoint(candidate_root / filename)
    baseline = _read_checkpoint(baseline_root / filename)
    plans = [plan_from_dict(plan) for plan in stored["candidate_plans"]]
    workload = workload_for_task(task, manifest)
    graph = build_workload_graph(workload, task["scale"])
    mapping = object_map_for_workload(records, workload)
    runtime = active_task_contract(task, contract, manifest)
    runtime["active_profiling_protocol"].update(uncertainty_variant="distance_floor",
        profiling_cost_variant="family_wall_median")
    pool = measurement_pool(graph, mapping, task["train_object_ids"])
    ranker = BatchedPlacementRanker(graph, plans, 2, ((0, 1), (1, 0)))
    oracle = ranker.rank(_oracle_profile(records, hardware, mapping))
    control = controls[task["task_id"]]
    if summarize_trajectory(baseline["trajectory"], len(pool)) != control["cells"]["distance_refit"]:
        raise ValueError("frozen baseline trajectory disagrees with published compact results")
    trajectory = _run_policy_trajectory("srop_refit", graph, plans, records, runtime, hardware,
        mapping, pool, oracle, split_id=task["split_id"], random_seed=None,
        registry=default_fusion_registry(), exchange_permutations=((0, 1), (1, 0)), ranker=ranker,
        stop_on_recommendation=True)
    if abs(trajectory[0]["top1_regret_ms"] - control["initial_regret_ms"]) > 1e-9:
        raise AssertionError("cost normalization changed initial mean predictions")
    full, _ = _active_profile(records, runtime, hardware, mapping,
                             split_id=task["split_id"], revealed_object_ids=pool)
    ranked = ranker.rank(full)
    oracle_costs = {item.plan_id: item.makespan for item in oracle}
    if max(abs(item.makespan-oracle_costs[item.plan_id]) for item in ranked) > 1e-12 or ranking_metrics(oracle, ranked)["strict_top_k_recall"] != 1.0:
        raise AssertionError("full-reveal predictions do not recover Oracle candidate costs")
    by_id = {row["object_id"]: row for row in records}
    # Hidden costs are used by the evaluator here, never by acquisition.
    horizon = sum(by_id[key]["measurement"]["profiling_wall_time_s"] for key in pool)
    cells = {"analytic_repetitions": copy.deepcopy(control["cells"]["distance_refit"]),
             "family_wall_median": summarize_trajectory(trajectory, len(pool))}
    for name, trace in zip(CELLS, (baseline["trajectory"], trajectory)):
        cells[name]["actual_cost_budget"] = time_budget_summary(trace, horizon)
    row = {key: task[key] for key in ("task_id", "split_id", "model_id", "scale", "initial_train_count", "subset_seed")}
    row.update(initial_regret_ms=control["initial_regret_ms"], candidate_plans=len(plans),
               measurement_pool_objects=len(pool), cells=cells)
    return {"task_id": task["task_id"], "compact": row, "trajectory": trajectory}


def summarize(tasks):
    if not tasks:
        return {}
    cells = {}
    for name in CELLS:
        points = [row["cells"][name] for row in tasks]
        final = [point["curve"][-1] for point in points]
        cells[name] = {"mean_count_auc_ms": statistics.fmean(point["regret_budget_auc_ms"] for point in points),
            "mean_step_actual_cost_auc_ms": statistics.fmean(point["actual_cost_budget"]["step_regret_time_auc_ms"] for point in points),
            "wrong_stop_configs": sum(point["stop_regret_ms"] > 1e-9 for point in points),
            "mean_measurements": statistics.fmean(point["stop_budget"] for point in points),
            **{f"mean_{metric}": statistics.fmean(point[metric] for point in final)
               for metric in ("top1_regret_ms", "profiling_wall_time_s", "algorithm_wall_time_s")}}
    paired = {}
    for metric in ("regret_budget_auc_ms", "step_regret_time_auc_ms"):
        def value(cell):
            return cell[metric] if metric == "regret_budget_auc_ms" else cell["actual_cost_budget"][metric]
        delta = [value(row["cells"][CELLS[1]]) - value(row["cells"][CELLS[0]]) for row in tasks]
        paired[metric] = {"lower": sum(x < -1e-9 for x in delta), "higher": sum(x > 1e-9 for x in delta),
                          "tied": sum(abs(x) <= 1e-9 for x in delta)}
    return {"configs": len(tasks), "cells": cells, "paired_new_minus_old": paired,
        "changed_selection_prefix_configs": sum(row["cells"][CELLS[0]]["selected_object_ids"] !=
            row["cells"][CELLS[1]]["selected_object_ids"] for row in tasks)}


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    directory = root / "data/bw1000_e2"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("workers and limit must be positive")
    def read(name):
        return json.loads((directory / name).read_text(encoding="utf-8"))
    baseline = read("e2c_factorial_evaluation.json")
    parent = read("e2c_graph_wide_diagnostics.json")
    manifest, contract, suite = (read(name) for name in ("shape_manifest.json", "split_contract.json", "e2c_task_suite.json"))
    records = load_jsonl_records(directory / "profiling_dataset.jsonl")
    hardware = load_hardware(root / "data/bw1000_2card/bw1000_2card.calibrated.json")
    ids = {task["task_id"] for task in suite["tasks"]}
    if (suite != build_task_suite(manifest, contract) or baseline["execution_status"] != "complete" or
            baseline["scope"] != "all_frozen_configs" or {task["task_id"] for task in baseline["tasks"]} != ids or
            baseline["configuration"]["candidate_experiment_id"] != parent["experiment_id"] or
            baseline["configuration"]["suite_id"] != suite["suite_id"] or
            baseline["configuration"]["dataset_id"] != content_id("diagnostics-data", records) or
            parent["configuration"]["hardware"] != hardware_to_dict(hardware)):
        raise ValueError("frozen baseline, suite, hardware or dataset changed")
    config = {"baseline_report_id": content_id("cost-baseline", baseline), "candidate_experiment_id": parent["experiment_id"],
        "suite_id": suite["suite_id"], "dataset_id": content_id("diagnostics-data", records), "protocol": PROTOCOL,
        "cost_axis": "common per-task sum of pool wall costs; step curve, observation available after completion",
        "scope": "Oracle-containing fixed development pools; correlated configs; no blind-test or closed-loop claim",
        "timings": "new CPU times measured here; old CPU times copied, not a simultaneous benchmark",
        "source_sha256": hashlib.sha256(b"".join((Path(__file__).parent / name).read_bytes() for name in
            ("e2_cost_normalization.py", "e2_cost_evaluation.py", "e2_profiling_cost.py", "e2_srop_proxy.py",
             "e2_refit_acquisition.py", "e2_uncertainty.py", "e2_residual.py", "batched_placement_ranker.py",
             "e2_uncertainty_stopping.py", "e2_task_suite.py", "e2_diagnostics.py", "structural.py", "structure.py"))
             + (root / "calibration/e2_dataset.py").read_bytes()).hexdigest()}
    experiment_id = content_id("e2-cost-normalization", config)
    checkpoint_root = directory / "diagnostics_checkpoints" / experiment_id.split(":")[1][:16]
    candidate_root = directory / "diagnostics_checkpoints" / parent["experiment_id"].split(":")[1][:16]
    baseline_root = directory / "diagnostics_checkpoints" / baseline["experiment_id"].split(":")[1][:16]
    tasks = suite["tasks"] if args.limit is None else suite["tasks"][:args.limit]
    parents = {task["task_id"]: task for task in parent["tasks"]}
    controls = {task["task_id"]: task for task in baseline["tasks"]}
    results = {}
    for task in tasks:
        filename = task["task_id"].split(":")[1] + ".json.gz"
        candidates = _read_checkpoint(candidate_root / filename)
        control = _read_checkpoint(baseline_root / filename)
        if (candidates is None or control is None or
                candidates["zero"]["candidate_ids"] != parents[task["task_id"]]["zero"]["candidate_ids"] or
                control["compact"] != controls[task["task_id"]]):
            raise ValueError("candidate or baseline checkpoint missing/mismatched")
        stored = _read_checkpoint(checkpoint_root / filename)
        if stored is not None:
            results[task["task_id"]] = stored["compact"]
    predictions = evaluate_cost_predictions(records, manifest, contract, suite)

    def save_report():
        rows = [results[task["task_id"]] for task in tasks if task["task_id"] in results]
        _write_json(args.output, {"format": "fusion-placement-search.e2-cost-normalization", "version": "1.0",
            "experiment_id": experiment_id, "configuration": config, "prediction_evaluation": predictions,
            "scope": "all_frozen_configs" if len(tasks) == len(suite["tasks"]) else "partial_smoke_test",
            "execution_status": "complete" if len(rows) == len(tasks) else "running", "completed_configs": len(rows),
            "summary": {"all": summarize(rows),
                "positive_initial": summarize([row for row in rows if row["initial_regret_ms"] > 1e-9]),
                "zero_initial": summarize([row for row in rows if row["initial_regret_ms"] <= 1e-9]),
                "by_model": {model: summarize([row for row in rows if row["model_id"] == model])
                             for model in sorted({row["model_id"] for row in rows})}}, "tasks": rows})

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_initialize,
            initargs=((manifest, contract, records, hardware, candidate_root, baseline_root, controls),)) as executor:
        futures = {executor.submit(_run_task, task): task for task in tasks if task["task_id"] not in results}
        for future in as_completed(futures):
            task = futures[future]
            result = future.result()
            _write_checkpoint(checkpoint_root / (task["task_id"].split(":")[1] + ".json.gz"), result)
            results[task["task_id"]] = result["compact"]
            save_report()
            print(f"wall cost {len(results)}/{len(tasks)} {task['model_id']} {task['scale']} train={task['initial_train_count']}", flush=True)
    save_report()
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
