"""Complete the frozen 2x2 interval-by-acquisition stopping ablation."""

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
from .e2_budget_summary import METRICS
from .e2_diagnostics import _read_checkpoint, _write_checkpoint, _write_json, measurement_pool, workload_for_task
from .e2_srop_proxy import _active_profile, _oracle_profile, _run_policy_trajectory, build_workload_graph, object_map_for_workload
from .e2_task_suite import active_task_contract, build_task_suite
from .e2_uncertainty_stopping import summarize_trajectory


CELLS = ("legacy_proxy", "legacy_refit", "distance_proxy", "distance_refit")
_CONTEXT = None


def _initialize(context):
    global _CONTEXT
    _CONTEXT = context


def frozen_controls(interval_task, refit_task):
    refit = refit_task["policies"]["srop_refit"]
    trace = refit["traces"][0]
    return copy.deepcopy({
        "legacy_proxy": interval_task["variants"]["legacy"],
        "legacy_refit": {**refit["modes"]["stopping"], "stop_budget": trace["stop_budget"],
            "stop_regret_ms": trace["stop_regret_ms"],
            "selected_object_ids": trace["selected_object_ids"][:trace["stop_budget"]]},
        "distance_proxy": interval_task["variants"]["distance_floor"],
    })


def interaction(aucs):
    return (aucs["distance_refit"] - aucs["distance_proxy"]) - (aucs["legacy_refit"] - aucs["legacy_proxy"])


def _run_task(task):
    manifest, contract, records, hardware, candidate_root, controls_by_id = _CONTEXT
    stored = _read_checkpoint(candidate_root / (task["task_id"].split(":")[1] + ".json.gz"))
    plans = [plan_from_dict(plan) for plan in stored["candidate_plans"]]
    workload = workload_for_task(task, manifest)
    graph = build_workload_graph(workload, task["scale"])
    mapping = object_map_for_workload(records, workload)
    runtime = active_task_contract(task, contract, manifest)
    runtime["active_profiling_protocol"]["uncertainty_variant"] = "distance_floor"
    pool = measurement_pool(graph, mapping, task["train_object_ids"])
    ranker = BatchedPlacementRanker(graph, plans, 2, ((0, 1), (1, 0)))
    oracle = ranker.rank(_oracle_profile(records, hardware, mapping))
    trajectory = _run_policy_trajectory("srop_refit", graph, plans, records, runtime, hardware,
        mapping, pool, oracle, split_id=task["split_id"], random_seed=None,
        registry=default_fusion_registry(), exchange_permutations=((0, 1), (1, 0)), ranker=ranker,
        stop_on_recommendation=True)
    initial_regret = controls_by_id[task["task_id"]]["initial_regret_ms"]
    if abs(trajectory[0]["top1_regret_ms"] - initial_regret) > 1e-9:
        raise AssertionError("mean predictions changed at initial public state")
    full, _ = _active_profile(records, runtime, hardware, mapping,
                             split_id=task["split_id"], revealed_object_ids=pool)
    ranked = ranker.rank(full)
    oracle_costs = {item.plan_id: item.makespan for item in oracle}
    if max(abs(item.makespan - oracle_costs[item.plan_id]) for item in ranked) > 1e-12 or ranking_metrics(oracle, ranked)["strict_top_k_recall"] != 1.0:
        raise AssertionError("full-reveal profile does not recover all Oracle candidates")
    cells = copy.deepcopy(controls_by_id[task["task_id"]]["cells"])
    cells["distance_refit"] = summarize_trajectory(trajectory, len(pool))
    row = {key: task[key] for key in ("task_id", "split_id", "model_id", "scale", "initial_train_count", "subset_seed")}
    row.update({"initial_regret_ms": initial_regret, "candidate_plans": len(plans),
        "measurement_pool_objects": len(pool), "cells": cells,
        "auc_interaction_ms": interaction({name: cell["regret_budget_auc_ms"] for name, cell in cells.items()})})
    return {"task_id": task["task_id"], "compact": row, "trajectory": trajectory}


def summarize(tasks):
    if not tasks:
        return {}
    cells = {}
    for cell in CELLS:
        curve = []
        for budget in (0, 1, 2, 4, 8, 16):
            points = [next(point for point in task["cells"][cell]["curve"] if point["requested_budget"] == budget) for task in tasks]
            curve.append({"requested_budget": budget,
                **{key: statistics.fmean(point[key] for point in points) for key in ("measurements_mean", *METRICS)},
                "harmed_configs": sum(point["top1_regret_ms"] > task["initial_regret_ms"] + 1e-9 for task, point in zip(tasks, points))})
        cells[cell] = {"mean_regret_budget_auc_ms": statistics.fmean(task["cells"][cell]["regret_budget_auc_ms"] for task in tasks),
            "wrong_stop_configs": sum(task["cells"][cell]["stop_regret_ms"] > 1e-9 for task in tasks),
            "mean_stop_budget": statistics.fmean(task["cells"][cell]["stop_budget"] for task in tasks), "curve": curve}
    comparisons = {}
    for baseline in CELLS[:-1]:
        delta = [task["cells"]["distance_refit"]["regret_budget_auc_ms"] - task["cells"][baseline]["regret_budget_auc_ms"] for task in tasks]
        old_wrong = {task["task_id"] for task in tasks if task["cells"][baseline]["stop_regret_ms"] > 1e-9}
        new_wrong = {task["task_id"] for task in tasks if task["cells"]["distance_refit"]["stop_regret_ms"] > 1e-9}
        comparisons[baseline] = {"lower_auc_configs": sum(value < -1e-9 for value in delta),
            "higher_auc_configs": sum(value > 1e-9 for value in delta), "tied_configs": sum(abs(value) <= 1e-9 for value in delta),
            "fixed_wrong_stops": len(old_wrong-new_wrong), "introduced_wrong_stops": len(new_wrong-old_wrong)}
    return {"configs": len(tasks), "cells": cells, "new_cell_vs_controls": comparisons,
        "mean_auc_interaction_ms": statistics.fmean(task["auc_interaction_ms"] for task in tasks),
        "interaction_interpretation": "descriptive difference-in-differences only; correlated configurations, no statistical significance claim"}


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
    intervals = json.loads((directory / "e2c_uncertainty_stopping.json").read_text(encoding="utf-8"))
    refit = json.loads((directory / "e2c_refit_evaluation.json").read_text(encoding="utf-8"))
    baseline = json.loads((directory / "e2c_graph_wide_budget_curves.json").read_text(encoding="utf-8"))
    parent = json.loads((directory / "e2c_graph_wide_diagnostics.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "shape_manifest.json").read_text(encoding="utf-8"))
    contract = json.loads((directory / "split_contract.json").read_text(encoding="utf-8"))
    suite = json.loads((directory / "e2c_task_suite.json").read_text(encoding="utf-8"))
    records = load_jsonl_records(directory / "profiling_dataset.jsonl")
    hardware = load_hardware(root / "data/bw1000_2card/bw1000_2card.calibrated.json")
    ids = {task["task_id"] for task in suite["tasks"]}
    if (suite != build_task_suite(manifest, contract) or
            any(report["execution_status"] != "complete" or report["scope"] != "all_frozen_configs" or
                {task["task_id"] for task in report["tasks"]} != ids for report in (intervals, refit, baseline)) or
            intervals["configuration"]["baseline_report_id"] != content_id("uncertainty-stop-baseline", baseline) or
            refit["configuration"]["baseline_report_id"] != content_id("refit-baseline", baseline) or
            baseline["configuration"]["parent_report_id"] != content_id("budget-parent-report", parent) or
            baseline["configuration"]["dataset_id"] != content_id("diagnostics-data", records) or
            parent["configuration"]["hardware"] != hardware_to_dict(hardware)):
        raise ValueError("frozen controls or data changed")
    refit_by_id = {task["task_id"]: task for task in refit["tasks"]}
    controls = {task["task_id"]: {"initial_regret_ms": task["initial_regret_ms"],
        "cells": frozen_controls(task, refit_by_id[task["task_id"]])} for task in intervals["tasks"]}
    config = {"interval_control_report_id": content_id("factorial-interval-control", intervals),
        "refit_control_report_id": content_id("factorial-refit-control", refit),
        "candidate_experiment_id": parent["experiment_id"], "suite_id": suite["suite_id"],
        "dataset_id": baseline["configuration"]["dataset_id"],
        "new_cell": "distance_floor+srop_refit; no formula, sampling, cost or stop-rule changes",
        "scope": "actual stop-only run; frozen Oracle-containing development pools; three controls retained from prior runs",
        "costs": "new cell timings measured here; control timings copied, not simultaneous benchmark",
        "source_sha256": hashlib.sha256(b"".join((Path(__file__).parent / name).read_bytes() for name in
            ("e2_factorial_evaluation.py", "e2_uncertainty_stopping.py", "e2_refit_acquisition.py", "e2_uncertainty.py",
             "e2_residual.py", "e2_srop_proxy.py", "batched_placement_ranker.py", "structural.py", "structure.py"))).hexdigest()}
    experiment_id = content_id("e2-factorial", config)
    checkpoint_root = directory / "diagnostics_checkpoints" / experiment_id.split(":")[1][:16]
    candidate_root = directory / "diagnostics_checkpoints" / parent["experiment_id"].split(":")[1][:16]
    tasks = suite["tasks"] if args.limit is None else suite["tasks"][:args.limit]
    parent_by_id = {task["task_id"]: task for task in parent["tasks"]}
    results = {}
    for task in tasks:
        filename = task["task_id"].split(":")[1] + ".json.gz"
        candidate = _read_checkpoint(candidate_root / filename)
        if candidate is None or candidate["zero"]["candidate_ids"] != parent_by_id[task["task_id"]]["zero"]["candidate_ids"]:
            raise ValueError("candidate checkpoints missing or mismatched")
        stored = _read_checkpoint(checkpoint_root / filename)
        if stored is not None:
            results[task["task_id"]] = stored["compact"]

    def save_report():
        rows = [results[task["task_id"]] for task in tasks if task["task_id"] in results]
        _write_json(args.output, {"format": "fusion-placement-search.e2-factorial-evaluation", "version": "1.0",
            "experiment_id": experiment_id, "configuration": config,
            "scope": "all_frozen_configs" if len(tasks) == len(suite["tasks"]) else "partial_smoke_test",
            "execution_status": "complete" if len(rows) == len(tasks) else "running", "completed_configs": len(rows),
            "summary": {"all": summarize(rows),
                "positive_initial": summarize([row for row in rows if row["initial_regret_ms"] > 1e-9]),
                "zero_initial": summarize([row for row in rows if row["initial_regret_ms"] <= 1e-9]),
                "by_model": {model: summarize([row for row in rows if row["model_id"] == model]) for model in sorted({row["model_id"] for row in rows})}},
            "tasks": rows})

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_initialize,
            initargs=((manifest, contract, records, hardware, candidate_root, controls),)) as executor:
        futures = {executor.submit(_run_task, task): task for task in tasks if task["task_id"] not in results}
        for future in as_completed(futures):
            task = futures[future]
            result = future.result()
            _write_checkpoint(checkpoint_root / (task["task_id"].split(":")[1] + ".json.gz"), result)
            results[task["task_id"]] = result["compact"]
            save_report()
            print(f"factorial {len(results)}/{len(tasks)} {task['model_id']} {task['scale']} train={task['initial_train_count']}", flush=True)
    save_report()
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
