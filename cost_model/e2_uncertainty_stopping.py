"""Stop-only calibration ablation on every frozen pool; no hidden-label tuning."""

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
from .e2_uncertainty import PROTOCOL


_CONTEXT = None
NEW_VARIANTS = ("strict_family", "distance_floor")


def _initialize(context):
    global _CONTEXT
    _CONTEXT = context


def summarize_trajectory(trajectory, total_pool):
    curve = []
    for budget in range(17):
        point = trajectory[min(budget, len(trajectory)-1)]
        curve.append({"requested_budget": budget, "measurements_mean": point["budget"],
                      **{key: point[key] for key in METRICS}})
    auc = sum((left["top1_regret_ms"] + right["top1_regret_ms"]) * 0.5
              for left, right in zip(curve, curve[1:])) / 16.0
    final = trajectory[-1]
    return {"curve": curve, "regret_budget_auc_ms": auc,
        "stop_budget": final["budget"], "stop_regret_ms": final["top1_regret_ms"],
        "stop_reason": "nonpositive_score" if final["stop_recommended"] else "pool_exhausted",
        "remaining_objects": total_pool - final["budget"],
        "selected_object_ids": final["selected_object_ids"],
        "scores_before_measurement": [point["next_selection_score"] for point in trajectory[:-1]],
        "full_budget_recovers_all_candidates": True}


def _run_task(task):
    manifest, contract, records, hardware, candidate_root, legacy_by_id, candidate_by_id = _CONTEXT
    filename = task["task_id"].split(":")[1] + ".json.gz"
    stored = _read_checkpoint(candidate_root / filename)
    plans = [plan_from_dict(plan) for plan in stored["candidate_plans"]]
    workload = workload_for_task(task, manifest)
    graph = build_workload_graph(workload, task["scale"])
    object_map = object_map_for_workload(records, workload)
    runtime = active_task_contract(task, contract, manifest)
    pool = measurement_pool(graph, object_map, task["train_object_ids"])
    oracle = _oracle_profile(records, hardware, object_map)
    exchange = ((0, 1), (1, 0))
    ranker = BatchedPlacementRanker(graph, plans, 2, exchange)
    oracle_ranked = ranker.rank(oracle)
    legacy = legacy_by_id[task["task_id"]]["policies"]["srop_proxy"]
    legacy_trace = legacy["traces"][0]
    row = {key: task[key] for key in ("task_id", "split_id", "model_id", "scale", "initial_train_count", "subset_seed")}
    row.update({"initial_regret_ms": legacy_by_id[task["task_id"]]["initial_regret_ms"],
                "candidate_plans": len(plans), "measurement_pool_objects": len(pool),
                "variants": {"legacy": {**legacy["modes"]["stopping"],
                    "stop_budget": legacy_trace["stop_budget"], "stop_regret_ms": legacy_trace["stop_regret_ms"],
                    "selected_object_ids": legacy_trace["selected_object_ids"][:legacy_trace["stop_budget"]]}}})
    raw = {}
    oracle_costs = {item.plan_id: item.makespan for item in oracle_ranked}
    for variant in NEW_VARIANTS:
        current = copy.deepcopy(runtime)
        current["active_profiling_protocol"]["uncertainty_variant"] = variant
        # This changes only uncertainty; original five samples, cost proxy,
        # current-decision loss, tie-breaking and stop rule stay fixed.
        trajectory = _run_policy_trajectory("srop_proxy", graph, plans, records, current, hardware,
            object_map, pool, oracle_ranked, split_id=task["split_id"], random_seed=None,
            registry=default_fusion_registry(), exchange_permutations=exchange, ranker=ranker,
            stop_on_recommendation=True)
        if abs(trajectory[0]["top1_regret_ms"] - row["initial_regret_ms"]) > 1e-9:
            raise AssertionError("uncertainty changed the initial mean ranking")
        full, _ = _active_profile(records, current, hardware, object_map,
                                 split_id=task["split_id"], revealed_object_ids=pool)
        ranked = ranker.rank(full)
        if max(abs(item.makespan-oracle_costs[item.plan_id]) for item in ranked) > 1e-12 or ranking_metrics(oracle_ranked, ranked)["strict_top_k_recall"] != 1.0:
            raise AssertionError("full observed budget changed oracle recovery")
        row["variants"][variant] = summarize_trajectory(trajectory, len(pool))
        raw[variant] = trajectory
    return {"task_id": task["task_id"], "compact": row, "trajectories": raw}


def summarize(tasks):
    if not tasks:
        return {}
    variants = {}
    for variant in ("legacy", *NEW_VARIANTS):
        curve = []
        for budget in (0, 1, 2, 4, 8, 16):
            # The legacy parent publishes six budgets; new runs contain all
            # integer prefixes. AUC always uses the dense 0..16 horizon.
            points = [next(point for point in task["variants"][variant]["curve"] if point["requested_budget"] == budget) for task in tasks]
            curve.append({"requested_budget": budget,
                **{key: statistics.fmean(point[key] for point in points) for key in ("measurements_mean", *METRICS)},
                "harmed_configs": sum(point["top1_regret_ms"] > task["initial_regret_ms"] + 1e-9 for task, point in zip(tasks, points))})
        variants[variant] = {"mean_regret_budget_auc_ms": statistics.fmean(task["variants"][variant]["regret_budget_auc_ms"] for task in tasks),
            "wrong_stop_configs": sum(task["variants"][variant]["stop_regret_ms"] > 1e-9 for task in tasks),
            "mean_stop_budget": statistics.fmean(task["variants"][variant]["stop_budget"] for task in tasks), "curve": curve}
    comparisons = {}
    for variant in NEW_VARIANTS:
        delta = [task["variants"][variant]["regret_budget_auc_ms"] - task["variants"]["legacy"]["regret_budget_auc_ms"] for task in tasks]
        comparisons[variant] = {"lower_auc_configs": sum(value < -1e-9 for value in delta),
            "higher_auc_configs": sum(value > 1e-9 for value in delta), "tied_configs": sum(abs(value) <= 1e-9 for value in delta)}
    return {"configs": len(tasks), "positive_initial_regret_configs": sum(task["initial_regret_ms"] > 1e-9 for task in tasks),
            "variants": variants, "vs_legacy": comparisons}


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
    baseline = json.loads((directory / "e2c_graph_wide_budget_curves.json").read_text(encoding="utf-8"))
    parent = json.loads((directory / "e2c_graph_wide_diagnostics.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "shape_manifest.json").read_text(encoding="utf-8"))
    contract = json.loads((directory / "split_contract.json").read_text(encoding="utf-8"))
    suite = json.loads((directory / "e2c_task_suite.json").read_text(encoding="utf-8"))
    records = load_jsonl_records(directory / "profiling_dataset.jsonl")
    hardware = load_hardware(root / "data/bw1000_2card/bw1000_2card.calibrated.json")
    if (suite != build_task_suite(manifest, contract) or baseline["execution_status"] != "complete"
            or baseline["configuration"]["parent_report_id"] != content_id("budget-parent-report", parent)
            or baseline["configuration"]["dataset_id"] != content_id("diagnostics-data", records)
            or parent["configuration"]["hardware"] != hardware_to_dict(hardware)
            or {task["task_id"] for task in baseline["tasks"]} != {task["task_id"] for task in suite["tasks"]}):
        raise ValueError("frozen baseline inputs changed")
    config = {"protocol": PROTOCOL, "variants": NEW_VARIANTS, "acquisition": "unchanged srop_proxy",
        "baseline_report_id": content_id("uncertainty-stop-baseline", baseline),
        "suite_id": suite["suite_id"], "dataset_id": baseline["configuration"]["dataset_id"],
        "scope": "actual stop-only runs; dense common 0..16 AUC; original Oracle-containing development pools",
        "costs": "legacy timings reused; new timings measured in this CPU replay",
        "source_sha256": hashlib.sha256(b"".join((Path(__file__).parent / name).read_bytes() for name in
            ("e2_uncertainty_stopping.py", "e2_uncertainty.py", "e2_residual.py", "e2_srop_proxy.py", "batched_placement_ranker.py", "structural.py", "structure.py"))).hexdigest()}
    experiment_id = content_id("uncertainty-stopping", config)
    checkpoint_root = directory / "diagnostics_checkpoints" / experiment_id.split(":")[1][:16]
    candidate_root = directory / "diagnostics_checkpoints" / parent["experiment_id"].split(":")[1][:16]
    tasks = suite["tasks"] if args.limit is None else suite["tasks"][:args.limit]
    legacy_by_id = {task["task_id"]: task for task in baseline["tasks"]}
    candidate_by_id = {task["task_id"]: task for task in parent["tasks"]}
    results = {}
    for task in tasks:
        filename = task["task_id"].split(":")[1] + ".json.gz"
        candidate = _read_checkpoint(candidate_root / filename)
        if candidate is None or candidate["zero"]["candidate_ids"] != candidate_by_id[task["task_id"]]["zero"]["candidate_ids"]:
            raise ValueError("candidate checkpoint missing or mismatched")
        stored = _read_checkpoint(checkpoint_root / filename)
        if stored is not None:
            results[task["task_id"]] = stored["compact"]

    def save_report():
        rows = [results[task["task_id"]] for task in tasks if task["task_id"] in results]
        _write_json(args.output, {"format": "fusion-placement-search.e2-uncertainty-stopping", "version": "1.0",
            "experiment_id": experiment_id, "configuration": config,
            "scope": "all_frozen_configs" if len(tasks) == len(suite["tasks"]) else "partial_smoke_test",
            "execution_status": "complete" if len(rows) == len(tasks) else "running", "completed_configs": len(rows),
            "summary": {"all": summarize(rows),
                "positive_initial": summarize([row for row in rows if row["initial_regret_ms"] > 1e-9]),
                "zero_initial": summarize([row for row in rows if row["initial_regret_ms"] <= 1e-9]),
                "by_model": {model: summarize([row for row in rows if row["model_id"] == model]) for model in sorted({row["model_id"] for row in rows})}},
            "tasks": rows})

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_initialize,
            initargs=((manifest, contract, records, hardware, candidate_root, legacy_by_id, candidate_by_id),)) as executor:
        futures = {executor.submit(_run_task, task): task for task in tasks if task["task_id"] not in results}
        for future in as_completed(futures):
            task = futures[future]
            result = future.result()
            _write_checkpoint(checkpoint_root / (task["task_id"].split(":")[1] + ".json.gz"), result)
            results[task["task_id"]] = result["compact"]
            save_report()
            print(f"uncertainty {len(results)}/{len(tasks)} {task['model_id']} {task['scale']} train={task['initial_train_count']}", flush=True)
    save_report()
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
