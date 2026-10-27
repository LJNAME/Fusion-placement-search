"""Replay full active-profiling trajectories on the frozen graph-wide pools."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

from calibration.e2_manifest import content_id
from calibration.e2_records import load_jsonl_records
from simulator.dataset import plan_from_dict
from simulator.hardware import hardware_to_dict, load_hardware

from .e2_budget_summary import BUDGETS, compact_task, grouped_summaries
from .batched_placement_ranker import BatchedPlacementRanker
from .e2_diagnostics import _read_checkpoint, _write_checkpoint, _write_json, workload_for_task
from .e2_srop_proxy import build_workload_graph, evaluate_fixed_candidate_srop
from .e2_task_suite import active_task_contract, build_task_suite


_CONTEXT = None


def _initialize(context):
    global _CONTEXT
    _CONTEXT = context


def _run_task(task):
    manifest, contract, records, hardware, base_root = _CONTEXT
    stored = _read_checkpoint(base_root / (task["task_id"].split(":")[1] + ".json.gz"))
    workload = workload_for_task(task, manifest)
    graph = build_workload_graph(workload, task["scale"])
    runtime = active_task_contract(task, contract, manifest)
    result = evaluate_fixed_candidate_srop(graph,
        [plan_from_dict(plan) for plan in stored["candidate_plans"]], records, runtime, hardware, workload,
        ranker_factory=BatchedPlacementRanker)
    # Tie-breaking, first selections and first-step regret must match the
    # frozen diagnostic before we interpret later-budget behavior.
    for name in ("uncertainty_only", "impact_only", "srop_proxy"):
        trajectory = result["policies"][name]["full_trajectories"][0]
        previous = stored["one_step"]["policies"][name]
        if (trajectory[0]["next_selection"] != previous["selected_object_id"] or
                abs(trajectory[1]["top1_regret_ms"] - previous["selected_regret_ms"]) > 1e-9):
            raise AssertionError("first step differs from the frozen policy diagnostic")
    for policy in result["policies"].values():
        for trajectory in policy["full_trajectories"]:
            if trajectory[-1]["strict_top_k_recall"] != 1.0:
                raise AssertionError("full budget must recover strict oracle top-k")
    return {"task_id": task["task_id"], "raw": result, "compact": compact_task(task, result)}


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    directory = root / "data/bw1000_e2"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-report", type=Path, default=directory / "e2c_graph_wide_diagnostics.json")
    parser.add_argument("--checkpoint-dir", type=Path, default=directory / "diagnostics_checkpoints")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("workers and limit must be positive")
    base = json.loads(args.base_report.read_text(encoding="utf-8"))
    manifest = json.loads((directory / "shape_manifest.json").read_text(encoding="utf-8"))
    contract = json.loads((directory / "split_contract.json").read_text(encoding="utf-8"))
    suite = json.loads((directory / "e2c_task_suite.json").read_text(encoding="utf-8"))
    records = load_jsonl_records(directory / "profiling_dataset.jsonl")
    hardware = load_hardware(root / "data/bw1000_2card/bw1000_2card.calibrated.json")
    if (suite != build_task_suite(manifest, contract) or base["execution_status"] != "complete"
            or base["configuration"]["suite_id"] != suite["suite_id"]
            or base["configuration"]["dataset_id"] != content_id("diagnostics-data", records)
            or base["configuration"]["hardware"] != hardware_to_dict(hardware)
            or base["configuration"].get("candidate_coverage") != "graph-wide-v1"
            or {item["task_id"] for item in base["tasks"]} != {task["task_id"] for task in suite["tasks"]}):
        raise ValueError("base must be the same complete graph-wide frozen experiment")
    if tuple(contract["active_profiling_protocol"]["budgets"]) != BUDGETS:
        raise ValueError("budget protocol changed")
    source_names = ("e2_budget_curves.py", "e2_budget_summary.py", "batched_placement_ranker.py", "e2_srop_proxy.py", "e2_residual.py",
                    "e2_audit.py", "e2_task_suite.py", "structural.py", "structure.py", "search_sensitivity.py")
    config = {
        "parent_experiment_id": base["experiment_id"], "parent_report_id": content_id("budget-parent-report", base),
        "suite_id": suite["suite_id"], "dataset_id": base["configuration"]["dataset_id"],
        "source_sha256": hashlib.sha256(b"".join((Path(__file__).parent / name).read_bytes() for name in source_names)).hexdigest(),
        "budgets": list(BUDGETS), "active_protocol": contract["active_profiling_protocol"],
        "candidate_scope": "unchanged graph-wide-v1 pools including Oracle-discovered candidates; development diagnostic",
        "random": "four seeded trajectories; seeds averaged within each configuration before aggregation",
        "auc": "normalized trapezoid over every integer budget on common 0..16 horizon; hold decision after stop/pool exhaustion",
        "costs": "recorded per-object profiling wall time summed sequentially; CPU algorithm times include next-selection computation at each prefix",
        "timing_limits": "offline replay CPU timing, not GPU training speedup; excludes original candidate generation and Oracle evaluation",
        "scoring_engine": "batched placement makespan recurrence; original scheduler fallback for mixed partitions or nonzero interference",
    }
    experiment_id = content_id("e2-budget-curves", config)
    checkpoint_root = args.checkpoint_dir / experiment_id.split(":")[1][:16]
    base_root = args.checkpoint_dir / base["experiment_id"].split(":")[1][:16]
    tasks = suite["tasks"] if args.limit is None else suite["tasks"][:args.limit]
    expected = {item["task_id"]: item for item in base["tasks"]}
    results = {}
    for task in tasks:
        filename = task["task_id"].split(":")[1] + ".json.gz"
        original = _read_checkpoint(base_root / filename)
        if original is None or original["zero"]["candidate_ids"] != expected[task["task_id"]]["zero"]["candidate_ids"]:
            raise ValueError("frozen graph-wide candidate checkpoints missing or mismatched")
        stored = _read_checkpoint(checkpoint_root / filename)
        if stored is not None:
            results[task["task_id"]] = stored["compact"]

    def save_report():
        published = [results[task["task_id"]] for task in tasks if task["task_id"] in results]
        _write_json(args.output, {"format": "fusion-placement-search.e2-budget-curves", "version": "1.0",
            "experiment_id": experiment_id, "configuration": config,
            "execution_status": "complete" if len(published) == len(tasks) else "running",
            "scope": "all_frozen_configs" if len(tasks) == len(suite["tasks"]) else "partial_smoke_test",
            "completed_configs": len(published), "total_frozen_configs": len(suite["tasks"]),
            "summary": grouped_summaries(published), "tasks": published})

    context = (manifest, contract, records, hardware, base_root)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_initialize, initargs=(context,)) as executor:
        futures = {executor.submit(_run_task, task): task for task in tasks if task["task_id"] not in results}
        for future in as_completed(futures):
            task = futures[future]
            result = future.result()
            _write_checkpoint(checkpoint_root / (task["task_id"].split(":")[1] + ".json.gz"), result)
            results[task["task_id"]] = result["compact"]
            save_report()
            print(f"curves {len(results)}/{len(tasks)} {task['model_id']} {task['scale']} train={task['initial_train_count']}", flush=True)
    save_report()
    print(f"report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
