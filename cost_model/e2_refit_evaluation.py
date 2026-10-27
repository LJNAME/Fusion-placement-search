"""Frozen-pool SROP-Refit ablation against the completed budget experiment."""

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
from simulator.dataset import plan_from_dict
from simulator.hardware import hardware_to_dict, load_hardware
from .batched_placement_ranker import BatchedPlacementRanker
from .e2_budget_summary import compact_task, grouped_summaries
from .e2_diagnostics import _read_checkpoint, _write_checkpoint, _write_json, workload_for_task
from .e2_srop_proxy import build_workload_graph, evaluate_fixed_candidate_srop
from .e2_task_suite import active_task_contract, build_task_suite


_CONTEXT = None


def _initialize(context):
    global _CONTEXT
    _CONTEXT = context


def _run_task(task):
    manifest, contract, records, hardware, candidate_root, baseline_by_id = _CONTEXT
    candidate = _read_checkpoint(candidate_root / (task["task_id"].split(":")[1] + ".json.gz"))
    workload = workload_for_task(task, manifest)
    graph = build_workload_graph(workload, task["scale"])
    runtime = active_task_contract(task, contract, manifest)
    runtime["active_profiling_protocol"]["policies"] = ["srop_refit"]
    runtime.pop("contract_id")
    runtime["contract_id"] = content_id("split-contract", runtime)
    raw = evaluate_fixed_candidate_srop(graph,
        [plan_from_dict(plan) for plan in candidate["candidate_plans"]], records, runtime, hardware, workload,
        ranker_factory=BatchedPlacementRanker)
    if raw["policies"]["srop_refit"]["full_trajectories"][0][-1]["strict_top_k_recall"] != 1.0:
        raise AssertionError("full refit budget must recover Oracle top-k")
    compact = copy.deepcopy(baseline_by_id[task["task_id"]])
    updated = compact_task(task, raw)
    if abs(updated["initial_regret_ms"] - compact["initial_regret_ms"]) > 1e-9:
        raise AssertionError("initial predictions changed")
    compact["policies"]["srop_refit"] = updated["policies"]["srop_refit"]
    compact["refit_evaluation_wall_time_s"] = raw["evaluation_wall_time_s"]
    return {"task_id": task["task_id"], "raw": raw, "compact": compact}


def refit_summaries(tasks):
    summary = grouped_summaries(tasks)
    groups = {"all_configs": tasks,
              "positive_initial_regret": [task for task in tasks if task["initial_regret_ms"] > 1e-9],
              "zero_initial_regret": [task for task in tasks if task["initial_regret_ms"] <= 1e-9]}
    groups.update({"model:" + model: [task for task in tasks if task["model_id"] == model]
                   for model in summary.get("by_model", {})})
    for key, subset in groups.items():
        target = summary["by_model"][key[6:]] if key.startswith("model:") else summary[key]
        if not subset:
            continue
        for mode in ("forced", "stopping"):
            comparisons = {}
            for policy in ("random", "uncertainty_only", "impact_only", "srop_proxy"):
                deltas = [task["policies"]["srop_refit"]["modes"][mode]["regret_budget_auc_ms"] -
                          task["policies"][policy]["modes"][mode]["regret_budget_auc_ms"] for task in subset]
                comparisons[policy] = {"refit_lower_auc_configs": sum(delta < -1e-9 for delta in deltas),
                    "refit_higher_auc_configs": sum(delta > 1e-9 for delta in deltas),
                    "tied_configs": sum(abs(delta) <= 1e-9 for delta in deltas),
                    "mean_refit_minus_baseline_auc_ms": statistics.fmean(deltas)}
            target["modes"][mode]["refit_auc_comparisons"] = comparisons
    return summary


def stop_diagnosis(tasks, parent):
    candidates = {task["task_id"]: task for task in parent["tasks"]}
    rows = []
    for task in tasks:
        old = task["policies"]["srop_proxy"]["traces"][0]
        if old["stop_regret_ms"] <= 1e-9:
            continue
        new = task["policies"]["srop_refit"]["traces"][0]
        observations = candidates[task["task_id"]]["one_step"]["observations"]
        helpful = [row for row in observations if row["retrained_gain_ms"] > 1e-9]
        rows.append({"task_id": task["task_id"], "model_id": task["model_id"], "scale": task["scale"],
            "initial_train_count": task["initial_train_count"], "subset_seed": task["subset_seed"],
            "old_stop_budget": old["stop_budget"], "old_stop_regret_ms": old["stop_regret_ms"],
            "new_initial_selection_score": new["scores_before_measurement"][0],
            "new_first_object_id": new["selected_object_ids"][0],
            "new_stop_budget": new["stop_budget"], "new_stop_regret_ms": new["stop_regret_ms"],
            "helpful_single_measurements": [{key: row[key] for key in
                ("object_id", "operator", "shape", "target_only_gain_ms", "retrained_gain_ms", "other_predictions_changed")}
                for row in helpful]})
    return rows


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    directory = root / "data/bw1000_e2"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=directory / "e2c_graph_wide_budget_curves.json")
    parser.add_argument("--checkpoint-dir", type=Path, default=directory / "diagnostics_checkpoints")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("workers and limit must be positive")
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    parent = json.loads((directory / "e2c_graph_wide_diagnostics.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "shape_manifest.json").read_text(encoding="utf-8"))
    contract = json.loads((directory / "split_contract.json").read_text(encoding="utf-8"))
    suite = json.loads((directory / "e2c_task_suite.json").read_text(encoding="utf-8"))
    records = load_jsonl_records(directory / "profiling_dataset.jsonl")
    hardware = load_hardware(root / "data/bw1000_2card/bw1000_2card.calibrated.json")
    if (suite != build_task_suite(manifest, contract) or baseline["execution_status"] != "complete"
            or baseline["scope"] != "all_frozen_configs"
            or baseline["configuration"]["parent_report_id"] != content_id("budget-parent-report", parent)
            or baseline["configuration"]["dataset_id"] != content_id("diagnostics-data", records)
            or parent["configuration"]["hardware"] != hardware_to_dict(hardware)
            or {task["task_id"] for task in baseline["tasks"]} != {task["task_id"] for task in suite["tasks"]}):
        raise ValueError("baseline must be the unchanged complete frozen budget study")
    files = ("e2_refit_evaluation.py", "e2_refit_acquisition.py", "e2_residual.py", "e2_srop_proxy.py",
             "e2_budget_summary.py", "e2_audit.py", "e2_task_suite.py", "batched_placement_ranker.py", "structure.py", "structural.py")
    config = {
        "baseline_experiment_id": baseline["experiment_id"], "baseline_report_id": content_id("refit-baseline", baseline),
        "candidate_experiment_id": parent["experiment_id"], "suite_id": suite["suite_id"],
        "dataset_id": baseline["configuration"]["dataset_id"],
        "source_sha256": hashlib.sha256(b"".join((Path(__file__).parent / name).read_bytes() for name in files)).hexdigest(),
        "method": "srop_refit-v1: hypothetical nearest-shape mean update; five original quantiles; unchanged analytic cost and zero-score stopping",
        "limitations": "update-aware decision-loss proxy, not Bayesian posterior/KG; Oracle-containing frozen development pools; baseline timing retained from previous run",
    }
    experiment_id = content_id("e2-refit-evaluation", config)
    checkpoint_root = args.checkpoint_dir / experiment_id.split(":")[1][:16]
    candidate_root = args.checkpoint_dir / parent["experiment_id"].split(":")[1][:16]
    tasks = suite["tasks"] if args.limit is None else suite["tasks"][:args.limit]
    baseline_by_id = {task["task_id"]: task for task in baseline["tasks"]}
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
        published = [results[task["task_id"]] for task in tasks if task["task_id"] in results]
        _write_json(args.output, {"format": "fusion-placement-search.e2-refit-evaluation", "version": "1.0",
            "experiment_id": experiment_id, "configuration": config,
            "execution_status": "complete" if len(published) == len(tasks) else "running",
            "scope": "all_frozen_configs" if len(tasks) == len(suite["tasks"]) else "partial_smoke_test",
            "completed_configs": len(published), "summary": refit_summaries(published),
            "previous_wrong_stops": stop_diagnosis(published, parent), "tasks": published})

    context = (manifest, contract, records, hardware, candidate_root, baseline_by_id)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_initialize, initargs=(context,)) as executor:
        futures = {executor.submit(_run_task, task): task for task in tasks if task["task_id"] not in results}
        for future in as_completed(futures):
            task = futures[future]
            result = future.result()
            _write_checkpoint(checkpoint_root / (task["task_id"].split(":")[1] + ".json.gz"), result)
            results[task["task_id"]] = result["compact"]
            save_report()
            print(f"refit {len(results)}/{len(tasks)} {task['model_id']} {task['scale']} train={task['initial_train_count']}", flush=True)
    save_report()
    print(f"report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
