"""Paired CPU benchmark and exact ranking audit of batched SROP scenarios."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time

import numpy as np

from calibration.e2_manifest import content_id
from calibration.e2_records import load_jsonl_records
from simulator import default_fusion_registry
from simulator.dataset import plan_from_dict
from simulator.hardware import hardware_to_dict, load_hardware
from .batched_placement_ranker import BatchedPlacementRanker
from .e2_diagnostics import _read_checkpoint, _write_checkpoint, _write_json, measurement_pool, workload_for_task
from .e2_srop_proxy import _active_profile, _oracle_profile, _run_policy_trajectory, build_workload_graph, object_map_for_workload
from .e2_task_suite import active_task_contract, build_task_suite
from .e2_uncertainty_stopping import summarize_trajectory


TIMING_KEYS = {"predictor_fit_wall_time_s", "selection_wall_time_s", "ranking_wall_time_s", "algorithm_wall_time_s"}
_CONTEXT = None


def semantic_trajectory(trace):
    return [{key: value for key, value in point.items() if key not in TIMING_KEYS} for point in trace]


class RankingAudit:
    """Compare full rankings in their original scenario order, outside timing."""
    def __init__(self, ranker, expected=None):
        self.ranker = ranker
        self.expected = expected
        self.stream = []
        self.candidate_scores = 0

    def _record(self, ranked):
        row = [(item.plan_id, item.makespan, item.confidence) for item in ranked]
        if self.expected is not None and (len(self.stream) >= len(self.expected) or
                                         row != self.expected[len(self.stream)]):
            raise AssertionError("full ranking, makespan or confidence changed")
        self.stream.append(row)
        self.candidate_scores += len(row)
        return ranked

    def rank(self, profile):
        return self._record(self.ranker.rank(profile))

    def rank_many(self, profiles):
        return [self._record(ranked) for ranked in self.ranker.rank_many(profiles)]


def _initialize(context):
    global _CONTEXT
    _CONTEXT = context


def _run_task(task):
    manifest, contract, records, hardware, candidate_root, baseline_root, controls, repeats, task_indices = _CONTEXT
    filename = task["task_id"].split(":")[1] + ".json.gz"
    stored = _read_checkpoint(candidate_root / filename)
    frozen = _read_checkpoint(baseline_root / filename)
    plans = [plan_from_dict(plan) for plan in stored["candidate_plans"]]
    workload = workload_for_task(task, manifest)
    graph = build_workload_graph(workload, task["scale"])
    mapping = object_map_for_workload(records, workload)
    runtime = active_task_contract(task, contract, manifest)
    runtime["active_profiling_protocol"]["uncertainty_variant"] = "distance_floor"
    pool = measurement_pool(graph, mapping, task["train_object_ids"])
    exchange = ((0, 1), (1, 0))
    prepared = BatchedPlacementRanker(graph, plans, 2, exchange)
    oracle = prepared.rank(_oracle_profile(records, hardware, mapping))

    def run(batched, ranker):
        current = copy.deepcopy(runtime)
        current["active_profiling_protocol"]["refit_batch_scenarios"] = batched
        return _run_policy_trajectory("srop_refit", graph, plans, records, current, hardware,
            mapping, pool, oracle, split_id=task["split_id"], random_seed=None,
            registry=default_fusion_registry(), exchange_permutations=exchange, ranker=ranker,
            stop_on_recommendation=True)

    # Untimed validation, including all hypothetical candidate rankings.
    scalar_audit = RankingAudit(prepared)
    reference = run(False, scalar_audit)
    expected = semantic_trajectory(reference)
    if expected != semantic_trajectory(frozen["trajectory"]):
        raise AssertionError("scalar reference differs from frozen factorial decisions")
    batch_audit = RankingAudit(BatchedPlacementRanker(graph, plans, 2, exchange), scalar_audit.stream)
    actual = run(True, batch_audit)
    if len(scalar_audit.stream) != len(batch_audit.stream) or semantic_trajectory(actual) != expected:
        raise AssertionError("batching changed a score, selected object, metric or stopping result")
    control = controls[task["task_id"]]["cells"]["distance_refit"]
    if summarize_trajectory(frozen["trajectory"], len(pool)) != control:
        raise AssertionError("frozen control checkpoint mismatched")
    full, _ = _active_profile(records, runtime, hardware, mapping,
                             split_id=task["split_id"], revealed_object_ids=pool)
    full_ranked = prepared.rank(full)
    if [(row.plan_id, row.makespan) for row in full_ranked] != [(row.plan_id, row.makespan) for row in oracle]:
        raise AssertionError("full-reveal profile did not recover Oracle rankings and costs")

    runs = []
    for repeat in range(repeats):
        order = (False, True) if (task_indices[task["task_id"]]+repeat) % 2 == 0 else (True, False)
        for batched in order:
            ranker = BatchedPlacementRanker(graph, plans, 2, exchange)
            started = time.perf_counter()
            trace = run(batched, ranker)
            elapsed = time.perf_counter()-started
            if semantic_trajectory(trace) != expected:
                raise AssertionError("clean timed replay changed semantics")
            runs.append({"repeat": repeat, "variant": "scenario_batch" if batched else "scalar_scenarios",
                "order": ["scenario_batch" if item else "scalar_scenarios" for item in order],
                "run_wall_time_s": elapsed, "preparation_wall_time_s": ranker.preparation_wall_time_s,
                **{key: trace[-1][key] for key in TIMING_KEYS}})
    row = {key: task[key] for key in ("task_id", "split_id", "model_id", "scale", "initial_train_count", "subset_seed")}
    row.update({"nodes": len(graph.nodes), "candidate_plans": len(prepared.identifiers),
        "initial_regret_ms": reference[0]["top1_regret_ms"], "stop_budget": reference[-1]["budget"],
        "stop_regret_ms": reference[-1]["top1_regret_ms"],
        "frozen_count_auc_ms": control["regret_budget_auc_ms"],
        "ranking_lists_checked": len(batch_audit.stream), "candidate_scores_checked": batch_audit.candidate_scores,
        "ranking_stream_id": content_id("scoring-ranking-stream", scalar_audit.stream),
        "semantic_trajectory_id": content_id("scoring-semantics", expected),
        "semantic_trajectory": expected, "timed_runs": runs})
    return {"task_id": task["task_id"], "compact": row}


def summarize(tasks):
    if not tasks:
        return {}
    variants = {}
    per_task = {}
    for variant in ("scalar_scenarios", "scenario_batch"):
        per_task[variant] = [{key: statistics.median(run[key] for run in task["timed_runs"] if run["variant"] == variant)
                            for key in ("run_wall_time_s", *sorted(TIMING_KEYS))} for task in tasks]
        variants[variant] = {f"mean_{key}": statistics.fmean(point[key] for point in per_task[variant])
                            for key in per_task[variant][0]}
    ratios = [old["algorithm_wall_time_s"]/new["algorithm_wall_time_s"]
              for old, new in zip(per_task["scalar_scenarios"], per_task["scenario_batch"])]
    return {"configs": len(tasks), "variants": variants,
        "mean_time_ratio": variants["scalar_scenarios"]["mean_algorithm_wall_time_s"]/
                           variants["scenario_batch"]["mean_algorithm_wall_time_s"],
        "median_paired_speedup": statistics.median(ratios), "min_paired_speedup": min(ratios),
        "faster_configs": sum(ratio > 1 for ratio in ratios),
        "ranking_lists_checked": sum(task["ranking_lists_checked"] for task in tasks),
        "candidate_scores_checked": sum(task["candidate_scores_checked"] for task in tasks),
        "wrong_stop_configs": sum(task["stop_regret_ms"] > 1e-9 for task in tasks),
        "unchanged_mean_count_auc_ms": statistics.fmean(task["frozen_count_auc_ms"] for task in tasks)}


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    directory = root / "data/bw1000_e2"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.repeats < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("workers, repeats and limit must be positive")
    def read(name):
        return json.loads((directory / name).read_text(encoding="utf-8"))
    baseline, parent = read("e2c_factorial_evaluation.json"), read("e2c_graph_wide_diagnostics.json")
    manifest, contract, suite = (read(name) for name in ("shape_manifest.json", "split_contract.json", "e2c_task_suite.json"))
    records = load_jsonl_records(directory / "profiling_dataset.jsonl")
    hardware = load_hardware(root / "data/bw1000_2card/bw1000_2card.calibrated.json")
    ids = {task["task_id"] for task in suite["tasks"]}
    if (suite != build_task_suite(manifest, contract) or baseline["execution_status"] != "complete" or
            baseline["scope"] != "all_frozen_configs" or {task["task_id"] for task in baseline["tasks"]} != ids or
            baseline["configuration"]["candidate_experiment_id"] != parent["experiment_id"] or
            baseline["configuration"]["dataset_id"] != content_id("diagnostics-data", records) or
            parent["configuration"]["hardware"] != hardware_to_dict(hardware)):
        raise ValueError("frozen inputs changed")
    config = {"baseline_report_id": content_id("scoring-baseline", baseline), "suite_id": suite["suite_id"],
        "dataset_id": content_id("diagnostics-data", records), "candidate_experiment_id": parent["experiment_id"],
        "workers": args.workers, "repeats_per_variant": args.repeats,
        "protocol": "untimed full ranking audit, then fresh paired CPU replays with alternating AB/BA order; no audit inside timed runs",
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
                        "processor": platform.processor(), "logical_cpus": os.cpu_count()},
        "limitations": "concurrent workers share CPU; all fixed development pools contain Oracle candidates; replay CPU time is not GPU end-to-end speedup",
        "source_sha256": hashlib.sha256(b"".join((Path(__file__).parent / name).read_bytes() for name in
            ("e2_scoring_benchmark.py", "e2_refit_acquisition.py", "batched_placement_ranker.py", "e2_residual.py",
             "e2_uncertainty.py", "e2_srop_proxy.py", "e2_audit.py", "e2_task_suite.py", "e2_uncertainty_stopping.py"))
             + (root / "calibration/e2_dataset.py").read_bytes()).hexdigest()}
    experiment_id = content_id("e2-scoring-benchmark", config)
    folders = directory / "diagnostics_checkpoints"
    checkpoint_root = folders / experiment_id.split(":")[1][:16]
    candidate_root = folders / parent["experiment_id"].split(":")[1][:16]
    baseline_root = folders / baseline["experiment_id"].split(":")[1][:16]
    tasks = suite["tasks"] if args.limit is None else suite["tasks"][:args.limit]
    parents = {row["task_id"]: row for row in parent["tasks"]}
    controls = {row["task_id"]: row for row in baseline["tasks"]}
    results = {}
    for task in tasks:
        filename = task["task_id"].split(":")[1] + ".json.gz"
        candidates = _read_checkpoint(candidate_root / filename)
        control = _read_checkpoint(baseline_root / filename)
        if (candidates is None or control is None or control["compact"] != controls[task["task_id"]] or
                candidates["zero"]["candidate_ids"] != parents[task["task_id"]]["zero"]["candidate_ids"]):
            raise ValueError("candidate or control checkpoint missing/mismatched")
        stored = _read_checkpoint(checkpoint_root / filename)
        if stored is not None:
            results[task["task_id"]] = stored["compact"]

    def save_report():
        rows = [results[task["task_id"]] for task in tasks if task["task_id"] in results]
        _write_json(args.output, {"format": "fusion-placement-search.e2-scoring-benchmark", "version": "1.0",
            "experiment_id": experiment_id, "configuration": config,
            "scope": "all_frozen_configs" if len(tasks) == len(suite["tasks"]) else "partial_smoke_test",
            "execution_status": "complete" if len(rows) == len(tasks) else "running", "completed_configs": len(rows),
            "summary": {"all": summarize(rows), "by_scale": {scale: summarize([row for row in rows if row["scale"] == scale])
                for scale in sorted({row["scale"] for row in rows})},
                "by_model": {model: summarize([row for row in rows if row["model_id"] == model])
                             for model in sorted({row["model_id"] for row in rows})}}, "tasks": rows})

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_initialize,
            initargs=((manifest, contract, records, hardware, candidate_root, baseline_root, controls, args.repeats,
                       {task["task_id"]: index for index, task in enumerate(suite["tasks"])}),)) as executor:
        futures = {executor.submit(_run_task, task): task for task in tasks if task["task_id"] not in results}
        for future in as_completed(futures):
            task = futures[future]
            result = future.result()
            _write_checkpoint(checkpoint_root / (task["task_id"].split(":")[1] + ".json.gz"), result)
            results[task["task_id"]] = result["compact"]
            save_report()
            print(f"scoring {len(results)}/{len(tasks)} {task['model_id']} {task['scale']}", flush=True)
    save_report()
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
