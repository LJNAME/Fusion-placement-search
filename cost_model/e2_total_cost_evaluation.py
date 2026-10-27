"""Shared-engine CPU-plus-recorded-profiling comparison on frozen E2 pools."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
import os
from pathlib import Path
import platform

from calibration.e2_dataset import make_dataset_view
from calibration.e2_manifest import content_id
from calibration.e2_records import load_jsonl_records
from simulator import default_fusion_registry
from simulator.dataset import plan_from_dict
from simulator.hardware import hardware_to_dict, load_hardware
from .batched_placement_ranker import BatchedPlacementRanker
from .e2_counterfactual_batch import counterfactual_scores
from .e2_diagnostics import _read_checkpoint, _write_checkpoint, _write_json, measurement_pool, workload_for_task
from .e2_scoring_benchmark import RankingAudit, TIMING_KEYS
from .e2_srop_proxy import (_active_profile, _oracle_profile, _run_policy_trajectory,
    _object_impact_ms_per_relative_error, _expected_decision_regret_ms, build_workload_graph, object_map_for_workload)
from .e2_task_suite import active_task_contract, build_task_suite
from .e2_total_cost import HORIZONS_S, PRIMARY_HORIZON_S, stopped_prefix, task_summary, summarize_tasks


RANDOM_SEEDS = (0, 1, 7, 23)
SPECS = [("random", seed) for seed in RANDOM_SEEDS] + [(name, None) for name in
    ("uncertainty_only", "impact_only", "srop_proxy", "srop_refit", "initial_only")]
_CONTEXT = None


def semantic_trace(trace):
    return [{key: value for key, value in point.items()
             if key not in TIMING_KEYS | {"decision_ready_algorithm_wall_time_s"}} for point in trace]


def compact_task(task, runs, preparation_s, reference_regret_ms, audit, reference_plan_id):
    row = {key: task[key] for key in ("task_id", "split_id", "model_id", "scale", "initial_train_count", "subset_seed")}
    row.update({"reference_plan_id": reference_plan_id, "reference_regret_ms": reference_regret_ms,
        "initial_regret_ms": runs[0]["trace"][0]["top1_regret_ms"], "preparation_s": preparation_s,
        "modes": task_summary(runs, reference_regret_ms), "initial_batch_audit": audit,
        "timed_runs": [{"policy": run["policy"], "seed": run["seed"], "repeat": run["repeat"],
            "raw_cpu_s": run["trace"][-1]["algorithm_wall_time_s"],
            "profiling_s": run["trace"][-1]["profiling_wall_time_s"],
            "measurements": run["trace"][-1]["budget"],
            "semantic_id": content_id("total-cost-semantic", semantic_trace(run["trace"]))} for run in runs]})
    return row


def _initialize(context):
    global _CONTEXT
    _CONTEXT = context


def _run_task(task):
    manifest, contract, records, hardware, candidate_root, proxy_root, frozen, repeats, indices = _CONTEXT
    filename = task["task_id"].split(":")[1] + ".json.gz"
    stored = _read_checkpoint(candidate_root / filename)
    proxy_control = _read_checkpoint(proxy_root / filename)["trajectories"]["distance_floor"]
    plans = [plan_from_dict(plan) for plan in stored["candidate_plans"]]
    workload = workload_for_task(task, manifest)
    graph = build_workload_graph(workload, task["scale"])
    mapping = object_map_for_workload(records, workload)
    runtime = active_task_contract(task, contract, manifest)
    runtime["active_profiling_protocol"].update(uncertainty_variant="distance_floor",
        refit_batch_scenarios=True, counterfactual_batch_scenarios=True)
    pool = measurement_pool(graph, mapping, task["train_object_ids"])
    exchange = ((0, 1), (1, 0))
    ranker = BatchedPlacementRanker(graph, plans, 2, exchange)
    preparation_s = ranker.preparation_wall_time_s
    reference_ids = [key for key, snapshot in ranker.prepared.items()
                     if all(group.device == 0 and len(group.node_ids) == 1 for group in snapshot.structure.groups)]
    if len(reference_ids) != 1:
        raise ValueError("a unique public singleton all-device-zero reference is required")
    reference_id = reference_ids[0]
    oracle = ranker.rank(_oracle_profile(records, hardware, mapping))
    reference_regret = max(0.0, next(row.makespan for row in oracle if row.plan_id == reference_id)-oracle[0].makespan)*1e3

    # Untimed audit of both newly batched baselines, on each initial public state.
    profile, uncertainty = _active_profile(records, runtime, hardware, mapping,
                                          split_id=task["split_id"], revealed_object_ids=())
    initial = ranker.rank(profile)
    audit = {}
    protocol = runtime["active_profiling_protocol"]
    for policy in ("impact_only", "srop_proxy"):
        scalar = RankingAudit(ranker)
        if policy == "impact_only":
            values = {key: _object_impact_ms_per_relative_error(graph, plans, profile, key,
                delta=protocol["impact_relative_delta"], registry=default_fusion_registry(),
                exchange_permutations=exchange, ranker=scalar, baseline_ranked=initial) for key in pool}
        else:
            values = {key: _expected_decision_regret_ms(graph, plans, profile, key, uncertainty[key], initial,
                standard_normal_quantiles=protocol["srop_proxy_standard_normal_quantiles"],
                q90_normalizer=protocol["uncertainty_q90_normalizer"], registry=default_fusion_registry(),
                exchange_permutations=exchange, ranker=scalar) for key in pool}
        batch = RankingAudit(ranker, scalar.stream)
        actual = counterfactual_scores(policy, profile, uncertainty, pool, protocol, batch, initial)
        if actual != values or len(batch.stream) != len(scalar.stream):
            raise AssertionError("baseline batching changed a score or full ranking")
        audit[policy] = {"ranking_lists": len(batch.stream), "candidate_scores": batch.candidate_scores,
                         "scores_id": content_id("total-cost-initial-scores", values)}

    runs = []
    reference_semantics = {}
    for repeat in range(repeats):
        offset = indices[task["task_id"]] % len(SPECS)
        order = SPECS[offset:]+SPECS[:offset]
        if repeat % 2:
            order = list(reversed(order))
        for policy, seed in order:
            current = copy.deepcopy(runtime)
            trace = _run_policy_trajectory("uncertainty_only" if policy == "initial_only" else policy,
                graph, plans, records, current, hardware, mapping, () if policy == "initial_only" else pool,
                oracle, split_id=task["split_id"], random_seed=seed, registry=default_fusion_registry(),
                exchange_permutations=exchange, ranker=ranker, capture_decision_timing=True)
            semantics = semantic_trace(trace)
            key = (policy, seed)
            if key in reference_semantics and semantics != reference_semantics[key]:
                raise AssertionError("timing repeats changed a policy trajectory")
            reference_semantics[key] = semantics
            if policy == "srop_refit" and semantic_trace(stopped_prefix(trace, stopping=True)) != frozen[task["task_id"]]["semantic_trajectory"]:
                raise AssertionError("Refit stopping prefix differs from the frozen scoring experiment")
            if policy == "srop_proxy" and semantic_trace(stopped_prefix(trace, stopping=True)) != semantic_trace(proxy_control):
                raise AssertionError("Proxy stopping prefix differs from its frozen distance control")
            if policy != "initial_only" and (trace[-1]["top1_regret_ms"] > 1e-9 or trace[-1]["strict_top_k_recall"] != 1.0):
                raise AssertionError("full observed pool did not recover the Oracle ranking")
            runs.append({"policy": policy, "seed": seed, "repeat": repeat,
                         "preparation_s": preparation_s, "trace": trace})
    row = compact_task(task, runs, preparation_s, reference_regret, audit, reference_id)
    return {"task_id": task["task_id"], "compact": row, "runs": runs}


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    directory = root / "data/bw1000_e2"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.repeats < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("workers, repeats and limit must be positive")
    def read(name):
        return json.loads((directory/name).read_text(encoding="utf-8"))
    controls, proxy, parent = (read(name) for name in
        ("e2c_scoring_benchmark.json", "e2c_uncertainty_stopping.json", "e2c_graph_wide_diagnostics.json"))
    manifest, contract, suite = (read(name) for name in ("shape_manifest.json", "split_contract.json", "e2c_task_suite.json"))
    records = load_jsonl_records(directory/"profiling_dataset.jsonl")
    hardware = load_hardware(root/"data/bw1000_2card/bw1000_2card.calibrated.json")
    ids = {task["task_id"] for task in suite["tasks"]}
    if (suite != build_task_suite(manifest, contract) or any(report["execution_status"] != "complete" or
            report["scope"] != "all_frozen_configs" or {row["task_id"] for row in report["tasks"]} != ids
            for report in (controls, proxy)) or controls["configuration"]["dataset_id"] != content_id("diagnostics-data", records) or
            controls["configuration"]["candidate_experiment_id"] != parent["experiment_id"] or
            parent["configuration"]["hardware"] != hardware_to_dict(hardware)):
        raise ValueError("frozen data or controls changed")
    config = {"suite_id": suite["suite_id"], "dataset_id": content_id("diagnostics-data", records),
        "scoring_control_id": content_id("total-cost-scoring-control", controls),
        "proxy_control_id": content_id("total-cost-proxy-control", proxy), "candidate_experiment_id": parent["experiment_id"],
        "workers": args.workers, "repeats": args.repeats, "random_seeds": list(RANDOM_SEEDS),
        "protocol": {"horizons_s": list(HORIZONS_S), "primary_horizon_s": PRIMARY_HORIZON_S,
            "clock": "serial: common fixed preparation + freshly measured CPU stages + recorded BW1000 profiling wall costs",
            "initial_fee": "per repeat, charge identical initial fit/rank time from initial-only to every policy; retain raw stage timings; only initialization is aligned",
            "decision": "retain public singleton all-device-zero plan until initial fit/rank; retain previous decision during acquisition, profiling and refit",
            "primary": "forced pool replay, decisions evaluated at common fixed time budgets",
            "secondary": "prefix replay at first nonpositive score; Random runs to exhaustion; no future outcomes affect cutoff",
            "controls": "initial-only fits/ranks once; reference-only holds public all-device-zero plan at zero incremental cost",
            "model": "same nearest-shape mean, distance-floor uncertainty, analytic x360 denominator, five samples",
            "order": "rotate strategy/seed order by task; reverse in alternate repeat; no audit in timing",
            "aggregation": "average seeds/repeats within task, then configurations equally",
            "scope": "development fixed Oracle-containing pools; existing training labels are sunk; not measured end-to-end wall time or independent blind tasks"},
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "logical_cpus": os.cpu_count()},
        "source_sha256": hashlib.sha256(b"".join((Path(__file__).parent/name).read_bytes() for name in
            ("e2_total_cost_evaluation.py", "e2_total_cost.py", "e2_counterfactual_batch.py", "e2_srop_proxy.py",
             "e2_refit_acquisition.py", "batched_placement_ranker.py", "e2_residual.py", "e2_uncertainty.py",
             "e2_audit.py", "e2_scoring_benchmark.py", "e2_task_suite.py"))+(root/"calibration/e2_dataset.py").read_bytes()).hexdigest()}
    experiment_id = content_id("e2-total-cost", config)
    folders = directory/"diagnostics_checkpoints"
    checkpoint_root = folders/experiment_id.split(":")[1][:16]
    candidate_root = folders/parent["experiment_id"].split(":")[1][:16]
    proxy_root = folders/proxy["experiment_id"].split(":")[1][:16]
    tasks = suite["tasks"] if args.limit is None else suite["tasks"][:args.limit]
    parents = {row["task_id"]: row for row in parent["tasks"]}
    frozen = {row["task_id"]: row for row in controls["tasks"]}
    results = {}
    for task in tasks:
        filename = task["task_id"].split(":")[1]+".json.gz"
        candidate = _read_checkpoint(candidate_root/filename)
        if candidate is None or candidate["zero"]["candidate_ids"] != parents[task["task_id"]]["zero"]["candidate_ids"] or _read_checkpoint(proxy_root/filename) is None:
            raise ValueError("candidate or Proxy checkpoints missing/mismatched")
        stored = _read_checkpoint(checkpoint_root/filename)
        if stored is not None:
            results[task["task_id"]] = stored["compact"]

    def save():
        rows = [results[task["task_id"]] for task in tasks if task["task_id"] in results]
        _write_json(args.output, {"format": "fusion-placement-search.e2-total-cost", "version": "1.0",
            "configuration": config, "experiment_id": experiment_id,
            "scope": "all_frozen_configs" if len(tasks) == len(suite["tasks"]) else "partial_smoke_test",
            "execution_status": "complete" if len(rows) == len(tasks) else "running", "completed_configs": len(rows),
            "summary": {"all": summarize_tasks(rows),
                "positive_initial": summarize_tasks([row for row in rows if row["initial_regret_ms"] > 1e-9]),
                "zero_initial": summarize_tasks([row for row in rows if row["initial_regret_ms"] <= 1e-9]),
                "by_model": {name: summarize_tasks([row for row in rows if row["model_id"] == name]) for name in sorted({row["model_id"] for row in rows})},
                "by_scale": {name: summarize_tasks([row for row in rows if row["scale"] == name]) for name in sorted({row["scale"] for row in rows})}},
            "tasks": rows})

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_initialize,
            initargs=((manifest, contract, records, hardware, candidate_root, proxy_root, frozen, args.repeats,
                       {task["task_id"]: index for index, task in enumerate(suite["tasks"])}),)) as executor:
        futures = {executor.submit(_run_task, task): task for task in tasks if task["task_id"] not in results}
        for future in as_completed(futures):
            task = futures[future]
            result = future.result()
            _write_checkpoint(checkpoint_root/(task["task_id"].split(":")[1]+".json.gz"), result)
            results[task["task_id"]] = result["compact"]
            save()
            print(f"total cost {len(results)}/{len(tasks)} {task['model_id']} {task['scale']}", flush=True)
    save()
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
