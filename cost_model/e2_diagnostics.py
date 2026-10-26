"""Run frozen E2 zero-budget and one-measurement diagnostics on CPU.

Oracle-derived candidate structures are allowed only in this fixed-pool
development diagnostic. Acquisition scores are computed before labels are
revealed, and marginal gains are evaluated separately using hidden labels.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

from calibration.e2_dataset import make_dataset_view
from calibration.e2_manifest import content_id
from calibration.e2_records import load_jsonl_records
from simulator.dataset import plan_from_dict, plan_to_dict
from simulator.fusion_rules import default_fusion_registry
from simulator.hardware import hardware_to_dict, load_hardware

from .e2_audit import PreparedCandidateRanker, ranking_metrics
from .e2_residual import _spearman
from .e2_srop_proxy import (
    E2Workload, ObjectCounterfactualProfile, _active_profile,
    _expected_decision_regret_ms, _object_impact_ms_per_relative_error,
    _oracle_profile, _profiling_cost_proxy_s, build_workload_graph,
    object_map_for_workload,
)
from .e2_task_suite import active_task_contract, build_task_suite
from .search_sensitivity import device_exchange_permutations, generate_candidate_pool, plan_id


FORMAT = "fusion-placement-search.e2-frozen-diagnostics"
VERSION = "1.0"
REGRET_TOLERANCE_MS = 1e-9  # Same absolute tolerance as the 1e-12 second rank ties.


def workload_for_task(task, manifest):
    model = next(item for item in manifest["models"] if item["model_id"] == task["model_id"])
    return E2Workload(task["task_id"], task["split_id"], task["model_id"],
                      model["model_family"], int(model["spec"]["seq_length"]))


def measurement_pool(graph, object_map, initial_train_ids):
    used = {object_map[str(node.extra.get("local_name", node.id))]
            for node in graph.nodes.values()
            if str(node.extra.get("local_name", node.id)) in object_map}
    return tuple(sorted(used - set(initial_train_ids)))


def _rank_setup(graph, plans, records, hardware, object_map):
    oracle = _oracle_profile(records, hardware, object_map)
    exchange = device_exchange_permutations(oracle.device_classes)
    ranker = PreparedCandidateRanker(graph, plans, hardware.n_devs, exchange)
    oracle_ranked = ranker.rank(oracle)
    return ranker, oracle_ranked, exchange


def evaluate_zero_budget(task, plans, records, manifest, contract, hardware, *, graph=None):
    started = time.perf_counter()
    workload = workload_for_task(task, manifest)
    graph = graph or build_workload_graph(workload, task["scale"])
    runtime = active_task_contract(task, contract, manifest)
    object_map = object_map_for_workload(records, workload)
    pool = measurement_pool(graph, object_map, task["train_object_ids"])
    ranker, oracle_ranked, _ = _rank_setup(graph, plans, records, hardware, object_map)
    baseline, uncertainty = _active_profile(records, runtime, hardware, object_map,
                                            split_id=workload.split_id, revealed_object_ids=())
    ranked = ranker.rank(baseline)
    metrics = ranking_metrics(oracle_ranked, ranked)
    by_id = {record["object_id"]: record for record in records}
    errors = [abs(math.log(by_id[key]["measurement"]["duration_s"] / baseline.durations_s[key]))
              for key in pool]
    coverage = statistics.fmean(float(error <= uncertainty[key]) for key, error in zip(pool, errors)) if pool else None
    full, _ = _active_profile(records, runtime, hardware, object_map,
                             split_id=workload.split_id, revealed_object_ids=pool)
    full_ranked = ranker.rank(full)
    full_metrics = ranking_metrics(oracle_ranked, full_ranked)
    oracle_by_id = {item.plan_id: item.makespan for item in oracle_ranked}
    full_error = max(abs(item.makespan - oracle_by_id[item.plan_id]) for item in full_ranked)
    if full_error > 1e-12 or full_metrics["strict_top_k_recall"] != 1.0:
        raise AssertionError("full active budget must recover every candidate's oracle cost and ranking")
    original_split = next(split for split in contract["splits"] if split["split_id"] == task["split_id"])
    unseen = set(original_split["hidden_object_ids"])
    runner_up = next((item.makespan for item in oracle_ranked
                      if item.makespan > oracle_ranked[0].makespan + 1e-12), None)
    return {
        "task_id": task["task_id"], "split_id": task["split_id"],
        "model_id": task["model_id"], "scale": task["scale"],
        "nodes": len(graph.nodes), "edges": len(graph.edges),
        "initial_train_count": task["initial_train_count"], "subset_seed": task["subset_seed"],
        "candidate_plans": len(ranked), "measurement_pool_objects": len(pool),
        "original_unseen_pool_objects": len(set(pool) & unseen),
        "withheld_original_train_pool_objects": len(set(pool) & set(original_split["train_object_ids"])),
        "measurement_pool_object_ids": list(pool),
        "positive_initial_regret": metrics["top1_regret_ms"] > REGRET_TOLERANCE_MS,
        "unmeasured_log_mae": statistics.fmean(errors) if errors else None,
        "unmeasured_interval_coverage": coverage,
        "oracle_best_ms": oracle_ranked[0].makespan * 1e3,
        "oracle_next_strict_cost_margin_ms": ((runner_up - oracle_ranked[0].makespan) * 1e3
                                               if runner_up is not None else None),
        "full_budget_max_cost_error_s": full_error,
        "full_budget_recovers_oracle": True,
        "candidate_ids": [item.plan_id for item in ranked],
        "evaluation_wall_time_s": time.perf_counter() - started,
        **metrics,
    }


def acquisition_scores(graph, plans, profile, uncertainty, public_records, pool, ranker, exchange, protocol):
    """No oracle labels enter this function; input rows are masked public views."""

    by_id = {record["object_id"]: record for record in public_records}
    registry = default_fusion_registry()
    baseline_ranked = ranker.rank(profile)
    rows = {}
    for key in pool:
        started = time.perf_counter()
        q90 = uncertainty[key]
        cost = _profiling_cost_proxy_s(by_id[key], int(protocol["profiling_repetitions"]))
        impact = _object_impact_ms_per_relative_error(
            graph, plans, profile, key, delta=float(protocol["impact_relative_delta"]),
            registry=registry, exchange_permutations=exchange,
            ranker=ranker, baseline_ranked=baseline_ranked)
        expected = _expected_decision_regret_ms(
            graph, plans, profile, key, q90, baseline_ranked,
            standard_normal_quantiles=protocol["srop_proxy_standard_normal_quantiles"],
            q90_normalizer=float(protocol["uncertainty_q90_normalizer"]),
            registry=registry, exchange_permutations=exchange, ranker=ranker)
        rows[key] = {
            "uncertainty_q90_log": q90,
            "uncertainty_relative": math.expm1(q90),
            "impact_ms_per_relative_error": impact,
            "expected_decision_regret_ms": expected,
            "profiling_cost_proxy_s": cost,
            "uncertainty_only": math.expm1(q90) / cost,
            "impact_only": impact / cost,
            "srop_proxy": expected / cost,
            "acquisition_wall_time_s": time.perf_counter() - started,
        }
    return rows


def evaluate_one_step(task, plans, records, manifest, contract, hardware, *, graph=None):
    """Reveal each possible next object separately; never feed outcomes into acquisition."""

    started = time.perf_counter()
    workload = workload_for_task(task, manifest)
    graph = graph or build_workload_graph(workload, task["scale"])
    runtime = active_task_contract(task, contract, manifest)
    object_map = object_map_for_workload(records, workload)
    pool = measurement_pool(graph, object_map, task["train_object_ids"])
    ranker, oracle_ranked, exchange = _rank_setup(graph, plans, records, hardware, object_map)
    baseline, uncertainty = _active_profile(records, runtime, hardware, object_map,
                                            split_id=workload.split_id, revealed_object_ids=())
    baseline_ranked = ranker.rank(baseline)
    initial = ranking_metrics(oracle_ranked, baseline_ranked)
    public_records = make_dataset_view(records, runtime, split_id=workload.split_id,
                                       mode="active_calibration", role="sampler")
    scores = acquisition_scores(graph, plans, baseline, uncertainty, public_records,
                                 pool, ranker, exchange, runtime["active_profiling_protocol"])
    # All acquisition scores are fixed before any hidden measurement is inspected here.
    by_id = {record["object_id"]: record for record in records}
    observations = []
    for key in pool:
        true_s = float(by_id[key]["measurement"]["duration_s"])
        predicted_s = baseline.durations_s[key]
        target_only = ObjectCounterfactualProfile(baseline, key, true_s / predicted_s - 1.0)
        direct_ranked = ranker.rank(target_only)
        direct = ranking_metrics(oracle_ranked, direct_ranked)
        updated, _ = _active_profile(records, runtime, hardware, object_map,
                                      split_id=workload.split_id, revealed_object_ids=(key,))
        updated_ranked = ranker.rank(updated)
        actual = ranking_metrics(oracle_ranked, updated_ranked)
        others = [other for other in pool if other != key]
        other_changes = [abs(updated.durations_s[other] - baseline.durations_s[other]) for other in others]
        old_errors = [abs(math.log(by_id[other]["measurement"]["duration_s"] / baseline.durations_s[other])) for other in others]
        new_errors = [abs(math.log(by_id[other]["measurement"]["duration_s"] / updated.durations_s[other])) for other in others]
        observations.append({
            "object_id": key, "operator": by_id[key]["kernel_key"]["operator"],
            "shape": by_id[key]["kernel_key"]["shape"],
            "predicted_duration_s": predicted_s, "measured_duration_s": true_s,
            "actual_abs_log_error": abs(math.log(true_s / predicted_s)),
            "inside_q90_interval": abs(math.log(true_s / predicted_s)) <= uncertainty[key],
            "profiling_wall_time_s": by_id[key]["measurement"].get("profiling_wall_time_s"),
            **scores[key],
            "target_only_regret_ms": direct["top1_regret_ms"],
            "retrained_regret_ms": actual["top1_regret_ms"],
            "target_only_gain_ms": initial["top1_regret_ms"] - direct["top1_regret_ms"],
            "retrained_gain_ms": initial["top1_regret_ms"] - actual["top1_regret_ms"],
            "retrained_gain_ms_per_cost_proxy_s": (initial["top1_regret_ms"] - actual["top1_regret_ms"]) / scores[key]["profiling_cost_proxy_s"],
            "model_update_additional_gain_ms": direct["top1_regret_ms"] - actual["top1_regret_ms"],
            "top1_decision_changed": actual["predicted_top1_plan_id"] != initial["predicted_top1_plan_id"],
            "retrained_top1_plan_id": actual["predicted_top1_plan_id"],
            "retrained_strict_top_k_recall": actual["strict_top_k_recall"],
            "retrained_best_of_k_regret_ms": actual["best_of_k_regret_ms"],
            "other_predictions_changed": sum(change > 1e-15 for change in other_changes),
            "other_predictions_max_change_s": max(other_changes, default=0.0),
            "other_log_mae_before": statistics.fmean(old_errors) if old_errors else None,
            "other_log_mae_after": statistics.fmean(new_errors) if new_errors else None,
        })
    gains = [row["retrained_gain_ms"] for row in observations]
    best_gain = max(gains, default=0.0)
    policies = {}
    for policy in ("uncertainty_only", "impact_only", "srop_proxy"):
        if observations:
            selected = min(observations, key=lambda row: (-row[policy], row["object_id"]))
            policies[policy] = {
                "selected_object_id": selected["object_id"],
                "selected_operator": selected["operator"],
                "selected_gain_ms": selected["retrained_gain_ms"],
                "selected_regret_ms": selected["retrained_regret_ms"],
                "missed_best_single_step_gain_ms": best_gain - selected["retrained_gain_ms"],
                "stop_recommended": selected[policy] <= 0.0,
                "selection_score": selected[policy],
                "score_gain_spearman": _spearman([row[policy] for row in observations], gains),
            }
    policies["random"] = {
        "definition": "exact uniform expectation over all one-object choices",
        "gain_ms_mean": statistics.fmean(gains) if gains else 0.0,
        "regret_ms_mean": statistics.fmean(row["retrained_regret_ms"] for row in observations) if observations else initial["top1_regret_ms"],
    }
    return {
        "task_id": task["task_id"], "initial_top1_regret_ms": initial["top1_regret_ms"],
        "measurement_objects": len(observations),
        "improving_objects": sum(gain > REGRET_TOLERANCE_MS for gain in gains),
        "harmful_objects": sum(gain < -REGRET_TOLERANCE_MS for gain in gains),
        "best_single_step_gain_ms": best_gain,
        "model_update_changes_outcome_objects": sum(abs(row["model_update_additional_gain_ms"]) > REGRET_TOLERANCE_MS for row in observations),
        "uncertainty_vs_actual_error_spearman": _spearman(
            [row["uncertainty_q90_log"] for row in observations],
            [row["actual_abs_log_error"] for row in observations]) if observations else 0.0,
        "policies": policies, "observations": observations,
        "evaluation_wall_time_s": time.perf_counter() - started,
    }


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _write_checkpoint(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    temporary.replace(path)


def _read_checkpoint(path):
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def summarize(results, total_tasks):
    zeros = [result["zero"] for result in results]
    def group_summary(items):
        return {
            "configs": len(items),
            "positive_initial_regret_configs": sum(item["positive_initial_regret"] for item in items),
            "initial_regret_ms_mean": statistics.fmean(item["top1_regret_ms"] for item in items) if items else None,
            "initial_normalized_regret_max": max((item["normalized_top1_regret"] for item in items), default=0.0),
            "strict_top_k_recall_mean": statistics.fmean(item["strict_top_k_recall"] for item in items) if items else None,
        }
    by_budget = {}
    for key in ("16", "32", "full"):
        subset = [item for item in zeros if (str(item["initial_train_count"]) if item["subset_seed"] is not None else "full") == key]
        by_budget[key] = group_summary(subset)
    diagnostics = [result["one_step"] for result in results if "one_step" in result]
    return {
        "total_frozen_configs": total_tasks, "completed_zero_configs": len(zeros),
        "completed_one_step_configs": len(diagnostics),
        "unique_model_scale_groups": len({(item["model_id"], item["scale"]) for item in zeros}),
        "all_configs": group_summary(zeros),
        "by_initial_train_budget": by_budget,
        "by_scale": {scale: group_summary([item for item in zeros if item["scale"] == scale]) for scale in ("layer", "model")},
        "full_budget_recovery_configs": sum(item["full_budget_recovers_oracle"] for item in zeros),
        "configs_with_improving_single_measurement": sum(item["improving_objects"] > 0 for item in diagnostics),
        "configs_where_model_update_changes_outcome": sum(item["model_update_changes_outcome_objects"] > 0 for item in diagnostics),
        "one_step_policy_mean_gain_ms": {
            policy: statistics.fmean(item["policies"][policy]["selected_gain_ms"] for item in diagnostics if policy in item["policies"])
            for policy in ("uncertainty_only", "impact_only", "srop_proxy")
        } if diagnostics else {},
    }


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=root / "data/bw1000_e2/e2c_task_suite.json")
    parser.add_argument("--manifest", type=Path, default=root / "data/bw1000_e2/shape_manifest.json")
    parser.add_argument("--contract", type=Path, default=root / "data/bw1000_e2/split_contract.json")
    parser.add_argument("--dataset", type=Path, default=root / "data/bw1000_e2/profiling_dataset.jsonl")
    parser.add_argument("--hardware", type=Path, default=root / "data/bw1000_2card/bw1000_2card.calibrated.json")
    parser.add_argument("--checkpoint-dir", type=Path, default=root / "data/bw1000_e2/diagnostics_checkpoints")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("zero", "one-step", "all"), default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--seeds", default="0,1,7,23")
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--rollouts", type=int, default=3)
    args = parser.parse_args(argv)
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if suite != build_task_suite(manifest, contract):
        raise ValueError("frozen task suite differs from its label-independent construction")
    records = load_jsonl_records(args.dataset)
    hardware = load_hardware(args.hardware)
    source_names = ("e2_diagnostics.py", "e2_srop_proxy.py", "e2_audit.py", "e2_task_suite.py", "e2_residual.py", "structure.py", "structural.py")
    source_hash = hashlib.sha256(b"".join((Path(__file__).parent / name).read_bytes() for name in source_names)).hexdigest()
    config = {
        "suite_id": suite["suite_id"], "dataset_id": content_id("diagnostics-data", records),
        "hardware": hardware_to_dict(hardware), "source_sha256": source_hash,
        "candidates_per_source": args.candidates, "seeds": [int(value) for value in args.seeds.split(",")],
        "iterations": args.iterations, "rollouts": args.rollouts,
        "diagnostic_oracle_union": True, "oracle_labels_available_to_acquisition": False,
        "candidate_scope": "predicted+oracle search union; development diagnostic only",
        "random_one_step": "exact uniform expectation over all objects",
        "regret_tolerance_ms": REGRET_TOLERANCE_MS,
    }
    experiment_id = content_id("e2-diagnostics-experiment", config)
    checkpoint_root = args.checkpoint_dir / experiment_id.split(":")[1][:16]
    tasks = suite["tasks"] if args.limit is None else suite["tasks"][:args.limit]
    if not tasks:
        raise ValueError("at least one task is required")
    results = {}
    for task in tasks:
        stored = _read_checkpoint(checkpoint_root / (task["task_id"].split(":")[1] + ".json.gz"))
        if stored is not None:
            results[task["task_id"]] = stored
    def save_report():
        published = [{key: value for key, value in results[task["task_id"]].items() if key != "candidate_plans"}
                     for task in tasks if task["task_id"] in results]
        summary = summarize(published, len(suite["tasks"]))
        complete = summary["completed_zero_configs"] == len(tasks) and (
            args.phase == "zero" or summary["completed_one_step_configs"] == len(tasks))
        _write_json(args.output, {"format": FORMAT, "version": VERSION,
            "experiment_id": experiment_id, "configuration": config,
            "scope": "all_frozen_configs" if len(tasks) == len(suite["tasks"]) else "partial_smoke_test",
            "execution_status": "complete" if complete else "running",
            "requested_phase": args.phase,
            "summary": summary, "tasks": published})
    # Finish zero-budget screening for every task before one-step diagnostics.
    for index, task in enumerate(tasks, 1):
        if task["task_id"] in results:
            continue
        workload = workload_for_task(task, manifest)
        graph = build_workload_graph(workload, task["scale"])
        runtime = active_task_contract(task, contract, manifest)
        object_map = object_map_for_workload(records, workload)
        predicted, _ = _active_profile(records, runtime, hardware, object_map,
                                       split_id=workload.split_id, revealed_object_ids=())
        oracle = _oracle_profile(records, hardware, object_map)
        generation_started = time.perf_counter()
        oracle_key = content_id("oracle-pool", {"model_id": task["model_id"], "scale": task["scale"]}).split(":")[1]
        oracle_path = checkpoint_root / ("oracle-" + oracle_key + ".json.gz")
        cached_oracle = _read_checkpoint(oracle_path)
        oracle_pool_reused = cached_oracle is not None
        if cached_oracle is None:
            oracle_plans, oracle_metadata = generate_candidate_pool(graph, oracle,
                candidate_count=args.candidates, seeds=config["seeds"],
                iterations=args.iterations, rollouts=args.rollouts)
            cached_oracle = {"plans": [plan_to_dict(plan) for plan in oracle_plans], "metadata": oracle_metadata}
            _write_checkpoint(oracle_path, cached_oracle)
        predicted_plans, predicted_metadata = generate_candidate_pool(graph, predicted,
            candidate_count=args.candidates, seeds=config["seeds"],
            iterations=args.iterations, rollouts=args.rollouts)
        exchange = device_exchange_permutations(predicted.device_classes)
        plans_by_id = {plan_id(plan, exchange): plan for plan in predicted_plans}
        oracle_plans = [plan_from_dict(plan) for plan in cached_oracle["plans"]]
        plans_by_id.update({plan_id(plan, exchange): plan for plan in oracle_plans})
        plans = list(plans_by_id.values())
        result = {"task_id": task["task_id"], "candidate_plans": [plan_to_dict(plan) for plan in plans],
            "candidate_generation_wall_time_s": time.perf_counter() - generation_started,
            "candidate_pool": {"predicted": predicted_metadata, "oracle": cached_oracle["metadata"],
                               "union_candidates": len(plans), "oracle_pool_reused": oracle_pool_reused},
            "zero": evaluate_zero_budget(task, plans, records, manifest, contract, hardware, graph=graph)}
        results[task["task_id"]] = result
        _write_checkpoint(checkpoint_root / (task["task_id"].split(":")[1] + ".json.gz"), result)
        save_report()
        zero = result["zero"]
        print(f"zero {index}/{len(tasks)} {task['model_id']} {task['scale']} train={task['initial_train_count']} pool={zero['measurement_pool_objects']} regret_ms={zero['top1_regret_ms']:.9f}", flush=True)
    save_report()
    if args.phase in ("one-step", "all"):
        for index, task in enumerate(tasks, 1):
            result = results[task["task_id"]]
            if "one_step" in result:
                continue
            plans = [plan_from_dict(plan) for plan in result["candidate_plans"]]
            result["one_step"] = evaluate_one_step(task, plans, records, manifest, contract, hardware)
            _write_checkpoint(checkpoint_root / (task["task_id"].split(":")[1] + ".json.gz"), result)
            save_report()
            diagnostic = result["one_step"]
            print(f"one-step {index}/{len(tasks)} {task['model_id']} {task['scale']} train={task['initial_train_count']} improving={diagnostic['improving_objects']} best_gain_ms={diagnostic['best_single_step_gain_ms']:.9f}", flush=True)
    print(json.dumps(summarize([results[task["task_id"]] for task in tasks], len(suite["tasks"])), indent=2), flush=True)
    print(f"report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
