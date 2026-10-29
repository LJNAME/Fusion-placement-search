"""D0-D2 development diagnosis using exposed kernel and GPU plan labels.

Every subset refits the existing predictor and reselects from the entire
measured physical candidate pool. This is restricted exact selection, not
unrestricted MCTS re-search or an active-policy comparison.
"""

import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import random
import statistics
import time

from calibration.e2_manifest import content_id
from calibration.e2_records import load_environment_registry, load_jsonl_records
from calibration.evaluate_gpu_holdout import evaluate_measurements
from calibration.gpu_holdout import ROOT, load_freeze
from calibration.reference_validation import check as check_reference, DEFAULT_BUNDLE
from cost_model.e2_srop_proxy import (
    E2Workload, _active_profile, _oracle_profile, object_map_for_workload,
)
from cost_model.reference_semantics import HostSubmissionProfile
from cost_model.reference_srop import ReferenceCandidateRanker, runtime_contract
from execution.audit_gpu_holdout import audit_execution, write_json
from simulator.dataset import graph_from_dict, plan_from_dict
from simulator.hardware import hardware_from_dict


def subset_design(keys, pair_limit=2):
    """All singletons/pairs plus fixed prefixes and the full endpoint."""
    keys = tuple(sorted(keys))
    if pair_limit != 2:
        raise ValueError("this protocol exhausts pairs only; higher orders use fixed prefixes")
    states = {(), keys}
    states.update((key,) for key in keys)
    states.update(itertools.combinations(keys, 2))
    orders = {"key_ascending": list(keys), "key_descending": list(reversed(keys))}
    shuffled = list(keys)
    random.Random(0).shuffle(shuffled)
    orders["shuffle_seed0"] = shuffled
    for order in orders.values():
        states.update(tuple(sorted(order[:count])) for count in range(2, len(keys) + 1))
    return sorted(states, key=lambda state: (len(state), state)), orders


def assess_subsets(evaluate, keys, profiling_costs, *, epsilons=(.001, .005, .01)):
    """Evaluator-owned opportunity enumeration, never a deployable policy."""
    keys = tuple(sorted(keys))
    if len(set(keys)) != len(keys):
        raise ValueError("measurement keys must be unique")
    states, orders = subset_design(keys)
    rows, by_state = [], {}
    for state in states:
        row = dict(evaluate(state))
        if not math.isfinite(row["gpu_relative_regret"]) or row["gpu_relative_regret"] < 0:
            raise ValueError("GPU regret must be finite and non-negative")
        missing = [key for key in state if profiling_costs.get(key) is None]
        row.update({"revealed_object_ids": list(state), "budget": len(state),
            "epsilon_success": {str(e): row["gpu_relative_regret"] <= e for e in epsilons},
            "recorded_replica_profiling_wall_sum_s": None if missing else sum(profiling_costs[key] for key in state),
            "profiling_cost_missing_keys": missing})
        by_state[state] = row
        rows.append(row)
    initial = by_state[()]
    for row in rows:
        row["selection_changed_from_initial"] = row["selected_plan_id"] != initial["selected_plan_id"]
        row["gpu_regret_reduction_from_initial"] = initial["gpu_relative_regret"] - row["gpu_relative_regret"]
    singles = [row for row in rows if row["budget"] == 1]
    pairs = [row for row in rows if row["budget"] == 2]
    gains = lambda items: sum(row["gpu_regret_reduction_from_initial"] > 1e-12 for row in items)
    best = min(rows, key=lambda row: (row["gpu_relative_regret"], row["budget"], row["revealed_object_ids"]))
    nonmonotone = []
    for state in states:
        for key in state:
            parent = tuple(value for value in state if value != key)
            if parent not in by_state:
                continue
            previous, current = by_state[parent], by_state[state]
            if current["gpu_relative_regret"] > previous["gpu_relative_regret"] + 1e-12:
                nonmonotone.append({"parent_revealed_ids": list(parent), "added_key": key,
                    "parent_regret": previous["gpu_relative_regret"],
                    "child_regret": current["gpu_relative_regret"]})
    full = by_state[keys]
    return {"D0": initial,
        "D1": {"states": len(singles), "improving_states": gains(singles),
               "selection_changes": sum(row["selection_changed_from_initial"] for row in singles)},
        "D2": {"pair_states": len(pairs), "improving_pairs": gains(pairs),
            "pair_only_improvements": sum(row["gpu_regret_reduction_from_initial"] > 1e-12
                and all(by_state[(key,)]["gpu_regret_reduction_from_initial"] <= 1e-12
                        for key in row["revealed_object_ids"]) for row in pairs),
            "full_reveal": full, "best_enumerated_subset_diagnostic_only": best,
            "nonmonotone_edges": nonmonotone, "fixed_orders": orders},
        "decision_by_epsilon": {str(e): (
            "C_initial_epsilon_near_optimal" if initial["gpu_relative_regret"] <= e else
            "A_enumerated_update_reaches_epsilon" if best["gpu_relative_regret"] <= e else
            "B_no_epsilon_recovery_found_in_enumerated_subsets") for e in epsilons},
        "coverage": {"evaluated_subsets": len(states), "all_possible_subsets": 2 ** len(keys),
                     "exhaustive": len(states) == 2 ** len(keys)},
        "states": rows}


def exposed_records(freeze, kernel_report, raw_records=None):
    """Attach evaluator labels to frozen query features, with explicit lineage."""
    if kernel_report["freeze_id"] != freeze["freeze_id"]:
        raise ValueError("kernel report belongs to another freeze")
    measured = {row["object_id"]: row for row in kernel_report["objects"]}
    if len(measured) != len(kernel_report["objects"]) or set(measured) != set(freeze["zero_shot_predictions"]):
        raise ValueError("kernel labels must cover each frozen query exactly once")
    raw = {row["record_id"]: row for row in (raw_records or []) if row["status"] == "ok"}
    result, costs = [], {}
    for query in freeze["query_features"]:
        row = copy.deepcopy(query)
        key = row["object_id"]
        label = measured[key]
        value, analytic = float(label["measured_s"]), float(row["analytic"]["estimate_s"])
        if not math.isfinite(value) or value <= 0 or not math.isfinite(analytic) or analytic <= 0:
            raise ValueError("kernel durations and analytic estimates must be positive")
        ids = label["raw_record_ids"]
        source = [raw[identifier] for identifier in ids if identifier in raw]
        if any(item["object_id"] != key for item in source):
            raise ValueError("kernel raw-record lineage belongs to another object")
        wall = [item["provenance"].get("profiling_wall_time_s") for item in source]
        costs[key] = sum(wall) if len(source) == len(ids) and ids and all(v is not None for v in wall) else None
        row["measurement"] = {"duration_s": value, "raw_record_ids": ids,
                              "profiling_wall_time_s": costs[key]}
        row["target"] = {"log_residual": math.log(value / analytic), "ratio": value / analytic}
        result.append(row)
    return result, costs


def diagnose_case(freeze, workload, train, queries, costs, contract, measured, host):
    graph = graph_from_dict(workload["graph"])
    hardware = hardware_from_dict(freeze["hardware"])
    model_id = workload["model_id"]
    records = [*train, *queries]
    train_ids, hidden_ids = {row["object_id"] for row in train}, {row["object_id"] for row in queries}
    if train_ids & hidden_ids:
        raise ValueError("frozen query measurement keys overlap initial training")
    runtime = runtime_contract(contract)
    split_id = "exposed-frozen-query-development"
    runtime["splits"] = [{"split_id": split_id, "train_object_ids": sorted(train_ids),
                          "hidden_object_ids": sorted(hidden_ids)}]
    runtime["contract_id"] = content_id("refit-opportunity-runtime", {k: v for k, v in runtime.items() if k != "contract_id"})
    descriptor = E2Workload(model_id, split_id, model_id, workload["model_family"], workload["sequence_length"])
    # Query-only usages avoid borrowing another model's similarly named node.
    mapping = object_map_for_workload(queries, descriptor)
    pool = sorted({mapping[str(node.extra.get("local_name", node.id))]
                   for node in graph.nodes.values() if node.op_type != "input"})
    plans = [plan_from_dict(workload["candidate_plans"][key]) for key in workload["execution_plan_ids"]]
    ranker = ReferenceCandidateRanker(graph, plans, hardware.n_devs, host)
    if set(ranker.plans) != set(measured):
        raise ValueError("physical candidate and GPU measurement IDs differ")
    model_oracle = ranker.rank(_oracle_profile(records, hardware, mapping))
    oracle_by_id = {row.plan_id: row.makespan for row in model_oracle}
    fixed_model_best, best_gpu = model_oracle[0].makespan, min(measured.values())
    initial_durations = {}

    def evaluate(state):
        started = time.perf_counter()
        profile, _ = _active_profile(records, runtime, hardware, mapping,
                                    split_id=split_id, revealed_object_ids=state)
        update_s = time.perf_counter() - started
        if not state:
            initial_durations.update(profile.durations_s)
            for key in pool:
                if not math.isclose(profile.durations_s[key], freeze["zero_shot_predictions"][key]["nearest_s"],
                                    rel_tol=1e-10, abs_tol=1e-15):
                    raise ValueError("initial predictor no longer reproduces frozen kernel predictions")
        started = time.perf_counter()
        ranked = ranker.rank(profile)
        rerank_s = time.perf_counter() - started
        selected = ranked[0].plan_id
        return {"selected_plan_id": selected, "predicted_selected_s": ranked[0].makespan,
            "gpu_selected_s": measured[selected], "gpu_relative_regret": measured[selected] / best_gpu - 1,
            "model_relative_regret": oracle_by_id[selected] / fixed_model_best - 1,
            "model_metric_reference": "fixed_all_kernel_labels_reference_model_on_measured_pool",
            "predictor_update_wall_time_s": update_s, "candidate_rescore_wall_time_s": rerank_s,
            "unrestricted_research_wall_time_s": None,
            "changed_unmeasured_graph_keys": sum(not math.isclose(profile.durations_s[key], initial_durations[key],
                rel_tol=1e-12, abs_tol=1e-15) for key in pool if key not in state)}

    report = assess_subsets(evaluate, pool, costs)
    return {"task_id": model_id, "freeze_id": freeze["freeze_id"], "nodes": len(graph.nodes),
        "measured_candidates": len(measured), "hidden_keys": len(pool), "gpu_best_measured_s": best_gpu,
        "selection_scope": "exact_reselection_over_fixed_measured_physical_pool_not_global_MCTS",
        "ranker_preparation_wall_time_s": ranker.preparation_wall_time_s,
        "runtime_contract_id": runtime["contract_id"],
        "initial_visible_objects": len(train_ids), **report}


def analyze(root=ROOT):
    root = Path(root)
    current, _ = check_reference(root / "data/reference_validation_v1")
    host = HostSubmissionProfile(**current["host_profile"])
    train_path, contract_path = root / "data/bw1000_e2/profiling_dataset.jsonl", root / "data/bw1000_e2/split_contract.json"
    train = load_jsonl_records(train_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    cases = [
        (root / "data/bw1000_gpu_holdout_v1", root / "data/bw1000_gpu_holdout_v1/results_received_20261028", False),
        (root / "data/reference_validation_v1", root / "data/received_reference_validation_v1/reference_validation_v1_results", True),
    ]
    workloads, inputs, audits = [], [train_path, contract_path], []
    for bundle, results, has_raw in cases:
        freeze, manifest = load_freeze(bundle)
        if set(freeze["train_object_ids"]) != {row["object_id"] for row in train}:
            raise ValueError("initial training differs from sealed training IDs")
        read = lambda name: json.loads((results / name).read_text(encoding="utf-8"))
        kernel, execution = read("zero_shot_report.json"), read("execution_report.json")
        audit = audit_execution(freeze, load_jsonl_records(results / "execution_samples.jsonl"),
                                read("execution_environment.json"), execution)
        inputs.extend([bundle / "freeze.json", bundle / "manifest.json", *[results / name for name in
            ("zero_shot_report.json", "execution_report.json", "execution_samples.jsonl", "execution_environment.json")]])
        raw = None
        if has_raw:
            raw = load_jsonl_records(results / "raw_measurements.jsonl")
            reproduced = evaluate_measurements(freeze, manifest, raw, load_environment_registry(results / "environments.json"))
            if reproduced["objects"] != kernel["objects"]:
                raise ValueError("kernel raw records do not reproduce stored labels")
            inputs.extend([results / "raw_measurements.jsonl", results / "environments.json"])
        queries, costs = exposed_records(freeze, kernel, raw)
        audits.append({"freeze_id": freeze["freeze_id"], "execution_records": audit["records"],
                       "kernel_raw_audited": has_raw, "missing_profiling_costs": sum(v is None for v in costs.values())})
        for workload in freeze["workloads"]:
            actual = next(row for row in execution["workloads"] if row["model_id"] == workload["model_id"])
            measured = {row["plan_id"]: row["median_s"] for row in actual["candidates"]}
            row = diagnose_case(freeze, workload, train, queries, costs, contract, measured, host)
            workloads.append(row)
            print(f"{row['task_id']}: initial_gpu_regret_pct={row['D0']['gpu_relative_regret']*100:.6f} "
                  f"single_gains={row['D1']['improving_states']} pair_gains={row['D2']['improving_pairs']} "
                  f"states={row['coverage']['evaluated_subsets']}", flush=True)
    # A3 targets already occur in the 112-row initial training set. Do not
    # fabricate a cold-start pool by hiding these already available labels.
    from calibration.decoder_competition import check as check_a3
    from execution.decoder_competition import summarize
    from execution.analyze_reference_validation import compare_report
    a3_bundle, a3_results = root / "data/a3_decoder_competition_v1", root / "data/received_a3_decoder_competition_v1"
    a3 = check_a3(a3_bundle)
    original = json.loads((a3_results / "competition_report.json").read_text(encoding="utf-8"))
    compare_report(summarize(a3, load_jsonl_records(a3_results / "execution_samples.jsonl"),
                   json.loads((a3_results / "environment.json").read_text(encoding="utf-8"))), original)
    controls = []
    for task in a3["workloads"]:
        actual = next(row for row in original["workloads"] if row["task_id"] == task["task_id"])
        by_id = {row["plan_id"]: row["median_s"] for row in actual["candidates"]}
        # Preserve the sealed pre-label choice, including its tie rule.
        scores = task["initial_scores_s"]
        selected = min(key for key, value in scores.items() if value <= min(scores.values()) + 1e-12)
        if not set(task["kernel_durations_s"]) <= {row["object_id"] for row in train}:
            raise ValueError("A3 control unexpectedly contains unseen kernel keys")
        controls.append({"task_id": task["task_id"], "selected_plan_id": selected,
            "initial_gpu_relative_regret": by_id[selected] / min(by_id.values()) - 1,
            "hidden_keys": 0, "D1_D2": "not_applicable_all_kernel_keys_already_in_initial_data",
            "baseline": "sealed_A3_prelabel_prediction_not_redefined_by_this_diagnosis"})
    inputs.extend([a3_bundle / "freeze.json", *[a3_results / name for name in
                   ("competition_report.json", "execution_samples.jsonl", "environment.json")]])
    return {"format": "fusion-placement-search.reference-refit-opportunity", "version": "1.0",
        "scope": "D0_D1_D2_exposed_development_diagnosis_no_active_policy_or_independent_benefit_claim",
        "epsilons": [.001, .005, .01], "primary_epsilon": .01,
        "summary": {"refit_tasks": len(workloads), "already_seen_A3_controls": len(controls),
            "initial_gpu_within_1pct": sum(row["D0"]["gpu_relative_regret"] <= .01 for row in workloads),
            "tasks_with_single_key_gpu_improvement": sum(row["D1"]["improving_states"] > 0 for row in workloads),
            "tasks_with_pair_gpu_improvement": sum(row["D2"]["improving_pairs"] > 0 for row in workloads),
            "evaluated_subsets": sum(row["coverage"]["evaluated_subsets"] for row in workloads)},
        "workloads": workloads, "already_seen_controls": controls, "input_audits": audits,
        "input_sha256": {str(path.relative_to(root)).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in inputs},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "timing_environment": {"python": platform.python_version(), "platform": platform.platform()},
        "limitations": [
            "GPU regret is relative to measured frozen physical candidates only; no new GPU execution or global MCTS re-search.",
            "D1/D2 enumerate evaluator-chosen reveals; their best subset is not an attainable active-policy result.",
            "All singletons and pairs are covered. Larger subsets use three fixed prefixes and full reveal, not exhaustive enumeration.",
            "Full reveal is an endpoint, not a bound on partial-update benefits or regressions.",
            "Old kernel raw journals are unavailable: their labels use stored reports and actual profiling costs remain null.",
            "Known costs sum recorded replica wall times, excluding startup and reordering; not elapsed parallel collection time.",
            "Host and link models are fixed, pooled kernel times and operator-body mismatch limitations remain.",
            "D3 strategy comparisons and SROP superiority are not evaluated.",
        ]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/reference_refit_opportunity.v1.json")
    args = parser.parse_args(argv)
    report = analyze()
    write_json(args.output, report)
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
