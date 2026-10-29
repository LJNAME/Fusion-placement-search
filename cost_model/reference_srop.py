"""Fixed-candidate SROP replay with the existing Reference execution model.

This adapter changes neither the latency predictor nor acquisition formulas.
Without independent plan measurements, regret is against a fully revealed
kernel profile under the same model, never against actual GPU execution.
"""

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

from calibration.e2_manifest import content_id
from calibration.e2_dataset import make_dataset_view
from calibration.e2_records import load_jsonl_records
from cost_model.e2_srop_proxy import (
    WORKLOADS, _oracle_profile, _run_policy_trajectory,
    build_workload_graph, object_map_for_workload,
)
from cost_model.e2_total_cost import task_summary
from cost_model.reference_semantics import HostSubmissionProfile, ReferenceEagerCostModel
from cost_model.search_sensitivity import _RankedPlan, plan_id
from execution.audit_gpu_holdout import write_json
from simulator import default_fusion_registry, make_reference_plan
from simulator.dataset import graph_to_dict, plan_to_dict
from simulator.hardware import hardware_to_dict, load_hardware


POLICIES = ("random", "uncertainty_only", "impact_only", "srop_proxy", "srop_refit")
BASELINE_SOURCES = (
    "cost_model/reference_srop.py", "cost_model/reference_semantics.py",
    "cost_model/e2_srop_proxy.py", "cost_model/e2_residual.py", "cost_model/e2_uncertainty.py",
    "cost_model/e2_refit_acquisition.py", "cost_model/e2_counterfactual_batch.py",
    "calibration/e2_dataset.py",
)


class ReferenceCandidateRanker:
    """Own candidate snapshots and preserve physical device directions.

    rank_many is a scalar compatibility adapter, not a vectorized scheduler.
    The existing model still validates and builds its timeline on every call.
    """

    def __init__(self, graph, plans, n_devs, host):
        started = time.perf_counter()
        self.graph = copy.deepcopy(graph)
        self.model = ReferenceEagerCostModel(copy.deepcopy(host))
        self.n_devs = n_devs
        self.plans = {plan_id(plan): copy.deepcopy(plan) for plan in plans}
        if not self.plans:
            raise ValueError("at least one candidate is required")
        if any(len(group.node_ids) != 1 for plan in self.plans.values() for group in plan.groups):
            raise ValueError("Reference SROP requires executable singleton groups")
        self.preparation_wall_time_s = time.perf_counter() - started
        self.calls = 0
        self.scoring_wall_time_s = 0.0

    def rank(self, profile):
        if profile.n_devs != self.n_devs:
            raise ValueError("profile device count differs from candidate device count")
        started = time.perf_counter()
        ranked = []
        for identifier, plan in self.plans.items():
            estimate = self.model.predict(self.graph, plan, profile)
            ranked.append(_RankedPlan(identifier, plan, estimate.makespan,
                                      estimate.critical_path, estimate.confidence))
        ranked.sort(key=lambda row: (row.makespan, row.plan_id))
        self.calls += 1
        self.scoring_wall_time_s += time.perf_counter() - started
        return ranked

    def rank_many(self, profiles):
        return [self.rank(profile) for profile in profiles]


def structural_candidates(graph, n_devs):
    """Public structural probes only; no prediction or hidden label is read."""
    plans = [make_reference_plan(graph, device=device) for device in range(n_devs)]
    root = plans[0]
    cuts = sorted({max(1, len(root.groups) * part // 4) for part in (1, 2, 3)})
    for source in range(n_devs):
        for target in range(n_devs):
            if source == target:
                continue
            for cut in cuts:
                plan = copy.deepcopy(root)
                for index, group in enumerate(plan.groups):
                    group.device = source if index < cut else target
                plans.append(plan)
    return list({plan_id(plan): plan for plan in plans}.values())


def runtime_contract(contract):
    """One public predictor/update/uncertainty protocol for every policy."""
    result = copy.deepcopy(contract)
    result["parent_contract_id"] = result.pop("contract_id", None)
    result["active_profiling_protocol"].update(
        uncertainty_variant="distance_floor", profiling_cost_variant="analytic_repetitions",
        candidate_scope="fixed_inputs_no_candidate_generation_inside_replay",
        refit_batch_scenarios=True, counterfactual_batch_scenarios=True,
        policies=list(POLICIES),
    )
    result["contract_id"] = content_id("reference-srop-runtime", result)
    return result


def replay(graph, plans, records, contract, hardware, workload, host, *,
           random_seeds=(0,), measured_plan_times=None):
    """Replay all policies to pool exhaustion; ignore stop recommendations.

    measured_plan_times, when supplied, is evaluator-owned and indexed by
    physical plan_id. It is never consulted to choose candidates or queries.
    """
    if not random_seeds:
        raise ValueError("at least one random seed is required")
    runtime = runtime_contract(contract)
    mapping = object_map_for_workload(records, workload)
    timed = [node for node in graph.nodes.values() if node.op_type != "input"]
    missing = [node.id for node in timed if str(node.extra.get("local_name", node.id)) not in mapping]
    if missing:
        raise ValueError(f"timed operators missing full measurement keys: {missing}")
    used = {mapping[str(node.extra.get("local_name", node.id))] for node in timed}
    split = next(row for row in runtime["splits"] if row["split_id"] == workload.split_id)
    if set(split["train_object_ids"]) & set(split["hidden_object_ids"]):
        raise ValueError("training and hidden measurement keys overlap")
    pool = tuple(sorted(used & set(split["hidden_object_ids"])))
    ranker = ReferenceCandidateRanker(graph, plans, hardware.n_devs, host)
    candidates = list(ranker.plans.values())
    reference_id = plan_id(make_reference_plan(graph, device=0))
    if reference_id not in ranker.plans:
        raise ValueError("the public all-device-zero reference must be present")
    # Evaluator-only work is outside the strategy's CPU clock.
    if measured_plan_times is None:
        truth = ranker.rank(_oracle_profile(records, hardware, mapping))
        metric_reference = "full_kernel_labels_under_fixed_reference_model_not_gpu_plan_truth"
    else:
        if set(measured_plan_times) != set(ranker.plans):
            raise ValueError("GPU measurements must cover exactly the fixed physical candidates")
        if any(not math.isfinite(value) or value <= 0 for value in measured_plan_times.values()):
            raise ValueError("GPU plan times must be finite and positive")
        truth = sorted((_RankedPlan(key, ranker.plans[key], value, (), 1.0)
                        for key, value in measured_plan_times.items()),
                       key=lambda row: (row.makespan, row.plan_id))
        metric_reference = "supplied_gpu_plan_measurements_evaluator_only"
    reference_regret = max(0., next(row.makespan for row in truth if row.plan_id == reference_id)
                           - truth[0].makespan) * 1000
    exchange = (tuple(range(hardware.n_devs)),)
    specs = [("initial_only", None), *(("random", seed) for seed in random_seeds),
             *((policy, None) for policy in POLICIES if policy != "random")]
    runs = []
    for policy, seed in specs:
        trace = _run_policy_trajectory(
            "uncertainty_only" if policy == "initial_only" else policy,
            graph, candidates, records, runtime, hardware, mapping,
            () if policy == "initial_only" else pool, truth,
            split_id=workload.split_id, random_seed=seed, registry=default_fusion_registry(),
            exchange_permutations=exchange, ranker=ranker, capture_decision_timing=True,
        )
        runs.append({"policy": policy, "seed": seed, "repeat": 0,
                     "preparation_s": ranker.preparation_wall_time_s, "trace": trace})
    baseline = {
        "predictor": "measured_exact_else_analytic_times_nearest_log_residual",
        "prediction_protocol": runtime["prediction_protocol"],
        "active_protocol": runtime["active_profiling_protocol"],
        "host_profile": asdict(host), "hardware": hardware_to_dict(hardware),
        "graph": graph_to_dict(graph),
        "physical_candidates": {key: plan_to_dict(plan) for key, plan in sorted(ranker.plans.items())},
        "split": split, "runtime_contract_id": runtime["contract_id"],
        "initial_visible_dataset_id": content_id("reference-srop-visible-dataset", make_dataset_view(
            records, runtime, split_id=workload.split_id, mode="zero_shot", role="predictor")),
        "implementation_sha256": {
            name: hashlib.sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
            for name in BASELINE_SOURCES
        },
    }
    return {
        "format": "fusion-placement-search.reference-srop-replay", "version": "1.0",
        "scope": "exposed_development_fixed_candidate_interface_replay",
        "candidate_provenance": "CLI uses graph-only structural probes; other API inputs require external audit",
        "metric_reference": metric_reference, "workload_id": workload.workload_id,
        "baseline_id": content_id("reference-srop-baseline", baseline), "baseline": baseline,
        "candidate_count": len(candidates), "measurement_pool_ids": list(pool),
        "physical_device_policy": "identity_only_no_exchange_deduplication",
        "timing_environment": {"python": platform.python_version(), "platform": platform.platform(),
                               "logical_cpus": os.cpu_count(), "passes": 1},
        "runs": runs,
        "forced_budget_cost_summary": task_summary(runs, reference_regret)["forced"],
        "limitations": [
            "Existing exposed data; not an independent validation of SROP benefit.",
            "Fixed public structural probes; no candidate discovery or re-search cost is measured.",
            "Kernel durations and analytic links retain the existing symmetric pooled profile.",
            "Host parameters remain fixed; operator-body coverage remains unverified.",
            "One local CPU pass plus recorded profiling wall times; no GPU was run by this adapter.",
            "Model-oracle regret is not actual GPU regret; supplied GPU labels need external provenance audit.",
            "Stop recommendations are recorded but not used; uncertainty is not a safety certificate.",
            "rank_many runs scalar Reference predictions; no batching speedup is claimed.",
        ],
    }


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=WORKLOADS, default="model_3b_s256")
    parser.add_argument("--host-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = {
        "dataset": root / "data/bw1000_e2/profiling_dataset.jsonl",
        "contract": root / "data/bw1000_e2/split_contract.json",
        "hardware": root / "data/bw1000_2card/bw1000_2card.calibrated.json",
        "host": args.host_profile,
    }
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    workload = WORKLOADS[args.workload]
    graph = build_workload_graph(workload, "layer")
    hardware = load_hardware(paths["hardware"])
    report = replay(graph, structural_candidates(graph, hardware.n_devs),
                    load_jsonl_records(paths["dataset"]), read(paths["contract"]), hardware,
                    workload, HostSubmissionProfile(**read(paths["host"])))
    report["input_sha256"] = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for name, path in paths.items()}
    report["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(args.output, report)
    print(f"candidates={report['candidate_count']} hidden_keys={len(report['measurement_pool_ids'])} "
          f"runs={len(report['runs'])} metric_reference={report['metric_reference']}")


if __name__ == "__main__":
    main()
