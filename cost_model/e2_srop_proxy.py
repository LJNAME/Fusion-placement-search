"""E2c fixed-candidate diagnostic for search-regret-aware profiling.

The candidate set is the union of plans found by the current predictor and by
the oracle. Acquisition scores use these candidate structures with public
predicted costs. Thus oracle-derived candidate discovery information is
present: this is a diagnostic, not evidence of an oracle-free search loop.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from calibration.e2_dataset import make_dataset_view
from calibration.e2_records import load_jsonl_records
from simulator import OpGraph, Plan, default_fusion_registry
from simulator.hardware import HardwareSpec, load_hardware
from simulator.model_graphs import build_decoder_layer, build_model_graph

from .e2_residual import NearestShapePredictor
from .hardware_profile import AnalyticHardwareProfile, GroupCost, TransferCost
from .search_evaluation import qwen25_3b_spec, qwen3_spec
from .search_regret_profiling import _aggregate_policy_runs
from .e2_audit import PreparedCandidateRanker, ranking_metrics, selection_diagnostics, summarize_metrics
from .search_sensitivity import (
    SymmetricHardwareProfile,
    _rank_plans,
    device_exchange_permutations,
    generate_candidate_pool,
    plan_id,
)
from .structure import GroupStructure


REPORT_FORMAT = "fusion-placement-search.e2-srop-proxy"
REPORT_VERSION = "1.1"


@dataclass(frozen=True)
class E2Workload:
    workload_id: str
    split_id: str
    model_id: str
    model_family: str
    sequence_length: int


WORKLOADS = {
    "model_3b_s256": E2Workload(
        workload_id="model_3b_s256",
        split_id="model-qwen06-to-qwen25-3b",
        model_id="qwen2.5-3b-s256-b1",
        model_family="qwen2.5-3b",
        sequence_length=256,
    ),
    "sequence_qwen06_s1024": E2Workload(
        workload_id="sequence_qwen06_s1024",
        split_id="sequence-short-to-1024",
        model_id="qwen3-0.6b-s1024-b1",
        model_family="qwen3-0.6b",
        sequence_length=1024,
    ),
    "sequence_qwen3b_s1024": E2Workload(
        workload_id="sequence_qwen3b_s1024",
        split_id="sequence-short-to-1024",
        model_id="qwen2.5-3b-s1024-b1",
        model_family="qwen2.5-3b",
        sequence_length=1024,
    ),
}


def build_workload_graph(workload: E2Workload, scale: str) -> OpGraph:
    if workload.model_family == "qwen3-0.6b":
        spec = dataclass_replace(qwen3_spec(28), seq_length=workload.sequence_length)
    elif workload.model_family == "qwen2.5-3b":
        spec = dataclass_replace(qwen25_3b_spec(), seq_length=workload.sequence_length)
    else:
        raise ValueError(f"unsupported E2 workload family: {workload.model_family}")
    if scale == "layer":
        graph = build_decoder_layer(spec, layer_id=0)
    elif scale == "model":
        graph = build_model_graph(spec)
    else:
        raise ValueError(f"unknown E2 workload scale: {scale}")
    graph.name = f"e2-{workload.workload_id}-{scale}"
    return graph


def object_map_for_workload(
    records: Sequence[Mapping[str, Any]], workload: E2Workload
) -> dict[str, str]:
    """Map graph-local operator names to complete E2 measurement objects."""

    result: dict[str, str] = {}
    for record in records:
        for usage in record["usages"]:
            if usage["model_id"] != workload.model_id:
                continue
            local_name = str(usage["local_name"])
            object_id = str(record["object_id"])
            previous = result.get(local_name)
            if previous is not None and previous != object_id:
                raise ValueError(
                    f"workload local name maps to multiple E2 objects: {local_name}"
                )
            result[local_name] = object_id
    if not result:
        raise ValueError(f"no E2 objects found for workload: {workload.model_id}")
    return result


class DatasetLatencyProfile:
    """Use E2 object durations for singleton groups and analytic link costs."""

    def __init__(
        self,
        hardware: HardwareSpec,
        object_by_local_name: Mapping[str, str],
        durations_s: Mapping[str, float],
        sources: Mapping[str, str],
    ) -> None:
        model_classes = tuple(
            f"model:{device.name.strip().lower()}" if device.name.strip() else f"physical:{index}"
            for index, device in enumerate(hardware.devices)
        )
        self.fallback = SymmetricHardwareProfile(
            AnalyticHardwareProfile(hardware), model_classes
        )
        self.device_classes = model_classes
        self.object_by_local_name = dict(object_by_local_name)
        self.durations_s = {str(key): float(value) for key, value in durations_s.items()}
        self.sources = {str(key): str(value) for key, value in sources.items()}
        self._group_costs: dict[str, GroupCost] = {}
        self._fallback_costs: dict[tuple, GroupCost] = {}
        self._transfer_costs: dict[tuple, TransferCost] = {}

    @property
    def n_devs(self) -> int:
        return self.fallback.n_devs

    @property
    def overlap_send(self) -> bool:
        return self.fallback.overlap_send

    @property
    def interference(self) -> float:
        return self.fallback.interference

    def max_memory(self, device: int) -> float:
        return self.fallback.max_memory(device)

    def object_id_for_group(self, graph: OpGraph, group: GroupStructure) -> str | None:
        if len(group.node_ids) != 1:
            return None
        node = graph.nodes[group.node_ids[0]]
        local_name = str(node.extra.get("local_name", node.id))
        return self.object_by_local_name.get(local_name)

    def estimate_group(self, graph: OpGraph, group: GroupStructure) -> GroupCost:
        object_id = self.object_id_for_group(graph, group)
        if object_id is None or object_id not in self.durations_s:
            cache_key = (id(graph), group.node_ids, group.device, group.pattern_id)
            if cache_key not in self._fallback_costs:
                self._fallback_costs[cache_key] = self.fallback.estimate_group(graph, group)
            return self._fallback_costs[cache_key]
        if object_id in self._group_costs:
            return self._group_costs[object_id]
        duration = self.durations_s[object_id]
        source = self.sources.get(object_id, "e2_unknown")
        exact = source in {"train_exact", "revealed_exact", "oracle_exact"}
        cost = GroupCost(
            duration=duration,
            compute_time=duration,
            memory_time=0.0,
            launch_time=0.0,
            confidence=0.95 if exact else 0.70,
            source=source,
            ood_reasons=() if exact else (f"e2:{object_id}:predicted",),
        )
        self._group_costs[object_id] = cost
        return cost

    def estimate_transfer(self, src_device: int, dst_device: int, nbytes: float) -> TransferCost:
        key = (src_device, dst_device, nbytes)
        if key not in self._transfer_costs:
            self._transfer_costs[key] = self.fallback.estimate_transfer(src_device, dst_device, nbytes)
        return self._transfer_costs[key]

    def link_key(self, src_device: int, dst_device: int) -> tuple[int, int]:
        return self.fallback.link_key(src_device, dst_device)


class ObjectCounterfactualProfile:
    def __init__(
        self, base: DatasetLatencyProfile, target_object_id: str, relative_delta: float
    ) -> None:
        self.base = base
        self.target_object_id = str(target_object_id)
        self.multiplier = 1.0 + float(relative_delta)
        if self.multiplier <= 0.0:
            raise ValueError("counterfactual multiplier must be positive")
        self.device_classes = base.device_classes

    @property
    def n_devs(self) -> int:
        return self.base.n_devs

    @property
    def overlap_send(self) -> bool:
        return self.base.overlap_send

    @property
    def interference(self) -> float:
        return self.base.interference

    def max_memory(self, device: int) -> float:
        return self.base.max_memory(device)

    def estimate_group(self, graph: OpGraph, group: GroupStructure) -> GroupCost:
        cost = self.base.estimate_group(graph, group)
        if self.base.object_id_for_group(graph, group) != self.target_object_id:
            return cost
        return GroupCost(
            duration=cost.duration * self.multiplier,
            compute_time=cost.compute_time * self.multiplier,
            memory_time=cost.memory_time * self.multiplier,
            launch_time=cost.launch_time * self.multiplier,
            confidence=cost.confidence,
            source=f"{cost.source}+e2_counterfactual",
            ood_reasons=cost.ood_reasons,
        )

    def estimate_transfer(self, src_device: int, dst_device: int, nbytes: float) -> TransferCost:
        return self.base.estimate_transfer(src_device, dst_device, nbytes)

    def link_key(self, src_device: int, dst_device: int) -> tuple[int, int]:
        return self.base.link_key(src_device, dst_device)


def _oracle_profile(
    records: Sequence[Mapping[str, Any]],
    hardware: HardwareSpec,
    object_map: Mapping[str, str],
) -> DatasetLatencyProfile:
    return DatasetLatencyProfile(
        hardware,
        object_map,
        {
            str(record["object_id"]): float(record["measurement"]["duration_s"])
            for record in records
        },
        {str(record["object_id"]): "oracle_exact" for record in records},
    )


def _active_profile(
    records: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    hardware: HardwareSpec,
    object_map: Mapping[str, str],
    *,
    split_id: str,
    revealed_object_ids: Iterable[str],
) -> tuple[DatasetLatencyProfile, dict[str, float]]:
    revealed = frozenset(str(value) for value in revealed_object_ids)
    view = make_dataset_view(
        records,
        contract,
        split_id=split_id,
        mode="active_calibration",
        role="predictor",
        revealed_object_ids=revealed,
    )
    visible = [record for record in view if "target" in record]
    predictor = NearestShapePredictor().fit(visible, contract["prediction_protocol"])
    durations: dict[str, float] = {}
    sources: dict[str, str] = {}
    uncertainty: dict[str, float] = {}
    for record in view:
        object_id = str(record["object_id"])
        if "measurement" in record:
            durations[object_id] = float(record["measurement"]["duration_s"])
            sources[object_id] = "revealed_exact" if object_id in revealed else "train_exact"
            continue
        prediction = predictor.predict(record)
        durations[object_id] = float(record["analytic"]["estimate_s"]) * math.exp(
            prediction.mean
        )
        sources[object_id] = "nearest_shape_predicted"
        uncertainty[object_id] = prediction.q90
    return DatasetLatencyProfile(hardware, object_map, durations, sources), uncertainty


def _object_impact_ms_per_relative_error(
    graph: OpGraph,
    plans: Sequence[Plan],
    profile: DatasetLatencyProfile,
    object_id: str,
    *,
    delta: float,
    registry,
    exchange_permutations: Sequence[Sequence[int]],
    ranker=None,
    baseline_ranked=None,
) -> float:
    baseline = baseline_ranked if baseline_ranked is not None else _rank(
        graph, plans, profile, registry, exchange_permutations, ranker
    )
    baseline_by_id = {item.plan_id: item.makespan for item in baseline}
    maximum = 0.0
    for relative_delta in (-abs(delta), abs(delta)):
        perturbed = _rank(
            graph,
            plans,
            ObjectCounterfactualProfile(profile, object_id, relative_delta),
            registry,
            exchange_permutations,
            ranker,
        )
        responses = [
            item.makespan - baseline_by_id[item.plan_id] for item in perturbed
        ]
        maximum = max(maximum, (max(responses) - min(responses)) * 1e3)
    return maximum / abs(delta)


def _expected_decision_regret_ms(
    graph: OpGraph,
    plans: Sequence[Plan],
    profile: DatasetLatencyProfile,
    object_id: str,
    uncertainty_q90_log: float,
    baseline_ranked,
    *,
    standard_normal_quantiles: Sequence[float],
    q90_normalizer: float,
    registry,
    exchange_permutations: Sequence[Sequence[int]],
    ranker=None,
) -> float:
    """Proxy EVPI: expected loss of keeping the current Top-1 decision.

    Only the queried object's duration changes in each hypothetical outcome;
    correlations and posterior updates of other objects are intentionally left
    for SROP-Posterior.
    """

    current_top1 = baseline_ranked[0].plan_id
    sigma = max(float(uncertainty_q90_log), 0.0) / float(q90_normalizer)
    regrets = []
    for quantile in standard_normal_quantiles:
        relative_delta = math.exp(float(quantile) * sigma) - 1.0
        sampled = _rank(
            graph,
            plans,
            ObjectCounterfactualProfile(profile, object_id, relative_delta),
            registry,
            exchange_permutations,
            ranker,
        )
        sampled_by_id = {item.plan_id: item.makespan for item in sampled}
        regrets.append(
            max(0.0, sampled_by_id[current_top1] - sampled[0].makespan) * 1e3
        )
    return statistics.fmean(regrets)


def _rank(graph, plans, profile, registry, exchange, ranker=None):
    if ranker is not None:
        return ranker.rank(profile)
    return _rank_plans(graph, plans, profile, registry, exchange)


def _profiling_cost_proxy_s(record: Mapping[str, Any], repetitions: int) -> float:
    return max(float(record["analytic"]["estimate_s"]) * int(repetitions), 1e-12)


def _actual_profiling_cost_s(record: Mapping[str, Any], repetitions: int) -> float:
    value = record["measurement"].get("profiling_wall_time_s")
    return (
        float(value)
        if value is not None
        else _profiling_cost_proxy_s(record, repetitions)
    )


def _area_under_curve(points: Sequence[tuple[float, float]]) -> float:
    ordered = sorted((float(x), float(y)) for x, y in points)
    if len(ordered) < 2 or ordered[-1][0] <= ordered[0][0]:
        return 0.0
    area = sum(
        (right_x - left_x) * (left_y + right_y) * 0.5
        for (left_x, left_y), (right_x, right_y) in zip(ordered, ordered[1:])
    )
    return area / (ordered[-1][0] - ordered[0][0])


def _run_policy_trajectory(
    policy: str,
    graph: OpGraph,
    plans: Sequence[Plan],
    records: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    hardware: HardwareSpec,
    object_map: Mapping[str, str],
    measurement_pool: Sequence[str],
    oracle_ranked,
    *,
    split_id: str,
    random_seed: int | None,
    registry,
    exchange_permutations: Sequence[Sequence[int]],
    ranker=None,
) -> list[dict[str, Any]]:
    active_protocol = contract["active_profiling_protocol"]
    delta = float(active_protocol["impact_relative_delta"])
    proxy_quantiles = tuple(
        float(value)
        for value in active_protocol["srop_proxy_standard_normal_quantiles"]
    )
    q90_normalizer = float(active_protocol["uncertainty_q90_normalizer"])
    repetitions = int(active_protocol["profiling_repetitions"])
    by_id = {str(record["object_id"]): record for record in records}
    revealed: list[str] = []
    if policy == "random":
        random_order = list(measurement_pool)
        random.Random(int(random_seed)).shuffle(random_order)
    else:
        random_order = []
    trajectory: list[dict[str, Any]] = []
    fit_elapsed = ranking_elapsed = selection_elapsed = 0.0
    previous_top1 = None
    for budget in range(len(measurement_pool) + 1):
        fit_started = time.perf_counter()
        profile, uncertainty = _active_profile(
            records,
            contract,
            hardware,
            object_map,
            split_id=split_id,
            revealed_object_ids=revealed,
        )
        fit_elapsed += time.perf_counter() - fit_started
        ranking_started = time.perf_counter()
        predicted_ranked = _rank(
            graph, plans, profile, registry, exchange_permutations, ranker
        )
        ranking_elapsed += time.perf_counter() - ranking_started
        metrics = ranking_metrics(
            oracle_ranked,
            predicted_ranked,
            top_k=min(5, len(plans)),
            tie_tolerance_s=1e-12,
        )
        remaining = [object_id for object_id in measurement_pool if object_id not in revealed]
        next_selection = None
        selection_score = None
        selected_uncertainty = None
        selected_impact = None
        selected_expected_regret = None
        selected_cost_proxy = None
        scores: dict[str, float] = {}
        selection_started = time.perf_counter()
        if remaining:
            if policy == "random":
                next_selection = next(item for item in random_order if item in remaining)
            else:
                impacts: dict[str, float] = {}
                if policy == "impact_only":
                    impacts = {
                        object_id: _object_impact_ms_per_relative_error(
                            graph,
                            plans,
                            profile,
                            object_id,
                            delta=delta,
                            registry=registry,
                            exchange_permutations=exchange_permutations,
                            ranker=ranker,
                            baseline_ranked=predicted_ranked,
                        )
                        for object_id in remaining
                    }
                expected_regrets: dict[str, float] = {}
                if policy == "srop_proxy":
                    expected_regrets = {
                        object_id: _expected_decision_regret_ms(
                            graph,
                            plans,
                            profile,
                            object_id,
                            uncertainty[object_id],
                            predicted_ranked,
                            standard_normal_quantiles=proxy_quantiles,
                            q90_normalizer=q90_normalizer,
                            registry=registry,
                            exchange_permutations=exchange_permutations,
                            ranker=ranker,
                        )
                        for object_id in remaining
                    }
                scores: dict[str, float] = {}
                for object_id in remaining:
                    cost_proxy = _profiling_cost_proxy_s(by_id[object_id], repetitions)
                    if policy == "uncertainty_only":
                        numerator = math.expm1(uncertainty[object_id])
                    elif policy == "impact_only":
                        numerator = impacts[object_id]
                    elif policy == "srop_proxy":
                        numerator = expected_regrets[object_id]
                    else:
                        raise ValueError(f"unknown E2 active policy: {policy}")
                    scores[object_id] = numerator / cost_proxy
                next_selection = min(remaining, key=lambda item: (-scores[item], item))
                selection_score = scores[next_selection]
                selected_impact = impacts.get(next_selection)
                selected_expected_regret = expected_regrets.get(next_selection)
            if next_selection is not None:
                selected_uncertainty = (
                    math.expm1(uncertainty[next_selection])
                    if next_selection in uncertainty
                    else None
                )
                selected_cost_proxy = _profiling_cost_proxy_s(
                    by_id[next_selection], repetitions
                )
        selection_elapsed += time.perf_counter() - selection_started
        audit = selection_diagnostics(scores, remaining)
        trajectory.append(
            {
                "budget": budget,
                "random_seed": random_seed,
                "selected_object_ids": list(revealed),
                "profiling_wall_time_s": sum(
                    _actual_profiling_cost_s(by_id[object_id], repetitions)
                    for object_id in revealed
                ),
                "next_selection": next_selection,
                "next_selection_score": selection_score,
                "next_uncertainty_relative": selected_uncertainty,
                "next_impact_ms_per_relative_error": selected_impact,
                "next_expected_decision_regret_ms": selected_expected_regret,
                "next_cost_proxy_s": selected_cost_proxy,
                "predictor_fit_wall_time_s": fit_elapsed,
                "selection_wall_time_s": selection_elapsed,
                "ranking_wall_time_s": ranking_elapsed,
                "algorithm_wall_time_s": fit_elapsed + selection_elapsed + ranking_elapsed,
                "top1_decision_changed": previous_top1 is not None and previous_top1 != predicted_ranked[0].plan_id,
                "selection_tie_break": "object_id_ascending" if scores else "random_order",
                **audit,
                **metrics,
            }
        )
        previous_top1 = predicted_ranked[0].plan_id
        if next_selection is None:
            break
        revealed.append(next_selection)
    return trajectory


def _policy_report(trajectories: Sequence[Sequence[Mapping[str, Any]]], budgets: Sequence[int]) -> dict[str, Any]:
    runs = [dict(point) for trajectory in trajectories for point in trajectory if int(point["budget"]) in budgets]
    by_budget = _aggregate_policy_runs(runs)
    all_points = [dict(point) for trajectory in trajectories for point in trajectory]
    all_by_budget = _aggregate_policy_runs(all_points)
    summarize_metrics(runs, by_budget)
    stopping_runs = []
    for trajectory in trajectories:
        stopped = next((point for point in trajectory if point["stop_recommended"]), trajectory[-1])
        stopping_runs.append({
            **dict(stopped),
            "stop_reason": "no_positive_proxy_value" if stopped["stop_recommended"] else "pool_exhausted",
            "saved_measurements": int(trajectory[-1]["budget"]) - int(stopped["budget"]),
        })
    budget_aucs = [
        _area_under_curve(
            [(float(point["budget"]), float(point["top1_regret_ms"])) for point in trajectory]
        )
        for trajectory in trajectories
    ]
    time_aucs = [
        _area_under_curve(
            [
                (float(point["profiling_wall_time_s"]), float(point["top1_regret_ms"]))
                for point in trajectory
            ]
        )
        for trajectory in trajectories
    ]
    stable_zero_budget = next(
        (
            int(item["budget"])
            for index, item in enumerate(all_by_budget)
            if all(
                float(later["top1_regret_ms_mean"]) <= 1e-12
                for later in all_by_budget[index:]
            )
        ),
        None,
    )
    stable_full_top_k_budget = next(
        (
            int(item["budget"])
            for index, item in enumerate(all_by_budget)
            if all(
                float(later["top_k_recall_mean"]) >= 1.0 - 1e-12
                for later in all_by_budget[index:]
            )
        ),
        None,
    )
    return {
        "runs": runs,
        "full_trajectories": trajectories,
        "stopping_mode": "offline prefix replay at first score-based recommendation; later outcomes do not affect the stop",
        "stopping_runs": stopping_runs,
        "by_budget": by_budget,
        "summary": {
            "regret_budget_auc_ms_mean": statistics.fmean(budget_aucs),
            "regret_profiling_time_auc_ms_mean": statistics.fmean(time_aucs),
            "first_zero_mean_regret_budget": next(
                (
                    int(item["budget"])
                    for item in all_by_budget
                    if float(item["top1_regret_ms_mean"]) <= 1e-12
                ),
                None,
            ),
            "first_full_mean_top_k_recall_budget": next(
                (
                    int(item["budget"])
                    for item in all_by_budget
                    if float(item["top_k_recall_mean"]) >= 1.0 - 1e-12
                ),
                None,
            ),
            "stable_zero_mean_regret_budget": stable_zero_budget,
            "stable_full_mean_top_k_recall_budget": stable_full_top_k_budget,
        },
    }


def evaluate_fixed_candidate_srop(
    graph: OpGraph,
    plans: Sequence[Plan],
    records: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    hardware: HardwareSpec,
    workload: E2Workload,
    *,
    random_seeds: Sequence[int] | None = None,
    use_prepared: bool = True,
    ranker_factory=PreparedCandidateRanker,
) -> dict[str, Any]:
    evaluation_started = time.perf_counter()
    registry = default_fusion_registry()
    object_map = object_map_for_workload(records, workload)
    split = next(item for item in contract["splits"] if item["split_id"] == workload.split_id)
    hidden = set(split["hidden_object_ids"])
    used_object_ids = {
        object_map[str(node.extra.get("local_name", node.id))]
        for node in graph.nodes.values()
        if str(node.extra.get("local_name", node.id)) in object_map
    }
    measurement_pool = tuple(sorted(used_object_ids & hidden))
    if not measurement_pool:
        raise ValueError(f"workload has no hidden E2 objects: {workload.workload_id}")
    oracle = _oracle_profile(records, hardware, object_map)
    exchange = device_exchange_permutations(oracle.device_classes)
    unique = {plan_id(plan, exchange): plan for plan in plans}
    candidates = list(unique.values())
    if len(candidates) < 2:
        raise ValueError("E2c requires at least two canonical candidate plans")
    ranker = ranker_factory(graph, candidates, hardware.n_devs, exchange, registry) if use_prepared else None
    oracle_ranked = _rank(graph, candidates, oracle, registry, exchange, ranker)
    active_protocol = contract["active_profiling_protocol"]
    seeds = tuple(
        int(seed)
        for seed in (
            random_seeds
            if random_seeds is not None
            else active_protocol["random_seeds"]
        )
    )
    total = len(measurement_pool)
    budgets = tuple(
        sorted(
            {0, total}
            | {min(max(int(value), 0), total) for value in active_protocol["budgets"]}
        )
    )
    policy_reports: dict[str, Any] = {}
    for policy in active_protocol["policies"]:
        policy_seeds: Sequence[int | None] = seeds if policy == "random" else (None,)
        trajectories = [
            _run_policy_trajectory(
                policy,
                graph,
                candidates,
                records,
                contract,
                hardware,
                object_map,
                measurement_pool,
                oracle_ranked,
                split_id=workload.split_id,
                random_seed=seed,
                registry=registry,
                exchange_permutations=exchange,
                ranker=ranker,
            )
            for seed in policy_seeds
        ]
        for trajectory in trajectories:
            if float(trajectory[-1]["top1_regret_ms"]) > 1e-9:
                raise AssertionError("full E2 active budget did not recover oracle top-1")
        policy_reports[policy] = _policy_report(trajectories, budgets)
    initial_regret = next(iter(policy_reports.values()))["runs"][0]["top1_regret_ms"]
    return {
        "workload_id": workload.workload_id,
        "split_id": workload.split_id,
        "model_id": workload.model_id,
        "graph_name": graph.name,
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "measurement_pool_objects": len(measurement_pool),
        "measurement_pool_object_ids": list(measurement_pool),
        "excluded_non_graph_hidden_objects": sorted((set(object_map.values()) & hidden) - used_object_ids),
        "candidate_plans": len(candidates),
        "oracle_top1_ms": oracle_ranked[0].makespan * 1e3,
        "oracle_top1_plan_id": oracle_ranked[0].plan_id,
        "budgets": list(budgets),
        "ranking_metrics": {
            "strict_top_k_recall": "exactly k candidates, ties broken by plan_id",
            "tie_aware_top_k_recall": "include all candidates tied at kth boundary",
            "best_of_k_regret_ms": "best oracle cost within predicted strict top-k minus oracle best",
        },
        "stopping_rule": "stop when max acquisition score <= 0; this is a proxy rule, not a safety certificate",
        "scoring_engine": "prepared_exact" if ranker else "reference",
        "initial_top1_regret_ms": initial_regret,
        "eligible_for_initial_regret_reduction_evidence": initial_regret > 1e-12,
        "evaluation_wall_time_s": time.perf_counter() - evaluation_started,
        "timing_scope": "algorithm times exclude oracle metric aggregation and candidate generation; evaluation includes both oracle metrics and preparation",
        "structure_preparation_wall_time_s": ranker.preparation_wall_time_s if ranker else 0.0,
        "ranking_calls": ranker.calls if ranker else None,
        "ranking_total_wall_time_s": ranker.scoring_wall_time_s if ranker else None,
        "policies": policy_reports,
    }


def diagnostic_candidate_union(
    graph: OpGraph,
    records: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    hardware: HardwareSpec,
    workload: E2Workload,
    *,
    candidate_count: int,
    seeds: Sequence[int],
    iterations: int,
    rollouts: int,
) -> tuple[list[Plan], dict[str, Any]]:
    object_map = object_map_for_workload(records, workload)
    predicted, _ = _active_profile(
        records,
        contract,
        hardware,
        object_map,
        split_id=workload.split_id,
        revealed_object_ids=(),
    )
    oracle = _oracle_profile(records, hardware, object_map)
    predicted_plans, predicted_metadata = generate_candidate_pool(
        graph,
        predicted,
        candidate_count=candidate_count,
        seeds=seeds,
        iterations=iterations,
        rollouts=rollouts,
        include_planned=False,
    )
    oracle_plans, oracle_metadata = generate_candidate_pool(
        graph,
        oracle,
        candidate_count=candidate_count,
        seeds=seeds,
        iterations=iterations,
        rollouts=rollouts,
        include_planned=False,
    )
    exchange = device_exchange_permutations(predicted.device_classes)
    unique: dict[str, Plan] = {}
    sources: dict[str, set[str]] = {}
    for source, source_plans in (
        ("predicted_search", predicted_plans),
        ("oracle_search", oracle_plans),
    ):
        for plan in source_plans:
            identifier = plan_id(plan, exchange)
            unique.setdefault(identifier, plan)
            sources.setdefault(identifier, set()).add(source)
    return list(unique.values()), {
        "diagnostic_oracle_union": True,
        "oracle_derived_candidates_in_diagnostic_pool": True,
        "oracle_labels_available_to_sampling": False,
        "predicted_search": predicted_metadata,
        "oracle_search": oracle_metadata,
        "union_candidate_plans": len(unique),
        "predicted_only": sum(value == {"predicted_search"} for value in sources.values()),
        "oracle_only": sum(value == {"oracle_search"} for value in sources.values()),
        "shared": sum(len(value) == 2 for value in sources.values()),
    }


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_int_csv(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in _parse_csv(value))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=root / "data" / "bw1000_e2" / "profiling_dataset.jsonl")
    parser.add_argument("--contract", type=Path, default=root / "data" / "bw1000_e2" / "split_contract.json")
    parser.add_argument("--hardware", type=Path, default=root / "data" / "bw1000_2card" / "bw1000_2card.calibrated.json")
    parser.add_argument("--workloads", default=",".join(WORKLOADS))
    parser.add_argument("--scale", choices=("layer", "model"), default="layer")
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--seeds", default="0,1,7,23")
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--rollouts", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-scoring", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    records = load_jsonl_records(args.dataset)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    hardware = load_hardware(args.hardware)
    workload_ids = _parse_csv(args.workloads)
    unknown = set(workload_ids) - set(WORKLOADS)
    if unknown:
        raise ValueError(f"unknown E2 workloads: {sorted(unknown)}")
    reports = []
    for workload_id in workload_ids:
        workload = WORKLOADS[workload_id]
        graph = build_workload_graph(workload, args.scale)
        candidate_started = time.perf_counter()
        plans, candidate_metadata = diagnostic_candidate_union(
            graph,
            records,
            contract,
            hardware,
            workload,
            candidate_count=args.candidates,
            seeds=_parse_int_csv(args.seeds),
            iterations=args.iterations,
            rollouts=args.rollouts,
        )
        candidate_metadata["generation_wall_time_s"] = time.perf_counter() - candidate_started
        report = evaluate_fixed_candidate_srop(
            graph, plans, records, contract, hardware, workload,
            use_prepared=not args.reference_scoring,
        )
        report["candidate_pool"] = candidate_metadata
        reports.append(report)
        print(
            f"{workload_id}: nodes={report['nodes']} candidates={report['candidate_plans']} "
            f"pool={report['measurement_pool_objects']}"
        )
        for policy, item in report["policies"].items():
            print(
                f"  {policy}: budget_auc_ms={item['summary']['regret_budget_auc_ms_mean']:.9f} "
                f"zero_budget={item['summary']['first_zero_mean_regret_budget']} "
                f"full_topk_budget={item['summary']['first_full_mean_top_k_recall_budget']}"
            )
    payload = {
        "format": REPORT_FORMAT,
        "version": REPORT_VERSION,
        "configuration": {
            "candidate_scope": "fixed diagnostic predicted+oracle union",
            "diagnostic_oracle_union": True,
            "oracle_derived_candidates_in_diagnostic_pool": True,
            "oracle_labels_available_to_sampling": False,
            "scale": args.scale,
            "candidate_count_per_source": args.candidates,
            "seeds": list(_parse_int_csv(args.seeds)),
            "iterations": args.iterations,
            "rollouts": args.rollouts,
            "active_profiling_protocol": contract["active_profiling_protocol"],
        },
        "workloads": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"report: {args.output}")
    print("Fusion E2c SROP-Proxy fixed-candidate diagnostic: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
