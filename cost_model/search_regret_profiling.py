"""E1 search-regret-aware offline profiling replay.

This module hides exact per-shape measurements behind the analytic profile,
then reveals a budgeted subset of them.  Candidate plans stay fixed so the
experiment measures re-ranking quality rather than candidate discovery.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import statistics
from typing import Iterable, Sequence

from simulator import FusionPatternRegistry, OpGraph, Plan, default_fusion_registry

from .hardware_profile import (
    AnalyticHardwareProfile,
    GroupCost,
    HardwareCostProfile,
    TransferCost,
    as_hardware_profile,
)
from .measured_profile import MeasuredHardwareProfile
from .search_evaluation import build_evaluation_graph
from .search_sensitivity import (
    ShapeKey,
    SymmetricHardwareProfile,
    _best_equivalence_set,
    _equivalence_ranks,
    _rank_plans,
    _top_k_equivalence_set,
    device_exchange_permutations,
    evaluate_counterfactual_sensitivity,
    generate_candidate_pool,
    infer_device_classes,
    infer_hardware_model_classes,
    plan_id,
    shape_key_for_group,
)
from .structure import analyze_plan_structure


REPORT_FORMAT = "fusion-placement-search.search-regret-profiling"
REPORT_VERSION = "1.0"
DEFAULT_BUDGETS = (0, 1, 2, 4, 8, 16)
DEFAULT_RANDOM_SEEDS = (0, 1, 7, 23)


class CalibrationReplayProfile:
    """Use measured costs only for ShapeKeys selected by a profiling policy."""

    def __init__(
        self,
        oracle: HardwareCostProfile,
        fallback: HardwareCostProfile,
        calibrated_keys: Iterable[ShapeKey],
    ) -> None:
        self.oracle = as_hardware_profile(oracle)
        self.fallback = as_hardware_profile(fallback)
        if self.oracle.n_devs != self.fallback.n_devs:
            raise ValueError("oracle and fallback must have the same device count")
        oracle_classes = infer_device_classes(self.oracle)
        fallback_classes = infer_device_classes(self.fallback)
        if oracle_classes != fallback_classes:
            raise ValueError("oracle and fallback must use the same device classes")
        self.device_classes = oracle_classes
        self.calibrated_keys = frozenset(calibrated_keys)

    @property
    def n_devs(self) -> int:
        return self.oracle.n_devs

    @property
    def overlap_send(self) -> bool:
        return self.oracle.overlap_send

    @property
    def interference(self) -> float:
        return self.oracle.interference

    def max_memory(self, device: int) -> float:
        return self.oracle.max_memory(device)

    def estimate_group(self, graph: OpGraph, group) -> GroupCost:
        key = shape_key_for_group(graph, group, self.device_classes)
        if key in self.calibrated_keys:
            cost = self.oracle.estimate_group(graph, group)
            return GroupCost(
                duration=cost.duration,
                compute_time=cost.compute_time,
                memory_time=cost.memory_time,
                launch_time=cost.launch_time,
                confidence=cost.confidence,
                source=f"{cost.source}+e1_revealed",
                ood_reasons=cost.ood_reasons,
            )
        cost = self.fallback.estimate_group(graph, group)
        return GroupCost(
            duration=cost.duration,
            compute_time=cost.compute_time,
            memory_time=cost.memory_time,
            launch_time=cost.launch_time,
            confidence=cost.confidence,
            source=f"{cost.source}+e1_hidden",
            ood_reasons=cost.ood_reasons,
        )

    def estimate_transfer(
        self,
        src_device: int,
        dst_device: int,
        nbytes: float,
    ) -> TransferCost:
        # Link calibration is held fixed; E1 currently selects operator shapes.
        return self.oracle.estimate_transfer(src_device, dst_device, nbytes)

    def link_key(self, src_device: int, dst_device: int) -> tuple[int, int]:
        return self.oracle.link_key(src_device, dst_device)


@dataclass(frozen=True)
class _ShapeRecord:
    key: ShapeKey
    oracle_duration: float
    fallback_duration: float
    occurrences: int
    plans_containing: int
    actual_log_residual: float
    signed_relative_error: float
    uncertainty: float = 0.0
    uncertainty_scope: str = ""
    uncertainty_peers: int = 0
    impact_ms_per_relative_error: float = 0.0

    @property
    def op_family(self) -> tuple[str, ...]:
        return tuple(operator.op_type for operator in self.key.operators)

    @property
    def shape_kind(self) -> tuple[str, ...]:
        return tuple(operator.shape_kind for operator in self.key.operators)

    @property
    def joint_score(self) -> float:
        return self.uncertainty * self.impact_ms_per_relative_error


def _collect_measured_shapes(
    graph: OpGraph,
    plans: Sequence[Plan],
    oracle: HardwareCostProfile,
    fallback: HardwareCostProfile,
    registry: FusionPatternRegistry,
    exchange_permutations: Sequence[Sequence[int]],
) -> list[_ShapeRecord]:
    device_classes = infer_device_classes(oracle)
    observations: dict[ShapeKey, dict[str, object]] = {}
    for plan in plans:
        identifier = plan_id(plan, exchange_permutations)
        structure = analyze_plan_structure(
            graph,
            plan,
            oracle.n_devs,
            registry=registry,
        )
        for group in structure.groups:
            exact = oracle.estimate_group(graph, group)
            if exact.duration <= 0.0 or "measured_exact_shape" not in exact.source:
                continue
            approximate = fallback.estimate_group(graph, group)
            key = shape_key_for_group(graph, group, device_classes)
            item = observations.setdefault(
                key,
                {
                    "oracle": [],
                    "fallback": [],
                    "occurrences": 0,
                    "plans": set(),
                },
            )
            item["oracle"].append(float(exact.duration))
            item["fallback"].append(float(approximate.duration))
            item["occurrences"] += 1
            item["plans"].add(identifier)

    records = []
    for key, item in sorted(observations.items()):
        oracle_duration = statistics.fmean(item["oracle"])
        fallback_duration = statistics.fmean(item["fallback"])
        if oracle_duration <= 0.0 or fallback_duration <= 0.0:
            continue
        records.append(
            _ShapeRecord(
                key=key,
                oracle_duration=oracle_duration,
                fallback_duration=fallback_duration,
                occurrences=int(item["occurrences"]),
                plans_containing=len(item["plans"]),
                actual_log_residual=abs(math.log(fallback_duration / oracle_duration)),
                signed_relative_error=fallback_duration / oracle_duration - 1.0,
            )
        )
    if not records:
        raise ValueError("E1 requires at least one measured exact ShapeKey")
    return records


def _with_leave_one_out_uncertainty(
    records: Sequence[_ShapeRecord],
) -> list[_ShapeRecord]:
    """Estimate target error from peer shapes without using its own residual."""

    if len(records) < 2:
        raise ValueError("leave-one-shape-out uncertainty requires at least two shapes")
    result = []
    for target in records:
        families = [
            (
                "operator_family",
                [item for item in records if item != target and item.op_family == target.op_family],
            ),
            (
                "shape_kind",
                [item for item in records if item != target and item.shape_kind == target.shape_kind],
            ),
            ("global", [item for item in records if item != target]),
        ]
        scope, peers = next((name, items) for name, items in families if items)
        uncertainty = statistics.median(item.actual_log_residual for item in peers)
        result.append(
            _ShapeRecord(
                **{
                    **target.__dict__,
                    "uncertainty": max(float(uncertainty), 1e-12),
                    "uncertainty_scope": scope,
                    "uncertainty_peers": len(peers),
                }
            )
        )
    return result


def _with_impact(
    graph: OpGraph,
    plans: Sequence[Plan],
    fallback: HardwareCostProfile,
    records: Sequence[_ShapeRecord],
    *,
    registry: FusionPatternRegistry,
    top_k: int,
    impact_delta: float,
) -> list[_ShapeRecord]:
    delta = abs(float(impact_delta))
    if not 0.0 < delta < 1.0:
        raise ValueError("impact_delta must be between zero and one")
    sensitivity = evaluate_counterfactual_sensitivity(
        graph,
        plans,
        fallback,
        perturbations=(-delta, delta),
        top_k=top_k,
        registry=registry,
    )
    by_key = {
        item["key_id"]: float(item["max_differential_response_ms"]) / delta
        for item in sensitivity["shape_sensitivities"]
    }
    return [
        _ShapeRecord(
            **{
                **record.__dict__,
                "impact_ms_per_relative_error": by_key.get(record.key.key_id, 0.0),
            }
        )
        for record in records
    ]


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("rank vectors must be non-empty and equally sized")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    left_energy = sum((x - left_mean) ** 2 for x in left)
    right_energy = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_energy * right_energy)
    if denominator == 0.0:
        return 1.0 if tuple(left) == tuple(right) else 0.0
    return float(numerator / denominator)


def _ranking_metrics(
    oracle_ranked,
    predicted_ranked,
    *,
    top_k: int,
    tie_tolerance_s: float,
) -> dict[str, object]:
    oracle_by_id = {item.plan_id: item for item in oracle_ranked}
    predicted_by_id = {item.plan_id: item for item in predicted_ranked}
    identifiers = sorted(oracle_by_id)
    if set(identifiers) != set(predicted_by_id):
        raise ValueError("oracle and predicted rankings cover different plans")
    oracle_best = _best_equivalence_set(oracle_ranked, tie_tolerance_s)
    predicted_best = _best_equivalence_set(predicted_ranked, tie_tolerance_s)
    effective_top_k = min(max(int(top_k), 1), len(oracle_ranked))
    oracle_top = _top_k_equivalence_set(
        oracle_ranked, effective_top_k, tie_tolerance_s
    )
    predicted_top = _top_k_equivalence_set(
        predicted_ranked, effective_top_k, tie_tolerance_s
    )
    selected = predicted_ranked[0].plan_id
    oracle_best_ms = oracle_ranked[0].makespan * 1e3
    selected_oracle_ms = oracle_by_id[selected].makespan * 1e3
    oracle_ranks = _equivalence_ranks(oracle_ranked, tie_tolerance_s)
    predicted_ranks = _equivalence_ranks(predicted_ranked, tie_tolerance_s)
    return {
        "predicted_top1_plan_id": selected,
        "top1_hit": not oracle_best.isdisjoint(predicted_best),
        "top1_regret_ms": max(0.0, selected_oracle_ms - oracle_best_ms),
        "oracle_top1_ms": oracle_best_ms,
        "selected_oracle_ms": selected_oracle_ms,
        "top_k_recall": len(oracle_top & predicted_top) / max(len(oracle_top), 1),
        "top_k_oracle_size": len(oracle_top),
        "top_k_predicted_size": len(predicted_top),
        "rank_correlation": _pearson(
            [float(oracle_ranks[item]) for item in identifiers],
            [float(predicted_ranks[item]) for item in identifiers],
        ),
    }


def _policy_orders(
    records: Sequence[_ShapeRecord],
    random_seeds: Sequence[int],
) -> dict[str, list[tuple[int | None, list[_ShapeRecord]]]]:
    def ordered(key):
        return sorted(records, key=lambda item: (-float(key(item)), item.key.key_id))

    policies = {
        "uncertainty": [(None, ordered(lambda item: item.uncertainty))],
        "impact": [
            (None, ordered(lambda item: item.impact_ms_per_relative_error))
        ],
        "regret_aware": [(None, ordered(lambda item: item.joint_score))],
        "random": [],
    }
    for seed in random_seeds:
        shuffled = list(records)
        random.Random(int(seed)).shuffle(shuffled)
        policies["random"].append((int(seed), shuffled))
    return policies


def _normalize_budgets(budgets: Sequence[int], total: int) -> tuple[int, ...]:
    values = {0, total}
    values.update(min(max(int(value), 0), total) for value in budgets)
    return tuple(sorted(values))


def _aggregate_policy_runs(runs: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    by_budget: dict[int, list[dict[str, object]]] = {}
    for run in runs:
        by_budget.setdefault(int(run["budget"]), []).append(run)
    result = []
    for budget in sorted(by_budget):
        items = by_budget[budget]
        regrets = [float(item["top1_regret_ms"]) for item in items]
        recalls = [float(item["top_k_recall"]) for item in items]
        correlations = [float(item["rank_correlation"]) for item in items]
        result.append(
            {
                "budget": budget,
                "runs": len(items),
                "top1_hit_rate": statistics.fmean(
                    float(bool(item["top1_hit"])) for item in items
                ),
                "top1_regret_ms_mean": statistics.fmean(regrets),
                "top1_regret_ms_min": min(regrets),
                "top1_regret_ms_max": max(regrets),
                "top_k_recall_mean": statistics.fmean(recalls),
                "rank_correlation_mean": statistics.fmean(correlations),
            }
        )
    return result


def evaluate_search_regret_profiling(
    graph: OpGraph,
    plans: Sequence[Plan],
    oracle: HardwareCostProfile,
    fallback: HardwareCostProfile,
    *,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    random_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
    top_k: int = 5,
    impact_delta: float = 0.20,
    tie_tolerance_s: float = 1e-12,
    registry: FusionPatternRegistry | None = None,
) -> dict[str, object]:
    """Replay budgeted calibration and report oracle ranking regret."""

    active_registry = registry or default_fusion_registry()
    base_oracle = as_hardware_profile(oracle)
    base_fallback = as_hardware_profile(fallback)
    device_classes = infer_device_classes(base_oracle)
    if device_classes != infer_device_classes(base_fallback):
        raise ValueError("oracle and fallback device classes differ")
    exchange = device_exchange_permutations(device_classes)
    unique = {plan_id(plan, exchange): plan for plan in plans}
    candidates = list(unique.values())
    if len(candidates) < 2:
        raise ValueError("E1 requires at least two distinct candidate plans")
    if not random_seeds:
        raise ValueError("random_seeds must not be empty")

    records = _collect_measured_shapes(
        graph,
        candidates,
        base_oracle,
        base_fallback,
        active_registry,
        exchange,
    )
    records = _with_leave_one_out_uncertainty(records)
    records = _with_impact(
        graph,
        candidates,
        base_fallback,
        records,
        registry=active_registry,
        top_k=top_k,
        impact_delta=impact_delta,
    )
    normalized_budgets = _normalize_budgets(budgets, len(records))
    oracle_ranked = _rank_plans(
        graph,
        candidates,
        base_oracle,
        active_registry,
        exchange,
    )
    policies = _policy_orders(records, random_seeds)
    policy_reports = {}
    for policy, orders in policies.items():
        runs = []
        for seed, order in orders:
            for budget in normalized_budgets:
                selected = order[:budget]
                replay = CalibrationReplayProfile(
                    base_oracle,
                    base_fallback,
                    (item.key for item in selected),
                )
                predicted = _rank_plans(
                    graph,
                    candidates,
                    replay,
                    active_registry,
                    exchange,
                )
                metrics = _ranking_metrics(
                    oracle_ranked,
                    predicted,
                    top_k=top_k,
                    tie_tolerance_s=tie_tolerance_s,
                )
                runs.append(
                    {
                        "budget": budget,
                        "random_seed": seed,
                        "selected_shape_ids": [item.key.key_id for item in selected],
                        **metrics,
                    }
                )
        by_budget = _aggregate_policy_runs(runs)
        zero_regret = next(
            (
                item["budget"]
                for item in by_budget
                if float(item["top1_regret_ms_mean"]) <= 1e-12
            ),
            None,
        )
        full_top_k = next(
            (
                item["budget"]
                for item in by_budget
                if float(item["top_k_recall_mean"]) >= 1.0 - 1e-12
            ),
            None,
        )
        policy_reports[policy] = {
            "selection_score": {
                "random": "uniform random permutation",
                "uncertainty": "leave-one-shape-out peer residual prior",
                "impact": "counterfactual differential response",
                "regret_aware": "uncertainty * counterfactual impact",
            }[policy],
            "runs": runs,
            "by_budget": by_budget,
            "summary": {
                "first_zero_mean_regret_budget": zero_regret,
                "first_full_mean_top_k_recall_budget": full_top_k,
                "mean_regret_across_budgets_ms": statistics.fmean(
                    float(item["top1_regret_ms_mean"]) for item in by_budget
                ),
                "mean_top_k_recall_across_budgets": statistics.fmean(
                    float(item["top_k_recall_mean"]) for item in by_budget
                ),
            },
        }

    all_keys = frozenset(record.key for record in records)
    full_profile = CalibrationReplayProfile(base_oracle, base_fallback, all_keys)
    full_ranked = _rank_plans(
        graph,
        candidates,
        full_profile,
        active_registry,
        exchange,
    )
    full_metrics = _ranking_metrics(
        oracle_ranked,
        full_ranked,
        top_k=top_k,
        tie_tolerance_s=tie_tolerance_s,
    )
    if full_metrics["top1_regret_ms"] > 1e-9 or full_metrics["top_k_recall"] < 1.0:
        raise AssertionError("revealing every measured shape did not recover oracle ranking")

    shape_reports = [
        {
            "key_id": item.key.key_id,
            "label": item.key.label,
            **item.key.as_dict(),
            "occurrences": item.occurrences,
            "plans_containing": item.plans_containing,
            "oracle_duration_us": item.oracle_duration * 1e6,
            "fallback_duration_us": item.fallback_duration * 1e6,
            "signed_relative_error": item.signed_relative_error,
            "actual_abs_log_residual": item.actual_log_residual,
            "uncertainty_estimate": item.uncertainty,
            "uncertainty_scope": item.uncertainty_scope,
            "uncertainty_peers": item.uncertainty_peers,
            "impact_ms_per_relative_error": item.impact_ms_per_relative_error,
            "regret_aware_score": item.joint_score,
        }
        for item in sorted(
            records,
            key=lambda value: (-value.joint_score, value.key.key_id),
        )
    ]
    return {
        "format": REPORT_FORMAT,
        "version": REPORT_VERSION,
        "graph_name": graph.name,
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "configuration": {
            "candidate_scope": "fixed canonical candidate re-ranking",
            "budgets": list(normalized_budgets),
            "random_seeds": [int(seed) for seed in random_seeds],
            "top_k": min(max(int(top_k), 1), len(candidates)),
            "impact_delta": float(impact_delta),
            "uncertainty_estimator": "leave-one-shape-out hierarchical median abs log residual",
            "uncertainty_scope_order": ["operator_family", "shape_kind", "global"],
            "device_classes": list(device_classes),
            "exchange_permutations": [list(item) for item in exchange],
        },
        "oracle": {
            "top1_plan_id": oracle_ranked[0].plan_id,
            "top1_ms": oracle_ranked[0].makespan * 1e3,
            "top1_margin_ms": (
                (oracle_ranked[1].makespan - oracle_ranked[0].makespan) * 1e3
                if len(oracle_ranked) > 1
                else 0.0
            ),
            "candidate_plans": len(oracle_ranked),
        },
        "profiled_shapes": shape_reports,
        "policies": policy_reports,
        "summary": {
            "candidate_plans": len(oracle_ranked),
            "profiled_shapes": len(records),
            "full_budget_recovers_oracle": True,
            "max_actual_abs_log_residual": max(
                item.actual_log_residual for item in records
            ),
            "max_impact_ms_per_relative_error": max(
                item.impact_ms_per_relative_error for item in records
            ),
        },
    }


def save_report(path: str | Path, report: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hardware",
        type=Path,
        default=root / "data" / "bw1000_2card" / "bw1000_2card.calibrated.json",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=root / "data" / "bw1000_2card" / "raw_measurements.json",
    )
    parser.add_argument(
        "--scale",
        choices=("small", "layer", "model", "model_3b"),
        default="layer",
    )
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--seeds", default="0,1,7,23")
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--rollouts", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument(
        "--random-seeds", default=",".join(map(str, DEFAULT_RANDOM_SEEDS))
    )
    parser.add_argument("--impact-delta", type=float, default=0.20)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    measured = MeasuredHardwareProfile.from_files(args.hardware, args.raw)
    device_classes = infer_hardware_model_classes(measured)
    oracle = SymmetricHardwareProfile(measured, device_classes)
    fallback = SymmetricHardwareProfile(
        AnalyticHardwareProfile(measured.hardware), device_classes
    )
    graph = build_evaluation_graph(args.scale)
    plans, pool = generate_candidate_pool(
        graph,
        oracle,
        candidate_count=args.candidates,
        seeds=_parse_csv_ints(args.seeds),
        iterations=args.iterations,
        rollouts=args.rollouts,
        include_planned=False,
    )
    report = evaluate_search_regret_profiling(
        graph,
        plans,
        oracle,
        fallback,
        budgets=_parse_csv_ints(args.budgets),
        random_seeds=_parse_csv_ints(args.random_seeds),
        top_k=args.top_k,
        impact_delta=args.impact_delta,
    )
    report["candidate_pool"] = pool
    report["hardware_name"] = measured.hardware.name
    save_report(args.output, report)

    summary = report["summary"]
    print(f"nodes: {report['nodes']}")
    print(f"candidates: {summary['candidate_plans']}")
    print(f"profiled_shapes: {summary['profiled_shapes']}")
    for policy in ("random", "uncertainty", "impact", "regret_aware"):
        entries = report["policies"][policy]["by_budget"]
        compact = ",".join(
            f"{item['budget']}:{item['top1_regret_ms_mean']:.6f}/"
            f"{item['top_k_recall_mean']:.3f}"
            for item in entries
        )
        print(f"{policy} budget:regret_ms/topk_recall={compact}")
    print(f"full_budget_recovers_oracle: {summary['full_budget_recovers_oracle']}")
    print(f"report: {args.output}")
    print("Fusion E1 search-regret profiling replay: OK")


__all__ = [
    "CalibrationReplayProfile",
    "DEFAULT_BUDGETS",
    "DEFAULT_RANDOM_SEEDS",
    "evaluate_search_regret_profiling",
    "save_report",
]


if __name__ == "__main__":
    main()
