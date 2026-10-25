"""E0 counterfactual shape-to-plan ranking sensitivity experiment.

The experiment keeps the candidate plans fixed and perturbs one reusable
operator-shape latency at a time.  It therefore measures whether a possible
profiling error can change the search decision, without yet implementing the
full active-profiling/VOI loop.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, replace
import hashlib
from itertools import permutations, product
import json
from pathlib import Path
from typing import Sequence

from simulator import (
    FusionPatternRegistry,
    OpGraph,
    Plan,
    default_fusion_registry,
    hardware_fingerprint,
    make_reference_plan,
)

from .hardware_profile import (
    GroupCost,
    HardwareCostProfile,
    TransferCost,
    as_hardware_profile,
)
from .legal_search import legal_mcts_search
from .measured_profile import MeasuredHardwareProfile
from .search_evaluation import build_evaluation_graph
from .structural import StructuralCostModel
from .structure import GroupStructure, analyze_plan_structure


REPORT_FORMAT = "fusion-placement-search.counterfactual-sensitivity"
REPORT_VERSION = "1.1"
DEFAULT_PERTURBATIONS = (-0.20, -0.10, -0.05, 0.05, 0.10, 0.20)


@dataclass(frozen=True, order=True)
class OperatorShape:
    """Node identity with layer/name removed so repeated shapes share one key."""

    op_type: str
    dtype: str
    shape_kind: str
    dimensions: tuple[tuple[str, int], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "op_type": self.op_type,
            "dtype": self.dtype,
            "shape_kind": self.shape_kind,
            "dimensions": {key: value for key, value in self.dimensions},
        }

    def label(self) -> str:
        dims = ",".join(f"{key}={value}" for key, value in self.dimensions)
        return f"{self.op_type}[{self.dtype}]({dims})"


@dataclass(frozen=True, order=True)
class ShapeKey:
    """Hardware-class reusable latency key used by the E0 perturbation."""

    device_class: str
    kernel_id: str
    operators: tuple[OperatorShape, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "device_class": self.device_class,
            "kernel_id": self.kernel_id,
            "operators": [operator.as_dict() for operator in self.operators],
        }

    @property
    def key_id(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def label(self) -> str:
        operators = "+".join(operator.label() for operator in self.operators)
        return (
            f"device_class={self.device_class}|kernel={self.kernel_id}|{operators}"
        )


def operator_shape(graph: OpGraph, node_id: str) -> OperatorShape:
    node = graph.nodes[node_id]
    extra = node.extra or {}
    if node.op_type == "matmul" and extra.get("shape") is not None:
        shape_kind = "matmul"
        raw_shape = extra.get("shape") or {}
        dimensions = tuple(
            (key, int(raw_shape[key]))
            for key in ("m", "n", "k")
            if key in raw_shape
        )
    elif extra.get("output_shape") is not None:
        output_shape = tuple(int(value) for value in extra.get("output_shape"))
        output_numel = 1
        for value in output_shape:
            output_numel *= value
        if node.op_type == "rmsnorm":
            shape_kind = "numel+inner"
            dimensions = (
                ("output_numel", output_numel),
                ("inner_size", output_shape[-1]),
            )
        else:
            shape_kind = "numel"
            dimensions = (("output_numel", output_numel),)
    else:
        shape_kind = "structural"
        dimensions = (
            ("flops", int(round(float(node.flops)))),
            ("out_bytes", int(round(float(node.out_bytes)))),
            ("param_bytes", int(round(float(node.param_bytes)))),
        )
    return OperatorShape(
        op_type=str(node.op_type).lower(),
        dtype=str(extra.get("dtype", "unknown")).lower(),
        shape_kind=shape_kind,
        dimensions=dimensions,
    )


def _default_device_classes(n_devs: int) -> tuple[str, ...]:
    return tuple(f"physical:{device}" for device in range(n_devs))


def infer_device_classes(profile: HardwareCostProfile) -> tuple[str, ...]:
    """Read explicit exchange classes; otherwise keep devices distinct."""

    explicit = getattr(profile, "device_classes", None)
    if explicit is not None:
        classes = tuple(str(value) for value in explicit)
        if len(classes) != profile.n_devs:
            raise ValueError("device_classes length must match n_devs")
        return classes
    return _default_device_classes(profile.n_devs)


def infer_hardware_model_classes(
    profile: HardwareCostProfile,
) -> tuple[str, ...]:
    """Infer exchange classes from configured hardware model names."""

    hardware = getattr(profile, "hardware", None)
    if hardware is None:
        return _default_device_classes(profile.n_devs)
    classes = []
    for device, spec in enumerate(hardware.devices):
        name = str(spec.name).strip().lower()
        classes.append(f"model:{name}" if name else f"physical:{device}")
    return tuple(classes)


def device_exchange_permutations(
    device_classes: Sequence[str],
) -> tuple[tuple[int, ...], ...]:
    """Return every device relabeling that preserves hardware classes."""

    classes = tuple(str(value) for value in device_classes)
    if not classes:
        raise ValueError("device_classes must not be empty")
    grouped: dict[str, list[int]] = {}
    for device, class_name in enumerate(classes):
        grouped.setdefault(class_name, []).append(device)
    groups = [tuple(grouped[name]) for name in sorted(grouped)]
    choices = [tuple(permutations(group)) for group in groups]
    result = []
    for selected in product(*choices):
        mapping = list(range(len(classes)))
        for original, permuted in zip(groups, selected):
            for source, target in zip(original, permuted):
                mapping[source] = target
        result.append(tuple(mapping))
    return tuple(sorted(set(result)))


def shape_key_for_group(
    graph: OpGraph,
    group: GroupStructure,
    device_classes: Sequence[str] | None = None,
) -> ShapeKey:
    # Singleton measurements in MeasuredHardwareProfile are selected by
    # op/dtype/shape, independent of whether the legal candidate was named
    # "unfused" or "rmsnorm_reference".  Real fused kernels remain distinct.
    kernel_id = "singleton" if len(group.node_ids) == 1 else group.pattern_id
    classes = (
        tuple(str(value) for value in device_classes)
        if device_classes is not None
        else _default_device_classes(group.device + 1)
    )
    if group.device < 0 or group.device >= len(classes):
        raise ValueError(f"device class missing for device {group.device}")
    return ShapeKey(
        device_class=classes[group.device],
        kernel_id=kernel_id,
        operators=tuple(operator_shape(graph, node_id) for node_id in group.node_ids),
    )


def _mean_group_cost(costs: Sequence[GroupCost]) -> GroupCost:
    count = float(len(costs))
    reasons = {reason for cost in costs for reason in cost.ood_reasons}
    sources = sorted({cost.source for cost in costs})
    return GroupCost(
        duration=sum(cost.duration for cost in costs) / count,
        compute_time=sum(cost.compute_time for cost in costs) / count,
        memory_time=sum(cost.memory_time for cost in costs) / count,
        launch_time=sum(cost.launch_time for cost in costs) / count,
        confidence=min(cost.confidence for cost in costs),
        source="symmetric_mean:" + "+".join(sources),
        ood_reasons=tuple(sorted(reasons)),
    )


def _mean_transfer_cost(costs: Sequence[TransferCost]) -> TransferCost:
    count = float(len(costs))
    reasons = {reason for cost in costs for reason in cost.ood_reasons}
    sources = sorted({cost.source for cost in costs})
    return TransferCost(
        duration=sum(cost.duration for cost in costs) / count,
        confidence=min(cost.confidence for cost in costs),
        source="symmetric_mean:" + "+".join(sources),
        ood_reasons=tuple(sorted(reasons)),
    )


class SymmetricHardwareProfile:
    """Average costs over exchangeable devices and directed-link orbits."""

    def __init__(
        self,
        base: HardwareCostProfile,
        device_classes: Sequence[str],
    ) -> None:
        self.base = as_hardware_profile(base)
        self.device_classes = tuple(str(value) for value in device_classes)
        if len(self.device_classes) != self.base.n_devs:
            raise ValueError("device_classes length must match n_devs")
        grouped: dict[str, list[int]] = {}
        for device, class_name in enumerate(self.device_classes):
            grouped.setdefault(class_name, []).append(device)
        self._class_devices = {
            name: tuple(devices) for name, devices in grouped.items()
        }
        self.permutations = device_exchange_permutations(self.device_classes)

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
        peers = self._class_devices[self.device_classes[device]]
        return min(self.base.max_memory(peer) for peer in peers)

    def estimate_group(self, graph: OpGraph, group: GroupStructure) -> GroupCost:
        peers = self._class_devices[self.device_classes[group.device]]
        costs = tuple(
            self.base.estimate_group(graph, replace(group, device=peer))
            for peer in peers
        )
        return _mean_group_cost(costs)

    def estimate_transfer(
        self,
        src_device: int,
        dst_device: int,
        nbytes: float,
    ) -> TransferCost:
        if src_device == dst_device:
            return self.base.estimate_transfer(src_device, dst_device, nbytes)
        src_class = self.device_classes[src_device]
        dst_class = self.device_classes[dst_device]
        pairs = tuple(
            (src, dst)
            for src in self._class_devices[src_class]
            for dst in self._class_devices[dst_class]
            if src != dst
        )
        if not pairs:
            return self.base.estimate_transfer(src_device, dst_device, nbytes)
        return _mean_transfer_cost(
            tuple(
                self.base.estimate_transfer(src, dst, nbytes)
                for src, dst in pairs
            )
        )

    def link_key(self, src_device: int, dst_device: int) -> tuple[int, int]:
        return self.base.link_key(src_device, dst_device)


class CounterfactualHardwareProfile:
    """Read-only profile view that scales one shared ShapeKey latency."""

    def __init__(
        self,
        base: HardwareCostProfile,
        target: ShapeKey,
        relative_delta: float,
    ) -> None:
        self.base = as_hardware_profile(base)
        self.device_classes = infer_device_classes(self.base)
        self.target = target
        self.relative_delta = float(relative_delta)
        self.multiplier = 1.0 + self.relative_delta
        if self.multiplier <= 0.0:
            raise ValueError("relative_delta must be greater than -1")

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
        if shape_key_for_group(graph, group, self.device_classes) != self.target:
            return cost
        return GroupCost(
            duration=cost.duration * self.multiplier,
            compute_time=cost.compute_time * self.multiplier,
            memory_time=cost.memory_time * self.multiplier,
            launch_time=cost.launch_time * self.multiplier,
            confidence=cost.confidence,
            source=f"{cost.source}+e0_counterfactual",
            ood_reasons=cost.ood_reasons,
        )

    def estimate_transfer(
        self,
        src_device: int,
        dst_device: int,
        nbytes: float,
    ) -> TransferCost:
        return self.base.estimate_transfer(src_device, dst_device, nbytes)

    def link_key(self, src_device: int, dst_device: int) -> tuple[int, int]:
        return self.base.link_key(src_device, dst_device)


def _plan_key(
    plan: Plan,
    device_permutation: Sequence[int] | None = None,
) -> tuple:
    mapping = (
        tuple(int(value) for value in device_permutation)
        if device_permutation is not None
        else None
    )
    return tuple(
        (
            str(group.extra.get("pattern_id", "legacy")),
            tuple(group.node_ids),
            mapping[int(group.device)] if mapping is not None else int(group.device),
        )
        for group in plan.groups
    )


def canonical_plan_key(
    plan: Plan,
    exchange_permutations: Sequence[Sequence[int]],
) -> tuple:
    keys = tuple(_plan_key(plan, permutation) for permutation in exchange_permutations)
    if not keys:
        raise ValueError("exchange_permutations must not be empty")
    return min(keys)


def plan_id(
    plan: Plan,
    exchange_permutations: Sequence[Sequence[int]] | None = None,
) -> str:
    key = (
        canonical_plan_key(plan, exchange_permutations)
        if exchange_permutations is not None
        else _plan_key(plan)
    )
    payload = json.dumps(key, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def permute_plan_devices(plan: Plan, permutation: Sequence[int]) -> Plan:
    result = copy.deepcopy(plan)
    mapping = tuple(int(value) for value in permutation)
    for group in result.groups:
        if group.device < 0 or group.device >= len(mapping):
            raise ValueError(f"device permutation missing device {group.device}")
        group.device = mapping[group.device]
    return result


def _max_exchange_score_delta(
    graph: OpGraph,
    plans: Sequence[Plan],
    profile: HardwareCostProfile,
    registry: FusionPatternRegistry,
    exchange_permutations: Sequence[Sequence[int]],
) -> float:
    model = StructuralCostModel(registry=registry)
    maximum = 0.0
    for plan in plans:
        baseline = float(model.predict(graph, plan, profile).makespan)
        for permutation in exchange_permutations:
            permuted = permute_plan_devices(plan, permutation)
            score = float(model.predict(graph, permuted, profile).makespan)
            maximum = max(maximum, abs(score - baseline))
    return maximum


@dataclass(frozen=True)
class _RankedPlan:
    plan_id: str
    plan: Plan
    makespan: float
    critical_path: tuple[str, ...]
    confidence: float


def _rank_plans(
    graph: OpGraph,
    plans: Sequence[Plan],
    profile: HardwareCostProfile,
    registry: FusionPatternRegistry,
    exchange_permutations: Sequence[Sequence[int]],
) -> list[_RankedPlan]:
    model = StructuralCostModel(registry=registry)
    ranked = []
    for plan in plans:
        estimate = model.predict(graph, plan, profile)
        ranked.append(
            _RankedPlan(
                plan_id=plan_id(plan, exchange_permutations),
                plan=plan,
                makespan=float(estimate.makespan),
                critical_path=tuple(estimate.critical_path),
                confidence=float(estimate.confidence),
            )
        )
    ranked.sort(key=lambda item: (item.makespan, item.plan_id))
    return ranked


def _strict_pairwise_flips(
    baseline: Sequence[_RankedPlan],
    perturbed: Sequence[_RankedPlan],
) -> int:
    old = {item.plan_id: item.makespan for item in baseline}
    new = {item.plan_id: item.makespan for item in perturbed}
    identifiers = sorted(old)
    flips = 0
    for left_index, left in enumerate(identifiers):
        for right in identifiers[left_index + 1 :]:
            old_difference = old[left] - old[right]
            new_difference = new[left] - new[right]
            if old_difference * new_difference < 0.0:
                flips += 1
    return flips


def _equivalence_ranks(
    ranked: Sequence[_RankedPlan],
    tolerance: float,
) -> dict[str, int]:
    result: dict[str, int] = {}
    group_start = 0
    group_score = ranked[0].makespan if ranked else 0.0
    for index, item in enumerate(ranked):
        if item.makespan - group_score > tolerance:
            group_start = index
            group_score = item.makespan
        result[item.plan_id] = group_start + 1
    return result


def _best_equivalence_set(
    ranked: Sequence[_RankedPlan],
    tolerance: float,
) -> set[str]:
    if not ranked:
        return set()
    threshold = ranked[0].makespan + tolerance
    return {item.plan_id for item in ranked if item.makespan <= threshold}


def _top_k_equivalence_set(
    ranked: Sequence[_RankedPlan],
    top_k: int,
    tolerance: float,
) -> set[str]:
    if not ranked:
        return set()
    boundary = ranked[min(top_k, len(ranked)) - 1].makespan + tolerance
    return {item.plan_id for item in ranked if item.makespan <= boundary}


def _shape_observations(
    graph: OpGraph,
    plans: Sequence[Plan],
    profile: HardwareCostProfile,
    registry: FusionPatternRegistry,
    device_classes: Sequence[str],
    exchange_permutations: Sequence[Sequence[int]],
) -> dict[ShapeKey, dict[str, object]]:
    observations: dict[ShapeKey, dict[str, object]] = {}
    for plan in plans:
        identifier = plan_id(plan, exchange_permutations)
        structure = analyze_plan_structure(
            graph,
            plan,
            profile.n_devs,
            registry=registry,
        )
        for group in structure.groups:
            cost = profile.estimate_group(graph, group)
            if cost.duration <= 0.0:
                continue
            key = shape_key_for_group(graph, group, device_classes)
            item = observations.setdefault(
                key,
                {
                    "occurrences": 0,
                    "plans": set(),
                    "confidences": [],
                    "sources": set(),
                },
            )
            item["occurrences"] += 1
            item["plans"].add(identifier)
            item["confidences"].append(float(cost.confidence))
            item["sources"].add(str(cost.source))
    return observations


def evaluate_counterfactual_sensitivity(
    graph: OpGraph,
    plans: Sequence[Plan],
    profile: HardwareCostProfile,
    *,
    perturbations: Sequence[float] = DEFAULT_PERTURBATIONS,
    top_k: int = 5,
    tie_tolerance_s: float = 1e-12,
    registry: FusionPatternRegistry | None = None,
) -> dict[str, object]:
    """Evaluate fixed candidates under one-shape-at-a-time perturbations."""

    active_registry = registry or default_fusion_registry()
    base_profile = as_hardware_profile(profile)
    device_classes = infer_device_classes(base_profile)
    exchange_permutations = device_exchange_permutations(device_classes)
    unique = {
        plan_id(plan, exchange_permutations): plan for plan in plans
    }
    candidates = list(unique.values())
    if len(candidates) < 2:
        raise ValueError("E0 requires at least two distinct candidate plans")
    deltas = tuple(float(value) for value in perturbations)
    if not deltas or any(value <= -1.0 or value == 0.0 for value in deltas):
        raise ValueError("perturbations must be non-zero and greater than -1")
    tolerance = float(tie_tolerance_s)
    if tolerance < 0.0:
        raise ValueError("tie_tolerance_s must be non-negative")

    max_exchange_delta = _max_exchange_score_delta(
        graph,
        candidates,
        base_profile,
        active_registry,
        exchange_permutations,
    )
    if len(exchange_permutations) > 1 and max_exchange_delta > 1e-12:
        raise ValueError(
            "profile violates declared device-exchange symmetry: "
            f"max score delta={max_exchange_delta:.6e}s"
        )

    baseline = _rank_plans(
        graph,
        candidates,
        base_profile,
        active_registry,
        exchange_permutations,
    )
    baseline_rank = _equivalence_ranks(baseline, tolerance)
    baseline_by_id = {item.plan_id: item for item in baseline}
    effective_top_k = min(max(int(top_k), 1), len(baseline))
    baseline_best = _best_equivalence_set(baseline, tolerance)
    baseline_top = _top_k_equivalence_set(
        baseline, effective_top_k, tolerance
    )
    observations = _shape_observations(
        graph,
        candidates,
        base_profile,
        active_registry,
        device_classes,
        exchange_permutations,
    )

    shape_reports = []
    for key in sorted(observations):
        observation = observations[key]
        runs = []
        for delta in deltas:
            perturbed_profile = CounterfactualHardwareProfile(
                base_profile, key, delta
            )
            ranked = _rank_plans(
                graph,
                candidates,
                perturbed_profile,
                active_registry,
                exchange_permutations,
            )
            perturbed_rank = _equivalence_ranks(ranked, tolerance)
            perturbed_by_id = {item.plan_id: item for item in ranked}
            responses = {
                identifier: perturbed_by_id[identifier].makespan
                - baseline_by_id[identifier].makespan
                for identifier in baseline_by_id
            }
            response_values = tuple(responses.values())
            critical_switches = sum(
                perturbed_by_id[identifier].critical_path
                != baseline_by_id[identifier].critical_path
                for identifier in baseline_by_id
            )
            max_rank_shift = max(
                abs(perturbed_rank[identifier] - baseline_rank[identifier])
                for identifier in baseline_by_id
            )
            perturbed_best = _best_equivalence_set(ranked, tolerance)
            top_set = _top_k_equivalence_set(
                ranked, effective_top_k, tolerance
            )
            runs.append(
                {
                    "relative_delta": delta,
                    "top1_plan_id": ranked[0].plan_id,
                    "top1_changed": baseline_best.isdisjoint(perturbed_best),
                    "optimal_set_changed": perturbed_best != baseline_best,
                    "optimal_set_size": len(perturbed_best),
                    "optimal_set_overlap": len(baseline_best & perturbed_best),
                    "top_k_membership_changed": top_set != baseline_top,
                    "max_rank_shift": max_rank_shift,
                    "strict_pairwise_flips": _strict_pairwise_flips(
                        baseline, ranked
                    ),
                    "critical_path_switches": critical_switches,
                    "max_abs_response_ms": max(
                        abs(value) for value in response_values
                    )
                    * 1e3,
                    "differential_response_ms": (
                        max(response_values) - min(response_values)
                    )
                    * 1e3,
                }
            )

        confidence_values = tuple(observation["confidences"])
        max_differential = max(
            float(run["differential_response_ms"]) for run in runs
        )
        shape_reports.append(
            {
                "key_id": key.key_id,
                "label": key.label,
                **key.as_dict(),
                "occurrences": int(observation["occurrences"]),
                "plans_containing": len(observation["plans"]),
                "sources": sorted(observation["sources"]),
                "confidence_min": min(confidence_values),
                "confidence_max": max(confidence_values),
                "uncertainty_proxy": 1.0 - min(confidence_values),
                "top1_flip": any(bool(run["top1_changed"]) for run in runs),
                "optimal_set_change": any(
                    bool(run["optimal_set_changed"]) for run in runs
                ),
                "top_k_flip": any(
                    bool(run["top_k_membership_changed"]) for run in runs
                ),
                "max_rank_shift": max(int(run["max_rank_shift"]) for run in runs),
                "strict_pairwise_flips": max(
                    int(run["strict_pairwise_flips"]) for run in runs
                ),
                "critical_path_switches": sum(
                    int(run["critical_path_switches"]) for run in runs
                ),
                "max_differential_response_ms": max_differential,
                "differential_response_ratio": max_differential
                / max(baseline[0].makespan * 1e3, 1e-12),
                "runs": runs,
            }
        )

    shape_reports.sort(
        key=lambda item: (
            not bool(item["top1_flip"]),
            not bool(item["top_k_flip"]),
            -int(item["max_rank_shift"]),
            -int(item["strict_pairwise_flips"]),
            -float(item["max_differential_response_ms"]),
            str(item["key_id"]),
        )
    )
    top1_flip_shapes = sum(bool(item["top1_flip"]) for item in shape_reports)
    optimal_set_change_shapes = sum(
        bool(item["optimal_set_change"]) for item in shape_reports
    )
    top_k_flip_shapes = sum(bool(item["top_k_flip"]) for item in shape_reports)
    max_rank_shift = max(
        (int(item["max_rank_shift"]) for item in shape_reports), default=0
    )
    critical_switches = sum(
        int(item["critical_path_switches"]) for item in shape_reports
    )
    if top1_flip_shapes:
        signal = "strong"
    elif top_k_flip_shapes or max_rank_shift:
        signal = "moderate"
    elif critical_switches:
        signal = "weak"
    else:
        signal = "none"

    return {
        "format": REPORT_FORMAT,
        "version": REPORT_VERSION,
        "graph_name": graph.name,
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "configuration": {
            "perturbations": list(deltas),
            "top_k": effective_top_k,
            "tie_tolerance_s": tolerance,
            "baseline_optimal_set_size": len(baseline_best),
            "shape_key_scope": "device_class+kernel+operator_shape",
            "device_classes": list(device_classes),
            "exchange_permutations": [
                list(permutation) for permutation in exchange_permutations
            ],
            "symmetry_enabled": len(exchange_permutations) > 1,
            "max_exchange_score_delta_ms": max_exchange_delta * 1e3,
        },
        "baseline_candidates": [
            {
                "order": order,
                "rank": baseline_rank[item.plan_id],
                "plan_id": item.plan_id,
                "makespan_ms": item.makespan * 1e3,
                "groups": len(item.plan.groups),
                "used_devices": sorted(
                    {int(group.device) for group in item.plan.groups}
                ),
                "critical_path": list(item.critical_path),
                "confidence": item.confidence,
            }
            for order, item in enumerate(baseline, start=1)
        ],
        "shape_sensitivities": shape_reports,
        "summary": {
            "candidate_plans": len(baseline),
            "shape_keys": len(shape_reports),
            "top1_flip_shapes": top1_flip_shapes,
            "optimal_set_change_shapes": optimal_set_change_shapes,
            "top_k_flip_shapes": top_k_flip_shapes,
            "max_rank_shift": max_rank_shift,
            "strict_pairwise_flips": max(
                (
                    int(item["strict_pairwise_flips"])
                    for item in shape_reports
                ),
                default=0,
            ),
            "critical_path_switches": critical_switches,
            "max_differential_response_ms": max(
                (
                    float(item["max_differential_response_ms"])
                    for item in shape_reports
                ),
                default=0.0,
            ),
            "ranking_signal": signal,
            "ranking_signal_detected": signal in {"strong", "moderate"},
        },
    }


class _CachedScore:
    def __init__(
        self,
        graph: OpGraph,
        profile: HardwareCostProfile,
        registry: FusionPatternRegistry,
        exchange_permutations: Sequence[Sequence[int]],
    ) -> None:
        self.graph = graph
        self.profile = profile
        self.model = StructuralCostModel(registry=registry)
        self.exchange_permutations = tuple(
            tuple(int(value) for value in permutation)
            for permutation in exchange_permutations
        )
        self.values: dict[tuple, float] = {}

    def __call__(self, plan: Plan) -> float:
        key = canonical_plan_key(plan, self.exchange_permutations)
        if key not in self.values:
            self.values[key] = float(
                self.model.predict(self.graph, plan, self.profile).makespan
            )
        return self.values[key]


def generate_candidate_pool(
    graph: OpGraph,
    profile: HardwareCostProfile,
    *,
    candidate_count: int = 20,
    seeds: Sequence[int] = (0, 1, 7, 23),
    iterations: int = 64,
    rollouts: int = 3,
    include_planned: bool = False,
    registry: FusionPatternRegistry | None = None,
) -> tuple[list[Plan], dict[str, object]]:
    """Collect competitive plans across MCTS seeds, then rank and deduplicate."""

    if candidate_count < 2:
        raise ValueError("candidate_count must be at least two")
    if not seeds:
        raise ValueError("seeds must not be empty")
    active_registry = registry or default_fusion_registry()
    device_classes = infer_device_classes(profile)
    exchange_permutations = device_exchange_permutations(device_classes)
    scorer = _CachedScore(
        graph, profile, active_registry, exchange_permutations
    )
    raw_collected: dict[str, Plan] = {}
    for device in range(profile.n_devs):
        reference = make_reference_plan(
            graph, device=device, registry=active_registry
        )
        raw_collected[plan_id(reference)] = reference
    for seed in seeds:
        plans = legal_mcts_search(
            graph,
            scorer,
            n_devs=profile.n_devs,
            iterations=iterations,
            rollouts=rollouts,
            fusion_probability=0.7,
            seed=int(seed),
            n_best=candidate_count,
            registry=active_registry,
            require_executable=not include_planned,
        )
        for plan in plans:
            raw_collected[plan_id(plan)] = plan
    collected: dict[str, Plan] = {}
    for plan in raw_collected.values():
        identifier = plan_id(plan, exchange_permutations)
        current = collected.get(identifier)
        if current is None or scorer(plan) < scorer(current):
            collected[identifier] = plan
    ranked = sorted(
        collected.values(),
        key=lambda plan: (
            scorer(plan),
            plan_id(plan, exchange_permutations),
        ),
    )
    selected = ranked[:candidate_count]
    if len(selected) < 2:
        raise RuntimeError("search produced fewer than two distinct candidate plans")
    metadata = {
        "selection": (
            "best canonical plans across independent MCTS seeds after "
            "device-exchange deduplication"
        ),
        "seeds": [int(seed) for seed in seeds],
        "iterations_per_seed": int(iterations),
        "rollouts": int(rollouts),
        "include_planned": bool(include_planned),
        "require_executable": not include_planned,
        "device_classes": list(device_classes),
        "exchange_permutations": [
            list(permutation) for permutation in exchange_permutations
        ],
        "raw_unique_plans_collected": len(raw_collected),
        "canonical_plans_collected": len(collected),
        "symmetry_duplicates_removed": len(raw_collected) - len(collected),
        "selected_plans": len(selected),
        "cost_model_evaluations": len(scorer.values),
    }
    return selected, metadata


def save_sensitivity(path: str | Path, report: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


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
    parser.add_argument(
        "--perturbations",
        default=",".join(str(value) for value in DEFAULT_PERTURBATIONS),
    )
    parser.add_argument(
        "--include-planned",
        action="store_true",
        help="include planned fusion placeholders; default is executable-only",
    )
    parser.add_argument(
        "--device-symmetry",
        choices=("auto", "none"),
        default="auto",
        help=(
            "auto averages devices with the same configured model name and "
            "deduplicates their label permutations"
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    measured_profile = MeasuredHardwareProfile.from_files(args.hardware, args.raw)
    if args.device_symmetry == "auto":
        device_classes = infer_hardware_model_classes(measured_profile)
        profile: HardwareCostProfile = SymmetricHardwareProfile(
            measured_profile, device_classes
        )
    else:
        device_classes = _default_device_classes(measured_profile.n_devs)
        profile = measured_profile
    graph = build_evaluation_graph(args.scale)
    plans, pool_metadata = generate_candidate_pool(
        graph,
        profile,
        candidate_count=args.candidates,
        seeds=_parse_csv_ints(args.seeds),
        iterations=args.iterations,
        rollouts=args.rollouts,
        include_planned=args.include_planned,
    )
    report = evaluate_counterfactual_sensitivity(
        graph,
        plans,
        profile,
        perturbations=_parse_csv_floats(args.perturbations),
        top_k=args.top_k,
    )
    report["hardware_name"] = measured_profile.hardware.name
    report["hardware_hash"] = hardware_fingerprint(measured_profile.hardware)
    report["device_symmetry"] = {
        "mode": args.device_symmetry,
        "device_classes": list(device_classes),
        "symmetric_profile": isinstance(profile, SymmetricHardwareProfile),
    }
    report["candidate_pool"] = pool_metadata
    save_sensitivity(args.output, report)

    summary = report["summary"]
    print(f"scope: {'planned-inclusive' if args.include_planned else 'executable-only'}")
    print(f"device_symmetry: {args.device_symmetry}")
    print(f"device_classes: {','.join(device_classes)}")
    print(f"nodes: {report['nodes']}")
    print(f"candidates: {summary['candidate_plans']}")
    print(f"shape_keys: {summary['shape_keys']}")
    print(f"top1_flip_shapes: {summary['top1_flip_shapes']}")
    print(f"top_k_flip_shapes: {summary['top_k_flip_shapes']}")
    print(f"max_rank_shift: {summary['max_rank_shift']}")
    print(f"strict_pairwise_flips: {summary['strict_pairwise_flips']}")
    print(
        "max_differential_response_ms: "
        f"{summary['max_differential_response_ms']:.6f}"
    )
    print(f"ranking_signal: {summary['ranking_signal']}")
    print(f"report: {args.output}")
    print("Fusion E0 counterfactual sensitivity evaluation: OK")


__all__ = [
    "CounterfactualHardwareProfile",
    "DEFAULT_PERTURBATIONS",
    "OperatorShape",
    "ShapeKey",
    "SymmetricHardwareProfile",
    "canonical_plan_key",
    "device_exchange_permutations",
    "evaluate_counterfactual_sensitivity",
    "generate_candidate_pool",
    "infer_device_classes",
    "infer_hardware_model_classes",
    "operator_shape",
    "permute_plan_devices",
    "plan_id",
    "save_sensitivity",
    "shape_key_for_group",
]


if __name__ == "__main__":
    main()
