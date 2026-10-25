"""E0 counterfactual shape-to-plan ranking sensitivity experiment.

The experiment keeps the candidate plans fixed and perturbs one reusable
operator-shape latency at a time.  It therefore measures whether a possible
profiling error can change the search decision, without yet implementing the
full active-profiling/VOI loop.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
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
REPORT_VERSION = "1.0"
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
    """Device-specific reusable latency key used by the E0 perturbation."""

    device: int
    kernel_id: str
    operators: tuple[OperatorShape, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "device": self.device,
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
        return f"device={self.device}|kernel={self.kernel_id}|{operators}"


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


def shape_key_for_group(graph: OpGraph, group: GroupStructure) -> ShapeKey:
    # Singleton measurements in MeasuredHardwareProfile are selected by
    # op/dtype/shape, independent of whether the legal candidate was named
    # "unfused" or "rmsnorm_reference".  Real fused kernels remain distinct.
    kernel_id = "singleton" if len(group.node_ids) == 1 else group.pattern_id
    return ShapeKey(
        device=int(group.device),
        kernel_id=kernel_id,
        operators=tuple(operator_shape(graph, node_id) for node_id in group.node_ids),
    )


class CounterfactualHardwareProfile:
    """Read-only profile view that scales one shared ShapeKey latency."""

    def __init__(
        self,
        base: HardwareCostProfile,
        target: ShapeKey,
        relative_delta: float,
    ) -> None:
        self.base = as_hardware_profile(base)
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
        if shape_key_for_group(graph, group) != self.target:
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


def _plan_key(plan: Plan) -> tuple:
    return tuple(
        (
            str(group.extra.get("pattern_id", "legacy")),
            tuple(group.node_ids),
            int(group.device),
        )
        for group in plan.groups
    )


def plan_id(plan: Plan) -> str:
    payload = json.dumps(_plan_key(plan), separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
) -> list[_RankedPlan]:
    model = StructuralCostModel(registry=registry)
    ranked = []
    for plan in plans:
        estimate = model.predict(graph, plan, profile)
        ranked.append(
            _RankedPlan(
                plan_id=plan_id(plan),
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


def _shape_observations(
    graph: OpGraph,
    plans: Sequence[Plan],
    profile: HardwareCostProfile,
    registry: FusionPatternRegistry,
) -> dict[ShapeKey, dict[str, object]]:
    observations: dict[ShapeKey, dict[str, object]] = {}
    for plan in plans:
        identifier = plan_id(plan)
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
            key = shape_key_for_group(graph, group)
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
    registry: FusionPatternRegistry | None = None,
) -> dict[str, object]:
    """Evaluate fixed candidates under one-shape-at-a-time perturbations."""

    active_registry = registry or default_fusion_registry()
    base_profile = as_hardware_profile(profile)
    unique = {plan_id(plan): plan for plan in plans}
    candidates = list(unique.values())
    if len(candidates) < 2:
        raise ValueError("E0 requires at least two distinct candidate plans")
    deltas = tuple(float(value) for value in perturbations)
    if not deltas or any(value <= -1.0 or value == 0.0 for value in deltas):
        raise ValueError("perturbations must be non-zero and greater than -1")

    baseline = _rank_plans(graph, candidates, base_profile, active_registry)
    baseline_rank = {item.plan_id: rank for rank, item in enumerate(baseline, start=1)}
    baseline_by_id = {item.plan_id: item for item in baseline}
    effective_top_k = min(max(int(top_k), 1), len(baseline))
    baseline_top = {item.plan_id for item in baseline[:effective_top_k]}
    observations = _shape_observations(
        graph, candidates, base_profile, active_registry
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
                graph, candidates, perturbed_profile, active_registry
            )
            perturbed_rank = {
                item.plan_id: rank for rank, item in enumerate(ranked, start=1)
            }
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
            top_set = {item.plan_id for item in ranked[:effective_top_k]}
            runs.append(
                {
                    "relative_delta": delta,
                    "top1_plan_id": ranked[0].plan_id,
                    "top1_changed": ranked[0].plan_id != baseline[0].plan_id,
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
            "shape_key_scope": "device+kernel+operator_shape",
        },
        "baseline_candidates": [
            {
                "rank": rank,
                "plan_id": item.plan_id,
                "makespan_ms": item.makespan * 1e3,
                "groups": len(item.plan.groups),
                "used_devices": sorted(
                    {int(group.device) for group in item.plan.groups}
                ),
                "critical_path": list(item.critical_path),
                "confidence": item.confidence,
            }
            for rank, item in enumerate(baseline, start=1)
        ],
        "shape_sensitivities": shape_reports,
        "summary": {
            "candidate_plans": len(baseline),
            "shape_keys": len(shape_reports),
            "top1_flip_shapes": top1_flip_shapes,
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
    ) -> None:
        self.graph = graph
        self.profile = profile
        self.model = StructuralCostModel(registry=registry)
        self.values: dict[tuple, float] = {}

    def __call__(self, plan: Plan) -> float:
        key = _plan_key(plan)
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
    scorer = _CachedScore(graph, profile, active_registry)
    collected: dict[str, Plan] = {}
    for device in range(profile.n_devs):
        reference = make_reference_plan(
            graph, device=device, registry=active_registry
        )
        collected[plan_id(reference)] = reference
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
            collected[plan_id(plan)] = plan
    ranked = sorted(collected.values(), key=lambda plan: (scorer(plan), plan_id(plan)))
    selected = ranked[:candidate_count]
    if len(selected) < 2:
        raise RuntimeError("search produced fewer than two distinct candidate plans")
    metadata = {
        "selection": "best unique plans across independent MCTS seeds",
        "seeds": [int(seed) for seed in seeds],
        "iterations_per_seed": int(iterations),
        "rollouts": int(rollouts),
        "include_planned": bool(include_planned),
        "require_executable": not include_planned,
        "unique_plans_collected": len(collected),
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
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    profile = MeasuredHardwareProfile.from_files(args.hardware, args.raw)
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
    report["hardware_name"] = profile.hardware.name
    report["hardware_hash"] = hardware_fingerprint(profile.hardware)
    report["candidate_pool"] = pool_metadata
    save_sensitivity(args.output, report)

    summary = report["summary"]
    print(f"scope: {'planned-inclusive' if args.include_planned else 'executable-only'}")
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
    "evaluate_counterfactual_sensitivity",
    "generate_candidate_pool",
    "operator_shape",
    "plan_id",
    "save_sensitivity",
    "shape_key_for_group",
]


if __name__ == "__main__":
    main()
