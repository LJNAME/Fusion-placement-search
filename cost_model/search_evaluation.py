"""Evaluate placement-only, fusion-only, sequential, and joint legal search."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass
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
from simulator.model_graphs import (
    TransformerModelSpec,
    build_decoder_layer,
    build_model_graph,
)
from simulator.dataset import plan_from_dict, plan_to_dict
from simulator.hardware import hardware_from_dict, hardware_to_dict
from simulator.hardware import load_hardware
from simulator.schema import OpNode, TensorEdge
from simulator.placement_artifacts import (
    PlacedOpGraph, graph_fingerprint, save_placed_op_graph,
)

from .exact_search import exact_legal_search
from .legal_search import legal_beam_search, legal_mcts_search
from .measured_profile import MeasuredHardwareProfile
from .hardware_profile import AnalyticHardwareProfile
from .structural import StructuralCostModel
from .structure import analyze_plan_structure


REPORT_FORMAT = "fusion-placement-search.search-evaluation"
REPORT_VERSION = "1.3"
INFEASIBLE_PENALTY_S = 1e6


def qwen3_spec(num_layers: int) -> TransformerModelSpec:
    return TransformerModelSpec(
        model_type="qwen3",
        num_layers=num_layers,
        hidden_size=1024,
        ffn_hidden_size=3072,
        num_attention_heads=16,
        num_query_groups=8,
        head_dim=128,
        seq_length=256,
        micro_batch_size=1,
        vocab_size=151936,
        dtype="bf16",
        normalization="RMSNorm",
        activation="silu",
        use_swiglu=True,
        qk_layernorm=True,
        tied_embeddings=True,
        attention_backend="unfused",
        attention_softmax_in_fp32=True,
        use_rotary_position_embeddings=True,
        linear_bias=False,
    )


def qwen25_3b_spec() -> TransformerModelSpec:
    """Official Qwen2.5-3B dimensions represented by the dense graph builder."""

    return TransformerModelSpec(
        model_type="qwen2.5-3b",
        num_layers=36,
        hidden_size=2048,
        ffn_hidden_size=11008,
        num_attention_heads=16,
        num_query_groups=2,
        head_dim=128,
        seq_length=256,
        micro_batch_size=1,
        vocab_size=151936,
        dtype="bf16",
        normalization="RMSNorm",
        activation="silu",
        use_swiglu=True,
        qk_layernorm=False,
        tied_embeddings=True,
        attention_backend="unfused",
        attention_softmax_in_fp32=True,
        use_rotary_position_embeddings=True,
        linear_bias=False,
    )


def _small_graph() -> OpGraph:
    full = build_decoder_layer(qwen3_spec(1), layer_id=0)
    selected = {
        "L000.mlp.gate_proj",
        "L000.mlp.up_proj",
        "L000.mlp.silu",
        "L000.mlp.gated_mul",
        "L000.mlp.down_proj",
    }
    graph = OpGraph(name="qwen3-mlp-search-evaluation")
    for node_id in full.topological_order():
        if node_id in selected:
            graph.add_node(copy.deepcopy(full.nodes[node_id]))
    for edge in full.edges:
        if edge.src in selected and edge.dst in selected:
            graph.add_edge(copy.deepcopy(edge))
    return graph


def build_evaluation_graph(scale: str) -> OpGraph:
    if scale == "small":
        return _small_graph()
    if scale == "layer":
        return build_decoder_layer(qwen3_spec(1), layer_id=0)
    if scale == "model":
        return build_model_graph(qwen3_spec(28))
    if scale == "model_3b":
        return build_model_graph(qwen25_3b_spec())
    raise ValueError(f"unknown evaluation scale: {scale}")


def _singleton_registry() -> FusionPatternRegistry:
    patterns = tuple(
        pattern
        for pattern in default_fusion_registry().patterns()
        if pattern.max_nodes == 1
    )
    return FusionPatternRegistry(patterns)


@dataclass(frozen=True)
class _Mode:
    name: str
    registry: FusionPatternRegistry
    require_executable: bool
    allowed_devices: tuple[int, ...]


def _modes(n_devs: int) -> tuple[_Mode, ...]:
    all_devices = tuple(range(n_devs))
    return (
        _Mode("placement_only", _singleton_registry(), True, all_devices),
        _Mode("fusion_only", default_fusion_registry(), False, (0,)),
        _Mode("sequential", default_fusion_registry(), False, all_devices),
        _Mode("joint", default_fusion_registry(), False, all_devices),
    )


def placement_then_fusion_mode(n_devs: int) -> _Mode:
    """Staged baseline: placement first (unfused), then fusion with devices frozen.

    Kept outside _modes() so existing four-mode reports are unchanged.
    """
    return _Mode("placement_then_fusion", default_fusion_registry(), False, tuple(range(n_devs)))


def _freeze_placement(placement):
    """Stage-two expand: each group takes its nodes' frozen device.

    A fused group whose nodes were placed on different devices is not
    representable without moving a node, so it is rejected (None).
    """
    devices = {n: g.device for g in placement.groups for n in g.node_ids}

    def expand(plan):
        result = copy.deepcopy(plan)
        result.name = "placement-then-fusion"
        for group in result.groups:
            targets = {devices[n] for n in group.node_ids}
            if len(targets) != 1:
                return None
            group.device = targets.pop()
        return result

    return expand, devices


def _plan_key(plan: Plan) -> tuple:
    return tuple(
        (
            group.extra.get("pattern_id"),
            tuple(group.node_ids),
            int(group.device),
        )
        for group in plan.groups
    )


def _partition_key(plan: Plan) -> tuple:
    return tuple((group.extra["pattern_id"], tuple(group.node_ids)) for group in plan.groups)


def _contract_partition(graph, template, registry, n_devs):
    """Restrict stage two to device choices; score only expanded original plans."""
    structure = analyze_plan_structure(graph, template, n_devs, registry=registry)
    macro = OpGraph(name=graph.name + ":fixed-partition")
    for group in structure.groups:
        macro.add_node(OpNode(str(group.index), "fixed_partition", group.flops,
                              group.output_bytes, group.parameter_bytes))
    for dep in structure.dependencies:
        macro.add_edge(TensorEdge(str(dep.src_group), str(dep.dst_group), dep.bytes))

    def expand(placement):
        devices = {int(g.node_ids[0]): g.device for g in placement.groups}
        result = copy.deepcopy(template)
        result.name = "sequential-fixed-partition"
        for index, group in enumerate(result.groups):
            group.device = devices[index]
        return result

    return macro, expand


class _CachedScorer:
    def __init__(self, graph: OpGraph, profile: MeasuredHardwareProfile, *, capacity=None) -> None:
        self.graph = graph
        self.profile = profile
        self.capacity = tuple(float(x) for x in (capacity if capacity is not None else
                              (d.max_mem for d in profile.hardware.devices)))
        if len(self.capacity) != profile.n_devs or any(not math.isfinite(x) or x <= 0 for x in self.capacity):
            raise ValueError("capacity must contain one finite positive byte limit per device")
        self.bind_capacity = True
        self.feasibility_cache: dict[tuple, bool] = {}
        self.last_makespan = 0.
        self.last_feasible = False
        self.model = StructuralCostModel()
        self.score_cache: dict[tuple, float] = {}
        self.cache_enabled = len(graph.nodes) <= 100
        self.calls = 0
        self.run_requests = 0
        self.run_model_evaluations = 0
        self.run_keys: set[tuple] = set()

    def start_run(self) -> None:
        self.run_requests = 0
        self.run_model_evaluations = 0
        self.run_keys.clear()

    def run_metrics(self) -> dict[str, int]:
        return {
            "scorer_requests": self.run_requests,
            "unique_requested_plans": len(self.run_keys),
            "cost_model_evaluations": self.run_model_evaluations,
        }

    def estimate(self, plan: Plan):
        return self.model.predict(self.graph, plan, self.profile)

    def is_feasible(self, estimate) -> bool:
        if len(estimate.peak_memory) != len(self.capacity):
            raise ValueError("peak-memory and capacity dimensions differ")
        return all(peak <= limit for peak, limit in zip(estimate.peak_memory, self.capacity))

    def _score(self, makespan, feasible):
        self.last_makespan, self.last_feasible = float(makespan), bool(feasible)
        score = float(makespan) + (INFEASIBLE_PENALTY_S if self.bind_capacity and not feasible else 0.)
        if not math.isfinite(score):
            raise ValueError("search score must remain finite")
        return score

    def __call__(self, plan: Plan) -> float:
        key = _plan_key(plan)
        self.run_requests += 1
        self.run_keys.add(key)
        if not self.cache_enabled:
            self.calls += 1
            self.run_model_evaluations += 1
            estimate = self.estimate(plan)
            return self._score(estimate.makespan, self.is_feasible(estimate))
        if key not in self.score_cache:
            estimate = self.estimate(plan)
            self.score_cache[key] = float(estimate.makespan)
            self.feasibility_cache[key] = self.is_feasible(estimate)
            self.calls += 1
            self.run_model_evaluations += 1
        return self._score(self.score_cache[key], self.feasibility_cache[key])


def _candidate_summary(
    plan: Plan,
    scorer: _CachedScorer,
    registry: FusionPatternRegistry,
    rank: int,
) -> dict[str, object]:
    estimate = scorer.estimate(plan)
    fused = [group for group in plan.groups if len(group.node_ids) > 1]
    placeholders = [
        group
        for group in plan.groups
        if registry.get(str(group.extra["pattern_id"])).implementation_status == "planned"
    ]
    signature_payload = json.dumps(_plan_key(plan), separators=(",", ":"))
    return {
        "rank": rank,
        "makespan_s": estimate.makespan,
        "peak_memory_bytes": list(estimate.peak_memory),
        "capacity_bytes": list(scorer.capacity),
        "feasible": scorer.is_feasible(estimate),
        "communication_time_s": estimate.communication_time,
        "cross_device_bytes": estimate.cross_device_bytes,
        "confidence": estimate.confidence,
        "out_of_distribution": estimate.out_of_distribution,
        "ood_reasons": list(estimate.ood_reasons),
        "groups": len(plan.groups),
        "fused_groups": len(fused),
        "placeholder_groups": len(placeholders),
        "used_devices": sorted({group.device for group in plan.groups}),
        "patterns": [str(group.extra["pattern_id"]) for group in fused],
        "plan_signature": "sha256:"
        + hashlib.sha256(signature_payload.encode("utf-8")).hexdigest(),
        "plan": plan_to_dict(plan),
        "covered_nodes": sum(len(group.node_ids) for group in plan.groups),
        "deployable": False,
    }


def _run_search(
    method: str,
    graph: OpGraph,
    scorer: _CachedScorer,
    mode: _Mode,
    *,
    seed: int,
    beam: int,
    rollouts: int,
    mcts_iterations: int,
    top_k: int,
) -> tuple[list[Plan], float, dict[str, object]]:
    if method not in {"exact", "beam", "mcts"}:
        raise ValueError(f"unknown search method: {method}")
    if method == "mcts" and mode.name == "sequential" and mcts_iterations < 2:
        raise ValueError("sequential MCTS needs at least two total iterations")
    scorer.start_run()
    started = time.perf_counter()
    stages = []

    def stage(name, search_graph, rules, devices, iterations, count, expand=lambda p: p,
              enforce_capacity=True):
        before = scorer.run_metrics()
        conflicts = [0]
        requested = {}
        feasible_keys = set()
        retained = {}
        def score(plan):
            original = expand(plan)
            if original is None:  # placement_then_fusion: fusion across frozen devices
                conflicts[0] += 1
                return 2 * INFEASIBLE_PENALTY_S
            key = _plan_key(original)
            value = scorer(original)
            requested[key] = scorer.last_makespan
            if scorer.last_feasible:
                feasible_keys.add(key)
            if not enforce_capacity or scorer.last_feasible:
                if key not in retained:
                    retained[key] = (scorer.last_makespan, original)
                    if len(retained) > count:
                        # Stable ties retain first observations, matching MCTS.
                        worst = max(reversed(retained), key=lambda k: retained[k][0])
                        del retained[worst]
            return value
        common = dict(graph=search_graph, scorer=score, n_devs=scorer.profile.n_devs,
                      registry=rules, require_executable=mode.require_executable,
                      allowed_devices=devices)
        stage_started = time.perf_counter()
        details = {"stage": name, "seed": seed, "allowed_devices": list(devices),
                   "capacity_bound": enforce_capacity}
        previous_binding = scorer.bind_capacity
        scorer.bind_capacity = enforce_capacity
        try:
            if method == "exact":
                exact = exact_legal_search(**common, top_k=count)
                plans = list(exact.plans)
                details.update(search_space_size=exact.search_space_size, evaluated_plans=exact.evaluated_plans)
            elif method == "beam":
                plans = legal_beam_search(**common, beam=beam, rollouts=rollouts,
                                         fusion_probability=.7, seed=seed)[:count]
                details.update(beam=beam, rollouts=rollouts,
                    reference_policy="best_finite_uniform_singleton_on_allowed_cards_in_this_stage_space",
                    reference_candidates=len(devices))
            else:
                plans = legal_mcts_search(**common, iterations=iterations, rollouts=rollouts,
                                         fusion_probability=.7, seed=seed, n_best=count)
                details.update(iterations=iterations, rollouts=rollouts)
        finally:
            scorer.bind_capacity = previous_binding
        # A finite search penalty guides exploration; final admission is strict.
        # Include feasible rollouts even if the engine returned only invalid plans.
        expanded = [e for e in (expand(p) for p in plans) if e is not None]
        selected = { _plan_key(p): p for p in expanded
                     if not enforce_capacity or _plan_key(p) in feasible_keys }
        selected.update({k: p for k, (_, p) in retained.items() if k not in selected})
        plans = sorted(selected.values(), key=lambda p: requested[_plan_key(p)])[:count]
        after = scorer.run_metrics()
        details.update(elapsed_s=time.perf_counter()-stage_started,
            scorer_requests=after["scorer_requests"]-before["scorer_requests"],
            unique_requested_plans=len(requested),
            unique_feasible_plans=len(feasible_keys), frozen_device_conflicts=conflicts[0],
            cost_model_evaluations=after["cost_model_evaluations"]-before["cost_model_evaluations"])
        details["reason"] = None if plans else (
            "no_feasible_in_space" if method == "exact" else "evaluated_no_feasible")
        stages.append(details)
        return plans

    if mode.name == "sequential":
        first = mcts_iterations // 2
        # This is a partition proposal, not the final single-device placement.
        template = stage("fusion", graph, mode.registry, (0,), first, 1, enforce_capacity=False)[0]
        macro, expand = _contract_partition(graph, template, mode.registry, scorer.profile.n_devs)
        plans = stage("placement", macro, _singleton_registry(), mode.allowed_devices,
                      mcts_iterations-first, top_k, expand)
        if any(_partition_key(plan) != _partition_key(template) for plan in plans):
            raise AssertionError("sequential placement changed the frozen fusion partition")
    elif mode.name == "placement_then_fusion":
        first = mcts_iterations // 2
        # Same split as sequential; stage one is the placement_only space.
        placed = stage("placement", graph, _singleton_registry(), mode.allowed_devices, first, 1)
        if not placed:
            plans = []
        else:
            expand, frozen_devices = _freeze_placement(placed[0])
            # Device choice is fixed by expand, so stage two only explores fusion.
            plans = stage("fusion", graph, mode.registry, (0,), mcts_iterations-first, top_k, expand)
            if any(g.device != frozen_devices[n] for p in plans for g in p.groups for n in g.node_ids):
                raise AssertionError("placement_then_fusion fusion stage moved a node")
    else:
        plans = stage(mode.name, graph, mode.registry, mode.allowed_devices, mcts_iterations, top_k)
    elapsed = time.perf_counter() - started
    budget = {"stages": stages, **scorer.run_metrics(), "capacity_bytes": list(scorer.capacity),
              "infeasible_penalty_s": INFEASIBLE_PENALTY_S,
              "reason": stages[-1]["reason"],
              "feasibility_scope": "frozen_partition" if mode.name == "sequential" else "mode_space"}
    if method == "mcts":
        budget.update(iterations=mcts_iterations, rollouts=rollouts)
    elif method == "beam":
        budget.update(beam=beam, rollouts=rollouts,
            budget_semantics="beam width per stage; not a fixed additive work budget")
    else:
        budget["evaluated_plans"] = sum(s["evaluated_plans"] for s in stages)
    if mode.name == "sequential":
        budget["frozen_partition"] = [dict(pattern_id=p, node_ids=list(nodes))
                                       for p,nodes in _partition_key(template)]
    if mode.name == "placement_then_fusion" and placed:
        budget["frozen_placement"] = {n: d for n, d in sorted(frozen_devices.items())}
    return plans, elapsed, budget


def evaluate_scale(
    scale: str,
    profile: MeasuredHardwareProfile,
    *,
    seed: int = 23,
    beam: int = 8,
    rollouts: int = 3,
    mcts_iterations: int = 64,
    top_k: int = 4,
    include_exact: bool = True,
    include_model_beam: bool = False,
) -> dict[str, object]:
    graph = build_evaluation_graph(scale)
    if mcts_iterations < 2:
        raise ValueError("four-mode evaluation needs at least two MCTS iterations")
    is_large_model = scale in {"model", "model_3b"}
    capacity = tuple(d.max_mem for d in profile.hardware.devices)
    scorer = _CachedScorer(graph, profile, capacity=capacity)
    reference_registry = default_fusion_registry()
    baseline = make_reference_plan(graph, device=0, registry=reference_registry)
    result: dict[str, object] = {
        "scale": scale,
        "graph_name": graph.name,
        "graph_hash": graph_fingerprint(graph),
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "parameter_elements": int(
            sum(float(node.param_bytes) for node in graph.nodes.values()) / 2.0
        ),
        "parameter_bytes": sum(
            float(node.param_bytes) for node in graph.nodes.values()
        ),
        "configuration": {
            "seed": seed,
            "beam": beam,
            "rollouts": rollouts,
            "mcts_iterations": mcts_iterations,
            "top_k": top_k,
            "capacity_bytes": list(capacity),
            "include_exact": include_exact,
            "include_model_beam": include_model_beam,
            "fusion_probability": .7,
            "sequential_fusion_device": 0,
            "sequential_iteration_split": [mcts_iterations//2, mcts_iterations-mcts_iterations//2],
        },
        "baseline": _candidate_summary(
            baseline, scorer, reference_registry, rank=1
        ),
        "searches": [],
        "skipped_methods": (
            []
            if not is_large_model or include_model_beam
            else [
                {
                    "method": "beam",
                    "reason": "direct full-graph rollout expansion is quadratic in graph depth",
                }
            ]
        ),
    }

    for mode in _modes(profile.n_devs):
        exact_score = None
        if include_exact and scale == "small":
            exact_plans, exact_elapsed, exact_budget = _run_search(
                "exact", graph, scorer, mode, seed=seed, beam=beam, rollouts=rollouts,
                mcts_iterations=mcts_iterations, top_k=top_k,
            )
            exact_score = min((scorer(plan) for plan in exact_plans), default=None)
            result["searches"].append(
                {
                    "mode": mode.name,
                    "method": "exact",
                    "elapsed_s": exact_elapsed,
                    "search_space_size": (None if mode.name == "sequential"
                                          else exact_budget["stages"][0]["search_space_size"]),
                    "budget": exact_budget,
                    "optimality_gap": None if mode.name == "sequential" or not exact_plans else 0.,
                    "reference_kind": ("two_stage_exact_not_global_optimum" if mode.name == "sequential"
                                       else "restricted_space_exact_optimum"),
                    "reference_gap": 0. if exact_plans else None,
                    "plan": plan_to_dict(exact_plans[0]) if exact_plans else None,
                    "reason": exact_budget["reason"],
                    "candidates": [
                        _candidate_summary(plan, scorer, mode.registry, rank)
                        for rank, plan in enumerate(exact_plans, start=1)
                    ],
                }
            )

        methods = (
            ("beam", "mcts")
            if not is_large_model or include_model_beam
            else ("mcts",)
        )
        for method in methods:
            plans, elapsed, budget = _run_search(
                method,
                graph,
                scorer,
                mode,
                seed=seed,
                beam=beam,
                rollouts=rollouts,
                mcts_iterations=mcts_iterations,
                top_k=top_k,
            )
            best_score = min((scorer(plan) for plan in plans), default=None)
            gap = None
            reference_gap = None
            if exact_score is not None and best_score is not None:
                reference_gap = (best_score-exact_score)/max(abs(exact_score), 1e-12)
            if exact_score is not None and best_score is not None and mode.name != "sequential":
                gap = max(best_score - exact_score, 0.0) / max(abs(exact_score), 1e-12)
            result["searches"].append(
                {
                    "mode": mode.name,
                    "method": method,
                    "elapsed_s": elapsed,
                    "search_space_size": None,
                    "budget": budget,
                    "plan": plan_to_dict(plans[0]) if plans else None,
                    "reason": budget["reason"],
                    "optimality_gap": gap,
                    "reference_gap": reference_gap,
                    "reference_kind": (None if exact_score is None else
                        "two_stage_exact_not_global_optimum" if mode.name == "sequential"
                        else "restricted_space_exact_optimum"),
                    "candidates": [
                        _candidate_summary(plan, scorer, mode.registry, rank)
                        for rank, plan in enumerate(plans, start=1)
                    ],
                }
            )
    baseline_summary = result["baseline"]
    searches = result["searches"]
    assert isinstance(baseline_summary, dict)
    assert isinstance(searches, list)
    baseline_makespan = float(baseline_summary["makespan_s"])
    baseline_summary["speedup_vs_unfused"] = 1.0 if baseline_summary["feasible"] else None
    baseline_summary["improvement_percent"] = 0.0 if baseline_summary["feasible"] else None
    for search in searches:
        assert isinstance(search, dict)
        candidates = search["candidates"]
        assert isinstance(candidates, list)
        for candidate in candidates:
            assert isinstance(candidate, dict)
            makespan = float(candidate["makespan_s"])
            candidate["speedup_vs_unfused"] = baseline_makespan / makespan if baseline_summary["feasible"] else None
            candidate["improvement_percent"] = (
                (baseline_makespan - makespan) / baseline_makespan * 100.0
            ) if baseline_summary["feasible"] else None
    result["cost_model_evaluations"] = scorer.calls
    return result


def evaluate_searches(
    profile: MeasuredHardwareProfile,
    scales: Sequence[str],
    **kwargs,
) -> dict[str, object]:
    return {
        "format": REPORT_FORMAT,
        "version": REPORT_VERSION,
        "hardware_hash": hardware_fingerprint(profile.hardware),
        "hardware_name": profile.hardware.name,
        "hardware_snapshot": hardware_to_dict(profile.hardware),
        "profile_policy": {"kind": "analytic" if isinstance(profile, AnalyticHardwareProfile) else "measured",
                           "measurement_contract_id": getattr(profile, "measurement_contract_id", None),
                           "allow_unbound": getattr(profile, "allow_unbound", False)},
        "search_implementation_sha256": {name: hashlib.sha256(
            (Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest() for name in (
                "cost_model/legal_search.py", "cost_model/exact_search.py", "cost_model/structural.py",
                "cost_model/structure.py", "cost_model/hardware_profile.py", "simulator/fusion_rules.py")},
        "placeholder_policy": (
            "fusion_only, sequential and joint include built-in planned patterns for search "
            "evaluation; results are not deployable"
        ),
        "scales": [evaluate_scale(scale, profile, **kwargs) for scale in scales],
    }


def save_evaluation(path: str | Path, report: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    hardware = hardware_from_dict(report["hardware_snapshot"])
    for scale in report["scales"]:
        graph = build_evaluation_graph(scale["scale"])
        for search in scale["searches"]:
            if not search["candidates"]:
                search["best_placed_op_graph"] = None
                continue
            best = search["candidates"][0]
            relative = Path(output.stem + ".artifacts") / scale["scale"] / (search["mode"] + "_" + search["method"] + ".json")
            placed = PlacedOpGraph(graph, hardware, plan_from_dict(best["plan"]), rank=1,
                predicted_makespan=best["makespan_s"], predicted_peak_memory=tuple(best["peak_memory_bytes"]),
                extra={"deployable": False, "scope": "fixed_profile_development_search_evaluation",
                       "profile_policy": report["profile_policy"], "configuration": scale["configuration"]})
            save_placed_op_graph(output.parent / relative, placed)
            search["best_placed_op_graph"] = relative.as_posix()
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


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
    parser.add_argument("--scales", default="small,layer,model")
    parser.add_argument("--profile", choices=("measured", "analytic"), default="measured",
                        help="analytic uses only hardware snapshot; measured retains strict label admission")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--beam", type=int, default=8)
    parser.add_argument("--rollouts", type=int, default=3)
    parser.add_argument("--mcts-iterations", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--measurement-contract-id")
    parser.add_argument("--allow-unbound", action="store_true",
                        help="diagnostic only: permit inadmissible labels with warnings and low confidence")
    parser.add_argument(
        "--model-beam",
        action="store_true",
        help="also run direct Beam on large full-model graphs (slow and memory intensive)",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scales = tuple(item.strip() for item in args.scales.split(",") if item.strip())
    unknown = sorted(set(scales) - {"small", "layer", "model", "model_3b"})
    if not scales or unknown:
        raise ValueError(f"invalid --scales value; unknown={unknown}")
    if args.profile == "analytic":
        if args.allow_unbound or args.measurement_contract_id:
            raise ValueError("analytic profile has no measured labels; omit measurement admission flags")
        profile = AnalyticHardwareProfile(load_hardware(args.hardware))
    else:
        profile = MeasuredHardwareProfile.from_files(args.hardware, args.raw,
            measurement_contract_id=args.measurement_contract_id, allow_unbound=args.allow_unbound)
    beam = min(args.beam, 2) if args.quick else args.beam
    rollouts = 1 if args.quick else args.rollouts
    mcts_iterations = min(args.mcts_iterations, 16) if args.quick else args.mcts_iterations
    top_k = min(args.top_k, 2) if args.quick else args.top_k
    report = evaluate_searches(
        profile,
        scales,
        seed=args.seed,
        beam=beam,
        rollouts=rollouts,
        mcts_iterations=mcts_iterations,
        top_k=top_k,
        include_model_beam=args.model_beam,
    )
    used_inputs = [("hardware", args.hardware)]
    if args.profile == "measured":
        used_inputs.append(("raw", args.raw))
    report["inputs"] = {name: {"path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for name,path in
        used_inputs}
    report["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    save_evaluation(args.output, report)
    for scale in report["scales"]:
        print(
            f"{scale['scale']}: nodes={scale['nodes']} edges={scale['edges']} "
            f"searches={len(scale['searches'])} "
            f"cost_model_evaluations={scale['cost_model_evaluations']}"
        )
        for search in scale["searches"]:
            best = search["candidates"][0]
            print(
                f"  {search['mode']}/{search['method']}: "
                f"{best['makespan_s'] * 1e3:.6f} ms "
                f"groups={best['groups']} placeholders={best['placeholder_groups']} "
                f"elapsed={search['elapsed_s']:.3f}s"
            )
        for skipped in scale["skipped_methods"]:
            print(f"  skipped {skipped['method']}: {skipped['reason']}")
    print(f"report: {args.output}")
    print("Fusion M6 search evaluation: OK")


if __name__ == "__main__":
    main()
