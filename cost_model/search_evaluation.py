"""Evaluate unfused, fusion-only, placement-only, and joint legal search."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
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

from .exact_search import exact_legal_search
from .legal_search import legal_beam_search, legal_mcts_search
from .measured_profile import MeasuredHardwareProfile
from .structural import StructuralCostModel


REPORT_FORMAT = "fusion-placement-search.search-evaluation"
REPORT_VERSION = "1.1"


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
        _Mode("joint", default_fusion_registry(), False, all_devices),
    )


def _plan_key(plan: Plan) -> tuple:
    return tuple(
        (
            group.extra.get("pattern_id"),
            tuple(group.node_ids),
            int(group.device),
        )
        for group in plan.groups
    )


class _CachedScorer:
    def __init__(self, graph: OpGraph, profile: MeasuredHardwareProfile) -> None:
        self.graph = graph
        self.profile = profile
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

    def __call__(self, plan: Plan) -> float:
        key = _plan_key(plan)
        self.run_requests += 1
        self.run_keys.add(key)
        if not self.cache_enabled:
            self.calls += 1
            self.run_model_evaluations += 1
            return float(self.estimate(plan).makespan)
        if key not in self.score_cache:
            self.score_cache[key] = float(self.estimate(plan).makespan)
            self.calls += 1
            self.run_model_evaluations += 1
        return self.score_cache[key]


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
    scorer.start_run()
    started = time.perf_counter()
    common = {
        "graph": graph,
        "scorer": scorer,
        "n_devs": scorer.profile.n_devs,
        "registry": mode.registry,
        "require_executable": mode.require_executable,
        "allowed_devices": mode.allowed_devices,
    }
    if method == "beam":
        plans = legal_beam_search(
            **common,
            beam=beam,
            rollouts=rollouts,
            fusion_probability=0.7,
            seed=seed,
        )[:top_k]
        budget = {"beam": beam, "rollouts": rollouts}
    elif method == "mcts":
        plans = legal_mcts_search(
            **common,
            iterations=mcts_iterations,
            rollouts=rollouts,
            fusion_probability=0.7,
            seed=seed,
            n_best=top_k,
        )
        budget = {"iterations": mcts_iterations, "rollouts": rollouts}
    else:
        raise ValueError(f"unknown search method: {method}")
    elapsed = time.perf_counter() - started
    budget.update(scorer.run_metrics())
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
    is_large_model = scale in {"model", "model_3b"}
    scorer = _CachedScorer(graph, profile)
    reference_registry = default_fusion_registry()
    baseline = make_reference_plan(graph, device=0, registry=reference_registry)
    result: dict[str, object] = {
        "scale": scale,
        "graph_name": graph.name,
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
            exact = exact_legal_search(
                graph,
                scorer,
                n_devs=profile.n_devs,
                top_k=top_k,
                registry=mode.registry,
                require_executable=mode.require_executable,
                allowed_devices=mode.allowed_devices,
            )
            exact_score = exact.optimal_score
            result["searches"].append(
                {
                    "mode": mode.name,
                    "method": "exact",
                    "elapsed_s": exact.elapsed_seconds,
                    "search_space_size": exact.search_space_size,
                    "budget": {"evaluated_plans": exact.evaluated_plans},
                    "optimality_gap": 0.0,
                    "candidates": [
                        _candidate_summary(plan, scorer, mode.registry, rank)
                        for rank, plan in enumerate(exact.plans, start=1)
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
            best_score = min(scorer(plan) for plan in plans)
            gap = None
            if exact_score is not None:
                gap = max(best_score - exact_score, 0.0) / max(abs(exact_score), 1e-12)
            result["searches"].append(
                {
                    "mode": mode.name,
                    "method": method,
                    "elapsed_s": elapsed,
                    "search_space_size": None,
                    "budget": budget,
                    "optimality_gap": gap,
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
    baseline_summary["speedup_vs_unfused"] = 1.0
    baseline_summary["improvement_percent"] = 0.0
    for search in searches:
        assert isinstance(search, dict)
        candidates = search["candidates"]
        assert isinstance(candidates, list)
        for candidate in candidates:
            assert isinstance(candidate, dict)
            makespan = float(candidate["makespan_s"])
            candidate["speedup_vs_unfused"] = baseline_makespan / makespan
            candidate["improvement_percent"] = (
                (baseline_makespan - makespan) / baseline_makespan * 100.0
            )
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
        "placeholder_policy": (
            "fusion_only and joint include built-in planned patterns for search "
            "evaluation; results are not deployable"
        ),
        "scales": [evaluate_scale(scale, profile, **kwargs) for scale in scales],
    }


def save_evaluation(path: str | Path, report: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--beam", type=int, default=8)
    parser.add_argument("--rollouts", type=int, default=3)
    parser.add_argument("--mcts-iterations", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--quick", action="store_true")
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
    profile = MeasuredHardwareProfile.from_files(args.hardware, args.raw)
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
