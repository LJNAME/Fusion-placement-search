"""Verify executable-only exact, Beam, and MCTS with the BW1000 profile."""

from __future__ import annotations

import copy
from pathlib import Path

from cost_model import (
    MeasuredHardwareProfile,
    StructuralCostModel,
    exact_legal_search,
    legal_beam_search,
    legal_mcts_search,
    measure_optimality_gap,
)
from simulator import OpGraph, validate_fusion_plan
from simulator.model_graphs import TransformerModelSpec, build_decoder_layer


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "bw1000_2card" / "raw_measurements.json"
HARDWARE = ROOT / "data" / "bw1000_2card" / "bw1000_2card.calibrated.json"


def _spec() -> TransformerModelSpec:
    return TransformerModelSpec(
        model_type="qwen3",
        num_layers=1,
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


def _qwen_mlp_graph() -> OpGraph:
    full = build_decoder_layer(_spec(), layer_id=0)
    selected = {
        "L000.mlp.gate_proj",
        "L000.mlp.up_proj",
        "L000.mlp.silu",
        "L000.mlp.gated_mul",
        "L000.mlp.down_proj",
    }
    graph = OpGraph(name="qwen3-mlp-executable-search")
    for node_id in full.topological_order():
        if node_id in selected:
            graph.add_node(copy.deepcopy(full.nodes[node_id]))
    for edge in full.edges:
        if edge.src in selected and edge.dst in selected:
            graph.add_edge(copy.deepcopy(edge))
    return graph


def verify_executable_search() -> dict[str, object]:
    graph = _qwen_mlp_graph()
    profile = MeasuredHardwareProfile.from_files(HARDWARE, RAW)
    model = StructuralCostModel()
    scorer = lambda plan: model.predict(graph, plan, profile)
    scalar_scorer = lambda plan: scorer(plan).makespan

    unrestricted = exact_legal_search(
        graph,
        scorer,
        n_devs=profile.n_devs,
        top_k=4,
        require_executable=False,
    )
    executable = exact_legal_search(
        graph,
        scorer,
        n_devs=profile.n_devs,
        top_k=4,
        require_executable=True,
    )
    beam = legal_beam_search(
        graph,
        scalar_scorer,
        n_devs=profile.n_devs,
        beam=8,
        rollouts=3,
        fusion_probability=0.7,
        seed=23,
        require_executable=True,
    )
    mcts = legal_mcts_search(
        graph,
        scalar_scorer,
        n_devs=profile.n_devs,
        iterations=64,
        rollouts=2,
        fusion_probability=0.7,
        seed=23,
        n_best=8,
        require_executable=True,
    )
    beam_gap = measure_optimality_gap(executable, beam, scorer)
    mcts_gap = measure_optimality_gap(executable, mcts, scorer)

    if unrestricted.search_space_size != 48:
        raise AssertionError(f"unexpected unrestricted space: {unrestricted.search_space_size}")
    if executable.search_space_size != 32:
        raise AssertionError(f"unexpected executable space: {executable.search_space_size}")
    for plan in executable.plans + tuple(beam) + tuple(mcts):
        validate_fusion_plan(
            graph,
            plan,
            n_devs=profile.n_devs,
            require_executable=True,
        )
    if beam_gap.relative_gap > 1e-12 or mcts_gap.relative_gap > 1e-12:
        raise AssertionError(
            f"search did not close executable exact gap: beam={beam_gap}, mcts={mcts_gap}"
        )

    return {
        "nodes": len(graph.nodes),
        "unrestricted_space": unrestricted.search_space_size,
        "executable_space": executable.search_space_size,
        "unrestricted_optimal_ms": f"{unrestricted.optimal_score * 1e3:.6f}",
        "executable_optimal_ms": f"{executable.optimal_score * 1e3:.6f}",
        "executable_groups": len(executable.best_plan.groups),
        "beam_gap": f"{beam_gap.relative_gap:.6f}",
        "mcts_gap": f"{mcts_gap.relative_gap:.6f}",
        "patterns": ",".join(
            group.extra["pattern_id"] for group in executable.best_plan.groups
        ),
    }


def main() -> None:
    for key, value in verify_executable_search().items():
        print(f"{key}: {value}")
    print("Fusion executable-only measured-search verification: OK")


if __name__ == "__main__":
    main()
