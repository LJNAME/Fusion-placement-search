"""Search and save an executable Qwen3 single-layer placed-op-graph."""

from __future__ import annotations

import argparse
from pathlib import Path

from cost_model import MeasuredHardwareProfile, StructuralCostModel, legal_beam_search
from simulator import PlacedOpGraph, save_placed_op_graph
from simulator.model_graphs import TransformerModelSpec, build_decoder_layer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", required=True, type=Path)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--beam", type=int, default=8)
    parser.add_argument("--rollouts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


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


def main() -> None:
    args = _parse_args()
    graph = build_decoder_layer(_spec(), layer_id=0)
    profile = MeasuredHardwareProfile.from_files(args.hardware, args.raw)
    model = StructuralCostModel()
    scorer = lambda plan: model.predict(graph, plan, profile)
    plans = legal_beam_search(
        graph,
        lambda plan: scorer(plan).makespan,
        n_devs=profile.n_devs,
        beam=args.beam,
        rollouts=args.rollouts,
        fusion_probability=0.7,
        seed=args.seed,
        require_executable=True,
    )
    estimate = scorer(plans[0])
    placed = PlacedOpGraph(
        graph=graph,
        hardware=profile.hardware,
        plan=plans[0],
        rank=1,
        predicted_makespan=estimate.makespan,
        predicted_peak_memory=estimate.peak_memory,
        extra={
            "scope": "qwen3-single-layer-synthetic-forward",
            "search_method": "beam",
            "seed": args.seed,
            "cost_model": estimate.model_name,
        },
    )
    save_placed_op_graph(args.output, placed)
    used_devices = sorted({group.device for group in placed.plan.groups})
    print(f"nodes: {len(graph.nodes)}")
    print(f"groups: {len(placed.plan.groups)}")
    print(f"used_logical_devices: {used_devices}")
    print(f"predicted_makespan_ms: {estimate.makespan * 1e3:.6f}")
    print(f"artifact: {args.output}")
    print("Fusion M5 reference artifact preparation: OK")


if __name__ == "__main__":
    main()
