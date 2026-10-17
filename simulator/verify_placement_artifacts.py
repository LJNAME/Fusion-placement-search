"""CPU verification entry point for M2 placement artifacts."""

from __future__ import annotations

from tempfile import TemporaryDirectory
from pathlib import Path

from cost_model import legal_beam_search
from simulator import (
    DeviceSpec,
    HardwareSpec,
    SearchCandidate,
    SearchResult,
    load_placed_op_graph,
    load_search_result,
    placed_from_search_result,
    save_placed_op_graph,
    save_search_result,
)
from simulator.model_graphs import TransformerModelSpec, build_decoder_layer


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


def verify_placement_artifacts() -> dict[str, object]:
    graph = build_decoder_layer(_spec(), layer_id=0)
    hardware = HardwareSpec(
        name="verification-hardware",
        devices=[
            DeviceSpec(1e12, 1e11, 16 * 1024**3, name="device0"),
            DeviceSpec(8e11, 8e10, 16 * 1024**3, name="device1"),
        ],
    )
    plans = legal_beam_search(
        graph,
        lambda plan: float(len(plan.groups)),
        n_devs=2,
        beam=4,
        rollouts=1,
        fusion_probability=1.0,
        seed=17,
    )
    result = SearchResult(
        graph=graph,
        hardware=hardware,
        objective={"metric": "predicted_makespan", "direction": "minimize"},
        constraints={},
        search_method="beam",
        budget={"beam": 4, "rollouts": 1},
        seed=17,
        candidates=[
            SearchCandidate(
                rank=index + 1,
                plan=plan,
                predicted_makespan=float(len(plan.groups)),
                predicted_peak_memory=(128 * 1024**2, 128 * 1024**2),
            )
            for index, plan in enumerate(plans)
        ],
    )

    with TemporaryDirectory() as directory:
        search_path = Path(directory) / "search-result.json"
        placed_path = Path(directory) / "placed-op-graph.json"
        save_search_result(search_path, result)
        restored_result = load_search_result(search_path)
        placed = placed_from_search_result(restored_result, rank=1)
        save_placed_op_graph(placed_path, placed)
        restored_placed = load_placed_op_graph(placed_path)

    return {
        "nodes": len(restored_placed.graph.nodes),
        "edges": len(restored_placed.graph.edges),
        "candidates": len(restored_result.candidates),
        "best_groups": len(restored_placed.plan.groups),
        "graph_hash": restored_result.graph_hash,
        "hardware_hash": restored_result.hardware_hash,
        "candidate_id": restored_placed.candidate_id,
    }


def main() -> None:
    result = verify_placement_artifacts()
    for key, value in result.items():
        print(f"{key}: {value}")
    print("Fusion M2 artifact verification: OK")


if __name__ == "__main__":
    main()
