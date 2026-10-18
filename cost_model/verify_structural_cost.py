"""CPU verification entry point for the M3a structural cost layer."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from cost_model import StructuralCostModel, legal_beam_search
from simulator import (
    DeviceSpec,
    HardwareSpec,
    SearchResult,
    load_search_result,
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


def _hardware(rate_scale: float = 1.0) -> HardwareSpec:
    shape_model = {
        "matmul": {
            "compute": {
                "curve": "table",
                "points": [[128, 0.5], [1024, 0.8], [4096, 0.9]],
            }
        }
    }
    return HardwareSpec(
        name="synthetic-calibrated",
        devices=[
            DeviceSpec(
                100e12 * rate_scale,
                800e9 * rate_scale,
                64e9,
                name="fast",
                extra={"calibrated": True},
            ),
            DeviceSpec(
                70e12 * rate_scale,
                600e9 * rate_scale,
                32e9,
                name="slow",
                extra={"calibrated": True},
            ),
        ],
        comm_bw=20e9 * rate_scale,
        comm_latency=5e-6,
        launch_overhead=4e-6,
        fusion_overhead=2e-7,
        link_bw=20e9 * rate_scale,
        interference=0.1,
        extra={"shape_model": shape_model},
    )


def verify_structural_cost() -> dict[str, object]:
    graph = build_decoder_layer(_spec(), layer_id=0)
    plans = legal_beam_search(
        graph,
        lambda plan: float(len(plan.groups)),
        n_devs=2,
        beam=4,
        rollouts=1,
        fusion_probability=1.0,
        seed=19,
    )
    plan = plans[0]
    for index, group in enumerate(plan.groups):
        group.device = index % 2
    model = StructuralCostModel()
    estimate = model.predict(graph, plan, _hardware())
    faster = model.predict(graph, plan, _hardware(rate_scale=2.0))
    candidate = estimate.as_search_candidate(1, plan)

    result = SearchResult(
        graph=graph,
        hardware=_hardware(),
        objective={"metric": "predicted_makespan", "direction": "minimize"},
        constraints={},
        search_method="beam+structural-v1",
        budget={"beam": 4, "rollouts": 1},
        seed=19,
        candidates=[candidate],
    )
    with TemporaryDirectory() as directory:
        path = Path(directory) / "m3a-search-result.json"
        save_search_result(path, result)
        restored = load_search_result(path)

    if not faster.makespan < estimate.makespan:
        raise AssertionError("faster hardware did not reduce predicted makespan")
    if candidate.predicted_makespan != estimate.makespan:
        raise AssertionError("CostEstimate to SearchCandidate bridge changed makespan")
    if len(estimate.peak_memory) != 2:
        raise AssertionError("per-device peak memory is incomplete")
    if estimate.cross_device_bytes <= 0.0:
        raise AssertionError("verification plan did not exercise cross-device transfer")
    if restored.candidates[0].extra["cost_model"] != "structural-v1":
        raise AssertionError("M2 artifact round trip lost M3 cost metadata")

    return {
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "groups": len(plan.groups),
        "makespan_s": f"{estimate.makespan:.9f}",
        "peak_memory": [int(value) for value in estimate.peak_memory],
        "cross_device_bytes": int(estimate.cross_device_bytes),
        "critical_path_groups": len(estimate.critical_path),
        "confidence": estimate.confidence,
        "out_of_distribution": estimate.out_of_distribution,
    }


def main() -> None:
    result = verify_structural_cost()
    for key, value in result.items():
        print(f"{key}: {value}")
    print("Fusion M3a structural cost verification: OK")


if __name__ == "__main__":
    main()
