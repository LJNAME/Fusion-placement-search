"""Verify M3b against the committed BW1000 two-card calibration artifacts."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from cost_model import MeasuredHardwareProfile, StructuralCostModel, analyze_plan_structure
from simulator import load_hardware, make_reference_plan
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


def verify_measured_profile() -> dict[str, object]:
    graph = build_decoder_layer(_spec(), layer_id=0)
    plan = make_reference_plan(graph)
    for index, group in enumerate(plan.groups):
        group.device = index & 1

    measured = MeasuredHardwareProfile.from_files(HARDWARE, RAW)
    structure = analyze_plan_structure(graph, plan, measured.n_devs)
    group_costs = [measured.estimate_group(graph, group) for group in structure.groups]
    sources = Counter(cost.source for cost in group_costs)
    measured_estimate = StructuralCostModel().predict(graph, plan, measured)
    aggregate_estimate = StructuralCostModel().predict(
        graph,
        plan,
        load_hardware(HARDWARE),
    )

    if sources["measured_exact_shape"] != 14:
        raise AssertionError(f"unexpected exact-shape coverage: {sources}")
    if sources["logical_input"] != 1:
        raise AssertionError(f"logical input was not recognized: {sources}")
    if sources["analytic_fallback"] != 6:
        raise AssertionError(f"unexpected fallback coverage: {sources}")
    if not measured_estimate.makespan > aggregate_estimate.makespan:
        raise AssertionError("full-shape profile did not separate from the K-only aggregate")
    if measured_estimate.confidence != 0.25:
        raise AssertionError(f"unexpected confidence: {measured_estimate.confidence}")

    return {
        "nodes": len(graph.nodes),
        "groups": len(plan.groups),
        "measured_exact_groups": sources["measured_exact_shape"],
        "logical_groups": sources["logical_input"],
        "fallback_groups": sources["analytic_fallback"],
        "aggregate_makespan_s": f"{aggregate_estimate.makespan:.9f}",
        "measured_makespan_s": f"{measured_estimate.makespan:.9f}",
        "measured_confidence": measured_estimate.confidence,
        "measured_ood": measured_estimate.out_of_distribution,
        "ood_reasons": list(measured_estimate.ood_reasons),
    }


def main() -> None:
    for key, value in verify_measured_profile().items():
        print(f"{key}: {value}")
    print("Fusion M3b measured-profile verification: OK")


if __name__ == "__main__":
    main()
