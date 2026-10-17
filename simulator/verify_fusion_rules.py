"""CPU-only smoke test for the strict Fusion M1a contract."""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Dict

from .fusion_rules import (
    enumerate_fusion_candidates,
    make_reference_plan,
)
from .model_graphs import TransformerModelSpec, build_model_graph, validate_model_graph
from .validate import validate_fusion_plan


def qwen3_smoke_spec(num_layers: int = 28) -> TransformerModelSpec:
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


def verify_qwen3_fusion_rules(num_layers: int = 28) -> Dict[str, object]:
    spec = qwen3_smoke_spec(num_layers)
    graph = build_model_graph(spec)
    summary = validate_model_graph(graph, spec)
    candidates = enumerate_fusion_candidates(graph)
    counts = Counter(candidate.pattern_id for candidate in candidates)
    reference = make_reference_plan(graph)
    validate_fusion_plan(
        graph,
        reference,
        n_devs=1,
        require_executable=True,
    )
    if counts["silu_mul"] != num_layers:
        raise AssertionError(
            f"expected {num_layers} silu_mul candidates, got {counts['silu_mul']}"
        )
    return {
        "nodes": summary.node_count,
        "edges": summary.edge_count,
        "candidates": len(candidates),
        "patterns": dict(sorted(counts.items())),
        "reference_groups": len(reference.groups),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=28)
    args = parser.parse_args()
    result = verify_qwen3_fusion_rules(args.layers)
    for key, value in result.items():
        print(f"{key}: {value}")
    print("strict executable reference: OK")
    print("Fusion M1a verification: OK")


if __name__ == "__main__":
    main()
