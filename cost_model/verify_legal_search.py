"""CPU-only smoke test for legal-candidate Beam and MCTS."""

from __future__ import annotations

from collections import Counter

from simulator import enumerate_fusion_candidates, validate_fusion_plan
from simulator.model_graphs import TransformerModelSpec, build_decoder_layer

from .legal_search import legal_beam_search, legal_mcts_search


def _qwen3_layer_spec() -> TransformerModelSpec:
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
    graph = build_decoder_layer(_qwen3_layer_spec(), layer_id=0)
    candidates = enumerate_fusion_candidates(graph)
    scorer = lambda plan: float(len(plan.groups))
    beam = legal_beam_search(
        graph,
        scorer,
        n_devs=2,
        beam=4,
        rollouts=1,
        fusion_probability=1.0,
        seed=0,
    )
    mcts = legal_mcts_search(
        graph,
        scorer,
        n_devs=2,
        iterations=30,
        rollouts=1,
        fusion_probability=1.0,
        seed=0,
        n_best=4,
    )
    for plan in beam + mcts:
        validate_fusion_plan(graph, plan, n_devs=2)

    counts = Counter(candidate.pattern_id for candidate in candidates)
    beam_patterns = Counter(group.extra["pattern_id"] for group in beam[0].groups)
    mcts_patterns = Counter(group.extra["pattern_id"] for group in mcts[0].groups)
    if beam_patterns["silu_mul"] != 1 or mcts_patterns["silu_mul"] != 1:
        raise AssertionError("legal search did not retain the Qwen3 silu_mul candidate")

    print(f"nodes: {len(graph.nodes)}")
    print(f"candidates: {len(candidates)}")
    print(f"candidate patterns: {dict(sorted(counts.items()))}")
    print(f"beam plans: {len(beam)}, best groups: {len(beam[0].groups)}")
    print(f"mcts plans: {len(mcts)}, best groups: {len(mcts[0].groups)}")
    print("strict plan validation: OK")
    print("Fusion M1b verification: OK")


if __name__ == "__main__":
    main()
