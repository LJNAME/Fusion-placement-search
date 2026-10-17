"""Strict fusion-pattern validation and candidate enumeration tests."""

from __future__ import annotations

import unittest

from simulator import (
    FusionGroup,
    FusionRuleError,
    OpGraph,
    OpNode,
    Plan,
    TensorEdge,
    enumerate_fusion_candidates,
    make_reference_plan,
    validate_fusion_plan,
)
from simulator.model_graphs import TransformerModelSpec, build_decoder_layer, build_model_graph
from simulator.verify_fusion_rules import verify_qwen3_fusion_rules


def _node(node_id: str, op_type: str, shape=(8,), dtype="bf16") -> OpNode:
    return OpNode(
        id=node_id,
        op_type=op_type,
        out_bytes=16.0,
        extra={
            "output_shape": shape,
            "dtype": dtype,
            "phase": "forward",
            "layer": 0,
        },
    )


def _small_swiglu_graph(with_external_consumer: bool = False) -> OpGraph:
    graph = OpGraph(name="small-swiglu")
    for node in (
        _node("gate", "matmul"),
        _node("up", "matmul"),
        _node("silu", "silu"),
        _node("mul", "mul"),
        _node("out", "matmul"),
    ):
        graph.add_node(node)
    graph.add_edge(TensorEdge("gate", "silu", 16.0))
    graph.add_edge(TensorEdge("silu", "mul", 16.0))
    graph.add_edge(TensorEdge("up", "mul", 16.0))
    graph.add_edge(TensorEdge("mul", "out", 16.0))
    if with_external_consumer:
        graph.add_node(_node("side", "add"))
        graph.add_edge(TensorEdge("silu", "side", 16.0))
    return graph


def _qwen3_spec() -> TransformerModelSpec:
    return TransformerModelSpec(
        model_type="qwen3",
        num_layers=28,
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


class TestFusionCandidates(unittest.TestCase):
    def test_small_graph_finds_silu_mul(self):
        candidates = enumerate_fusion_candidates(_small_swiglu_graph())
        keys = {(candidate.pattern_id, candidate.node_ids) for candidate in candidates}
        self.assertIn(("silu_mul", ("silu", "mul")), keys)
        self.assertNotIn(("elementwise_chain", ("silu", "mul")), keys)
        self.assertEqual(
            {candidate.node_ids[0] for candidate in candidates if len(candidate.node_ids) == 1},
            {"gate", "up", "silu", "mul", "out"},
        )

    def test_generic_elementwise_chain_is_available(self):
        graph = OpGraph(name="elementwise")
        graph.add_node(_node("add", "add"))
        graph.add_node(_node("mul", "mul"))
        graph.add_edge(TensorEdge("add", "mul", 16.0))
        keys = {
            (candidate.pattern_id, candidate.node_ids)
            for candidate in enumerate_fusion_candidates(graph)
        }
        self.assertIn(("elementwise_chain", ("add", "mul")), keys)

    def test_internal_fanout_rejects_fusion(self):
        candidates = enumerate_fusion_candidates(_small_swiglu_graph(True))
        keys = {(candidate.pattern_id, candidate.node_ids) for candidate in candidates}
        self.assertNotIn(("silu_mul", ("silu", "mul")), keys)

    def test_qwen3_layer_and_model_candidates(self):
        layer_candidates = enumerate_fusion_candidates(build_decoder_layer(_qwen3_spec(), 0))
        self.assertIn(
            ("silu_mul", ("L000.mlp.silu", "L000.mlp.gated_mul")),
            {(candidate.pattern_id, candidate.node_ids) for candidate in layer_candidates},
        )

        model_candidates = enumerate_fusion_candidates(build_model_graph(_qwen3_spec()))
        silu_mul = [candidate for candidate in model_candidates if candidate.pattern_id == "silu_mul"]
        self.assertEqual(len(silu_mul), 28)


class TestStrictFusionValidation(unittest.TestCase):
    def test_reference_plan_is_executable_and_complete(self):
        graph = _small_swiglu_graph()
        plan = make_reference_plan(graph)
        validate_fusion_plan(graph, plan, n_devs=1, require_executable=True)
        self.assertEqual(len(plan.groups), len(graph.nodes))

    def test_valid_planned_silu_mul_group(self):
        graph = _small_swiglu_graph()
        plan = Plan(
            groups=[
                FusionGroup(["gate"], 0, {"pattern_id": "unfused"}),
                FusionGroup(["up"], 0, {"pattern_id": "unfused"}),
                FusionGroup(
                    ["silu", "mul"],
                    0,
                    {
                        "pattern_id": "silu_mul",
                        "backend": "template",
                        "implementation_status": "planned",
                    },
                ),
                FusionGroup(["out"], 0, {"pattern_id": "unfused"}),
            ]
        )
        validate_fusion_plan(graph, plan, n_devs=1)
        with self.assertRaisesRegex(FusionRuleError, "no executable backend"):
            validate_fusion_plan(graph, plan, n_devs=1, require_executable=True)

    def test_missing_pattern_is_rejected(self):
        graph = _small_swiglu_graph()
        plan = Plan([FusionGroup([node_id], 0) for node_id in graph.topological_order()])
        with self.assertRaisesRegex(FusionRuleError, "missing extra.pattern_id"):
            validate_fusion_plan(graph, plan, n_devs=1)

    def test_wrong_pattern_and_backend_are_rejected(self):
        graph = _small_swiglu_graph()
        plan = make_reference_plan(graph)
        plan.groups[0].extra["pattern_id"] = "silu_mul"
        with self.assertRaisesRegex(FusionRuleError, "expects 2..2 nodes"):
            validate_fusion_plan(graph, plan, n_devs=1)

        plan = make_reference_plan(graph)
        plan.groups[0].extra["backend"] = "wrong"
        with self.assertRaisesRegex(FusionRuleError, "declares backend"):
            validate_fusion_plan(graph, plan, n_devs=1)


class TestFusionVerificationEntryPoint(unittest.TestCase):
    def test_default_qwen3_smoke_result(self):
        result = verify_qwen3_fusion_rules()
        self.assertEqual(result["nodes"], 591)
        self.assertEqual(result["edges"], 730)
        self.assertEqual(result["candidates"], 619)
        self.assertEqual(result["patterns"]["silu_mul"], 28)
        self.assertEqual(result["reference_groups"], 591)


if __name__ == "__main__":
    unittest.main()
