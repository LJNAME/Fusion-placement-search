"""Legal fusion-candidate Beam/MCTS tests; no GPU or torch is required."""

from __future__ import annotations

import unittest

from cost_model.legal_search import (
    LegalFusionSearchSpace,
    legal_beam_search,
    legal_mcts_search,
)
from simulator import OpGraph, OpNode, TensorEdge, validate_fusion_plan
from simulator.model_graphs import TransformerModelSpec, build_decoder_layer


def _node(node_id: str, op_type: str) -> OpNode:
    return OpNode(
        id=node_id,
        op_type=op_type,
        out_bytes=16.0,
        extra={
            "output_shape": (8,),
            "dtype": "bf16",
            "phase": "forward",
            "layer": 0,
        },
    )


def _swiglu_graph() -> OpGraph:
    graph = OpGraph(name="swiglu-search")
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
    return graph


def _qwen3_spec() -> TransformerModelSpec:
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


def _plan_key(plan):
    return tuple(
        (group.extra["pattern_id"], tuple(group.node_ids), group.device)
        for group in plan.groups
    )


class TestLegalSearchSpace(unittest.TestCase):
    def test_non_contiguous_linear_order_candidate_contracts_to_valid_plan(self):
        graph = _swiglu_graph()
        self.assertEqual(graph.topological_order(), ["gate", "silu", "up", "mul", "out"])
        space = LegalFusionSearchSpace(graph, n_devs=1)
        state = space.initial_state()

        def choose(pattern_id, node_ids):
            return next(
                action
                for action in space.actions(state)
                if action.candidate.pattern_id == pattern_id
                and action.candidate.node_ids == node_ids
            )

        state = space.apply(state, choose("unfused", ("gate",)))
        state = space.apply(state, choose("silu_mul", ("silu", "mul")))
        state = space.apply(state, choose("unfused", ("up",)))
        state = space.apply(state, choose("unfused", ("out",)))
        plan = space.to_plan(state)

        self.assertEqual(
            [group.node_ids for group in plan.groups],
            [["gate"], ["up"], ["silu", "mul"], ["out"]],
        )
        validate_fusion_plan(graph, plan, n_devs=1)

    def test_partial_state_cannot_be_converted(self):
        space = LegalFusionSearchSpace(_swiglu_graph(), n_devs=1)
        with self.assertRaisesRegex(ValueError, "partial state"):
            space.to_plan(space.initial_state())


class TestLegalBeamAndMCTS(unittest.TestCase):
    def setUp(self):
        self.graph = _swiglu_graph()
        self.scorer = lambda plan: float(len(plan.groups))

    def _assert_strict(self, plans, n_devs=2):
        self.assertTrue(plans)
        for plan in plans:
            validate_fusion_plan(self.graph, plan, n_devs=n_devs)
            self.assertEqual(plan.extra["search_space"], "legal_fusion_v1")

    def test_beam_finds_silu_mul_and_is_deterministic(self):
        first = legal_beam_search(
            self.graph,
            self.scorer,
            n_devs=2,
            beam=8,
            rollouts=3,
            fusion_probability=0.7,
            seed=7,
        )
        second = legal_beam_search(
            self.graph,
            self.scorer,
            n_devs=2,
            beam=8,
            rollouts=3,
            fusion_probability=0.7,
            seed=7,
        )
        self._assert_strict(first)
        self.assertEqual([_plan_key(plan) for plan in first], [_plan_key(plan) for plan in second])
        self.assertTrue(
            any(
                group.extra["pattern_id"] == "silu_mul"
                for group in first[0].groups
            )
        )

    def test_mcts_returns_only_strict_plans(self):
        plans = legal_mcts_search(
            self.graph,
            self.scorer,
            n_devs=2,
            iterations=40,
            rollouts=2,
            fusion_probability=0.7,
            seed=3,
            n_best=5,
        )
        self._assert_strict(plans)
        self.assertTrue(
            any(
                group.extra["pattern_id"] == "silu_mul"
                for group in plans[0].groups
            )
        )

    def test_qwen3_layer_beam_smoke(self):
        graph = build_decoder_layer(_qwen3_spec(), layer_id=0)
        plans = legal_beam_search(
            graph,
            lambda plan: float(len(plan.groups)),
            n_devs=1,
            beam=4,
            rollouts=1,
            fusion_probability=1.0,
            seed=0,
        )
        self.assertTrue(plans)
        validate_fusion_plan(graph, plans[0], n_devs=1)
        self.assertEqual(len(plans[0].groups), 20)
        self.assertIn(
            ["L000.mlp.silu", "L000.mlp.gated_mul"],
            [group.node_ids for group in plans[0].groups],
        )


if __name__ == "__main__":
    unittest.main()
