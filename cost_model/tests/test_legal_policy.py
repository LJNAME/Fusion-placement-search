"""Variable legal-action policy/value tests."""

from __future__ import annotations

import unittest

import torch

from cost_model import (
    LegalFusionSearchSpace,
    LegalPolicyValueNet,
    TorchLegalPolicyValueEvaluator,
    encode_legal_policy_state,
    legal_mcts_search,
    legal_policy_value_loss,
)
from simulator import OpGraph, OpNode, TensorEdge, validate_fusion_plan


def _node(node_id: str, op_type: str) -> OpNode:
    return OpNode(
        id=node_id,
        op_type=op_type,
        flops=100.0,
        out_bytes=16.0,
        param_bytes=8.0,
        extra={"output_shape": (8,), "dtype": "bf16", "phase": "forward", "layer": 0},
    )


def _swiglu_graph() -> OpGraph:
    graph = OpGraph(name="legal-policy-swiglu")
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


class TestLegalPolicyEncoding(unittest.TestCase):
    def setUp(self):
        self.space = LegalFusionSearchSpace(_swiglu_graph(), n_devs=2)
        self.patterns = tuple(
            pattern.pattern_id for pattern in self.space.registry.patterns()
        )

    def test_action_axis_tracks_current_legal_candidates(self):
        root = self.space.initial_state()
        root_encoded = encode_legal_policy_state(self.space, root)
        self.assertEqual(root_encoded.action_features.shape[0], 2)
        first = self.space.apply(root, self.space.actions(root)[0])
        next_encoded = encode_legal_policy_state(self.space, first)
        self.assertEqual(next_encoded.action_features.shape[0], 4)
        self.assertEqual(len(next_encoded.action_ids), 4)
        self.assertEqual(len(set(next_encoded.action_ids)), 4)
        self.assertTrue(any(action_id.startswith("silu_mul|") for action_id in next_encoded.action_ids))

    def test_network_and_evaluator_preserve_variable_axis(self):
        torch.manual_seed(7)
        net = LegalPolicyValueNet(2, self.patterns, hidden=16)
        evaluator = TorchLegalPolicyValueEvaluator(net)
        state = self.space.apply(
            self.space.initial_state(), self.space.actions(self.space.initial_state())[0]
        )
        actions = self.space.actions(state)
        value, priors = evaluator(self.space, state, actions)
        self.assertTrue(torch.isfinite(torch.tensor(value)))
        self.assertEqual(len(priors), len(actions))
        self.assertAlmostEqual(sum(priors), 1.0, places=6)

    def test_policy_value_loss_accepts_each_states_own_action_count(self):
        value = torch.tensor(0.2, requires_grad=True)
        logits = torch.tensor([0.1, 0.2, -0.1, 0.4], requires_grad=True)
        target = torch.tensor([0.0, 0.5, 0.0, 0.5])
        loss = legal_policy_value_loss(value, logits, torch.tensor(0.0), target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logits.grad)

    def test_mcts_consumes_variable_policy_priors(self):
        action_counts = set()

        def evaluator(space, state, actions):
            action_counts.add(len(actions))
            weights = [2.0 if len(action.candidate.node_ids) > 1 else 1.0 for action in actions]
            total = sum(weights)
            return 4.0, [weight / total for weight in weights]

        plans = legal_mcts_search(
            self.space.graph,
            lambda plan: float(len(plan.groups) + sum(group.device for group in plan.groups)),
            n_devs=2,
            iterations=20,
            rollouts=1,
            seed=3,
            n_best=4,
            policy_value_evaluator=evaluator,
            value_mix=0.25,
        )
        self.assertIn(2, action_counts)
        self.assertIn(4, action_counts)
        validate_fusion_plan(self.space.graph, plans[0], n_devs=2)


if __name__ == "__main__":
    unittest.main()
