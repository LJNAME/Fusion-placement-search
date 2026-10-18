"""M4a exhaustive legal-fusion and placement oracle tests."""

from __future__ import annotations

import unittest

from cost_model import (
    CostEstimate,
    ExactSearchLimitExceeded,
    LegalFusionSearchSpace,
    count_exact_search_space,
    exact_legal_search,
    legal_beam_search,
    measure_optimality_gap,
)
from simulator import OpGraph, OpNode, TensorEdge, validate_fusion_plan


def _node(node_id: str, op_type: str) -> OpNode:
    return OpNode(
        id=node_id,
        op_type=op_type,
        flops=100.0,
        out_bytes=16.0,
        extra={
            "output_shape": (8,),
            "dtype": "bf16",
            "phase": "forward",
            "layer": 0,
        },
    )


def _chain(n_nodes: int) -> OpGraph:
    graph = OpGraph(name=f"chain-{n_nodes}")
    for index in range(n_nodes):
        graph.add_node(_node(f"n{index}", "matmul"))
        if index:
            graph.add_edge(TensorEdge(f"n{index - 1}", f"n{index}", 16.0))
    return graph


def _swiglu_graph() -> OpGraph:
    graph = OpGraph(name="swiglu-exact")
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


def _plan_key(plan):
    return tuple(
        (group.extra["pattern_id"], tuple(group.node_ids), group.device)
        for group in plan.groups
    )


class TestExactSpaceCounting(unittest.TestCase):
    def test_singleton_chain_has_all_device_assignments(self):
        space = LegalFusionSearchSpace(_chain(3), n_devs=2)
        self.assertEqual(count_exact_search_space(space), 8)

    def test_swiglu_counts_fused_and_unfused_covers(self):
        space = LegalFusionSearchSpace(_swiglu_graph(), n_devs=2)
        self.assertEqual(count_exact_search_space(space), 48)


class TestExactLegalSearch(unittest.TestCase):
    def test_finds_known_global_optimum(self):
        graph = _swiglu_graph()

        def scorer(plan):
            return float(10 * len(plan.groups) + sum(group.device for group in plan.groups))

        result = exact_legal_search(graph, scorer, n_devs=2, top_k=5)
        self.assertEqual(result.search_space_size, 48)
        self.assertEqual(result.evaluated_plans, 48)
        self.assertEqual(result.feasible_plans, 48)
        self.assertEqual(result.optimal_score, 40.0)
        self.assertEqual(len(result.best_plan.groups), 4)
        self.assertTrue(all(group.device == 0 for group in result.best_plan.groups))
        self.assertIn("silu_mul", [group.extra["pattern_id"] for group in result.best_plan.groups])
        validate_fusion_plan(graph, result.best_plan, n_devs=2)

    def test_top_k_is_deterministic(self):
        graph = _swiglu_graph()
        scorer = lambda plan: float(len(plan.groups))
        first = exact_legal_search(graph, scorer, n_devs=2, top_k=6)
        second = exact_legal_search(graph, scorer, n_devs=2, top_k=6)
        self.assertEqual(first.scores, second.scores)
        self.assertEqual(
            [_plan_key(plan) for plan in first.plans],
            [_plan_key(plan) for plan in second.plans],
        )

    def test_feasibility_filter_preserves_exact_count(self):
        graph = _chain(3)
        result = exact_legal_search(
            graph,
            lambda plan: float(sum(group.device for group in plan.groups)),
            n_devs=2,
            is_feasible=lambda plan: any(group.device == 1 for group in plan.groups),
        )
        self.assertEqual(result.search_space_size, 8)
        self.assertEqual(result.evaluated_plans, 8)
        self.assertEqual(result.feasible_plans, 7)
        self.assertEqual(result.optimal_score, 1.0)

    def test_cost_estimate_memory_overflow_is_infeasible(self):
        graph = _chain(3)

        def scorer(plan):
            overflow = any(group.device == 1 for group in plan.groups)
            reasons = ("device:1:memory_overflow",) if overflow else ()
            return CostEstimate(
                makespan=float(1 + sum(group.device for group in plan.groups)),
                peak_memory=(10.0, 20.0 if overflow else 0.0),
                critical_path=("fg0000",),
                communication_time=0.0,
                computation_time=1.0,
                per_device_time=(1.0, 0.0),
                cross_device_bytes=0.0,
                confidence=1.0,
                out_of_distribution=overflow,
                ood_reasons=reasons,
                details={"overflow_devices": [1] if overflow else []},
            )

        result = exact_legal_search(graph, scorer, n_devs=2)
        self.assertEqual(result.evaluated_plans, 8)
        self.assertEqual(result.feasible_plans, 1)
        self.assertTrue(all(group.device == 0 for group in result.best_plan.groups))

    def test_rejects_node_and_evaluation_limits_before_scoring(self):
        calls = 0

        def scorer(plan):
            nonlocal calls
            calls += 1
            return 0.0

        with self.assertRaisesRegex(ExactSearchLimitExceeded, "at most 15 nodes"):
            exact_legal_search(_chain(16), scorer, n_devs=1)
        with self.assertRaisesRegex(ExactSearchLimitExceeded, "requires 32768"):
            exact_legal_search(
                _chain(15),
                scorer,
                n_devs=2,
                max_evaluations=1000,
            )
        self.assertEqual(calls, 0)

    def test_beam_gap_uses_exact_reference(self):
        graph = _swiglu_graph()
        scorer = lambda plan: float(10 * len(plan.groups) + sum(group.device for group in plan.groups))
        exact = exact_legal_search(graph, scorer, n_devs=2)
        beam = legal_beam_search(
            graph,
            scorer,
            n_devs=2,
            beam=8,
            rollouts=2,
            seed=5,
        )
        gap = measure_optimality_gap(exact, beam, scorer)
        self.assertGreaterEqual(gap.relative_gap, 0.0)
        self.assertEqual(gap.optimal_score, exact.optimal_score)
        self.assertGreaterEqual(gap.candidate_score, exact.optimal_score)


if __name__ == "__main__":
    unittest.main()
