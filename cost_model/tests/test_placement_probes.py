from __future__ import annotations

import copy
import unittest

from cost_model.placement_probes import graph_wide_candidates
from cost_model.search_evaluation import qwen3_spec
from cost_model.search_sensitivity import device_exchange_permutations, plan_id
from simulator import make_reference_plan, validate_fusion_plan
from simulator.model_graphs import build_model_graph


class PlacementProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = build_model_graph(qwen3_spec(8))
        cls.base = make_reference_plan(cls.graph)
        cls.classes = ("model:bw", "model:bw")

    def test_all_original_candidates_preserved_without_mutation(self):
        original = copy.deepcopy(self.base)
        plans, metadata = graph_wide_candidates(self.graph, [self.base], self.classes)
        exchange = device_exchange_permutations(self.classes)
        self.assertIn(plan_id(self.base, exchange), {plan_id(plan, exchange) for plan in plans})
        self.assertEqual(self.base, original)
        self.assertGreater(metadata["added_candidates"], 0)

    def test_all_layers_and_non_input_nodes_have_placement_variation(self):
        plans, metadata = graph_wide_candidates(self.graph, [self.base], self.classes)
        moved = {key for plan in plans for group in plan.groups if group.device == 1 for key in group.node_ids
                 if plan.extra.get("placement_probe", {}).get("kind") == "all-depth-singleton"}
        expected = {key for key, node in self.graph.nodes.items() if node.op_type != "input"}
        self.assertEqual(moved, expected)
        self.assertEqual(metadata["layer_ids"], list(range(8)))

    def test_fork_chain_reaches_last_layer_and_remains_executable(self):
        plans, _ = graph_wide_candidates(self.graph, [self.base], self.classes)
        chain_plans = [plan for plan in plans if plan.extra.get("placement_probe", {}).get("kind") == "all-depth-chain"]
        self.assertTrue(chain_plans)
        for plan in plans:
            validate_fusion_plan(self.graph, plan, 2, require_executable=True)
        self.assertTrue(any(any(self.graph.nodes[key].extra.get("layer") == 7
                                for group in plan.groups if group.device == 1 for key in group.node_ids)
                            for plan in chain_plans))

    def test_exchange_mirrors_removed_and_output_deterministic(self):
        mirror = copy.deepcopy(self.base)
        for group in mirror.groups:
            group.device = 1
        first, metadata = graph_wide_candidates(self.graph, [self.base, mirror], self.classes)
        second, _ = graph_wide_candidates(self.graph, [self.base, mirror], self.classes)
        exchange = device_exchange_permutations(self.classes)
        identifiers = [plan_id(plan, exchange) for plan in first]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertEqual(metadata["base_candidates"], 1)
        self.assertEqual(identifiers, [plan_id(plan, exchange) for plan in second])

    def test_device_limit_explicit(self):
        with self.assertRaisesRegex(ValueError, "exactly two"):
            graph_wide_candidates(self.graph, [self.base], ("bw",))


if __name__ == "__main__":
    unittest.main()
