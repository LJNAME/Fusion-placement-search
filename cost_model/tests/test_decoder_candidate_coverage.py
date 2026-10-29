import copy
import math
import unittest

from cost_model.decoder_candidate_coverage import audit_candidates, expand_directions, structured_candidates
from cost_model.placement_probes import graph_wide_candidates
from cost_model.search_sensitivity import plan_id
from execution.verify_decoder_stack import placed_stack, small_spec
from simulator import FusionGroup, Plan, make_reference_plan


class DecoderCoverageTests(unittest.TestCase):
    def setUp(self):
        self.placed = placed_stack(small_spec(2))
        self.graph = self.placed.graph
        self.reference = self.placed.plan
        self.hardware = self.placed.hardware

    def test_exchange_coverage_alone_does_not_imply_branch_coverage(self):
        single = audit_candidates(self.graph, [self.reference], self.hardware)
        self.assertEqual(single["missing_exchange_partners"], 1)
        both = audit_candidates(self.graph, expand_directions([self.reference]), self.hardware)
        self.assertEqual(both["missing_exchange_partners"], 0)
        self.assertEqual(both["single_device_directions"], [0, 1])
        self.assertTrue(all(branch["missing_projection_patterns"] for branch in both["branches"]))

    def test_all_depth_variation_can_miss_local_combinations(self):
        base, _ = graph_wide_candidates(self.graph, [self.reference], ("bw", "bw"))
        base = expand_directions(base)
        before = audit_candidates(self.graph, base, self.hardware)
        self.assertTrue(any(b["missing_isolated_branch_masks"] for b in before["branches"]))
        saved = copy.deepcopy(base)
        expanded = structured_candidates(self.graph, base)
        self.assertEqual(base, saved)
        self.assertTrue({plan_id(p) for p in base} <= {plan_id(p) for p in expanded})
        after = audit_candidates(self.graph, expanded, self.hardware)
        self.assertFalse(after["missing_topology_prefix_cuts"])
        self.assertFalse(after["missing_whole_layer_cuts"])
        self.assertEqual(after["missing_exchange_partners"], 0)
        self.assertFalse(after["failures"])
        for branch in after["branches"]:
            self.assertFalse(branch["missing_projection_patterns"])
            self.assertFalse(branch["missing_colocated_chain_patterns"])
            self.assertFalse(branch["missing_isolated_branch_masks"])

    def test_planned_fusion_is_search_legal_but_not_reference_evidence(self):
        graph = self.graph
        groups = copy.deepcopy(make_reference_plan(graph).groups)
        nodes = ["L000.mlp.silu", "L000.mlp.gated_mul"]
        replaced = []
        for group in groups:
            if group.node_ids == [nodes[0]]:
                continue
            if group.node_ids == [nodes[1]]:
                replaced.append(FusionGroup(nodes, 0, {"pattern_id": "silu_mul"}))
            else:
                replaced.append(group)
        result = audit_candidates(graph, [Plan(replaced)], self.hardware)
        self.assertEqual(result["search_legal_candidates"], 1)
        self.assertEqual(result["registry_executable_candidates"], 0)
        self.assertEqual(result["reference_eligible_unique_candidates"], 0)
        self.assertEqual(result["failures"][0]["stage"], "registry_executability")

    def test_invalid_coverage_cannot_pollute_valid_direction_counts(self):
        bad = copy.deepcopy(self.reference)
        bad.groups.pop()
        result = audit_candidates(self.graph, [self.reference, bad], self.hardware)
        self.assertEqual(result["input_candidates"], 2)
        self.assertEqual(result["search_legal_candidates"], 1)
        self.assertEqual(result["reference_eligible_unique_candidates"], 1)
        self.assertEqual(result["missing_exchange_partners"], 1)

    def test_copy_structure_uses_full_tensors_not_declared_slices(self):
        plan = copy.deepcopy(self.reference)
        for group in plan.groups:
            if group.node_ids == ["L000.self_attention.q_proj"]:
                group.device = 1
        for edge in self.graph.edges:
            edge.bytes = 1
        result = audit_candidates(self.graph, [plan], self.hardware)
        row = result["candidates"][0]
        devices = {key: group.device for group in plan.groups for key in group.node_ids}
        crosses = [edge for edge in self.graph.edges if devices[edge.src] != devices[edge.dst]]
        expected = sum(math.prod(self.graph.nodes[e.src].extra["output_shape"]) * 4 for e in crosses)
        self.assertEqual(row["full_tensor_copy_bytes"], expected)
        self.assertEqual(row["host_copy_dispatches"], len(crosses))
        self.assertEqual(row["two_sided_copy_barriers_in_reference_model"], len(crosses))
        self.assertGreater(row["full_tensor_copy_bytes"], len(crosses))


if __name__ == "__main__":
    unittest.main()
