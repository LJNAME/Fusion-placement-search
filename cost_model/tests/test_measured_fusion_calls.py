"""Matching complete recipes, multi-branch semantics, and charged search traces."""
import copy
import unittest

from cost_model.fusion_library import load_library
from cost_model.hardware_profile import AnalyticHardwareProfile
from cost_model.measured_fusion_calls import MeasuredFusionCalls
from cost_model.search_evaluation import build_evaluation_graph
from cost_model.legal_search import LegalFusionSearchSpace, rollout_legal_plan
from cost_model.structure import analyze_plan_structure
from simulator import make_reference_plan
from simulator.dataset import plan_to_dict, plan_from_dict
from simulator.fusion_rules import enumerate_fusion_candidates, FusionRuleError
from simulator.hardware import load_hardware
from simulator.validate import validate_fusion_plan
from tools.evaluate_stage_search import HARDWARE
from tools.evaluate_full_joint import FullLedger, run_one


class MeasuredCalls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = load_library()
        cls.profile = MeasuredFusionCalls(AnalyticHardwareProfile(load_hardware(HARDWARE)), cls.library)

    def candidate(self, graph, pid):
        return next(c for c in enumerate_fusion_candidates(graph, self.library.registry()) if c.pattern_id == pid)

    def test_3b_two_patterns_match_on_both_cards_without_duplicating_measurements(self):
        graph = build_evaluation_graph("model_3b")
        for pid, expected in (("lightop_silu_mul", 57.235140800476074), ("lightop_add_rmsnorm", 13.612760305404663)):
            candidate = self.candidate(graph, pid)
            for device in (0, 1):
                group = candidate.to_group(device)
                binding = self.profile.match(graph, group)
                self.assertAlmostEqual(binding["median_us"], expected)
                self.assertEqual(binding["measured_on"], 0)
                self.assertEqual(binding["target_independently_measured"], device == 0)
                self.assertAlmostEqual(self.profile.estimate_group(graph, group).duration, expected * 1e-6)

    def test_other_shapes_and_known_layout_or_eps_conflicts_do_not_hit(self):
        for scale in ("model", "model_3b"):
            for field, value in (("layout", "strided"), ("strides", [1, 1, 1]), ("eps", 1e-5)):
                graph = build_evaluation_graph(scale)
                c = self.candidate(graph, "lightop_add_rmsnorm")
                graph.nodes[c.node_ids[0]].extra[field] = value
                self.assertIsNone(self.profile.match(graph, c.to_group(0)))
        graph = build_evaluation_graph("model")
        self.assertIsNone(self.profile.match(graph, self.candidate(graph, "lightop_silu_mul").to_group(0)))

    def test_hardware_mismatch_disables_measurement_reuse(self):
        profile = copy.deepcopy(self.profile)
        profile.device_match[1] = False
        graph = build_evaluation_graph("model_3b")
        self.assertIsNone(profile.match(graph, self.candidate(graph, "lightop_add_rmsnorm").to_group(1)))

    def test_cli_relative_measurement_directory_produces_an_audit(self):
        profile = MeasuredFusionCalls(self.profile.base, self.library, "data/fusion_timing_expansion_20261109_01")
        audit = profile.audit_graph(build_evaluation_graph("layer"), self.library.registry())
        self.assertEqual(audit["joint_qk_structural_groups"], 1)
        self.assertEqual(len(audit["measurement_files"]), 38)  # process.log files are not JSON

    def test_complete_call_annotation_survives_export_and_adds_pack_workspace(self):
        graph = build_evaluation_graph("model_3b")
        # Use the search contraction, not an invalid manually reordered cover.
        import random
        space = LegalFusionSearchSpace(graph, 2, self.library.registry(), allowed_devices=(0,))
        plan = rollout_legal_plan(space, space.initial_state(), random.Random(0), fusion_probability=1.)
        self.profile.annotate_plan(graph, plan)
        restored = plan_from_dict(plan_to_dict(plan))
        validate_fusion_plan(graph, restored, 2, registry=self.library.registry())
        groups = [g for g in restored.groups if g.extra.get("pattern_id") == "lightop_silu_mul"]
        self.assertEqual(len(groups), 36)
        self.assertEqual(groups[0].extra["workspace_bytes"], 1 * 256 * 11008 * 4)
        self.assertEqual(groups[0].extra["measured_complete_call"]["recipe_id"], "split_inputs_cat_out_then_lightop.v1")
        structures = analyze_plan_structure(graph, restored, 2, registry=self.library.registry()).groups
        for group in structures:
            if group.pattern_id == "lightop_silu_mul":
                self.assertAlmostEqual(self.profile.estimate_group(graph, group).duration, 57.235140800476074e-6)

    def test_qk_registered_but_missing_or_wrong_graph_semantics_blocks_it(self):
        graph = build_evaluation_graph("model")
        pattern = self.library.registry().get("lightop_qk_rmsnorm_rope")
        groups = list(pattern.candidate_node_ids(graph))
        self.assertEqual(len(groups), 28)
        self.assertFalse(any(pattern.matches(graph, ids) for ids in groups))
        self.declare_qk(graph)
        graph.extra["fusion_contracts"][pattern.pattern_id]["is_neox"] = False
        self.assertFalse(pattern.matches(graph, groups[0]))

    @staticmethod
    def declare_qk(graph):
        graph.extra["fusion_contracts"] = {"lightop_qk_rmsnorm_rope": {
            "theta": 1e6, "rotary_dim": 128, "is_neox": True, "eps": 1e-6,
            "residual_q": None, "residual_k": None, "source": "explicit CPU test declaration; not production evidence"}}

    def test_joint_qk_is_one_four_node_candidate_and_one_measured_call(self):
        graph = build_evaluation_graph("layer")
        self.declare_qk(graph)
        candidate = self.candidate(graph, "lightop_qk_rmsnorm_rope")
        self.assertEqual(len(candidate.node_ids), 4)
        self.assertAlmostEqual(self.profile.match(graph, candidate.to_group(0))["median_us"], 15.76956033706665)
        import random
        space = LegalFusionSearchSpace(graph, 2, self.library.registry(), allowed_devices=(0,))
        plan = rollout_legal_plan(space, space.initial_state(), random.Random(0), fusion_probability=1.)
        validate_fusion_plan(graph, plan, 2, registry=self.library.registry())
        self.assertEqual(sum(g.extra["pattern_id"] == "lightop_qk_rmsnorm_rope" for g in plan.groups), 1)
        graph.nodes[candidate.node_ids[0]].extra["output_shape"] = (1, 128, 16, 128)
        self.assertFalse(self.library.registry().get(candidate.pattern_id).matches(graph, candidate.node_ids))

    def test_mcts_tree_diagnostics_do_not_create_extra_scoring_calls(self):
        graph = build_evaluation_graph("layer")
        capacity = [d.max_mem for d in self.profile.hardware.devices]
        row, plan = run_one(graph, self.profile, self.library.registry(), capacity, "joint_hybrid", 0, 8, "unit")
        self.assertEqual(row["budget_used"], 8)
        self.assertEqual(len(row["trace"]), 8)
        observed = [r for r in row["trace"] if "tree_choices" in r]
        self.assertTrue(observed)
        self.assertTrue(all(r["tree_assigned_nodes"] + r["rollout_completed_nodes"] == len(graph.nodes) for r in observed))
        self.assertLess(max(r["tree_assigned_nodes"] for r in observed), len(graph.nodes))


if __name__ == "__main__":
    unittest.main()
