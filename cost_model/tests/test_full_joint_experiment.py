"""Full-graph budget plumbing and tied-weight reference placements."""
from pathlib import Path
import unittest
from unittest.mock import patch

from cost_model.fusion_library import load_library
from cost_model.hardware_profile import AnalyticHardwareProfile
from cost_model.htae import HTAECostModel
from cost_model.search_evaluation import qwen3_spec
from simulator import make_reference_plan
from simulator.hardware import load_hardware
from simulator.model_graphs import build_model_graph
from tools.evaluate_full_joint import METHODS, assert_shared_colocated, reference_plans, run_one

ROOT = Path(__file__).resolve().parents[2]


class FullJointExperiment(unittest.TestCase):
    def setUp(self):
        self.graph = build_model_graph(qwen3_spec(2))
        self.registry = load_library().registry("full")
        self.profile = AnalyticHardwareProfile(load_hardware(ROOT / "data/bw1000_2card/bw1000_2card.calibrated.json"))
        self.capacity = [d.max_mem for d in self.profile.hardware.devices]

    def test_all_eight_references_keep_shared_head_on_embedding_owner(self):
        refs = reference_plans(self.graph, self.registry)
        self.assertEqual(len(refs), 8)
        model = HTAECostModel(registry=self.registry)
        for name, plan in refs.items():
            assert_shared_colocated(self.graph, plan)
            dev = {n: g.device for g in plan.groups for n in g.node_ids}
            self.assertEqual(dev["embedding"], dev["lm_head"])
            if name.startswith("half_layers_owner0"):
                self.assertEqual(dev["L001.mlp.up_proj"], 1)
                self.assertEqual(dev["lm_head"], 0)
            self.assertGreater(model.predict(self.graph, plan, self.profile).makespan, 0.)

    def test_remote_shared_weight_without_replica_is_rejected(self):
        plan = make_reference_plan(self.graph, 0)
        next(g for g in plan.groups if "lm_head" in g.node_ids).device = 1
        with self.assertRaisesRegex(AssertionError, "shared parameter split"):
            assert_shared_colocated(self.graph, plan)

    def test_four_methods_use_real_budget_without_exact_count(self):
        with patch("cost_model.exact_search.count_exact_search_space", side_effect=AssertionError("no Exact on full graphs")):
            for method in METHODS:
                with self.subTest(method=method):
                    row, plan = run_one(self.graph, self.profile, self.registry, self.capacity, method, 0, 4, "unit")
                    self.assertEqual(row["budget_used"], 4)
                    self.assertEqual(len({x["candidate_id"] for x in row["trace"]}), 4)
                    self.assertEqual([r["phase"] for r in row["trace"][:2]], ["initialization"] * 2)
                    self.assertIsNotNone(plan)
                    assert_shared_colocated(self.graph, plan)


if __name__ == "__main__":
    unittest.main()
