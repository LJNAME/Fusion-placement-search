"""Budget censoring, cold random prefixes and conservative measured witnesses."""
import copy
from pathlib import Path
import unittest

from cost_model.fusion_library import load_library
from cost_model.hardware_profile import AnalyticHardwareProfile
from cost_model.legal_search import LegalFusionSearchSpace
from cost_model.exact_search import count_exact_search_space
from cost_model.search_evaluation import build_evaluation_graph
from simulator.hardware import load_hardware
from tools.audit_coupling_evidence import contract_without_uuid, placement_candidate
from tools.evaluate_hybrid_budget import assess, summarize, trial
from tools.evaluate_mcts_medium import extract_graph

ROOT = Path(__file__).resolve().parents[2]


class BudgetQuality(unittest.TestCase):
    def test_first_hit_and_censor_are_distinct(self):
        trace = [{"evaluation": 1, "best_ms": None}, {"evaluation": 2, "best_ms": 12.},
                 {"evaluation": 3, "best_ms": 10.05}, {"evaluation": 4, "best_ms": 10.}]
        early = assess(trace[:3], 10.)
        full = assess(trace, 10.)
        self.assertEqual(early["first_1pct_evaluation"], 3)
        self.assertIsNone(early["first_exact_evaluation"])
        self.assertEqual(full["first_exact_evaluation"], 4)

    def test_failed_runs_remain_in_success_denominator(self):
        rows = [{"best": None, "exact_gap_percent": None, "first_exact_evaluation": None,
                 "first_1pct_evaluation": None, "budget_used": 8},
                {"best": {"makespan_ms": 10.}, "exact_gap_percent": 0.,
                 "first_exact_evaluation": 3, "first_1pct_evaluation": 3, "budget_used": 8}]
        s = summarize(rows)
        self.assertEqual((s["runs"], s["feasible_runs"], s["within_1pct_hits"]), (2, 1, 1))
        self.assertEqual(s["first_1pct_evaluations"], [None, 3])

    def test_all_method_prefixes_are_budget_independent_and_count_initialization(self):
        graph = extract_graph(build_evaluation_graph("model_3b"), "mlp")
        registry = load_library().registry("full")
        profile = AnalyticHardwareProfile(load_hardware(ROOT / "data/bw1000_2card/bw1000_2card.calibrated.json"))
        capacity = [d.max_mem for d in profile.hardware.devices]
        for method in ("joint_node", "joint_hybrid", "random_node", "random_hybrid"):
            policy = "hybrid" if method.endswith("hybrid") else "node"
            size = count_exact_search_space(LegalFusionSearchSpace(graph, 2, registry, placement_policy=policy))
            a = trial(graph, profile, registry, capacity, method, 8, 0, size)
            b = trial(graph, profile, registry, capacity, method, 16, 0, size)
            self.assertEqual(a["trace"], b["trace"][:8])
            self.assertEqual((a["unique_candidates"], b["unique_candidates"]), (8, 16))
            self.assertEqual([r["phase"] for r in a["trace"][:2]], ["initialization"] * 2)


class MeasuredEvidence(unittest.TestCase):
    def test_point_estimate_flip_with_overlap_is_not_confirmed(self):
        cells = {"0": {"native": {"median_us": 10., "min_us": 9., "max_us": 12.},
                       "fused": {"median_us": 7.25, "min_us": 7.2, "max_us": 7.3}},
                 "1": {"native": {"median_us": 11., "min_us": 10., "max_us": 13.},
                       "fused": {"median_us": 7.23, "min_us": 7.2, "max_us": 8.}}}
        row = placement_candidate(cells)
        self.assertTrue(row["point_estimate_placement_flip"])
        self.assertFalse(any(row["observed_ranges_separated"].values()))
        self.assertFalse(row["confirmed_coupling"])

    def test_contract_comparison_removes_only_uuid_without_mutating_receipt(self):
        a = {"environment": {"device_uuid": "old", "architecture": "gfx936"}, "rounds": 7}
        original = copy.deepcopy(a)
        b = copy.deepcopy(a)
        b["environment"]["device_uuid"] = "new"
        self.assertEqual(contract_without_uuid(a), contract_without_uuid(b))
        b["environment"]["architecture"] = "other"
        self.assertNotEqual(contract_without_uuid(a), contract_without_uuid(b))
        self.assertEqual(a, original)


if __name__ == "__main__":
    unittest.main()
