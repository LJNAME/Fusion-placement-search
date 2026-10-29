import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from calibration.e2_records import load_jsonl_records
from cost_model.reference_refit_opportunity import (
    assess_subsets, diagnose_case, exposed_records, subset_design,
)
from cost_model.reference_semantics import HostSubmissionProfile


class SubsetOpportunityTests(unittest.TestCase):
    def test_all_pairs_and_label_independent_prefixes_are_present(self):
        keys = tuple("abcdef")
        states, orders = subset_design(keys)
        self.assertEqual(len([s for s in states if len(s) == 1]), 6)
        self.assertEqual(len([s for s in states if len(s) == 2]), 15)
        self.assertIn(keys, states)
        self.assertEqual(orders["key_descending"], list(reversed(keys)))

    def test_pair_gain_can_disappear_at_full_reveal(self):
        def evaluate(state):
            good = set(state) == {"x", "y"}
            return {"selected_plan_id": "fast" if good else "slow",
                    "gpu_relative_regret": 0. if good else .1}
        result = assess_subsets(evaluate, ("x", "y", "z"), {"x": 1., "y": 2., "z": None})
        self.assertEqual(result["D1"]["improving_states"], 0)
        self.assertEqual(result["D2"]["pair_only_improvements"], 1)
        self.assertEqual(result["D2"]["full_reveal"]["gpu_relative_regret"], .1)
        self.assertEqual(len(result["D2"]["nonmonotone_edges"]), 1)
        self.assertEqual(result["decision_by_epsilon"]["0.01"], "A_enumerated_update_reaches_epsilon")
        self.assertIsNone(result["D2"]["full_reveal"]["recorded_replica_profiling_wall_sum_s"])
        self.assertEqual(result["D2"]["best_enumerated_subset_diagnostic_only"]["recorded_replica_profiling_wall_sum_s"], 3.)

    def test_initial_near_optimal_remains_zero_measurement_control(self):
        result = assess_subsets(lambda state: {"selected_plan_id": "a", "gpu_relative_regret": .002},
                                ("x",), {"x": None})
        self.assertEqual(result["decision_by_epsilon"]["0.01"], "C_initial_epsilon_near_optimal")
        self.assertEqual(result["decision_by_epsilon"]["0.001"], "B_no_epsilon_recovery_found_in_enumerated_subsets")

    def test_pair_scan_does_not_claim_exhaustive_higher_order_coverage(self):
        result = assess_subsets(lambda state: {"selected_plan_id": "a", "gpu_relative_regret": .1},
                                tuple("abcdef"), {})
        self.assertFalse(result["coverage"]["exhaustive"])
        self.assertLess(result["coverage"]["evaluated_subsets"], 64)


class RealRefitInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[2]
        read = lambda p: json.loads(p.read_text(encoding="utf-8"))
        cls.freeze = read(root / "data/reference_validation_v1/freeze.json")
        results = root / "data/received_reference_validation_v1/reference_validation_v1_results"
        cls.kernel = read(results / "zero_shot_report.json")
        cls.queries, cls.costs = exposed_records(cls.freeze, cls.kernel)
        cls.train = load_jsonl_records(root / "data/bw1000_e2/profiling_dataset.jsonl")
        cls.contract = read(root / "data/bw1000_e2/split_contract.json")
        cls.workload = copy.deepcopy(cls.freeze["workloads"][0])
        cls.workload["execution_plan_ids"] = cls.workload["execution_plan_ids"][:2]
        cls.measured = {key: .001 + index * .001 for index, key in enumerate(cls.workload["execution_plan_ids"])}
        cls.host = HostSubmissionProfile(**cls.freeze["host_profile"])

    def run_small_case(self, measured=None, queries=None):
        def small_design(keys):
            return [(), (keys[0],), tuple(keys)], {"test_only": list(keys)}
        with patch("cost_model.reference_refit_opportunity.subset_design", side_effect=small_design):
            return diagnose_case(self.freeze, self.workload, self.train,
                self.queries if queries is None else queries, self.costs, self.contract,
                self.measured if measured is None else measured, self.host)

    def test_refits_reproduce_frozen_initial_predictor_and_full_model_endpoint(self):
        result = self.run_small_case()
        self.assertGreater(result["hidden_keys"], 1)
        self.assertAlmostEqual(result["D2"]["full_reveal"]["model_relative_regret"], 0.)
        self.assertIsNone(result["D0"]["unrestricted_research_wall_time_s"])
        self.assertEqual(result["selection_scope"], "exact_reselection_over_fixed_measured_physical_pool_not_global_MCTS")

    def test_gpu_labels_change_evaluation_but_never_selection(self):
        first = self.run_small_case()
        changed = self.run_small_case(measured={key: .003 - value for key, value in self.measured.items()})
        self.assertEqual([row["selected_plan_id"] for row in first["states"]],
                         [row["selected_plan_id"] for row in changed["states"]])
        self.assertNotEqual(first["D0"]["gpu_relative_regret"], changed["D0"]["gpu_relative_regret"])

    def test_hidden_kernel_labels_do_not_change_D0(self):
        first = self.run_small_case()
        queries = copy.deepcopy(self.queries)
        for row in queries:
            row["measurement"]["duration_s"] *= 50
            row["target"]["log_residual"] += 4
        changed = self.run_small_case(queries=queries)
        self.assertEqual(first["D0"]["selected_plan_id"], changed["D0"]["selected_plan_id"])
        self.assertEqual(first["D0"]["predicted_selected_s"], changed["D0"]["predicted_selected_s"])

    def test_missing_raw_costs_stay_unknown_and_mismatched_freeze_is_rejected(self):
        self.assertTrue(all(value is None for value in self.costs.values()))
        broken = {**self.kernel, "freeze_id": "wrong-freeze"}
        with self.assertRaisesRegex(ValueError, "another freeze"):
            exposed_records(self.freeze, broken)


if __name__ == "__main__":
    unittest.main()
