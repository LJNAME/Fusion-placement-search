import unittest

from cost_model.host_decision_stability import aggregate_host, decision_summary, scenarios
from cost_model.reference_semantics import HostSubmissionProfile
from cost_model.search_sensitivity import permute_plan_devices, plan_id
from execution.verify_reference import _placed


class HostDecisionTests(unittest.TestCase):
    def test_tiny_rank_flip_can_preserve_epsilon_decision_without_a_stop_certificate(self):
        scores = {"nominal": {"a0": 1., "a1": 1., "b0": 1.005, "b1": 1.005},
            "round_0": {"a0": 1.005, "a1": 1.005, "b0": 1., "b1": 1.},
            "zero": {"a0": 10., "a1": 10., "b0": 1., "b1": 1.}}
        classes = {"a0": "a", "a1": "a", "b0": "b", "b1": "b"}
        result = decision_summary(scores, classes, {"nominal": "sensitivity", "round_0": "sensitivity", "zero": "ablation"})
        self.assertEqual(result["nominal_physical_top1_set"], ["a0", "a1"])
        self.assertEqual(result["class_top1_change_scenarios"], 1)
        self.assertEqual(result["max_strict_class_pairwise_flips"], 1)
        self.assertEqual(result["ablation_scenarios_excluded"], 1)
        self.assertAlmostEqual(result["nominal_choice_worst_predicted_excess_pct"], .5)
        self.assertEqual(result["epsilon_sets"][0]["class_ids_within_epsilon_in_every_scenario"], [])
        self.assertEqual(result["epsilon_sets"][-1]["class_ids_within_epsilon_in_every_scenario"], ["a", "b"])

    def test_exchange_labels_are_equivalent_in_class_but_not_physical_identity(self):
        plan = _placed().plan
        mirrored = permute_plan_devices(plan, (1, 0))
        self.assertNotEqual(plan_id(plan), plan_id(mirrored))
        self.assertEqual(plan_id(plan, ((0, 1), (1, 0))), plan_id(mirrored, ((0, 1), (1, 0))))

    def test_large_possible_loss_cannot_be_hidden_by_an_epsilon_decision(self):
        result = decision_summary({"nominal": {"a": 1., "b": 1.1}, "other": {"a": 1.1, "b": 1.}},
            {"a": "a", "b": "b"}, {"nominal": "sensitivity", "other": "sensitivity"})
        self.assertEqual(result["epsilon_sets"][-1]["class_ids_within_epsilon_in_every_scenario"], [])
        self.assertAlmostEqual(result["finite_minimax_predicted_excess_pct"], 10.)

    def test_host_scenarios_remain_uncalibrated_and_zero_is_outside_sensitivity(self):
        nominal = HostSubmissionProfile(start_s=2e-6, finish_s=3e-6, copy_submit_s=8e-6,
            operator_submit_s={"local:input": 1e-6})
        other = HostSubmissionProfile(start_s=4e-6, finish_s=5e-6, copy_submit_s=12e-6,
            operator_submit_s={"local:input": 2e-6})
        rows, bounds = scenarios(nominal, [other, nominal, other])
        self.assertTrue(all(not x["host"].calibrated for x in rows))
        self.assertEqual(bounds["copy_submit_s"], [8e-6, 12e-6])
        self.assertEqual([x["name"] for x in rows if x["role"] == "ablation"], ["zero_host_ablation"])

    def test_aggregation_retains_bad_rounds_and_excludes_busy_gpu_wait(self):
        rows = []
        for repeat, value in enumerate((1e-5, 2e-5, 3e-5)):
            rows.append({"case_id": "fixture", "repeat": repeat, "catalog": {
                "node": {"kind": "node", "operator": "matmul", "local_name": "projection"}, "copy": {"kind": "copy"}},
                "probe_samples": [{"start_s": 1e-6, "idle_sync_s": 2e-6,
                    "busy_final_sync_s_not_a_host_parameter": 1., "operations": [
                        {"key": "node", "dispatch_step_s": value}, {"key": "copy", "dispatch_step_s": value * 2}]}]})
        host = aggregate_host(rows)
        self.assertEqual(host.finish_s, 2e-6)
        self.assertEqual(host.node_cost("matmul", "projection"), 2e-5)
        self.assertEqual(host.copy_submit_s, 4e-5)
        self.assertFalse(host.calibrated)


if __name__ == "__main__":
    unittest.main()
