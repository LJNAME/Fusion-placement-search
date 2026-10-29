import unittest

from cost_model.reference_profiling_opportunity import assess_replacements, decision_metrics


class ProfilingOpportunityTests(unittest.TestCase):
    def test_predicted_tie_keeps_fixed_physical_choice(self):
        result = decision_metrics({"a": .001, "b": .001 + 1e-13}, {"a": .002, "b": .001})
        self.assertEqual(result["selected_id"], "a")
        self.assertAlmostEqual(result["relative_regret"], 1.)
        self.assertEqual(result["tie_regret_range"], [0., 1.])

    def test_full_reveal_zero_improvement_does_not_rule_out_partial_gain(self):
        score = lambda d: {"a": .002, "b": (.004 - d["x"] + d["y"])}
        result = assess_replacements(score, {"x": .001, "y": .001},
                                     {"x": .004, "y": .004}, {"a": .003, "b": .001})
        self.assertEqual(result["full_reveal_realized_saving_ms"], 0.)
        self.assertEqual(result["single_key_positive_savings"], 1)
        self.assertAlmostEqual(result["max_single_key_saving_ms"], 2.)

    def test_exact_kernel_correction_can_harm_plan_choice(self):
        result = assess_replacements(lambda d: {"a": .002, "b": d["x"]},
            {"x": .003}, {"x": .001}, {"a": .001, "b": .002})
        self.assertEqual(result["single_key_harmful_changes"], 1)
        self.assertAlmostEqual(result["full_reveal_realized_saving_ms"], -1.)

    def test_small_actual_gap_is_epsilon_success_with_nonzero_regret(self):
        result = decision_metrics({"a": .001, "b": .002}, {"a": .001005, "b": .001})
        self.assertTrue(result["within_epsilon"])
        self.assertGreater(result["relative_regret"], 0.)
        self.assertEqual(result["epsilon_excess_ms"], 0.)


if __name__ == "__main__":
    unittest.main()
