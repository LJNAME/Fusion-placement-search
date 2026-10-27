import unittest

from cost_model.audit_e2_first_step import POLICIES, audit_first_step


class FirstStepAuditTests(unittest.TestCase):
    def make_report(self, gain, stop):
        policies = {name: {"selected_gain_ms": 0.0, "selected_regret_ms": 2.0,
                           "selected_object_id": "key", "selection_score": 1.0,
                           "stop_recommended": False} for name in POLICIES}
        policies["srop_proxy"].update(selected_gain_ms=gain, selected_regret_ms=2.0-gain,
                                      selection_score=0.0 if stop else 1.0, stop_recommended=stop)
        return {"experiment_id": "test", "tasks": [{"task_id": "a", "zero": {
            "split_id": "split", "model_id": "model", "scale": "layer", "initial_train_count": 16,
            "subset_seed": 0, "top1_regret_ms": 2.0}, "one_step": {"policies": policies}}]}

    def test_forced_zero_score_gain_does_not_count_as_policy_improvement(self):
        result = audit_first_step(self.make_report(1.0, True))
        self.assertEqual(result["policy_outcomes"]["forced_measurement"]["srop_proxy"]["improved_configs"], 1)
        self.assertEqual(result["policy_outcomes"]["with_stopping"]["srop_proxy"]["improved_configs"], 0)
        self.assertEqual(result["tasks"][0]["policies"]["srop_proxy"]["regret_with_stopping_ms"], 2.0)
        self.assertEqual(result["nonpositive_score_forced_improvements"]["srop_proxy"], 1)

    def test_harmful_measurement_is_retained(self):
        result = audit_first_step(self.make_report(-0.5, False))
        self.assertEqual(result["policy_outcomes"]["with_stopping"]["srop_proxy"]["harmed_configs"], 1)
        self.assertEqual(result["pairwise_comparisons"]["with_stopping"]["srop_proxy_vs_impact_only"]["worse_configs"], 1)


if __name__ == "__main__":
    unittest.main()
