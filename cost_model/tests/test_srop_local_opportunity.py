"""Selection/evaluation separation and exhaustive reveal effects."""
import copy
import unittest
from unittest.mock import patch

from cost_model.srop_local_opportunity import (
    KEYS, analytic_prior, context_diagnostic, mask_audit, run_case, select, visible_keys,
)


class TestLocalOpportunity(unittest.TestCase):
    def test_all_eight_masks_and_twelve_reveals(self):
        states, actions = mask_audit(dict(silu=5., mul=5., fused=12.),
            dict(silu=5., mul=5., fused=6.), dict(fused=6., unfused=10.))
        self.assertEqual(len(states), 8)
        self.assertEqual(len(actions), 12)
        self.assertEqual(visible_keys(3), ("silu", "mul"))
        self.assertEqual(visible_keys(4), ("fused",))
        # One useful action among three is a nontrivial selection problem.
        self.assertEqual(states[0]["available_actions"], 3)
        self.assertEqual(states[0]["beneficial_actions"], 1)
        self.assertEqual({a["key"] for a in actions if a["beneficial"]}, {"fused"})

    def test_zero_initial_regret_does_not_exclude_later_harmful_reveals(self):
        states, actions = mask_audit(dict(silu=10., mul=10., fused=9.),
            dict(silu=2., mul=2., fused=3.), dict(fused=3., unfused=5.))
        self.assertEqual(states[0]["reference_regret"], 0.)
        self.assertGreater(states[3]["reference_regret"], 0.)
        self.assertTrue(any(a["harmful"] for a in actions))
        repair = next(a for a in actions if a["visible_mask"] == 3 and a["key"] == "fused")
        self.assertEqual(repair["reference_gain_us"], 2.)

    def test_reference_does_not_enter_selector_and_gap_is_not_regret(self):
        prior, labels = dict(silu=5., mul=5., fused=11.), dict(silu=5., mul=5., fused=6.)
        a, _ = mask_audit(prior, labels, dict(fused=6., unfused=10.))
        b, _ = mask_audit(prior, labels, dict(fused=9., unfused=8.))
        self.assertEqual([r["choice"] for r in a], [r["choice"] for r in b])
        self.assertEqual(a[0]["predicted_unfused_minus_fused_us"], -1.)
        self.assertAlmostEqual(a[0]["reference_regret"], 4/6)
        with self.assertRaisesRegex(ValueError, "evaluator-only"):
            select(prior, {"chain": 1.})

    def test_context_branch_is_posthoc_and_cannot_mutate_main_results(self):
        prior, labels = dict(silu=7., mul=5., fused=11.), dict(silu=7., mul=5., fused=8.)
        states, _ = mask_audit(prior, labels, dict(fused=8., unfused=10.))
        before = copy.deepcopy(states)
        diagnostic = context_diagnostic(states, labels, dict(fused=8., unfused=10.))
        self.assertEqual(states, before)
        self.assertTrue(diagnostic["hidden_reference_used"])
        self.assertTrue(diagnostic["rows"][0]["choice_changed"])

    def test_real_transfer_prior_matches_historical_prefill_formula(self):
        prior, details = analytic_prior(0, 256)
        self.assertAlmostEqual(prior["fused"], 11.28762880762224, places=10)
        self.assertAlmostEqual(prior["silu"]+prior["mul"], 21.40284976849216, places=10)
        hypothesis, _ = analytic_prior(0, 256, launch_us=3.2)
        self.assertAlmostEqual(prior["fused"]-hypothesis["fused"], details["fused"]["launch_us"]-3.2)

    def test_B_stays_pending_and_round_replicates_share_fixed_evaluator(self):
        report = {"case_id": "fixture", "measurement_contract_id": "test", "rows": [
            {"round": i, "method": k, "graph_calls": 50,
             "graph_event_total_s": v*50/1e6}
            for i in range(8) for k,v in dict(silu=5., mul=5., fused=3.+i, chain=20.+i).items()]}
        fixed_prior = (dict(silu=6., mul=6., fused=15.), {})
        with patch("cost_model.srop_local_opportunity.analytic_prior", return_value=fixed_prior):
            result = run_case(report, 0, 256)
        self.assertEqual(result["B_contract_consistent"]["status"], "pending_independent_calibration")
        self.assertEqual(len(result["B_contract_consistent"]["states"]), 8)
        self.assertTrue(all(s["choice"] is None for s in result["B_contract_consistent"]["states"]))
        for scenario in result["scenarios"].values():
            self.assertFalse(scenario["counts_as_same_contract_SROP_evidence"])
            costs = [r["states"][0]["reference_cost_us"] for r in scenario["round_audits"]]
            self.assertEqual(len(set(costs)), 1)

    def test_ties_are_deterministic_and_invalid_costs_fail(self):
        self.assertEqual(select(dict(silu=1., mul=2., fused=3.), {})[0], "fused")
        for x in (0., float("nan"), -1.):
            with self.assertRaises(ValueError):
                select(dict(silu=x, mul=2., fused=3.), {})


if __name__ == "__main__":
    unittest.main()
