"""Finite capacity penalties, strict admission, and honest no-plan outcomes."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cost_model.estimate import CostEstimate
from cost_model.hardware_profile import AnalyticHardwareProfile
from cost_model.legal_search import _LegalMCTSNode
from cost_model.search_evaluation import (
    INFEASIBLE_PENALTY_S, _CachedScorer, _modes, _run_search, evaluate_searches, save_evaluation,
)
from simulator import OpGraph, OpNode, make_reference_plan
from simulator.hardware import load_hardware
from tools.verify_search_replay import verify

ROOT = Path(__file__).resolve().parents[2]


def profile():
    return AnalyticHardwareProfile(load_hardware(ROOT / "data/bw1000_2card/bw1000_2card.calibrated.json"))


def graph():
    return OpGraph(nodes={"a": OpNode("a", out_bytes=10.), "b": OpNode("b", out_bytes=10.)})


def estimate(seconds, peaks):
    return CostEstimate(seconds, tuple(peaks), (), 0., seconds, (seconds, 0.), 0., 1., False)


def search(method, g, scorer, mode="joint", iterations=16):
    m = next(m for m in _modes(2) if m.name == mode)
    return _run_search(method, g, scorer, m, seed=0, beam=2, rollouts=3,
                       mcts_iterations=iterations, top_k=1)


class TestSearchCapacity(unittest.TestCase):
    def test_explicit_capacity_and_raw_cache_are_independent_of_profile(self):
        p = profile(); g = graph(); plan = make_reference_plan(g, 0)
        scorer = _CachedScorer(g, p, capacity=(8., 8.))
        with patch.object(scorer.model, "predict", return_value=estimate(.01, (10., 0.))) as predict:
            # The profile reports ample memory and no overflow reason. The
            # explicit capacity, rather than that flag, is authoritative.
            with patch.object(p, "max_memory", side_effect=AssertionError("implicit capacity")):
                self.assertEqual(scorer(plan), INFEASIBLE_PENALTY_S + .01)
                scorer.bind_capacity = False
                self.assertEqual(scorer(plan), .01)
                scorer.bind_capacity = True
                self.assertEqual(scorer(plan), INFEASIBLE_PENALTY_S + .01)
            self.assertEqual(predict.call_count, 1)

    def test_feasible_rollout_wins_even_when_finite_penalty_is_not_a_global_bound(self):
        g = graph(); scorer = _CachedScorer(g, profile(), capacity=(8., 8.))
        invalid = make_reference_plan(g, 0)
        valid = copy.deepcopy(invalid); valid.groups[-1].device = 1
        def predict(_, plan, __):
            return estimate(2e6, (5., 5.)) if len({x.device for x in plan.groups}) == 2 else estimate(.01, (10., 0.))
        def engine(**kwargs):
            kwargs["scorer"](invalid); kwargs["scorer"](valid)
            return [invalid]
        with patch.object(scorer.model, "predict", side_effect=predict), \
             patch("cost_model.search_evaluation.legal_mcts_search", side_effect=engine):
            plans, _, budget = search("mcts", g, scorer)
        self.assertEqual([x.device for x in plans[0].groups], [0, 1])
        self.assertIsNone(budget["reason"])

    def test_mcts_miss_is_not_an_exact_infeasibility_proof(self):
        g = graph(); p = profile()
        reasons = {}
        for method in ("mcts", "exact", "beam"):
            scorer = _CachedScorer(g, p, capacity=(1., 1.))
            plans, _, budget = search(method, g, scorer)
            self.assertEqual(plans, [])
            reasons[method] = budget["reason"]
        self.assertEqual(reasons, dict(mcts="evaluated_no_feasible", beam="evaluated_no_feasible",
                                      exact="no_feasible_in_space"))

    def test_all_infeasible_rollouts_keep_rewards_and_ucb_values_finite(self):
        g = graph(); scorer = _CachedScorer(g, profile(), capacity=(1., 1.))
        original = _LegalMCTSNode.update
        updates = []
        def update(node, reward):
            self.assertTrue(math.isfinite(reward))
            original(node, reward)
            self.assertTrue(math.isfinite(node.q))
            updates.append(reward)
        with patch.object(_LegalMCTSNode, "update", update):
            plans, _, _ = search("mcts", g, scorer, iterations=64)
        self.assertFalse(plans)
        self.assertGreaterEqual(len(updates), 64)

    def test_sequential_partition_proposal_is_not_final_capacity_admission(self):
        g = graph(); scorer = _CachedScorer(g, profile(), capacity=(15., 15.))
        plans, _, budget = search("exact", g, scorer, mode="sequential")
        self.assertTrue(plans)
        self.assertEqual([s["capacity_bound"] for s in budget["stages"]], [False, True])
        self.assertEqual(len({x.device for x in plans[0].groups}), 2)
        self.assertEqual(budget["feasibility_scope"], "frozen_partition")

    def test_uncached_scorer_path_still_filters_by_explicit_capacity(self):
        g = graph(); scorer = _CachedScorer(g, profile(), capacity=(15., 15.))
        scorer.cache_enabled = False
        plans, _, budget = search("exact", g, scorer)
        self.assertTrue(plans)
        self.assertEqual(len({x.device for x in plans[0].groups}), 2)
        self.assertEqual(budget["cost_model_evaluations"], 4)
        self.assertIsNone(budget["reason"])

    def test_no_plan_report_exports_no_artifact_and_replays(self):
        p = profile()
        for device in p.hardware.devices:
            device.max_mem = 1.
        with tempfile.TemporaryDirectory() as directory:
            a = evaluate_searches(p, ["small"], seed=0, mcts_iterations=2, beam=2)
            b = evaluate_searches(p, ["small"], seed=0, mcts_iterations=2, beam=2)
            first, second = Path(directory) / "a/search.json", Path(directory) / "b/search.json"
            save_evaluation(first, a); save_evaluation(second, b)
            self.assertTrue(verify(first, second)["pass"])
            for row in json.loads(first.read_text())["scales"][0]["searches"]:
                self.assertIsNone(row["plan"])
                self.assertIsNone(row["best_placed_op_graph"])
                self.assertEqual(row["candidates"], [])
                self.assertEqual(row["reason"], "no_feasible_in_space" if row["method"] == "exact"
                                 else "evaluated_no_feasible")
            self.assertFalse(list(Path(directory).rglob("*.artifacts")))


if __name__ == "__main__":
    unittest.main()
