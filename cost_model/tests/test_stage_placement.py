"""Contiguous stage actions preserve fusion and budget semantics."""
import copy
from pathlib import Path
import unittest

from cost_model.exact_search import count_exact_search_space, exact_legal_search
from cost_model.fusion_library import load_library
from cost_model.hardware_profile import AnalyticHardwareProfile
from cost_model.legal_search import LegalChoice, LegalFusionSearchSpace, legal_mcts_search
from cost_model.search_evaluation import _plan_key, _singleton_registry, build_evaluation_graph
from simulator import OpGraph, OpNode, TensorEdge, make_reference_plan
from simulator.hardware import load_hardware
from tools.evaluate_mcts_medium import extract_graph
from tools.evaluate_stage_search import BudgetReached, Ledger, heuristic, oracle, walk

ROOT = Path(__file__).resolve().parents[2]


def chain(n=4):
    graph = OpGraph(name="unit-chain")
    for i in range(n):
        graph.add_node(OpNode(str(i), "add", 8., 16., 0., {"dtype": "bf16", "output_shape": (8,)}))
        if i:
            graph.add_edge(TensorEdge(str(i - 1), str(i), 16.))
    return graph


def contiguous(graph, plan):
    devices = {n: g.device for g in plan.groups for n in g.node_ids}
    stages = []
    for n in graph.topological_order():
        if not stages or devices[n] != stages[-1]:
            stages.append(devices[n])
    return len(stages) == len(set(stages))


class StageActions(unittest.TestCase):
    def test_matches_independent_filter_of_original_space_with_nonadjacent_fusion(self):
        graph = extract_graph(build_evaluation_graph("model_3b"), "mlp")
        registry = load_library().registry("full")
        old = LegalFusionSearchSpace(graph, 2, registry)
        new = LegalFusionSearchSpace(graph, 2, registry, placement_policy="contiguous_stages")
        expected = {_plan_key(p) for p in walk(old) if contiguous(graph, p)}
        plans = list(walk(new))
        self.assertEqual({_plan_key(p) for p in plans}, expected)
        self.assertEqual(len(plans), len(expected))
        self.assertEqual(count_exact_search_space(new), len(plans))
        self.assertTrue(any(len(g.node_ids) > 1 for p in plans for g in p.groups))
        self.assertTrue(any(p.extra["stage_devices"] == [1, 0] for p in plans))

    def test_all_singleton_cuts_both_card_orders_and_no_return(self):
        graph = chain()
        rules = _singleton_registry()
        space = LegalFusionSearchSpace(graph, 2, rules, placement_policy="contiguous_stages")
        self.assertEqual(count_exact_search_space(space), 8)
        self.assertEqual(count_exact_search_space(LegalFusionSearchSpace(graph, 2, rules)), 16)
        state = space.initial_state()
        state = space.apply(state, next(a for a in space.actions(state) if a.device == 0))
        state = space.apply(state, next(a for a in space.actions(state) if a.device == 1))
        legal = space.actions(state)
        self.assertEqual({a.device for a in legal}, {1})
        with self.assertRaisesRegex(ValueError, "completed stage"):
            space.apply(state, LegalChoice(legal[0].candidate, 0))

    def test_shared_endpoints_leave_only_uniform_plans(self):
        graph = chain()
        graph.nodes["0"].param_bytes = 16.
        graph.nodes["3"].extra["shared_param_with"] = "0"
        space = LegalFusionSearchSpace(graph, 2, _singleton_registry(), tie_shared_params=True,
                                       placement_policy="contiguous_stages")
        plans = list(walk(space))
        self.assertEqual(len(plans), 2)
        self.assertTrue(all(len({g.device for g in p.groups}) == 1 for p in plans))

    def test_device_relabel_preserves_complete_stage_set(self):
        graph = chain()
        rules = _singleton_registry()
        a = LegalFusionSearchSpace(graph, 2, rules, placement_policy="contiguous_stages")
        b = LegalFusionSearchSpace(graph, 2, rules, allowed_devices=(1, 0), placement_policy="contiguous_stages")
        expected = set()
        for p in walk(a):
            for g in p.groups:
                g.device = 1 - g.device
            expected.add(_plan_key(p))
        self.assertEqual(expected, {_plan_key(p) for p in walk(b)})

    def test_exact_and_mcts_use_stage_policy(self):
        graph = chain()
        rules = _singleton_registry()
        score = lambda p: 1. if {g.device for g in p.groups} == {0, 1} else 2.
        exact = exact_legal_search(graph, score, 2, registry=rules, placement_policy="contiguous_stages")
        found = legal_mcts_search(graph, score, 2, registry=rules, placement_policy="contiguous_stages",
                                  iterations=100, skip_exhausted=True, normalize_q=True)
        self.assertEqual(exact.search_space_size, 8)
        self.assertEqual(score(found[0]), exact.optimal_score)
        self.assertTrue(all(contiguous(graph, p) for p in found))


class StageBudget(unittest.TestCase):
    def setUp(self):
        self.graph = chain()
        self.rules = _singleton_registry()
        self.profile = AnalyticHardwareProfile(load_hardware(ROOT / "data/bw1000_2card/bw1000_2card.calibrated.json"))
        self.capacity = [d.max_mem for d in self.profile.hardware.devices]

    def test_cache_hits_and_reporting_never_hide_extra_calls(self):
        ledger = Ledger(self.graph, self.profile, self.rules, self.capacity, budget=2)
        p = make_reference_plan(self.graph, 0)
        ledger.evaluate(p)
        ledger.evaluate(copy.deepcopy(p))
        self.assertEqual(len(ledger.cache), 1)
        ledger.evaluate(make_reference_plan(self.graph, 1))
        changed = copy.deepcopy(p)
        changed.groups[-1].device = 1
        with self.assertRaises(BudgetReached):
            ledger.evaluate(changed)
        self.assertEqual(len(ledger.cache), 2)
        ledger.describe(ledger.best)
        self.assertEqual(len(ledger.cache), 2)

    def test_four_methods_share_real_budget_and_exact_gaps(self):
        exact = oracle(self.graph, self.profile, self.rules, self.capacity, "node")
        stage = oracle(self.graph, self.profile, self.rules, self.capacity, "contiguous_stages")
        for method in ("random", "sequential", "joint_mcts", "stage_joint_mcts"):
            with self.subTest(method=method):
                row = heuristic(self.graph, self.profile, self.rules, self.capacity, method, 8, 0, exact, stage)
                self.assertLessEqual(row["unique_candidates"], 8)
                self.assertEqual(len(row["trace"]), row["unique_candidates"])
                self.assertEqual(row["trace"][0]["phase"], "initialization")
                self.assertGreaterEqual(row["exact_gap_percent"], 0)
                if method != "sequential":
                    self.assertEqual(row["unique_candidates"], 8)


if __name__ == "__main__":
    unittest.main()
