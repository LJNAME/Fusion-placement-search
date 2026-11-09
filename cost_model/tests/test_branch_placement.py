"""DAG-derived branch units, mixed actions, overlap and counterfactual coupling."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cost_model.fusion_library import load_library
from cost_model.hardware_profile import AnalyticHardwareProfile
from cost_model.legal_search import LegalFusionSearchSpace, branch_regions
from cost_model.search_evaluation import _plan_key, build_evaluation_graph
from simulator import FusionGroup, Plan, make_reference_plan
from simulator.hardware import load_hardware
from tools.evaluate_branch_search import ObservedLedger, coupling, parallelism
from tools.evaluate_mcts_medium import extract_graph
from tools.evaluate_stage_search import walk

ROOT = Path(__file__).resolve().parents[2]


def positions(graph, plan):
    return {n: g.device for g in plan.groups for n in g.node_ids}


class BranchSpaces(unittest.TestCase):
    def test_real_mlp_regions_and_independent_complete_space_filter(self):
        graph = extract_graph(build_evaluation_graph("model_3b"), "mlp")
        registry = load_library().registry("full")
        regions = branch_regions(graph, graph.topological_order())
        self.assertEqual(len(regions), 1)
        r = regions[0]
        self.assertEqual(r["fork"], "L000.pre_mlp_norm")
        self.assertEqual(r["join"], "L000.mlp.gated_mul")
        self.assertEqual({tuple(p) for p in r["branches"]}, {
            ("L000.mlp.gate_proj", "L000.mlp.silu"), ("L000.mlp.up_proj",)})
        raw = list(walk(LegalFusionSearchSpace(graph, 2, registry)))
        branch_nodes = {n for path in r["branches"] for n in path}
        for policy in ("branch_aware", "hybrid"):
            expected = set()
            for plan in raw:
                dev = positions(graph, plan)
                if not all(len({dev[n] for n in path}) == 1 for path in r["branches"]):
                    continue
                if policy == "hybrid":
                    stages = []
                    for n in graph.topological_order():
                        if n not in branch_nodes and (not stages or dev[n] != stages[-1]):
                            stages.append(dev[n])
                    if len(stages) != len(set(stages)):
                        continue
                expected.add(_plan_key(plan))
            space = LegalFusionSearchSpace(graph, 2, registry, placement_policy=policy)
            found = list(walk(space))
            self.assertEqual(expected, {_plan_key(p) for p in found})
            self.assertEqual(len(found), len(expected))
            self.assertTrue(any(len(g.node_ids) > 1 for p in found for g in p.groups))

    def test_serial_fallback_and_device_permutation(self):
        registry = load_library().registry("full")
        graph = extract_graph(build_evaluation_graph("model_3b"), "serial_tail")
        self.assertEqual(branch_regions(graph, graph.topological_order()), [])
        for a, b in (("node", "branch_aware"), ("contiguous_stages", "hybrid")):
            self.assertEqual({_plan_key(p) for p in walk(LegalFusionSearchSpace(graph, 2, registry, placement_policy=a))},
                             {_plan_key(p) for p in walk(LegalFusionSearchSpace(graph, 2, registry, placement_policy=b))})
        graph = extract_graph(build_evaluation_graph("model_3b"), "mlp")
        for policy in ("branch_aware", "hybrid"):
            normal = list(walk(LegalFusionSearchSpace(graph, 2, registry, placement_policy=policy)))
            expected = set()
            for plan in normal:
                for g in plan.groups:
                    g.device = 1 - g.device
                expected.add(_plan_key(plan))
            swapped = LegalFusionSearchSpace(graph, 2, registry, placement_policy=policy, allowed_devices=(1, 0))
            self.assertEqual(expected, {_plan_key(p) for p in walk(swapped)})


class Diagnostics(unittest.TestCase):
    def test_overlap_counts_time_not_just_two_cards_used(self):
        plan = Plan(groups=[FusionGroup(["gate"], 0), FusionGroup(["up"], 1), FusionGroup(["join"], 0)])
        estimate = SimpleNamespace(makespan=4., details={"groups": [
            {"start": 0., "end": 2.}, {"start": 1., "end": 3.}, {"start": 3., "end": 4.}]})
        row = parallelism(plan, estimate, {"gate": "left", "up": "right"})
        self.assertEqual(row["branch_overlap_us"], 1e6)
        self.assertEqual(row["branch_overlap_fraction"], .25)
        self.assertEqual(row["max_parallel_branch_devices"], 2)
        estimate.details["groups"][1] = {"start": 2., "end": 3.}
        serial = parallelism(plan, estimate, {"gate": "left", "up": "right"})
        self.assertEqual(serial["branch_overlap_us"], 0.)
        self.assertEqual(serial["max_parallel_branch_devices"], 1)

    def test_legality_change_is_not_cost_preference_flip(self):
        n = (("native", ("a",)), ("native", ("b",)))
        f = (("fused", ("a", "b")),)
        row = lambda t: {"feasible": True, "makespan_ms": t}
        catalog = {n: {(0, 0): row(10), (1, 1): row(11), (0, 1): row(2)},
                   f: {(0, 0): row(9), (1, 1): row(10)}}
        p = coupling(catalog, ["a", "b"])["pairs"][0]
        self.assertTrue(p["legality_changes"])
        self.assertFalse(p["cost_interaction"])
        self.assertFalse(p["fusion_preference_flip"])
        self.assertFalse(p["best_placement_changes_on_shared_domain"])
        self.assertFalse(p["decision_coupling_observed"])

    def test_preference_and_placement_reversal_have_counterfactual_witness(self):
        n = (("native", ("a",)), ("native", ("b",)))
        f = (("fused", ("a", "b")),)
        row = lambda t: {"feasible": True, "makespan_ms": t}
        p = coupling({n: {(0, 0): row(10), (1, 1): row(8)}, f: {(0, 0): row(7), (1, 1): row(9)}}, ["a", "b"])["pairs"][0]
        self.assertTrue(p["fusion_preference_flip"])
        self.assertTrue(p["best_placement_changes_on_shared_domain"])
        self.assertIsNotNone(p["placement_ranking_reversal"])
        self.assertTrue(p["decision_coupling_observed"])

    def test_observations_reuse_the_one_charged_prediction(self):
        graph = extract_graph(build_evaluation_graph("model_3b"), "mlp")
        profile = AnalyticHardwareProfile(load_hardware(ROOT / "data/bw1000_2card/bw1000_2card.calibrated.json"))
        ledger = ObservedLedger(graph, profile, load_library().registry("full"), [d.max_mem for d in profile.hardware.devices], 64)
        with patch.object(ledger.model, "predict", wraps=ledger.model.predict) as predict:
            plan = make_reference_plan(graph, 0)
            row = ledger.evaluate(plan)
            ledger.evaluate(plan)
            self.assertEqual(predict.call_count, 1)
            self.assertEqual(row["branch_overlap_us"], 0.)


if __name__ == "__main__":
    unittest.main()
