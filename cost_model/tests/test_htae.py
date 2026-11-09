"""Layer-2 Proteus-style executor and the opt-in MCTS selection fixes."""
from pathlib import Path
import random
import unittest

from cost_model.exact_search import exact_legal_search
from cost_model.hardware_profile import AnalyticHardwareProfile, GroupCost, TransferCost
from cost_model.htae import HTAECostModel
from cost_model.legal_search import LegalFusionSearchSpace, legal_mcts_search, rollout_legal_plan
from cost_model.search_evaluation import build_evaluation_graph
from cost_model.structural import StructuralCostModel
from simulator import FusionGroup, OpGraph, OpNode, Plan, TensorEdge, make_reference_plan
from simulator.hardware import load_hardware

ROOT = Path(__file__).resolve().parents[2]


def bw1000():
    return AnalyticHardwareProfile(load_hardware(ROOT / "data/bw1000_2card/bw1000_2card.calibrated.json"))


class FixedProfile:
    """Every group takes ``g`` seconds, every transfer ``t``; optional shared bus."""

    def __init__(self, n_devs, g=1.0, t=1.0, shared_bus=False, overlap_send=False, interference=0.0):
        self.n_devs, self.g, self.t, self.shared_bus = n_devs, g, t, shared_bus
        self.overlap_send, self.interference = overlap_send, interference

    def max_memory(self, device):
        return 1e18

    def estimate_group(self, graph, group):
        return GroupCost(self.g, self.g, 0., 0., 1., "fixed")

    def estimate_transfer(self, src, dst, nbytes):
        return TransferCost(self.t, 1., "fixed")

    def link_key(self, src, dst):
        return (0, 0) if self.shared_bus else (src, dst)


def graph(nodes, edges):
    g = OpGraph(name="t")
    for n in nodes:
        g.add_node(OpNode(n, "add", 1., 8., 0., {}))
    for a, b in edges:
        g.add_edge(TensorEdge(a, b, 8.))
    return g


def plan(placement):
    return Plan(groups=[FusionGroup([n], d, {"pattern_id": "unfused"}) for n, d in placement])


class HTAEMatchesStructuralWhereSemanticsAgree(unittest.TestCase):
    def test_single_device_and_blocking_serial_split(self):
        p = bw1000()
        for scale in ("small", "layer"):
            g = build_evaluation_graph(scale)
            ref = make_reference_plan(g, 0)
            a, b = StructuralCostModel().predict(g, ref, p), HTAECostModel().predict(g, ref, p)
            self.assertEqual((a.makespan, a.peak_memory), (b.makespan, b.peak_memory))
            order = g.topological_order()
            half = set(order[:len(order) // 2])
            for group in ref.groups:
                group.device = 0 if group.node_ids[0] in half else 1
            a, b = StructuralCostModel().predict(g, ref, p), HTAECostModel().predict(g, ref, p)
            self.assertAlmostEqual(a.makespan, b.makespan, places=15)
            self.assertEqual(a.peak_memory, b.peak_memory)

    def test_ready_order_differs_from_plan_order_only_by_dispatch(self):
        # Greedy ready-order dispatch is not monotone (list-scheduling anomalies):
        # it may be faster or slower than plan order. Both must stay finite and
        # transfer the same bytes.
        g, p = build_evaluation_graph("layer"), bw1000()
        space = LegalFusionSearchSpace(g, 2, require_executable=False)
        rng = random.Random(0)
        for _ in range(4):
            q = rollout_legal_plan(space, space.initial_state(), rng, .7)
            a, b = StructuralCostModel().predict(g, q, p), HTAECostModel().predict(g, q, p)
            self.assertEqual(a.cross_device_bytes, b.cross_device_bytes)
            self.assertAlmostEqual(a.communication_time, b.communication_time, places=15)


class ProteusRuntimeBehaviors(unittest.TestCase):
    def test_async_feature_comm_overlaps_compute(self):
        # a(d0) -> b(d1); c(d0) independent. Blocking: c waits for the send.
        g = graph(["a", "b", "c"], [("a", "b")])
        p = plan([("a", 0), ("c", 0), ("b", 1)])
        blocking = HTAECostModel().predict(g, p, FixedProfile(2)).makespan
        asynchronous = HTAECostModel(feature_comm_blocks_compute=False).predict(g, p, FixedProfile(2)).makespan
        self.assertEqual(blocking, 3.0)      # a | send | c, b runs during c
        self.assertEqual(asynchronous, 3.0)  # a | send || c | b  -> b ends at 3
        g2 = graph(["a", "b", "c", "d"], [("a", "b"), ("c", "d")])
        p2 = plan([("a", 0), ("c", 0), ("b", 1), ("d", 0)])
        self.assertEqual(HTAECostModel().predict(g2, p2, FixedProfile(2)).makespan, 4.0)
        self.assertEqual(HTAECostModel(feature_comm_blocks_compute=False)
                         .predict(g2, p2, FixedProfile(2)).makespan, 3.0)

    def test_fair_bandwidth_sharing_on_one_link(self):
        # a(d0)->c(d2), b(d1)->c(d2) at the same time over one shared bus.
        g = graph(["a", "b", "c"], [("a", "c"), ("b", "c")])
        p = plan([("a", 0), ("b", 1), ("c", 2)])
        shared = HTAECostModel().predict(g, p, FixedProfile(3, shared_bus=True))
        separate = HTAECostModel().predict(g, p, FixedProfile(3))
        self.assertEqual(separate.makespan, 3.0)  # 1 compute + 1 transfer + 1 compute
        self.assertEqual(shared.makespan, 4.0)    # both transfers at half rate end at t=3
        self.assertEqual([t["end"] for t in shared.details["transfers"]], [3.0, 3.0])

    def test_overlap_factor_slows_both_overlapped_operators(self):
        g = graph(["a", "b", "c"], [("a", "b")])
        p = plan([("a", 0), ("c", 0), ("b", 1)])
        model = HTAECostModel(feature_comm_blocks_compute=False, overlap_factor=1.0)
        est = model.predict(g, p, FixedProfile(2))
        # send and c overlap from t=1, each at half speed: both end at t=3; b ends at 4.
        self.assertEqual(est.makespan, 4.0)

    def test_settings_default_to_the_hardware_snapshot(self):
        hw = bw1000()
        self.assertEqual(HTAECostModel().settings(hw), (True, 0.0))


class MCTSSelectionFixes(unittest.TestCase):
    def test_skip_exhausted_enumerates_the_space_and_reaches_exact(self):
        g = graph(["a", "b", "c", "d"], [("a", "b"), ("b", "c"), ("a", "d")])
        p = bw1000()
        model = StructuralCostModel()
        score = lambda q: model.predict(g, q, p).makespan  # noqa: E731
        exact = exact_legal_search(g, score, 2, top_k=1, require_executable=False)
        seen = set()

        def counting(q):
            seen.add(tuple((tuple(x.node_ids), x.device) for x in q.groups))
            return score(q)
        best = legal_mcts_search(g, counting, 2, iterations=10_000, rollouts=1, seed=0, n_best=1,
                                 require_executable=False, skip_exhausted=True, normalize_q=True)[0]
        self.assertEqual(len(seen), exact.search_space_size)  # stopped once everything was scored
        self.assertEqual(score(best), score(exact.plans[0]))

    def test_defaults_reproduce_historical_selection(self):
        g, p = build_evaluation_graph("small"), bw1000()
        model = StructuralCostModel()
        score = lambda q: model.predict(g, q, p).makespan  # noqa: E731
        kwargs = dict(iterations=24, rollouts=2, seed=3, n_best=3, require_executable=False)
        a = legal_mcts_search(g, score, 2, **kwargs)
        b = legal_mcts_search(g, score, 2, normalize_q=False, skip_exhausted=False, **kwargs)
        self.assertEqual([score(x) for x in a], [score(x) for x in b])


if __name__ == "__main__":
    unittest.main()
