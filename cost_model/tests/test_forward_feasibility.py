"""Soundness against every completion of a small tied fork, plus budget fairness."""
import random
import unittest
from types import SimpleNamespace

from cost_model.forward_feasibility import ForwardResidentBound, replay_plan
from cost_model.fusion_library import load_library
from cost_model.hardware_profile import AnalyticHardwareProfile
from cost_model.htae import HTAECostModel
from cost_model.legal_search import LegalFusionSearchSpace, legal_mcts_search, rollout_legal_plan
from cost_model.search_evaluation import qwen3_spec, _plan_key
from simulator import OpGraph, OpNode, TensorEdge, hardware_from_dict
from simulator.model_graphs import build_model_graph
from tools.audit_qwen3_feasibility import run_one
from tools.evaluate_stage_search import walk


def graph():
    g = OpGraph(name="tied_fork", extra={"dtype": "bf16"})
    for name, kind, param, out in (("embedding", "embedding", 100., 1.),
                                  ("gate", "matmul", 20., 1.),
                                  ("silu", "silu", 0., 1.),
                                  ("up", "matmul", 20., 1.),
                                  ("mul", "mul", 0., 1.),
                                  ("lm_head", "matmul", 0., 8.)):
        g.add_node(OpNode(name, kind, 1., out, param,
                         extra={"shared_param_with": "embedding"} if name == "lm_head" else {}))
    for src, dst in (("embedding", "gate"), ("gate", "silu"), ("embedding", "up"),
                     ("silu", "mul"), ("up", "mul"), ("mul", "lm_head")):
        g.add_edge(TensorEdge(src, dst, 1.))
    return g


def profile():
    return AnalyticHardwareProfile(hardware_from_dict({"name": "test", "comm_bw": 1e10,
        "comm_latency": 1e-6, "link_bw": 1e10,
        "devices": [{"name": f"d{i}", "flops": 1e12, "mem_bw": 1e11,
                     "max_mem": 1e12, "calibrated": True} for i in range(2)]}))


class ForwardFeasibility(unittest.TestCase):
    def test_lower_bound_never_exceeds_any_completion_peak(self):
        g, p = graph(), profile()
        for policy in ("node", "hybrid"):
            space = LegalFusionSearchSpace(g, 2, tie_shared_params=True, placement_policy=policy)
            bound, model = ForwardResidentBound((135., 135.)), HTAECostModel()
            feasible = 0
            for plan in walk(space):
                peak = model.predict(g, plan, p).peak_memory
                okay = all(v <= 135. for v in peak)
                feasible += okay
                for state in replay_plan(space, plan):
                    lower = bound.explain(space, state)["peak_lower_bound_bytes"]
                    self.assertTrue(all(lo <= hi + 1e-6 for lo, hi in zip(lower, peak)))
                    if okay:
                        self.assertFalse(bound(space, state))
            self.assertGreater(feasible, 0)

    def test_pruned_domain_keeps_every_feasible_plan(self):
        g, p = graph(), profile()
        plain = LegalFusionSearchSpace(g, 2, tie_shared_params=True, placement_policy="hybrid")
        bounded = LegalFusionSearchSpace(g, 2, tie_shared_params=True, placement_policy="hybrid",
                                        partial_prune=ForwardResidentBound((135., 135.)))
        kept = {_plan_key(plan) for plan in walk(bounded)}
        model = HTAECostModel()
        all_plans = list(walk(plain))
        self.assertLess(len(kept), len(all_plans))
        for plan in all_plans:
            if all(v <= 135. for v in model.predict(g, plan, p).peak_memory):
                self.assertIn(_plan_key(plan), kept)

    def test_random_rollout_uses_the_same_action_filter(self):
        space = LegalFusionSearchSpace(graph(), 2, tie_shared_params=True, placement_policy="hybrid",
                                       partial_prune=ForwardResidentBound((135., 135.)))
        rng = random.Random(23)
        for _ in range(20):
            plan = rollout_legal_plan(space, space.initial_state(), rng)
            for state in replay_plan(space, plan):
                self.assertFalse(space.partial_prune(space, state))

    def test_empty_constrained_root_returns_no_plan(self):
        g = graph()
        result = legal_mcts_search(g, None, 2, iterations=10, tie_shared_params=True,
            evaluator=lambda p: SimpleNamespace(feasible=False, latency=None),
            partial_prune=ForwardResidentBound((0., 0.)), skip_exhausted=True)
        self.assertEqual(result, [])

    def test_observer_does_not_change_default_search(self):
        traces = []
        observed = []
        for enabled in (False, True):
            trace = []
            def score(plan):
                trace.append(_plan_key(plan))
                return SimpleNamespace(feasible=True, latency=float(sum(g.device for g in plan.groups) + 1))
            legal_mcts_search(graph(), None, 2, iterations=10, rollouts=3, seed=7,
                tie_shared_params=True, evaluator=score,
                rollout_observer=(lambda s, state, p: observed.append(len(state.choices))) if enabled else None)
            traces.append(trace)
        self.assertEqual(traces[0], traces[1])
        self.assertTrue(observed)

    def test_both_arms_and_methods_share_charged_witness(self):
        g = build_model_graph(qwen3_spec(2))
        p, registry = profile(), load_library().registry("full")
        space = LegalFusionSearchSpace(g, 2, registry, placement_policy="hybrid", tie_shared_params=True)
        state = space.initial_state()
        while not space.is_terminal(state):
            node = space.anchor(state)
            d = int(node in space.branch_unit)
            action = next(a for a in space.actions(state) if a.device == d and len(a.candidate.node_ids) == 1)
            state = space.apply(state, action)
        witness = space.to_plan(state)
        prior_ids = set()
        for guided in (False, True):
            for method in ("joint_node", "joint_hybrid", "random_node", "random_hybrid"):
                row, _ = run_one(g, p, registry, (1e12, 1e12), witness, method, 0, guided, budget=4)
                self.assertEqual(row["budget_used"], 4)
                self.assertEqual(row["unique_candidates"], 4)
                self.assertEqual(row["trace"][2]["phase"], "shared_witness")
                prior_ids.add(row["trace"][2]["candidate_id"])
        self.assertEqual(len(prior_ids), 1)


if __name__ == "__main__":
    unittest.main()
