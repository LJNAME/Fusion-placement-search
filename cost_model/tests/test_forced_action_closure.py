"""Exact cover preservation, RNG equivalence, and decision-only tree expansion."""
from types import SimpleNamespace
import random
import unittest

from cost_model.decision_diagnostics import ActionAudit, explain_actions
from cost_model.exact_search import count_exact_search_space, exact_legal_search
from cost_model.fusion_library import load_library
from cost_model.legal_search import (LegalFusionSearchSpace, _LegalMCTSNode, _guided_rollout_legal_plan,
    _unfused_guided_rollout_legal_plan, _plan_key, legal_mcts_search, rollout_legal_plan)
from cost_model.search_evaluation import _singleton_registry, build_evaluation_graph
from simulator.dataset import plan_to_dict
from simulator import OpGraph, OpNode, TensorEdge
from tools.evaluate_mcts_medium import extract_graph
from tools.evaluate_stage_search import walk


def tied_chain():
    graph = OpGraph(name="tied-chain")
    for i in range(5):
        graph.add_node(OpNode(str(i), "add", 1., 2., 2. if i == 0 else 0., {"dtype": "bf16", "output_shape": (1,)}))
        if i:
            graph.add_edge(TensorEdge(str(i - 1), str(i), 2.))
    graph.nodes["4"].extra["shared_param_with"] = "0"
    return graph


def closed_walk(space):
    root = _LegalMCTSNode(space.forced_closure(space.initial_state()))
    def visit(node):
        if space.is_terminal(node.state):
            yield space.to_plan(node.state)
            return
        if len(space.actions(node.state)) == 1:
            raise AssertionError("unary state persisted in compressed tree")
        node.expand(space, forced_action_closure=True)
        for child in node.children:
            yield from visit(child)
    yield from visit(root)


class ForcedClosure(unittest.TestCase):
    def test_reference_singletons_are_classified_by_real_binding(self):
        space = LegalFusionSearchSpace(tied_chain(), 2, _singleton_registry(),
                                       placement_policy="hybrid", tie_shared_params=True)
        state = space.initial_state()
        self.assertEqual(explain_actions(space, state, space.actions(state))["kind"], "placement")
        state = space.apply(state, space.actions(state)[0])
        audit = ActionAudit()
        while not space.is_terminal(state):
            actions = space.actions(state)
            audit(space, state, actions)
            state = space.apply(state, actions[0])
        self.assertEqual(audit.to_dict()["forced_primary_causes"], {"pending_shared_parameter_blocks_stage_exit": 4})

    def test_closure_is_idempotent_and_preserves_every_original_choice(self):
        space = LegalFusionSearchSpace(tied_chain(), 1, _singleton_registry())
        state = space.forced_closure(space.initial_state())
        self.assertTrue(space.is_terminal(state))
        self.assertEqual(len(state.choices), 5)
        self.assertEqual(space.forced_closure(state), state)
        self.assertEqual(plan_to_dict(space.to_plan(state)), plan_to_dict(next(iter(walk(space)))))

    def test_all_four_action_domains_keep_identical_exact_covers(self):
        graph = extract_graph(build_evaluation_graph("model_3b"), "mlp")
        for policy in ("node", "contiguous_stages", "branch_aware", "hybrid"):
            with self.subTest(policy=policy):
                space = LegalFusionSearchSpace(graph, 2, load_library().registry(), placement_policy=policy)
                raw = {_plan_key(p): plan_to_dict(p) for p in walk(space)}
                compact = list(closed_walk(space))
                self.assertEqual(len(compact), count_exact_search_space(space))
                self.assertEqual(set(raw), {_plan_key(p) for p in compact})
                for plan in compact:
                    actual = plan_to_dict(plan)
                    expected = raw[_plan_key(plan)]
                    actual.pop("name", None); expected.pop("name", None)
                    self.assertEqual(actual, expected)

    def test_random_completions_and_rng_state_are_identical(self):
        graph = extract_graph(build_evaluation_graph("model_3b"), "mlp")
        for policy in ("node", "contiguous_stages", "branch_aware", "hybrid"):
            space = LegalFusionSearchSpace(graph, 2, load_library().registry(), placement_policy=policy)
            for seed in range(10):
                a, b = random.Random(seed), random.Random(seed)
                raw = rollout_legal_plan(space, space.initial_state(), a, .7)
                compact = rollout_legal_plan(space, space.initial_state(), b, .7, forced_action_closure=True)
                self.assertEqual(_plan_key(raw), _plan_key(compact))
                self.assertEqual(a.getstate(), b.getstate())
            for completion in (_guided_rollout_legal_plan, _unfused_guided_rollout_legal_plan):
                self.assertEqual(_plan_key(completion(space, space.initial_state())),
                                 _plan_key(completion(space, space.initial_state(), forced_action_closure=True)))

    def test_root_terminal_and_feasible_infeasible_leaves_match_exact(self):
        graph, registry = tied_chain(), _singleton_registry()
        for allowed in ((0,), (0, 1)):
            score = lambda p: 2. - p.groups[0].device
            oracle = exact_legal_search(graph, score, 2, registry=registry, allowed_devices=allowed,
                                         placement_policy="hybrid", tie_shared_params=True)
            for hard in (False, True):
                stats = {}
                kwargs = {"evaluator": lambda p: SimpleNamespace(feasible=True, latency=score(p))} if hard else {}
                plans = legal_mcts_search(graph, None if hard else score, 2, iterations=100,
                    registry=registry, allowed_devices=allowed, placement_policy="hybrid", tie_shared_params=True,
                    forced_action_closure=True, search_stats=stats, skip_exhausted=True, **kwargs)
                self.assertEqual(score(plans[0]), oracle.optimal_score)
                self.assertEqual(stats["unary_expansions"], 0)

    def test_sound_pruning_checks_intermediate_forced_states(self):
        graph, registry = tied_chain(), _singleton_registry()
        prune = lambda space, state: len(state.choices) >= 2 and state.choices[0].device == 0
        score = lambda plan: 1. if plan.groups[0].device == 1 else 100.
        plans = legal_mcts_search(graph, score, 2, registry=registry, iterations=100,
            placement_policy="hybrid", tie_shared_params=True, forced_action_closure=True,
            prune=prune, skip_exhausted=True)
        self.assertEqual(len(plans), 1)
        self.assertEqual({g.device for g in plans[0].groups}, {1})
        all_pruned = legal_mcts_search(graph, score, 2, registry=registry, iterations=5,
            forced_action_closure=True, allowed_devices=(0,), prune=lambda space, state: len(state.choices) == 2)
        self.assertEqual(all_pruned, [])

    def test_partial_pruning_dead_end_is_not_a_terminal_solution(self):
        graph = tied_chain()
        for closure in (False, True):
            result = legal_mcts_search(graph, None, 2, registry=_singleton_registry(), iterations=100,
                evaluator=lambda p: SimpleNamespace(feasible=False, latency=None),
                partial_prune=lambda space, state: len(state.choices) >= 2,
                forced_action_closure=closure, skip_exhausted=True)
            self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
