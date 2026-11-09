"""A1 shared memory view, A2 tied-parameter action constraint, A4 hard constraint."""
import json
from pathlib import Path
import tempfile
import unittest

from cost_model.exact_search import count_exact_search_space, exact_legal_search
from cost_model.hardware_profile import AnalyticHardwareProfile
from cost_model.legal_search import LegalFusionSearchSpace, legal_beam_search, legal_mcts_search
from cost_model.search_evaluation import (
    _CachedScorer, _modes, _run_search, evaluate_searches, save_evaluation,
)
from cost_model.training_memory import make_training_evaluator
from simulator import OpGraph, OpNode, TensorEdge, hardware_from_dict
from simulator.hardware import load_hardware
from simulator.placement_artifacts import load_placed_op_graph

ROOT = Path(__file__).resolve().parents[2]


def bw1000():
    return AnalyticHardwareProfile(load_hardware(ROOT / "data/bw1000_2card/bw1000_2card.calibrated.json"))


def tied_graph():
    """embedding -> mid -> lm_head; lm_head shares the embedding weight."""
    g = OpGraph(name="tied", extra={"dtype": "bf16"})
    g.add_node(OpNode("embedding", "embedding", 0.0, 64.0, 2000.0, extra={}))
    g.add_node(OpNode("mid", "matmul", 1e6, 64.0, 400.0, extra={}))
    g.add_node(OpNode("lm_head", "matmul", 1e6, 128.0, 0.0, extra={"shared_param_with": "embedding"}))
    g.add_edge(TensorEdge("embedding", "mid", 64.0))
    g.add_edge(TensorEdge("mid", "lm_head", 64.0))
    return g


def small_profile(max_mem=1e12):
    devices = [{"name": f"d{i}", "flops": 1e12, "mem_bw": 1e11, "max_mem": max_mem, "calibrated": True}
               for i in range(2)]
    return AnalyticHardwareProfile(hardware_from_dict(
        {"name": "t", "comm_bw": 1e10, "comm_latency": 1e-6, "link_bw": 1e10, "devices": devices}))


def device_of(plan, node):
    return next(g.device for g in plan.groups if node in g.node_ids)


def mode(name):
    return next(m for m in _modes(2) if m.name == name)


class TiedParameterActions(unittest.TestCase):
    def test_space_forbids_split_and_counts_only_legal_plans(self):
        g = tied_graph()
        free = LegalFusionSearchSpace(g, 2)
        tied = LegalFusionSearchSpace(g, 2, tie_shared_params=True)
        self.assertEqual(count_exact_search_space(free), 8)
        self.assertEqual(count_exact_search_space(tied), 4)
        state = tied.apply(tied.initial_state(), next(a for a in tied.actions(tied.initial_state())
                                                       if a.device == 1))
        state = tied.apply(state, tied.actions(state)[0])
        self.assertEqual({a.device for a in tied.actions(state)}, {1})
        with self.assertRaises(ValueError):
            tied.apply(state, next(a for a in free.actions(state) if a.device == 0))

    def test_every_engine_returns_only_colocated_tied_plans(self):
        g = tied_graph()
        scorer = lambda plan: float(sum(gr.device for gr in plan.groups))  # noqa: E731
        exact = exact_legal_search(g, scorer, 2, top_k=8, tie_shared_params=True)
        self.assertEqual(exact.search_space_size, 4)
        plans = (list(exact.plans)
                 + legal_beam_search(g, scorer, 2, beam=4, seed=0, tie_shared_params=True)
                 + legal_mcts_search(g, scorer, 2, iterations=32, seed=0, tie_shared_params=True))
        for plan in plans:
            self.assertEqual(device_of(plan, "embedding"), device_of(plan, "lm_head"))

    def test_default_space_is_unchanged(self):
        self.assertEqual(count_exact_search_space(LegalFusionSearchSpace(tied_graph(), 2)), 8)


class TrainingMemoryView(unittest.TestCase):
    def test_scorer_uses_training_evaluator_for_peak_and_feasibility(self):
        g, p = tied_graph(), small_profile()
        evaluator = make_training_evaluator(g, p, (1e12, 1e12))
        scorer = _CachedScorer(g, p, capacity=(1e12, 1e12), training_evaluator=evaluator)
        plans, _, _ = _run_search("exact", g, scorer, mode("joint"), seed=0, beam=2, rollouts=1,
                                  mcts_iterations=4, top_k=8)
        self.assertTrue(plans)
        for plan in plans:
            view, ev = scorer.memory_view(plan), evaluator(plan)
            self.assertTrue(ev.feasible)  # split tied plans are infeasible in training memory
            self.assertEqual(list(view.peak_memory), [ev.peak_per_device[d] for d in range(2)])
            self.assertEqual(view.makespan, ev.latency)

    def test_hard_mode_never_scores_infeasible_into_results_and_reports_none(self):
        g = tied_graph()
        # Static training state of all params on one device (2400 B / 2 * 18) overflows 15 kB.
        p = small_profile(max_mem=15_000.0)
        capacity = (15_000.0, 15_000.0)
        evaluator = make_training_evaluator(g, p, capacity)
        for method in ("exact", "beam", "mcts"):
            scorer = _CachedScorer(g, p, capacity=capacity, training_evaluator=evaluator,
                                   hard_constraint=True)
            plans, _, budget = _run_search(method, g, scorer, mode("joint"), seed=0, beam=2,
                                           rollouts=2, mcts_iterations=16, top_k=4)
            for plan in plans:
                self.assertTrue(scorer.memory_view(plan).feasible, method)
            self.assertIsNone(budget["infeasible_penalty_s"])
            self.assertEqual(budget["stages"][0]["constraint"], "hard")

    def test_hard_mode_with_no_feasible_plan_is_an_honest_empty_result(self):
        g, p = tied_graph(), small_profile(max_mem=10.0)
        evaluator = make_training_evaluator(g, p, (10.0, 10.0))
        reasons = {}
        for method in ("exact", "beam", "mcts"):
            scorer = _CachedScorer(g, p, capacity=(10.0, 10.0), training_evaluator=evaluator,
                                   hard_constraint=True)
            plans, _, budget = _run_search(method, g, scorer, mode("joint"), seed=0, beam=2,
                                           rollouts=2, mcts_iterations=8, top_k=2)
            self.assertEqual(plans, [])
            reasons[method] = budget["reason"]
        self.assertEqual(reasons, dict(exact="no_feasible_in_space", beam="evaluated_no_feasible",
                                       mcts="evaluated_no_feasible"))


class ReportAndExportShareOneResult(unittest.TestCase):
    def test_training_hard_report_artifact_and_candidates_agree(self):
        p = bw1000()
        report = evaluate_searches(p, ["small"], seed=0, mcts_iterations=4, beam=2, rollouts=1,
                                   top_k=2, memory_model="training", constraint="hard",
                                   tie_shared_params=True)
        scale = report["scales"][0]
        policy = scale["configuration"]["memory_policy"]
        self.assertEqual((policy["memory_model"], policy["constraint"]), ("training", "hard"))
        self.assertAlmostEqual(scale["configuration"]["capacity_bytes"][0], 0.95 * 68702699520.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            save_evaluation(path, report)
            saved = json.loads(path.read_text())
            for search in saved["scales"][0]["searches"]:
                for c in search["candidates"]:
                    self.assertTrue(c["feasible"])
                    self.assertTrue(c["memory_semantics"].startswith("training_lifetime_v1"))
                    self.assertTrue(all(x <= y for x, y in zip(c["peak_memory_bytes"], c["capacity_bytes"])))
                if search["candidates"]:
                    placed = load_placed_op_graph(path.parent / search["best_placed_op_graph"])
                    best = search["candidates"][0]
                    self.assertEqual(list(placed.predicted_peak_memory), best["peak_memory_bytes"])
                    self.assertEqual(placed.predicted_makespan, best["makespan_s"])

    def test_training_search_never_commits_to_unverified_fused_backward(self):
        # Layer graph has silu_mul; without a verified backward the evaluator rejects
        # it, so search must not propose it (sequential used to freeze it in stage one).
        report = evaluate_searches(bw1000(), ["layer"], seed=23, mcts_iterations=8, beam=2,
                                   rollouts=1, top_k=2, memory_model="training",
                                   constraint="hard", tie_shared_params=True)
        for search in report["scales"][0]["searches"]:
            self.assertIsNotNone(search["plan"], (search["mode"], search["method"]))
            self.assertTrue(all(len(g["node_ids"]) == 1 for g in search["plan"]["groups"]))

    def test_default_report_has_no_new_fields(self):
        report = evaluate_searches(bw1000(), ["small"], seed=0, mcts_iterations=2, beam=2, rollouts=1)
        scale = report["scales"][0]
        self.assertNotIn("memory_policy", scale["configuration"])
        candidate = scale["searches"][0]["candidates"][0]
        self.assertNotIn("memory_semantics", candidate)
        self.assertEqual(report["version"], "1.3")


if __name__ == "__main__":
    unittest.main()
