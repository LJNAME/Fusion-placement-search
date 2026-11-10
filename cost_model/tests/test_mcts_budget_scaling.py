"""Budget prefixes preserve policy, distinct-plan counting and unknown optima."""
import json
from pathlib import Path
import tempfile
import unittest

from cost_model.hardware_profile import AnalyticHardwareProfile
from cost_model.search_evaluation import _singleton_registry
from simulator import OpGraph, OpNode, TensorEdge
from simulator.hardware import load_hardware
from tools.evaluate_stage_search import HARDWARE
from tools.evaluate_mcts_budget_scaling import prefix_metrics, run_seed


def graph():
    result = OpGraph(name="four-node-budget-fixture")
    for i in range(4):
        result.add_node(OpNode(str(i), "add", 8., 16., 0., {"dtype": "bf16", "output_shape": (8,)}))
        if i:
            result.add_edge(TensorEdge(str(i - 1), str(i), 16.))
    return result


def trace(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class BudgetScaling(unittest.TestCase):
    def test_independent_short_run_matches_continuous_prefix(self):
        profile = AnalyticHardwareProfile(load_hardware(HARDWARE))
        capacity = [d.max_mem for d in profile.hardware.devices]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            short = run_seed(graph(), profile, _singleton_registry(), capacity, 7, (4,), root / "short")
            expected = trace(root / "short/trace.jsonl")
            long = run_seed(graph(), profile, _singleton_registry(), capacity, 7, (4, 8), root / "long", expected)
            actual = trace(root / "long/trace.jsonl")
            self.assertEqual([r["candidate_id"] for r in expected], [r["candidate_id"] for r in actual[:4]])
            self.assertEqual(short["checkpoints"][0]["distinct_feasible_plans"], 4)
            self.assertEqual(long["checkpoints"][-1]["distinct_feasible_plans"], 8)
            self.assertEqual(long["checkpoints"][0]["expanded_decision_nodes"], short["checkpoints"][0]["expanded_decision_nodes"])
            self.assertLessEqual(long["checkpoints"][0]["wall_s"], long["checkpoints"][-1]["wall_s"])
            self.assertEqual(long["actual_unique_evaluations"], 8)

    def test_infeasible_results_remain_null(self):
        profile = AnalyticHardwareProfile(load_hardware(HARDWARE))
        with tempfile.TemporaryDirectory() as tmp:
            result = run_seed(graph(), profile, _singleton_registry(), (0., 0.), 0, (4,), Path(tmp) / "run")
            row = result["checkpoints"][0]
            self.assertFalse(row["found_feasible"])
            self.assertEqual(row["distinct_feasible_plans"], 0)
            self.assertIsNone(row["first_feasible_evaluation"])
            self.assertIsNone(row["best_feasible_ms"])
            self.assertNotIn("best_plan", row)

    def test_prefix_never_uses_future_feasible_labels(self):
        rows = [{"candidate_id": str(i), "evaluation": i + 1, "feasible": i >= 2,
                 "makespan_ms": 4 - i, "elapsed_s": i + 1} for i in range(4)]
        before = prefix_metrics(rows, 2, {})
        after = prefix_metrics(rows, 4, {"branch_expansions": 9, "max_decision_depth": 3})
        self.assertFalse(before["found_feasible"])
        self.assertIsNone(before["best_feasible_ms"])
        self.assertEqual(after["first_feasible_evaluation"], 3)
        self.assertEqual(after["expanded_decision_nodes"], 9)
        self.assertEqual(after["max_decision_depth"], 3)


if __name__ == "__main__":
    unittest.main()
