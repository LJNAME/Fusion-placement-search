from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from calibration.e2_records import load_jsonl_records
from cost_model.e2_budget_summary import METRICS, compact_task, prefix_point, summarize_tasks
from cost_model.e2_diagnostics import workload_for_task
from cost_model.e2_srop_proxy import build_workload_graph, evaluate_fixed_candidate_srop
from cost_model.e2_task_suite import active_task_contract
from simulator import make_reference_plan
from simulator.hardware import load_hardware


def synthetic_result():
    trajectory = []
    for budget, regret in enumerate((1.0, 2.0, 0.0)):
        point = {key: 0.0 for key in METRICS}
        point.update(budget=budget, top1_regret_ms=regret, stop_recommended=budget == 0,
                     random_seed=None, next_selection_score=0.0,
                     selected_object_ids=["a", "b"][:budget])
        trajectory.append(point)
    return {"initial_top1_regret_ms": 1.0, "measurement_pool_objects": 2, "candidate_plans": 2,
            "structure_preparation_wall_time_s": 0.0, "evaluation_wall_time_s": 0.0,
            "policies": {name: {"full_trajectories": [copy.deepcopy(trajectory)]}
                         for name in ("random", "uncertainty_only", "impact_only", "srop_proxy")}}


class BudgetSummaryTests(unittest.TestCase):
    task = {"task_id": "a", "split_id": "split", "model_id": "model", "scale": "layer",
            "initial_train_count": 16, "subset_seed": 0}

    def test_stop_is_chosen_from_score_and_future_outcomes_do_not_change_it(self):
        trajectory = synthetic_result()["policies"]["srop_proxy"]["full_trajectories"][0]
        stopped = prefix_point(trajectory, 16, stopping=True)
        self.assertEqual(stopped["budget"], 0)
        self.assertEqual(stopped["top1_regret_ms"], 1.0)

    def test_auc_uses_common_horizon_and_exhausted_pool_holds_last_decision(self):
        compact = compact_task(self.task, synthetic_result())
        modes = compact["policies"]["srop_proxy"]["modes"]
        self.assertEqual(modes["forced"]["curve"][-1]["measurements_mean"], 2.0)
        self.assertEqual(modes["stopping"]["curve"][-1]["measurements_mean"], 0.0)
        self.assertAlmostEqual(modes["forced"]["regret_budget_auc_ms"], 2.5 / 16)
        self.assertEqual(modes["stopping"]["regret_budget_auc_ms"], 1.0)

    def test_harm_is_retained_at_intermediate_budgets(self):
        compact = compact_task(self.task, synthetic_result())
        summary = summarize_tasks([compact])
        curve = summary["modes"]["forced"]["policies"]["srop_proxy"]["curve"]
        self.assertEqual(curve[1]["harmed_configs"], 1)
        self.assertEqual(curve[-1]["harmed_configs"], 0)


class BudgetLabelIsolationTests(unittest.TestCase):
    def test_unrevealed_label_changes_cannot_change_later_selection_prefix(self):
        root = Path(__file__).resolve().parents[2]
        directory = root / "data/bw1000_e2"
        manifest = json.loads((directory / "shape_manifest.json").read_text(encoding="utf-8"))
        contract = json.loads((directory / "split_contract.json").read_text(encoding="utf-8"))
        suite = json.loads((directory / "e2c_task_suite.json").read_text(encoding="utf-8"))
        task = next(task for task in suite["tasks"] if task["model_id"] == "qwen2.5-3b-s256-b1"
                    and task["scale"] == "layer" and task["initial_train_count"] == 16)
        runtime = active_task_contract(task, contract, manifest)
        records = load_jsonl_records(directory / "profiling_dataset.jsonl")
        hardware = load_hardware(root / "data/bw1000_2card/bw1000_2card.calibrated.json")
        workload = workload_for_task(task, manifest)
        graph = build_workload_graph(workload, "layer")
        base = make_reference_plan(graph)
        other = copy.deepcopy(base)
        for group in other.groups[len(other.groups)//2:]:
            group.device = 1
        initial = evaluate_fixed_candidate_srop(graph, [base, other], records, runtime, hardware, workload, random_seeds=(0,))
        trajectory = initial["policies"]["srop_proxy"]["full_trajectories"][0]
        target = trajectory[-1]["selected_object_ids"][-1]
        altered = copy.deepcopy(records)
        record = next(record for record in altered if record["object_id"] == target)
        record["measurement"]["duration_s"] *= 20
        record["target"]["log_residual"] += 3
        changed = evaluate_fixed_candidate_srop(graph, [base, other], altered, runtime, hardware, workload, random_seeds=(0,))
        replay = changed["policies"]["srop_proxy"]["full_trajectories"][0]
        self.assertGreater(len(trajectory), 2)
        self.assertEqual([point["next_selection"] for point in trajectory[:-1]],
                         [point["next_selection"] for point in replay[:-1]])
        self.assertEqual([point["next_selection_score"] for point in trajectory[:-1]],
                         [point["next_selection_score"] for point in replay[:-1]])


if __name__ == "__main__":
    unittest.main()
