import copy
import json
import math
from pathlib import Path
import unittest

from calibration.e2_dataset import make_dataset_view
from calibration.e2_records import load_jsonl_records
from cost_model.e2_factorial_evaluation import frozen_controls, interaction
from cost_model.e2_diagnostics import measurement_pool, workload_for_task
from cost_model.e2_refit_acquisition import expected_refit_losses
from cost_model.e2_srop_proxy import _active_profile, build_workload_graph, object_map_for_workload
from cost_model.e2_task_suite import active_task_contract
from cost_model.batched_placement_ranker import BatchedPlacementRanker
from cost_model.placement_probes import graph_wide_candidates
from simulator import make_reference_plan
from simulator.hardware import load_hardware


class FactorialTests(unittest.TestCase):
    def test_additive_effects_have_zero_interaction_and_level_shift_cancels(self):
        cells = {"legacy_proxy": 10.0, "legacy_refit": 9.0, "distance_proxy": 8.0, "distance_refit": 7.0}
        self.assertEqual(interaction(cells), 0.0)
        cells["distance_refit"] = 2.0
        self.assertEqual(interaction(cells), -5.0)
        self.assertEqual(interaction({key: value+100 for key, value in cells.items()}), -5.0)

    def test_frozen_controls_are_owned_copies_and_use_stopped_prefix(self):
        intervals = {"variants": {"legacy": {"value": 1}, "distance_floor": {"value": 2}}}
        refit = {"policies": {"srop_refit": {"modes": {"stopping": {"regret_budget_auc_ms": 0.1}},
                   "traces": [{"stop_budget": 1, "stop_regret_ms": 0.0, "selected_object_ids": ["a", "b"]}]}}}
        saved = copy.deepcopy((intervals, refit))
        result = frozen_controls(intervals, refit)
        self.assertEqual(result["legacy_refit"]["selected_object_ids"], ["a"])
        result["legacy_proxy"]["value"] = 99
        self.assertEqual((intervals, refit), saved)

    def test_hidden_labels_do_not_enter_combined_acquisition_after_a_reveal(self):
        root = Path(__file__).resolve().parents[2]
        directory = root / "data/bw1000_e2"
        records = load_jsonl_records(directory / "profiling_dataset.jsonl")
        manifest = json.loads((directory / "shape_manifest.json").read_text(encoding="utf-8"))
        contract = json.loads((directory / "split_contract.json").read_text(encoding="utf-8"))
        suite = json.loads((directory / "e2c_task_suite.json").read_text(encoding="utf-8"))
        task = next(task for task in suite["tasks"] if task["model_id"] == "qwen2.5-3b-s512-b1"
                    and task["scale"] == "layer" and task["initial_train_count"] == 16 and task["subset_seed"] == 1)
        runtime = active_task_contract(task, contract, manifest)
        runtime["active_profiling_protocol"]["uncertainty_variant"] = "distance_floor"
        hardware = load_hardware(root / "data/bw1000_2card/bw1000_2card.calibrated.json")
        workload = workload_for_task(task, manifest)
        graph = build_workload_graph(workload, "layer")
        mapping = object_map_for_workload(records, workload)
        pool = measurement_pool(graph, mapping, task["train_object_ids"])
        revealed = (pool[0],)
        remaining = [key for key in pool if key not in revealed]
        altered = copy.deepcopy(records)
        for row in altered:
            if row["object_id"] not in set(task["train_object_ids"]) | set(revealed):
                row["measurement"]["duration_s"] *= 100
                row["target"]["log_residual"] += math.log(100)
        outputs = []
        for dataset in (records, altered):
            profile, q90 = _active_profile(dataset, runtime, hardware, mapping,
                split_id=task["split_id"], revealed_object_ids=revealed)
            public = make_dataset_view(dataset, runtime, split_id=task["split_id"],
                mode="active_calibration", role="sampler", revealed_object_ids=revealed)
            plans, _ = graph_wide_candidates(graph, [make_reference_plan(graph)], profile.device_classes)
            ranker = BatchedPlacementRanker(graph, plans, 2, ((0, 1), (1, 0)))
            outputs.append(expected_refit_losses(profile, q90, public, remaining, runtime["prediction_protocol"],
                runtime["active_profiling_protocol"], ranker, ranker.rank(profile)))
        self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
