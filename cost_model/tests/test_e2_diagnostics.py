from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from calibration.e2_dataset import make_dataset_view
from calibration.e2_records import load_jsonl_records
from cost_model.e2_audit import PreparedCandidateRanker
from cost_model.e2_diagnostics import (
    acquisition_scores, evaluate_one_step, evaluate_zero_budget, measurement_pool,
    workload_for_task, _read_checkpoint, _write_checkpoint,
)
from cost_model.e2_srop_proxy import _active_profile, build_workload_graph, object_map_for_workload
from cost_model.e2_task_suite import active_task_contract
from cost_model.search_sensitivity import device_exchange_permutations
from simulator import make_reference_plan
from simulator.hardware import load_hardware


class E2DiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[2]
        directory = root / "data/bw1000_e2"
        cls.manifest = json.loads((directory / "shape_manifest.json").read_text(encoding="utf-8"))
        cls.contract = json.loads((directory / "split_contract.json").read_text(encoding="utf-8"))
        suite = json.loads((directory / "e2c_task_suite.json").read_text(encoding="utf-8"))
        cls.task = next(task for task in suite["tasks"] if task["model_id"] == "qwen2.5-3b-s256-b1"
                        and task["scale"] == "layer" and task["initial_train_count"] == 16)
        cls.records = load_jsonl_records(directory / "profiling_dataset.jsonl")
        cls.hardware = load_hardware(root / "data/bw1000_2card/bw1000_2card.calibrated.json")
        cls.workload = workload_for_task(cls.task, cls.manifest)
        cls.graph = build_workload_graph(cls.workload, "layer")
        base = make_reference_plan(cls.graph)
        split = copy.deepcopy(base)
        for group in split.groups[len(split.groups)//2:]:
            group.device = 1
        cls.plans = [base, split]

    def test_low_budget_masks_withheld_train_and_can_reveal_it(self):
        before = copy.deepcopy(self.contract)
        runtime = active_task_contract(self.task, self.contract, self.manifest)
        split = runtime["splits"][0]
        mapping = object_map_for_workload(self.records, self.workload)
        pool = measurement_pool(self.graph, mapping, self.task["train_object_ids"])
        withheld = set(split["withheld_original_train_object_ids"]) & set(pool)
        self.assertTrue(withheld)
        view = make_dataset_view(self.records, runtime, split_id=self.task["split_id"],
                                  mode="active_calibration", role="predictor")
        self.assertEqual(sum("target" in row for row in view), 16)
        self.assertTrue(all("target" not in row for row in view if row["object_id"] in withheld))
        selected = sorted(withheld)[0]
        revealed = make_dataset_view(self.records, runtime, split_id=self.task["split_id"],
                                      mode="active_calibration", role="predictor", revealed_object_ids=(selected,))
        self.assertEqual(sum("target" in row for row in revealed), 17)
        self.assertEqual(self.contract, before)

    def test_full_budget_recovers_entire_oracle_ranking(self):
        result = evaluate_zero_budget(self.task, self.plans, self.records, self.manifest,
                                      self.contract, self.hardware, graph=self.graph)
        self.assertTrue(result["full_budget_recovers_oracle"])
        self.assertEqual(result["full_budget_max_cost_error_s"], 0.0)
        self.assertGreater(result["withheld_original_train_pool_objects"], 0)

    def test_hidden_labels_do_not_change_acquisition_scores(self):
        runtime = active_task_contract(self.task, self.contract, self.manifest)
        mapping = object_map_for_workload(self.records, self.workload)
        pool = measurement_pool(self.graph, mapping, self.task["train_object_ids"])
        altered = copy.deepcopy(self.records)
        for row in altered:
            if row["object_id"] not in set(self.task["train_object_ids"]):
                row["measurement"]["duration_s"] *= 100
                row["target"]["log_residual"] += 10
        outputs = []
        for records in (self.records, altered):
            profile, uncertainty = _active_profile(records, runtime, self.hardware, mapping,
                split_id=self.task["split_id"], revealed_object_ids=())
            exchange = device_exchange_permutations(profile.device_classes)
            ranker = PreparedCandidateRanker(self.graph, self.plans, 2, exchange)
            public = make_dataset_view(records, runtime, split_id=self.task["split_id"],
                mode="active_calibration", role="sampler")
            scores = acquisition_scores(self.graph, self.plans, profile, uncertainty, public,
                pool, ranker, exchange, runtime["active_profiling_protocol"])
            outputs.append({key: {name: value for name, value in row.items() if name != "acquisition_wall_time_s"}
                            for key, row in scores.items()})
        self.assertEqual(outputs[0], outputs[1])

    def test_one_step_covers_pool_without_using_outcomes_for_selection(self):
        result = evaluate_one_step(self.task, self.plans, self.records, self.manifest,
                                   self.contract, self.hardware, graph=self.graph)
        self.assertEqual(result["measurement_objects"], len(result["observations"]))
        for policy in ("uncertainty_only", "impact_only", "srop_proxy"):
            expected = min(result["observations"], key=lambda row: (-row[policy], row["object_id"]))
            self.assertEqual(result["policies"][policy]["selected_object_id"], expected["object_id"])
        self.assertAlmostEqual(result["policies"]["random"]["gain_ms_mean"],
            sum(row["retrained_gain_ms"] for row in result["observations"]) / result["measurement_objects"])

    def test_compressed_checkpoints_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.json.gz"
            self.assertIsNone(_read_checkpoint(path))
            _write_checkpoint(path, {"candidate_plans": [], "zero": {"regret": 0.0}})
            self.assertEqual(_read_checkpoint(path)["zero"]["regret"], 0.0)


if __name__ == "__main__":
    unittest.main()
