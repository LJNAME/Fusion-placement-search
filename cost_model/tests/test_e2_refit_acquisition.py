import copy
import json
import math
from pathlib import Path
import unittest

from calibration.e2_dataset import make_dataset_view
from calibration.e2_records import load_jsonl_records
from cost_model.batched_placement_ranker import BatchedPlacementRanker
from cost_model.e2_diagnostics import measurement_pool, workload_for_task
from cost_model.e2_refit_acquisition import NearestRefitResponse, expected_refit_losses
from cost_model.e2_residual import NearestShapePredictor
from cost_model.e2_srop_proxy import _active_profile, build_workload_graph, object_map_for_workload, evaluate_fixed_candidate_srop
from cost_model.e2_task_suite import active_task_contract
from cost_model.placement_probes import graph_wide_candidates
from simulator import make_reference_plan
from simulator.hardware import load_hardware


class RefitAcquisitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[2]
        directory = root / "data/bw1000_e2"
        cls.records = load_jsonl_records(directory / "profiling_dataset.jsonl")
        manifest = json.loads((directory / "shape_manifest.json").read_text(encoding="utf-8"))
        contract = json.loads((directory / "split_contract.json").read_text(encoding="utf-8"))
        suite = json.loads((directory / "e2c_task_suite.json").read_text(encoding="utf-8"))
        cls.task = next(t for t in suite["tasks"] if t["model_id"] == "qwen2.5-3b-s512-b1"
                        and t["scale"] == "layer" and t["initial_train_count"] == 16 and t["subset_seed"] == 1)
        cls.runtime = active_task_contract(cls.task, contract, manifest)
        cls.hardware = load_hardware(root / "data/bw1000_2card/bw1000_2card.calibrated.json")
        workload = workload_for_task(cls.task, manifest)
        cls.workload = workload
        cls.graph = build_workload_graph(workload, "layer")
        cls.mapping = object_map_for_workload(cls.records, workload)
        cls.pool = measurement_pool(cls.graph, cls.mapping, cls.task["train_object_ids"])
        cls.query = cls.mapping["mlp.gated_mul"]
        cls.profile, cls.uncertainty = _active_profile(cls.records, cls.runtime, cls.hardware, cls.mapping,
            split_id=cls.task["split_id"], revealed_object_ids=())
        cls.public = make_dataset_view(cls.records, cls.runtime, split_id=cls.task["split_id"],
            mode="active_calibration", role="sampler")

    def test_compiled_response_matches_literal_refit_for_all_unmeasured_rows(self):
        response = NearestRefitResponse(self.public, self.runtime["prediction_protocol"], self.query)
        self.assertTrue(response.affected_object_ids)
        for multiplier in (0.5, 1.0, 2.0):
            observed = self.profile.durations_s[self.query] * multiplier
            synthetic = copy.deepcopy(next(row for row in self.public if row["object_id"] == self.query))
            synthetic["target"] = {"log_residual": math.log(observed / synthetic["analytic"]["estimate_s"])}
            predictor = NearestShapePredictor().fit([row for row in self.public if "target" in row] + [synthetic],
                                                   self.runtime["prediction_protocol"])
            updated = response.profile(self.profile, observed)
            for row in self.public:
                key = row["object_id"]
                if "measurement" in row:
                    expected = row["measurement"]["duration_s"]
                elif key == self.query:
                    expected = observed
                else:
                    expected = row["analytic"]["estimate_s"] * math.exp(predictor.predict(row).mean)
                self.assertAlmostEqual(updated.durations_s[key], expected, delta=1e-15)

    def test_true_observation_response_matches_actual_predictor_update(self):
        response = NearestRefitResponse(self.public, self.runtime["prediction_protocol"], self.query)
        observed = next(row["measurement"]["duration_s"] for row in self.records if row["object_id"] == self.query)
        simulated = response.profile(self.profile, observed)
        actual, _ = _active_profile(self.records, self.runtime, self.hardware, self.mapping,
                                    split_id=self.task["split_id"], revealed_object_ids=(self.query,))
        for key in actual.durations_s:
            self.assertAlmostEqual(simulated.durations_s[key], actual.durations_s[key], delta=1e-15)

    def test_simulation_does_not_publish_or_mutate_synthetic_labels(self):
        before = copy.deepcopy(self.public)
        durations = dict(self.profile.durations_s)
        response = NearestRefitResponse(self.public, self.runtime["prediction_protocol"], self.query)
        response.profile(self.profile, self.profile.durations_s[self.query] * 2)
        self.assertEqual(self.public, before)
        self.assertEqual(self.profile.durations_s, durations)

    def test_unmasked_records_rejected(self):
        with self.assertRaisesRegex(ValueError, "unmeasured|masked"):
            NearestRefitResponse(self.records, self.runtime["prediction_protocol"], self.query)

    def test_hidden_labels_do_not_change_refit_acquisition(self):
        plans, _ = graph_wide_candidates(self.graph, [make_reference_plan(self.graph)], self.profile.device_classes)
        ranker = BatchedPlacementRanker(self.graph, plans, 2, ((0, 1), (1, 0)))
        baseline = ranker.rank(self.profile)
        altered = copy.deepcopy(self.records)
        for row in altered:
            if row["object_id"] not in self.task["train_object_ids"]:
                row["measurement"]["duration_s"] *= 100
                row["target"]["log_residual"] += math.log(100)
        public = make_dataset_view(altered, self.runtime, split_id=self.task["split_id"],
                                   mode="active_calibration", role="sampler")
        common = (self.profile, self.uncertainty)
        tail = (self.pool, self.runtime["prediction_protocol"], self.runtime["active_profiling_protocol"], ranker, baseline)
        self.assertEqual(expected_refit_losses(*common, self.public, *tail),
                         expected_refit_losses(*common, public, *tail))

    def test_unrevealed_labels_cannot_change_multistep_refit_selection(self):
        runtime = copy.deepcopy(self.runtime)
        runtime["active_profiling_protocol"]["policies"] = ["srop_refit"]
        reference = make_reference_plan(self.graph)
        split = copy.deepcopy(reference)
        for group in split.groups[len(split.groups)//2:]:
            group.device = 1
        common = (self.graph, [reference, split])
        first = evaluate_fixed_candidate_srop(*common, self.records, runtime, self.hardware, self.workload,
                                              ranker_factory=BatchedPlacementRanker)
        trajectory = first["policies"]["srop_refit"]["full_trajectories"][0]
        query = trajectory[-1]["selected_object_ids"][-1]
        altered = copy.deepcopy(self.records)
        for row in altered:
            if row["object_id"] == query:
                row["measurement"]["duration_s"] *= 20
                row["target"]["log_residual"] += math.log(20)
        second = evaluate_fixed_candidate_srop(*common, altered, runtime, self.hardware, self.workload,
                                               ranker_factory=BatchedPlacementRanker)
        changed = second["policies"]["srop_refit"]["full_trajectories"][0]
        self.assertGreater(len(trajectory), 2)
        self.assertEqual([(row["next_selection"], row["next_selection_score"]) for row in trajectory[:-1]],
                         [(row["next_selection"], row["next_selection_score"]) for row in changed[:-1]])


if __name__ == "__main__":
    unittest.main()
