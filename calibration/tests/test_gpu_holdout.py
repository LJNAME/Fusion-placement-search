import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from calibration.e2_manifest import content_id
from calibration.e2_records import build_environment_record, build_raw_measurement_record
from calibration.evaluate_gpu_holdout import evaluate_measurements
from calibration.gpu_holdout import build_freeze, load_freeze, file_hashes, SOURCES


class GPUHoldoutFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.freeze = build_freeze(iterations=4, seeds=(0,))

    def test_known_labels_are_disjoint_and_features_have_no_targets(self):
        fresh = {row["object_id"] for row in self.freeze["query_features"]}
        self.assertFalse(fresh & set(self.freeze["train_object_ids"]))
        self.assertEqual(len(self.freeze["train_object_ids"]), 112)
        self.assertEqual(len(self.freeze["workloads"]), 4)
        for row in self.freeze["query_features"]:
            self.assertNotIn("measurement", row)
            self.assertNotIn("target", row)
        for workload in self.freeze["workloads"]:
            self.assertIn(workload["predicted_top1_id"], workload["execution_plan_ids"])
            self.assertIn(workload["reference_plan_id"], workload["execution_plan_ids"])
            self.assertTrue(any(len({group["device"] for group in workload["candidate_plans"][key]["groups"]}) == 2
                                for key in workload["execution_plan_ids"]))
            self.assertLessEqual(len(workload["execution_plan_ids"]), 6)

    def test_bundle_tampering_and_labels_in_feature_store_are_rejected(self):
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory/"manifest.json").write_text(json.dumps(self.manifest))
            (directory/"freeze.json").write_text(json.dumps(self.freeze))
            self.assertEqual(load_freeze(directory)[0]["freeze_id"], self.freeze["freeze_id"])
            altered = copy.deepcopy(self.freeze)
            key = next(iter(altered["zero_shot_predictions"]))
            altered["zero_shot_predictions"][key]["nearest_s"] *= 10
            (directory/"freeze.json").write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, "changed"):
                load_freeze(directory)
            altered = copy.deepcopy(self.freeze)
            altered["query_features"][0]["measurement"] = {"duration_s": 1}
            altered["freeze_id"] = content_id("gpu-holdout-freeze", {k: v for k, v in altered.items() if k != "freeze_id"})
            (directory/"freeze.json").write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, "must not contain labels"):
                load_freeze(directory)

    def test_new_measurements_evaluate_sealed_predictions_without_mutating_them(self):
        environment = build_environment_record(hardware_class="BW1000", device_name="BW",
            total_memory_bytes=10**10, framework_version="2.9.0", accelerator_version="6.3.26093",
            python_version="fixture", platform="fixture", source_revision="cpu-test-only")
        records = []
        for key, prediction in self.freeze["zero_shot_predictions"].items():
            for device in (0, 1):
                records.append(build_raw_measurement_record(object_id=key, environment_id=environment["environment_id"],
                    physical_device_index=device, samples_s=[prediction["nearest_s"]*1.5]*7,
                    warmup=10, iterations_per_sample=50, run_id="cpu-schema-fixture", measured_at="fixture", profiling_wall_time_s=.1))
        before = copy.deepcopy(self.freeze)
        report = evaluate_measurements(self.freeze, self.manifest, records, {environment["environment_id"]: environment})
        self.assertEqual(self.freeze, before)
        self.assertAlmostEqual(report["all"]["nearest"]["mape"], 1/3)
        self.assertEqual(report["all"]["nearest"]["objects"], len(self.freeze["query_features"]))
        records[0]["timing"]["warmup"] = 2
        records[0]["record_id"] = content_id("raw-measurement", {k: v for k, v in records[0].items() if k != "record_id"})
        with self.assertRaisesRegex(ValueError, "quick/changed"):
            evaluate_measurements(self.freeze, self.manifest, records, {environment["environment_id"]: environment})
