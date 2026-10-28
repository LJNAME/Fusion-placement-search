import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from calibration.e2_manifest import content_id
from calibration.e2_records import build_environment_record, build_raw_measurement_record
from calibration.evaluate_gpu_holdout import evaluate_measurements
from calibration.reference_validation import build_freeze, check
from execution.evaluate_reference_validation import evaluate
from execution.gpu_holdout import summarize_execution
from execution.audit_gpu_holdout import write_json
from simulator import PlacedOpGraph
from simulator.dataset import graph_from_dict, plan_from_dict
from simulator.hardware import hardware_from_dict


class ReferenceValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.freeze = build_freeze(iterations=4, seeds=(0,))

    def test_layer_only_pool_and_full_key_vs_feature_novelty(self):
        f, m = self.freeze, self.manifest
        self.assertEqual((len(m["objects"]), len(f["query_features"])), (56, 52))
        self.assertEqual(f["feature_alias_audit"]["feature_novel_objects"], 49)
        self.assertEqual(len(f["feature_alias_audit"]["aliases"]), 3)
        self.assertFalse({r["object_id"] for r in f["query_features"]} & set(f["exposed_object_ids"]))
        allowed = {(w["model_id"], n["extra"]["local_name"]) for w in f["workloads"]
                   for n in w["graph"]["nodes"] if n["op_type"] != "input"}
        actual = {(u["model_id"], u["local_name"]) for r in f["query_features"] for u in r["usages"]}
        self.assertEqual(actual, allowed)
        self.assertTrue(all(u["occurrence_count"] == 1 for r in m["objects"] for u in r["usages"]))
        self.assertTrue(all("measurement" not in r and "target" not in r for r in f["query_features"]))

    def test_all_predicted_ties_and_physical_directions_are_scheduled(self):
        for w in self.freeze["workloads"]:
            measured = set(w["execution_plan_ids"])
            self.assertIn(w["reference_plan_id"], measured)
            for name, ties in w["method_top1_sets"].items():
                self.assertTrue(set(ties) <= measured)
                self.assertEqual(w["method_selected_ids"][name], min(ties))
            for key in measured:
                class_id = w["physical_to_class"][key]
                self.assertEqual(sum(c == class_id for p, c in w["physical_to_class"].items() if p in measured), 2)
        self.assertFalse(self.freeze["host_profile"]["calibrated"])
        self.assertFalse(self.freeze["protocol"]["epsilon_evaluation"]["stop_rule_enabled"])

    def test_tampering_and_rehashed_query_labels_are_rejected(self):
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            write_json(directory / "manifest.json", self.manifest)
            write_json(directory / "freeze.json", self.freeze)
            self.assertEqual(check(directory)[0]["freeze_id"], self.freeze["freeze_id"])
            changed = copy.deepcopy(self.freeze)
            changed["query_features"][0]["target"] = {"log_residual": 0.}
            changed["freeze_id"] = content_id("gpu-holdout-freeze", {k: v for k, v in changed.items() if k != "freeze_id"})
            write_json(directory / "freeze.json", changed)
            with self.assertRaisesRegex(ValueError, "must not contain labels"):
                check(directory)

    def test_epsilon_evaluator_preserves_preselected_physical_choice_and_alias_split(self):
        f, hardware, records = self.freeze, hardware_from_dict(self.freeze["hardware"]), []
        environment = {"host": "synthetic-CPU-schema-fixture-not-a-GPU-measurement"}
        environment_id = content_id("gpu-holdout-execution-environment", environment)
        for w in f["workloads"]:
            graph = graph_from_dict(w["graph"])
            for key in w["execution_plan_ids"]:
                placed = PlacedOpGraph(graph, hardware, plan_from_dict(w["candidate_plans"][key]))
                duration = .006 if key == w["method_selected_ids"]["reference_host"] else .004
                for repeat in range(2):
                    row = {"freeze_id": f["freeze_id"], "environment_id": environment_id,
                        "model_id": w["model_id"], "plan_id": key, "repeat": repeat,
                        "artifact_candidate_id": placed.candidate_id, "status": "ok", "median_s": duration,
                        "samples_s": [duration] * 20, "correctness": {"pass": True, "max_abs_error": 0.}, "peak_memory_bytes": [1, 1]}
                    row["record_id"] = content_id("gpu-holdout-execution-sample", row)
                    records.append(row)
        original = summarize_execution(f, records)
        original.update(environment=environment, environment_id=environment_id, execution_status="complete")
        kernel_env = build_environment_record(hardware_class="BW1000", device_name="BW", total_memory_bytes=10**10,
            framework_version="2.9.0", accelerator_version="6.3.26093", python_version="fixture", platform="fixture", source_revision="fixture")
        raw = [build_raw_measurement_record(object_id=key, environment_id=kernel_env["environment_id"],
                physical_device_index=device, samples_s=[pred["nearest_s"] * 1.5] * 7,
                warmup=10, iterations_per_sample=50, run_id="CPU-schema-only", measured_at="fixture", profiling_wall_time_s=.1)
               for key, pred in f["zero_shot_predictions"].items() for device in (0, 1)]
        kernel = evaluate_measurements(f, self.manifest, raw, {kernel_env["environment_id"]: kernel_env})
        before = copy.deepcopy(f)
        report = evaluate(f, records, environment, original, kernel)
        self.assertEqual(f, before)
        self.assertFalse(report["all_primary_within_1pct"])
        self.assertEqual(report["kernel_groups"]["all"]["nearest"]["objects"], 52)
        self.assertEqual(report["kernel_groups"]["feature_novel"]["nearest"]["objects"], 49)
        for row in report["workloads"]:
            primary = row["methods"]["reference_host"]
            self.assertAlmostEqual(primary["relative_regret"], .5)
            self.assertEqual(primary["tie_regret_range"], [0., .5])
            self.assertEqual(primary["selected_class_best_member_regret_diagnostic"], 0.)


if __name__ == "__main__":
    unittest.main()
