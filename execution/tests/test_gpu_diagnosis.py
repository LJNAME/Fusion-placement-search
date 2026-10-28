import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from calibration.e2_records import load_jsonl_records
from calibration.e2_manifest import content_id
from calibration.gpu_holdout import DEFAULT_BUNDLE, load_freeze
from execution.audit_gpu_holdout import audit_execution, representative_plans, timing_statistics
from execution.diagnose_gpu_holdout import CopyRecorder, capture_trace, main, prediction_breakdown
from execution.gpu_holdout import CanonicalReferenceExecutor
from execution.verify_reference import _placed


class GPUExecutionDiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.freeze, _ = load_freeze()
        base = DEFAULT_BUNDLE / "results_received_20261028"
        cls.records = load_jsonl_records(base / "execution_samples.jsonl")
        cls.environment = json.loads((base / "execution_environment.json").read_text())
        cls.original = json.loads((base / "execution_report.json").read_text())

    def test_raw_journal_reproduces_all_results_and_block_level_claims(self):
        result = audit_execution(self.freeze, self.records, self.environment, self.original)
        self.assertEqual(result["records"], 44)
        self.assertEqual(result["timing_samples"], 880)
        self.assertTrue(all(row["reference_is_best_in_both_rounds"] for row in result["workloads"]))
        self.assertEqual([row["complete_ranking_same_in_both_rounds"] for row in result["workloads"]], [True, True, False, False])

    def test_modified_raw_sample_is_detected_before_statistics(self):
        modified = copy.deepcopy(self.records)
        modified[0]["samples_s"][0] *= 2
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            audit_execution(self.freeze, modified, self.environment, self.original)

    def test_duplicate_execution_block_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate execution slots"):
            audit_execution(self.freeze, self.records + self.records[:1], self.environment, self.original)

    def test_long_tail_is_retained_and_not_mistaken_for_median(self):
        result = timing_statistics([.001] * 19 + [.066])
        self.assertEqual(result["median_ms"], 1)
        self.assertEqual(result["max_ms"], 66)
        self.assertEqual(result["samples_above_twice_median"], 1)
        self.assertGreater(result["mean_ms"], 4)

    def test_traced_executor_keeps_reference_outputs_and_exports_cpu_trace(self):
        placed = _placed()
        templates = {}
        parameters = {"seed": 1234, "rtol": .01, "atol": .001, "warmup": 1}
        with torch.no_grad():
            expected, _, _, _ = CanonicalReferenceExecutor(placed, ("cpu", "cpu"), 1234, templates).run_once(False)
        # Only test annotation/output behavior here, never claim CPU timings validate GPUs.
        with TemporaryDirectory() as temp, patch("torch.profiler.supported_activities", return_value={torch.profiler.ProfilerActivity.CPU}):
            path = Path(temp) / "fixture.trace.json"
            result = capture_trace(placed, ("cpu", "cpu"), parameters, templates, expected, path)
            self.assertTrue(result["correctness"]["pass"])
            self.assertEqual(result["correctness"]["max_abs_error"], 0)
            self.assertEqual(len(result["node_cpu_scopes"]), len(placed.graph.nodes))
            self.assertFalse(result["gpu_timeline_available"])
            self.assertTrue(json.loads(path.read_text(encoding="utf-8"))["traceEvents"])

    def test_copy_recorder_distinguishes_cast_from_cross_gpu_transfer(self):
        source = torch.ones((2, 3), dtype=torch.float32)
        recorder = CopyRecorder()
        with recorder:
            converted = source.to(torch.float64)
        self.assertEqual(len(recorder.rows), 1)
        row = recorder.rows[0]
        self.assertFalse(row["cross_gpu"])
        self.assertEqual(row["source_payload_bytes"], 24)
        self.assertEqual(row["destination_payload_bytes"], 48)
        self.assertTrue(torch.equal(converted, source.double()))

    def test_representatives_are_existing_measured_plans(self):
        selection = representative_plans(self.freeze, self.original, "qwen2.5-3b-s2048-b1")
        self.assertEqual([row["role"] for row in selection["plans"]], ["single_gpu_reference", "predicted_top1", "intermediate"])
        self.assertTrue(selection["plans"][2]["plan_id"].endswith("b3edd2d5"))

    def test_schedule_and_copy_sizes_reproduce_the_frozen_prediction(self):
        selection = representative_plans(self.freeze, self.original, "qwen2.5-3b-s2048-b1")
        breakdown = prediction_breakdown(self.freeze, selection)
        by_role = {row["role"]: row for row in breakdown["plans"]}
        expected_bytes = {"single_gpu_reference": 0, "predicted_top1": 51 * 1024**2, "intermediate": 2 * 1024**2}
        for role, byte_count in expected_bytes.items():
            row = by_role[role]
            self.assertEqual(sum(copy["payload_bytes"] for copy in row["reference_loop_expected_transfers"]), byte_count)
            self.assertEqual(row["frozen_prediction"]["cross_device_bytes"], byte_count)

    def test_gpu_diagnostic_command_has_no_cpu_fallback(self):
        with TemporaryDirectory() as temp, patch("execution.diagnose_gpu_holdout.load_freeze", return_value=({}, {})), patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "no HIP/CUDA"):
                main(["--out-dir", temp])

    def test_command_persists_all_nine_blocks_with_real_jsonl_writer(self):
        # Mock only GPU work. Exercise real command orchestration, hashes,
        # JSONL append, summary generation and report output together.
        measurement = {"samples_s": [.004] * 20, "median_s": .004,
            "correctness": {"pass": True, "max_abs_error": 0.0}, "peak_memory_bytes": [1, 1]}
        with TemporaryDirectory() as temp, \
                patch("execution.diagnose_gpu_holdout._validate_runtime"), \
                patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(name="BW", total_memory=1024)), \
                patch("torch.cuda.can_device_access_peer", return_value=True), \
                patch("execution.diagnose_gpu_holdout.CanonicalReferenceExecutor") as executor, \
                patch("execution.diagnose_gpu_holdout.measure_candidate", return_value=measurement), \
                patch("execution.diagnose_gpu_holdout.capture_trace", side_effect=lambda *args:
                    {"trace_file": args[-1].name, "gpu_timeline_available": False, "correctness": {"pass": True}}):
            executor.return_value.run_once.return_value = (torch.ones(1), [], [], .001)
            output = Path(temp) / "diagnosis"
            main(["--out-dir", str(output)])
            rows = load_jsonl_records(output / "baseline_samples.jsonl")
            report = json.loads((output / "diagnostic_report.json").read_text())
            environment = json.loads((output / "diagnostic_environment.json").read_text())
            self.assertEqual(len(rows), 9)
            self.assertEqual(len({row["record_id"] for row in rows}), 9)
            self.assertEqual({(row["role"], row["repeat"]) for row in rows},
                {(role, repeat) for role in ("single_gpu_reference", "predicted_top1", "intermediate") for repeat in range(3)})
            expected_environment_id = content_id("gpu-three-plan-diagnostic-environment", environment)
            for row in rows:
                self.assertEqual(row["freeze_id"], self.freeze["freeze_id"])
                self.assertEqual(row["environment_id"], expected_environment_id)
                self.assertEqual(row["record_id"], content_id("gpu-three-plan-diagnostic-baseline",
                    {key: value for key, value in row.items() if key != "record_id"}))
            self.assertEqual(sum(len(row["samples_s"]) for row in rows), 180)
            self.assertEqual(report["environment_id"], expected_environment_id)
            self.assertEqual(report["status"], "baselines_complete_traces_partial")
            self.assertEqual(len(report["traces"]), 3)
            self.assertTrue(all(row["pooled"]["samples"] == 60 for row in report["baselines"]))


if __name__ == "__main__":
    unittest.main()
