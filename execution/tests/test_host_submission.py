from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from calibration.calibrate_reference_host import main, measure_probe
from calibration.e2_manifest import content_id
from calibration.e2_records import load_jsonl_records
from calibration.gpu_holdout import load_freeze
from cost_model.reference_semantics import HostSubmissionProfile
from execution.diagnose_gpu_holdout import expected_transfers
from execution.gpu_holdout import CanonicalReferenceExecutor
from execution.host_submission import SubmissionProbeExecutor, summarize_blocks, training_cases
from execution.verify_reference import _placed


def fake_probe(placed, devices, parameters, templates, reference, *, mode, environment_id):
    catalog, operations = {}, []
    copies = expected_transfers(placed)
    for group in placed.plan.groups:
        node_id = group.node_ids[0]
        for i, item in enumerate(copies):
            if item["destination_node"] == node_id:
                key = f"copy:{i}"
                catalog[key] = {"kind": "copy", **item}
                operations.append({"key": key})
        node = placed.graph.nodes[node_id]
        key = f"node:{node_id}"
        body = {"node_id": node_id, "shape": node.extra["output_shape"], "fixture_only": True}
        catalog[key] = {"kind": "node", "operator": node.op_type, "local_name": node.extra.get("local_name", node_id),
            "body_id": content_id("fixture-body", body), "body_key": body}
        operations.append({"key": key})
    for index, operation in enumerate(operations):
        operation.update(start_s=index * 1e-5, api_call_s=5e-6, dispatch_step_s=1e-5, bookkeeping_after_s=5e-6,
            gpu_stream_window_s=2e-5 if mode == "window_probe" and catalog[operation["key"]].get("operator") != "input" else None)
    samples = [{"total_s": .004, "start_s": 2e-6, "idle_sync_s": 3e-6,
        "busy_final_sync_s_not_a_host_parameter": .003, "operations": copy.deepcopy(operations)} for _ in range(parameters["iterations"])]
    return {"samples_s": [.004] * parameters["iterations"], "probe_samples": samples,
        "catalog": catalog, "correctness": {"pass": True}}


class HostSubmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.freeze, cls.manifest = load_freeze()

    def test_training_cases_use_only_the_112_old_training_keys(self):
        cases = training_cases(self.freeze, self.manifest)
        self.assertEqual(len(cases), 16)
        self.assertEqual(len({row["case_id"] for row in cases}), 16)
        self.assertEqual(len({row["model_id"] for row in cases}), 4)
        measured = {key for row in cases for key in row["node_training_object_ids"].values() if key}
        self.assertTrue(measured <= set(self.freeze["train_object_ids"]))
        self.assertFalse(measured & set(self.freeze["zero_shot_predictions"]))
        for sequence in (768, 1280, 1536, 2048):
            with self.assertRaisesRegex(ValueError, "restricted"):
                training_cases(self.freeze, self.manifest, (sequence,))

    def test_probe_loop_matches_canonical_outputs_and_uses_no_device_events_in_cpu_mode(self):
        placed, templates = _placed(), {}
        with torch.no_grad():
            expected, _, _, _ = CanonicalReferenceExecutor(placed, ("cpu", "cpu"), 1234, templates).run_once(False)
            probe = SubmissionProbeExecutor(placed, ("cpu", "cpu"), 1234, templates)
            probe.run_probe()  # catalog warmup, excluded from timing comparison
            with patch("torch.cuda.Event", side_effect=AssertionError("CPU-only probe created events")):
                output, sample = probe.run_probe()
        self.assertTrue(torch.equal(expected, output))
        self.assertEqual(len(sample["operations"]), len(placed.graph.nodes))
        for operation in sample["operations"]:
            self.assertGreaterEqual(operation["dispatch_step_s"], operation["api_call_s"])
            self.assertGreaterEqual(operation["bookkeeping_after_s"], 0)
            self.assertIsNone(operation["gpu_stream_window_s"])
        self.assertTrue(all(item["body_key"]["executor_source_sha256"] for item in probe.catalog.values()))

    def test_probe_measurement_writes_separate_api_steps_and_windows_without_cpu_gpu_claims(self):
        placed = _placed()
        for group in placed.plan.groups:
            group.device = 0
        templates = {}
        parameters = {"seed": 1234, "rtol": .01, "atol": .001, "warmup": 1, "iterations": 2}
        with torch.no_grad():
            expected, _, _, _ = CanonicalReferenceExecutor(placed, ("cpu", "cpu"), 1234, templates).run_once(False)
        result = measure_probe(placed, ("cpu", "cpu"), parameters, templates, expected,
            mode="cpu_probe", environment_id="cpu-fixture")
        self.assertTrue(result["correctness"]["pass"])
        self.assertEqual(len(result["probe_samples"]), 2)
        self.assertEqual(len(result["catalog"]), 4)

    def test_summary_keeps_busy_gpu_wait_out_of_host_parameters_and_separates_case_windows(self):
        placed = _placed()
        blocks = []
        for case in ("a", "b"):
            for mode in ("baseline_before", "cpu_probe", "window_probe", "baseline_after"):
                result = fake_probe(placed, None, {"iterations": 2}, None, None, mode=mode, environment_id="fixture")
                blocks.append({"case_id": case, "model_id": "fixture", "plan_name": "placed", "mode": mode, **result})
        summary = summarize_blocks(blocks)
        host = HostSubmissionProfile(**summary["host_parameters"])
        self.assertTrue(host.calibrated)
        self.assertEqual(host.start_s, 2e-6)
        self.assertEqual(host.finish_s, 3e-6)  # not the .003 busy wait
        self.assertEqual(host.node_cost("matmul", "matmul"), 1e-5)
        self.assertEqual({row["case_id"] for row in summary["body_window_candidates"]}, {"a", "b"})
        # Failing observer-overhead checks retains candidate values but prevents
        # a calibrated profile; no favorable-case filtering.
        bad = copy.deepcopy(blocks)
        bad[1]["samples_s"] = [.008, .008]
        self.assertFalse(summarize_blocks(bad)["host_parameters"]["calibrated"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize_blocks(blocks + blocks[:1])

    def test_gpu_entry_cannot_fall_back_to_cpu(self):
        with TemporaryDirectory() as temp, patch("calibration.calibrate_reference_host.load_freeze", return_value=({}, {})), \
                patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "no HIP/CUDA"):
                main(["--out-dir", temp])

    def test_real_command_persists_all_64_blocks_and_exports_a_compatible_host_profile(self):
        with TemporaryDirectory() as temp, redirect_stdout(io.StringIO()), \
                patch("calibration.calibrate_reference_host._validate_runtime"), \
                patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(name="BW", total_memory=1024)), \
                patch("torch.cuda.can_device_access_peer", return_value=True), \
                patch("calibration.calibrate_reference_host.CanonicalReferenceExecutor") as executor, \
                patch("calibration.calibrate_reference_host.measure_candidate", return_value={"samples_s": [.004] * 20, "correctness": {"pass": True}}), \
                patch("calibration.calibrate_reference_host.measure_probe", side_effect=fake_probe):
            executor.return_value.run_once.return_value = (torch.ones(1), [], [], .001)
            output = Path(temp) / "host"
            main(["--out-dir", str(output)])
            records = load_jsonl_records(output / "blocks.jsonl")
            report = json.loads((output / "calibration_report.json").read_text())
            profile = HostSubmissionProfile(**json.loads((output / "host_submission_profile.json").read_text()))
            self.assertEqual(len(records), 64)
            self.assertEqual(len({row["record_id"] for row in records}), 64)
            self.assertEqual(sum(len(row["samples_s"]) for row in records), 1280)
            for row in records:
                self.assertEqual(row["record_id"], content_id("reference-host-calibration-block", {key: value for key, value in row.items() if key != "record_id"}))
            self.assertTrue(profile.calibrated)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(len(report["cases"]), 16)
            self.assertTrue(report["all_window_data_complete"])
            self.assertTrue((output / "body_window_candidates.json").exists())


if __name__ == "__main__":
    unittest.main()
