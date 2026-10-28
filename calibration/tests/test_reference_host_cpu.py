"""CPU orchestration checks; measured GPU calibration remains server-side."""

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

from calibration.calibrate_reference_host_cpu import audit_collection, build_protocol, main, summarize_cpu_blocks
from calibration.e2_records import load_jsonl_records
from calibration.gpu_holdout import load_freeze
from execution.tests.test_host_submission import fake_probe


def coherent_fake_probe(*args, **kwargs):
    if kwargs["mode"] != "cpu_probe":
        raise AssertionError("the CPU protocol requested device window measurements")
    row = fake_probe(*args, **kwargs)
    for sample in row["probe_samples"]:
        sample["busy_final_sync_s_not_a_host_parameter"] = sample["total_s"] - sample["start_s"] - sum(
            item["dispatch_step_s"] for item in sample["operations"])
    return row


class ReferenceHostCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        freeze, manifest = load_freeze()
        cls.protocol = build_protocol(freeze, manifest)

    def test_three_rounds_cover_every_old_case_in_a_fixed_order(self):
        p = self.protocol
        self.assertEqual(len(p["cases"]), 16)
        self.assertEqual(len(p["schedule"]), 144)
        orders = []
        for repeat in range(3):
            rows = [x for x in p["schedule"] if x["repeat"] == repeat]
            self.assertEqual(len(rows), 48)
            self.assertEqual({x["case_id"] for x in rows}, {x["case_id"] for x in p["cases"]})
            for index in range(0, 48, 3):
                self.assertEqual([x["mode"] for x in rows[index:index + 3]],
                    ["baseline_before", "cpu_probe", "baseline_after"])
            orders.append([x["case_id"] for x in rows if x["mode"] == "cpu_probe"])
        self.assertEqual(len({tuple(x) for x in orders}), 3)

    def test_real_journal_and_audit_cover_144_blocks_without_events_or_cpu_fallback(self):
        with TemporaryDirectory() as temp, redirect_stdout(io.StringIO()), \
                patch("calibration.calibrate_reference_host_cpu._validate_runtime"), \
                patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(name="BW", total_memory=1024)), \
                patch("torch.cuda.can_device_access_peer", return_value=True), \
                patch("torch.cuda.Event", side_effect=AssertionError("CPU protocol created device events")), \
                patch("calibration.calibrate_reference_host_cpu.CanonicalReferenceExecutor") as executor, \
                patch("calibration.calibrate_reference_host_cpu.measure_candidate", return_value={
                    "samples_s": [.004] * 20, "median_s": .004, "correctness": {"pass": True}}), \
                patch("calibration.calibrate_reference_host_cpu.measure_probe", side_effect=coherent_fake_probe):
            executor.return_value.run_once.return_value = (torch.ones(1), [], [], .001)
            directory = Path(temp) / "cpu"
            main(["--out-dir", str(directory)])
            audit = audit_collection(directory)
            self.assertEqual((audit["records"], audit["measured_forwards"]), (144, 2880))
            self.assertTrue(audit["profile_accepted"])
            rows = load_jsonl_records(directory / "blocks.jsonl")
            self.assertEqual(len({x["record_id"] for x in rows}), 144)
            profile = json.loads((directory / "host_submission_profile.json").read_text())
            self.assertEqual(profile["finish_s"], 3e-6)  # No busy final GPU tail.
            self.assertFalse((directory / "body_window_candidates.json").exists())
            main(["--audit-dir", str(directory)])

            # A single failed case-round is retained and prevents acceptance,
            # even though two other rounds of this case are clean.
            changed = copy.deepcopy(rows)
            bad = next(row for row in changed if row["mode"] == "baseline_after")
            bad["samples_s"] = [.0045] * 20
            summary = summarize_cpu_blocks(self.protocol, changed)
            self.assertEqual(summary["case_round_targets_pass"], 47)
            self.assertEqual(summary["case_targets_pass"], 15)
            self.assertFalse(summary["host_parameters"]["calibrated"])
            self.assertEqual(summary["host_parameters"]["finish_s"], 3e-6)
            with self.assertRaisesRegex(ValueError, "reordered or duplicate"):
                summarize_cpu_blocks(self.protocol, changed[:-1])

            # Audit rejects both timing corruption and a profile promoted to
            # calibrated=true independently of its raw acceptance evidence.
            host_path = directory / "host_submission_profile.json"
            profile["copy_submit_s"] *= 2
            host_path.write_text(json.dumps(profile))
            with self.assertRaisesRegex(ValueError, "host export mismatch"):
                audit_collection(directory)
            journal = directory / "blocks.jsonl"
            lines = journal.read_text().splitlines()
            record = json.loads(lines[0])
            record["samples_s"][0] *= 2
            lines[0] = json.dumps(record)
            journal.write_text("\n".join(lines) + "\n")
            with self.assertRaisesRegex(ValueError, "record hash mismatch"):
                audit_collection(directory)

    def test_gpu_entry_refuses_a_cpu_only_runtime(self):
        with TemporaryDirectory() as temp, patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "no HIP/CUDA"):
                main(["--out-dir", temp])

    def test_runtime_failure_preserves_completed_blocks(self):
        with TemporaryDirectory() as temp, redirect_stdout(io.StringIO()), \
                patch("calibration.calibrate_reference_host_cpu._validate_runtime"), \
                patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(name="BW", total_memory=1024)), \
                patch("torch.cuda.can_device_access_peer", return_value=True), \
                patch("calibration.calibrate_reference_host_cpu.CanonicalReferenceExecutor") as executor, \
                patch("calibration.calibrate_reference_host_cpu.measure_candidate", return_value={
                    "samples_s": [.004] * 20, "median_s": .004, "correctness": {"pass": True}}), \
                patch("calibration.calibrate_reference_host_cpu.measure_probe", side_effect=RuntimeError("fixture GPU failure")):
            executor.return_value.run_once.return_value = (torch.ones(1), [], [], .001)
            directory = Path(temp) / "partial"
            with self.assertRaisesRegex(RuntimeError, "fixture GPU failure"):
                main(["--out-dir", str(directory)])
            self.assertEqual(len(load_jsonl_records(directory / "blocks.jsonl")), 1)
            status = json.loads((directory / "collection_status.json").read_text())
            self.assertEqual((status["status"], status["records"]), ("incomplete", 1))


if __name__ == "__main__":
    unittest.main()
