"""Regression checks for the received GPU journal and its acceptance boundary."""

import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

from execution.audit_reference_host import audit


RECEIVED = Path(__file__).resolve().parents[2] / "data/bw1000_reference_host_train"


class ReferenceHostAuditTests(unittest.TestCase):
    def test_complete_gpu_records_do_not_override_failed_acceptance(self):
        result = audit(RECEIVED)
        self.assertEqual((result["records"], result["measured_forwards"]), (64, 1280))
        self.assertEqual((result["cpu_targets_pass"], result["window_targets_pass"]), (15, 0))
        self.assertFalse(result["profile_accepted"])
        self.assertEqual(len(result["training_reconstruction"]), 4)

    def test_corrupt_raw_timing_is_rejected_before_summary(self):
        with TemporaryDirectory() as temp:
            directory = Path(temp) / "received"
            shutil.copytree(RECEIVED, directory)
            path = directory / "blocks.jsonl"
            rows = path.read_text().splitlines()
            row = json.loads(rows[0])
            row["samples_s"][0] *= 2
            rows[0] = json.dumps(row)
            path.write_text("\n".join(rows) + "\n")
            with self.assertRaisesRegex(ValueError, "record hash"):
                audit(directory)

    def test_cannot_silently_promote_rejected_profile_or_drop_windows(self):
        for filename, field, value, message in (
            ("host_submission_profile.json", "calibrated", True, "host export"),
            ("body_window_candidates.json", "windows", [], "window export"),
        ):
            with self.subTest(filename=filename), TemporaryDirectory() as temp:
                directory = Path(temp) / "received"
                shutil.copytree(RECEIVED, directory)
                path = directory / filename
                row = json.loads(path.read_text())
                row[field] = value
                path.write_text(json.dumps(row))
                with self.assertRaisesRegex(ValueError, message):
                    audit(directory)


if __name__ == "__main__":
    unittest.main()
