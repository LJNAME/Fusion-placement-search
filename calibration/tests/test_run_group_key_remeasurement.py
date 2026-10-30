"""Workflow control tests use mocked subprocesses and acquire no GPU labels."""

import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from calibration.run_group_key_remeasurement import run_workflow


class GroupKeyWorkflowTests(unittest.TestCase):
    def test_complete_sequence_and_explicit_profile_export(self):
        with TemporaryDirectory() as directory:
            out = Path(directory) / "run"
            with patch("calibration.run_group_key_remeasurement.subprocess.run", return_value=SimpleNamespace(returncode=0)) as run, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_workflow(out, Path(directory) / "FlagScale"), 0)
            self.assertEqual(run.call_count, 5)
            commands = [c.args[0] for c in run.call_args_list]
            self.assertEqual([c[3] for c in commands], ["check", "collect", "verify", "collect", "verify"])
            self.assertIn("smoke", commands[1])
            self.assertIn("formal", commands[3])
            self.assertEqual(commands[-1][-2:], ["--out", str(out / "formal/verified_profile.json")])
            summary = json.loads((out / "workflow_summary.json").read_text())
            self.assertTrue(summary["complete"])
            self.assertFalse(summary["automatic_profile_writeback"])

    def test_smoke_failure_prevents_formal_measurement(self):
        with TemporaryDirectory() as directory:
            out = Path(directory) / "run"
            with patch("calibration.run_group_key_remeasurement.subprocess.run",
                       side_effect=[SimpleNamespace(returncode=0), SimpleNamespace(returncode=1)]) as run, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_workflow(out, Path(directory)), 1)
            self.assertEqual(run.call_count, 2)
            self.assertFalse((out / "formal_collect.log").exists())
            summary = json.loads((out / "workflow_summary.json").read_text())
            self.assertFalse(summary["complete"])
            self.assertEqual(summary["steps"][-1]["name"], "smoke_collect")

    def test_existing_output_refused_before_subprocess(self):
        with TemporaryDirectory() as directory:
            with patch("calibration.run_group_key_remeasurement.subprocess.run") as run:
                with self.assertRaises(FileExistsError):
                    run_workflow(Path(directory), Path(directory))
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
