"""CPU process-failure checks; fixture durations are never GPU evidence."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from calibration.ffn_b4 import plan, save
from calibration.ffn_b4_supervisor import supervise
from calibration.ffn_b4_policy import POLICY


def test_worker_hard_exit_preserves_19_cells_and_continues_without_retry():
    frozen = plan()
    calls = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        def run(command, **kwargs):
            group_id = command[command.index("--group")+1]
            calls.append(group_id)
            if group_id == "link_quartet":
                # A real OS-level worker exit, bypassing Python exception cleanup.
                return subprocess.run([sys.executable, "-c", "import os; print('fatal fixture', flush=True); os._exit(37)"], **kwargs)
            group = next(g for g in frozen["matrix"]["round_groups"] if g["group_id"] == group_id)
            worker_root = Path(command[command.index("--worker")+1])
            save(worker_root/"formal/grid.json", {"cells": {
                row: {"status": "quality_passed", "fixture_only": True} for row in group["members"]}})
            return subprocess.CompletedProcess(command, 0)
        result = supervise(frozen, root, Path("unused-fixture-root"), run_worker=run)
        assert result["complete"] and len(result["cells"]) == 19
        assert result["quality_failed"] == 4
        assert len(calls) == len(set(calls)) == 10
        assert calls[-1] == "copy/1572864/1->0"
        assert "fatal fixture" in (root/"groups/link_quartet/worker.log").read_text()
        assert all(result["cells"][r]["worker_returncode"] == 37 for r in
                   next(g for g in frozen["matrix"]["round_groups"] if g["group_id"] == "link_quartet")["members"])
        assert json.loads((root/"grid.json").read_text())["recorded_cells"] == 19


def test_missing_worker_grid_fails_its_group_without_losing_later_groups():
    frozen = plan()
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)
    with tempfile.TemporaryDirectory() as directory:
        result = supervise(frozen, directory, "unused-fixture-root", run_worker=run)
    assert len(calls) == 10 and result["recorded_cells"] == 19
    assert result["quality_failed"] == 19 and result["retries"] == 0
    assert not result["automatic_admission"]


def test_worker_entry_checks_parent_smoke_and_runs_only_requested_group():
    from calibration.ffn_b4_supervisor import main
    frozen = plan()
    runtime = SimpleNamespace(identity_record={"fixture_only": True})
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        worker = root/"formal/groups/link_quartet"
        save(worker/"parent_plan.json", frozen)
        save(root/"smoke/result.json", {"status": "passed", "cells": [
            {flag: True for flag in POLICY["smoke"]["required_booleans"]} for _ in range(4)]})
        with patch("sys.argv", ["worker", "--worker", str(worker), "--group", "link_quartet", "--flagscale-root", "fixture-root"]), \
             patch("calibration.ffn_b4.plan", return_value=frozen), \
             patch("calibration.ffn_b4_runtime.Runtime", return_value=runtime), \
             patch("calibration.ffn_b4.execute_formal") as execute:
            main()
        execute.assert_called_once_with(runtime, frozen, worker/"formal", group_id="link_quartet")
