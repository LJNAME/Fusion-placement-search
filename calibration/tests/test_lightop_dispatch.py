"""CPU orchestration fixtures only; never dispatch or admission evidence."""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from calibration import lightop_dispatch as dispatch
from tools import review_lightop_dispatch as audit


def test_plan_is_cpu_only_and_separates_trace_from_latency(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("plan used GPU"))
    assert dispatch.plan() == dispatch.plan()
    frozen = dispatch.plan()
    assert frozen["target_replays"] == 1 and frozen["calls_per_replay"] == 50
    assert frozen["marker_copy_bytes"] == [257, 263]
    assert not frozen["timing_labels_emitted"] and not frozen["automatic_admission"]
    assert frozen["collector"]["source_sha256"]
    dispatch.main(["plan", "--output", str(tmp_path)])
    assert json.loads((tmp_path / "plan.json").read_text()) == frozen
    with pytest.raises(SystemExit):
        dispatch.main(["plan", "--output", str(tmp_path)])


def test_markers_surround_exactly_one_graph_replay_without_other_work():
    events = []
    @contextmanager
    def stream(value):
        events.append("stream_start")
        yield
        events.append("stream_end")
    fake = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda _: events.append("sync"), stream=stream))
    markers = [SimpleNamespace(copy_=lambda host, non_blocking: events.append(("copy", host, non_blocking))) for _ in range(2)]
    graph = SimpleNamespace(replay=lambda: events.append("graph_replay"))
    dispatch.marked_replay(fake, graph, "fixture-stream", "cuda:0", [257, 263], markers)
    assert events == ["sync", "stream_start", ("copy", 257, True), "sync", "graph_replay", "sync",
                      ("copy", 263, True), "sync", "stream_end"]


def test_worker_cpu_failure_preserves_context_and_pool(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    with pytest.raises(RuntimeError, match="HIP GPU0 required"):
        dispatch.main(["worker", "--output", str(tmp_path)])
    result = json.loads((tmp_path / "failure.json").read_text())
    assert result["context"]["stage"] == "runtime_guard" and result["traceback"]
    assert result["pool"] == dispatch.POOL and not result["timing_labels_emitted"]


@pytest.mark.parametrize("failure", ["missing_profiler", "worker_crash", "trace_incomplete", "none"])
def test_parent_runs_one_profile_process_and_never_runs_timing(tmp_path, monkeypatch, failure):
    profiler = tmp_path / "fixture-hipprof"
    profiler.write_bytes(b"CPU test only")
    monkeypatch.setattr(dispatch.shutil, "which", lambda _: None if failure == "missing_profiler" else str(profiler))
    calls = []
    def command(args, **kwargs):
        calls.append(args)
        assert args[:2] == [str(profiler), "--hip-trace"]
        assert args[2] == "/fixture/launcher" and "worker" in args
        assert "benchmark" not in args and "run" not in args
        assert kwargs["check"] is False and kwargs["cwd"].name == "trace"
        return subprocess.CompletedProcess(args, -11 if failure == "worker_crash" else 0)
    def review(directory):
        if failure == "trace_incomplete":
            raise ValueError("fixture trace missing")
        return {"dispatch_observed": True, "scope": "CPU test fixture"}, []
    monkeypatch.setattr(audit, "review", review)
    output = tmp_path / "output"
    output.mkdir()
    if failure == "missing_profiler":
        with pytest.raises(RuntimeError, match="hipprof not found"):
            dispatch.run(output, "/fixture/launcher", {}, command)
        assert not calls
    else:
        result = dispatch.run(output, "/fixture/launcher", {}, command)
        assert len(calls) == 1 and result["attempts"] == 1
        assert result["pipeline_passed"] == (failure == "none")
        assert not result["automatic_admission"] and not result["timing_labels_emitted"]
        if failure == "trace_incomplete":
            assert "fixture trace missing" in result["trace_audit_error"]
