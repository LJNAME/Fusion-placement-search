"""Synthetic CPU ordering/identity tests, never HIP measurement evidence."""
import copy
import itertools
import subprocess

import pytest

from calibration import ffn_anchor_conditioning as collector
from calibration.ffn_anchor_control import pilot_plan
from calibration.tests.test_ffn_anchor_control import FakeRuntime


def synthetic(directory, cell, *, uuid="fixture-GPU0", first_us=12, monkeypatch=None):
    runtime, ticks, count = FakeRuntime(uuid), itertools.count(100), itertools.count()
    runtime.checkpoint = lambda stage, **k: runtime.events.append("checkpoint:" + stage)
    per_round = 7 + 2 * cell["conditioning_replays_per_endpoint"]
    cond = cell["conditioning_replays_per_endpoint"]
    def sample(case):
        assert not runtime.profiled
        index = next(count) % per_round
        is_conditioning = cond == 1 and index in (0, 5)
        is_first = index == cond
        value = 999 if is_conditioning else 20 if case.name != "anchor" else first_us if is_first else 10
        runtime.events.append("sample:" + case.name)
        return {**{k: next(ticks) for k in ("reset_begin", "timing_begin", "event_wait_end", "all_device_sync_end", "validation_end")},
            "numeric_ok": True, "calls": 50, "batch_ms": value / 20,
            "invariants": {"all_50_outputs": True}, "device": "cuda:0", "start_device": "cuda:0", "end_device": "cuda:0",
            "path_evidence_state": "pending_post_timing_trace"}
    def observe(gpu):
        assert gpu == 0
        runtime.events.append("telemetry")
        return {"fixture": True}
    frozen = collector.plan()
    result = collector.execute_cell(runtime, cell, directory, frozen,
        sample=sample, observe=observe, clock=lambda: next(ticks))
    return result, runtime, frozen


@pytest.mark.parametrize("condition", ["A", "B"])
def test_counts_separate_conditioning_from_all_three_measured_values(tmp_path, condition):
    cell = next(c for c in collector.plan()["design"]["cells"] if c["condition"] == condition)
    result, runtime, frozen = synthetic(tmp_path, cell)
    checked = collector.review_cell(tmp_path, cell, frozen)
    assert checked["measured_replays"] == 49
    assert checked["conditioning_replays"] == (14 if condition == "B" else 0)
    assert result["analysis"]["primary_pre_contrast_us"] == 2
    assert result["analysis"]["quality"]["target_median_us"] == 20
    assert runtime.events[-2:] == ["trace:down_proj@gpu0", "trace:anchor"]
    raw = (tmp_path / "samples.json").read_bytes()
    assert b"pending_post_timing_trace" in raw and raw == (tmp_path / "samples.json").read_bytes()
    if condition == "B":
        samples = collector.read(tmp_path / "samples.json")
        assert all(r["pre_conditioning"][0]["primary_measurement"] is False for r in samples)
        assert all(len(r["pre"]) == len(r["post"]) == 3 for r in samples)
        assert all(r["pre_conditioning"][0]["batch_ms"] * 20 == 999 for r in samples)


def test_no_disk_telemetry_or_checkpoint_is_inserted_between_conditioning_and_measurement(tmp_path, monkeypatch):
    cell = collector.plan()["design"]["cells"][1]
    writes = []
    original_save = collector.save
    active_runtime = []
    def save(path, value):
        if active_runtime: active_runtime[0].events.append("disk")
        writes.append(path.name)
        original_save(path, value)
    monkeypatch.setattr(collector, "save", save)
    # Obtain the runtime through the first sample without changing sampler code.
    from calibration.tests.test_ffn_anchor_control import FakeRuntime as Base
    class Runtime(Base):
        def __init__(self, *a):
            super().__init__(*a)
            active_runtime.append(self)
    monkeypatch.setattr(__import__(__name__, fromlist=["FakeRuntime"]), "FakeRuntime", Runtime)
    _, runtime, _ = synthetic(tmp_path, cell)
    start = runtime.events.index("checkpoint:conditioning_round")
    round_events = runtime.events[start:runtime.events.index("disk", start) + 1]
    assert round_events == ["checkpoint:conditioning_round", "telemetry"] + ["sample:anchor"] * 4 + ["sample:down_proj@gpu0"] + ["sample:anchor"] * 4 + ["telemetry", "disk"]
    assert writes.count("samples.json") == 7


@pytest.mark.parametrize("mutation", ["drop_conditioning", "extra_conditioning", "role", "order", "drop_measured", "contract", "trace", "warmup"])
def test_parent_rejects_changed_raw_semantics_even_after_file_hash_is_refreshed(tmp_path, mutation):
    cell = collector.plan()["design"]["cells"][1]
    synthetic(tmp_path, cell)
    result = collector.read(tmp_path / "result.json")
    if mutation == "contract":
        path = tmp_path / "identity.json"; value = collector.read(path)
        value["measurement_contract"]["recipe"]["conditioning_replays_per_endpoint"] = 0
    elif mutation == "trace":
        path = tmp_path / "target_trace.json"; value = collector.read(path)
        value["traceEvents"][0]["name"] = "changed"
        collector.save(path, value)
        result["trace_sha256"]["target"] = collector.digest(path)
    elif mutation == "warmup":
        path = tmp_path / "preparation.json"; value = collector.read(path)
        value["anchor_warmup_diagnostics"].pop()
    else:
        path = tmp_path / "samples.json"; value = collector.read(path)
        row = value[0]
        if mutation == "drop_conditioning": row["pre_conditioning"] = []
        elif mutation == "extra_conditioning": row["pre_conditioning"].append(copy.deepcopy(row["pre_conditioning"][0]))
        elif mutation == "role": row["pre_conditioning"][0]["primary_measurement"] = True
        elif mutation == "order": row["pre"][0]["reset_begin"] = 0
        else: row["pre"].pop(0)
        collector.save(path, value)
        result["samples_sha256"] = collector.digest(path)
    collector.save(path, value)
    collector.save(tmp_path / "result.json", result)
    with pytest.raises(ValueError): collector.review_cell(tmp_path, cell, collector.plan())


def test_fixed_abba_both_block_signs_and_quality_failures_remain_visible(tmp_path):
    calls = []
    frozen = collector.plan()
    def run(command, **kwargs):
        name = command[command.index("--worker") + 1]
        cell = next(c for c in frozen["design"]["cells"] if c["id"] == name)
        calls.append(name)
        synthetic(command[command.index("--output") + 1], cell, first_us=11 if name == "block1_B" else 14 if name == "block2_B" else 12)
        return subprocess.CompletedProcess(command, 0)
    summary = collector.collect(tmp_path, "fixture-unused", frozen, run_worker=run)
    assert calls == ["block1_A", "block1_B", "block2_B", "block2_A"]
    assert summary["pipeline_passed"] and any(r["status"] == "quality_failed" for r in summary["cells"].values())
    report = collector.read(tmp_path / "report.json")
    assert report["block_contrasts"]["block1"]["B_minus_A_us"] == -1
    assert report["block_contrasts"]["block2"]["B_minus_A_us"] == 2
    assert report["direction"] == "incomplete_mixed_or_zero_inconclusive"
    assert not report["protocol_adopted"] and not report["followup_matrix_open"]
    assert sum(r["conditioning_replays"] for r in summary["cells"].values()) == 28
    assert sum(r["measured_replays"] for r in summary["cells"].values()) == 196


@pytest.mark.parametrize("failure", ["crash", "uuid"])
def test_native_crash_or_changed_device_blocks_rest_without_retry(tmp_path, failure):
    frozen, calls = collector.plan(), []
    def run(command, **kwargs):
        name = command[command.index("--worker") + 1]; calls.append(name)
        if failure == "crash": return subprocess.CompletedProcess(command, -11)
        cell = next(c for c in frozen["design"]["cells"] if c["id"] == name)
        synthetic(command[command.index("--output") + 1], cell, uuid="changed" if len(calls) == 2 else "first")
        return subprocess.CompletedProcess(command, 0)
    result = collector.collect(tmp_path, "fixture-unused", frozen, run_worker=run)
    assert len(calls) == (1 if failure == "crash" else 2)
    assert not result["pipeline_passed"] and result["cells"]["block2_A"]["worker_attempts"] == 0
    package = collector.read(tmp_path / calls[-1] / "parent_failure.json")
    assert all(k in package for k in ("traceback", "pool_config", "memory", "worker_fault_traceback_log"))


def test_plan_default_is_diagnostic_and_old_pilot_source_identity_is_unchanged():
    frozen = collector.plan()
    assert frozen == collector.plan()
    assert frozen["default_phase"] == "plan" and not frozen["gpu_validated"]
    assert not frozen["automatic_admission"] and not frozen["followup_matrix_open"]
    assert pilot_plan() == collector.read(collector.ROOT / "data/ffn_anchor_control_v2/plan.json")
