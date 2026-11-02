"""CPU control-flow and failure tests; synthetic times are not GPU evidence."""
from contextlib import nullcontext
import copy
import itertools
import subprocess
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from calibration import ffn_anchor_control as pilot
from calibration import ffn_control_runtime as runtime_module
from calibration.ffn_control_runtime import ControlRuntime


class FakeCase:
    def __init__(self, owner, name):
        self.owner, self.name = owner, name
        self.identity = copy.deepcopy(owner.identity)
    def capture(self): self.owner.events.append("capture:" + self.name)
    def replay(self, _):
        self.owner.events.append("warmup:" + self.name)
        return {"elapsed_ms": .5}, {"all_50_outputs": True}
    def trace(self, path):
        self.owner.events.append("trace:" + self.name)
        self.owner.profiled = True
        pilot.save(path, {"synthetic_fixture": True, "traceEvents": [{"cat": "kernel", "name": "fixture"}] * 50})
        return {"path_trace_written": True, "host_staging_detected": False, "kernel_names": ["fixture"] * 50}


class FakeRuntime:
    def __init__(self, uuid="fixture-GPU0"):
        self.events, self.context, self.profiled = [], {}, False
        self.identity = {"devices": {"0": {"observed": {"uuid": uuid, "architecture": "fixture"},
            "environment": {"device_uuid": uuid, "torch": "fixture", "hip": "fixture", "triton": "fixture", "device_model": "fixture"}}},
            "actual_torch_build": {"fixture": True}, "loaded_kerv_source": {"fixture": True}}
    def anchor(self, gpu):
        assert gpu == 0
        return FakeCase(self, "anchor")
    def make(self, row): return FakeCase(self, row)
    def snapshots(self): return {"0": {"fixture": True}}
    def checkpoint(self, *a, **k): pass


def synthetic_cell(directory, cell, *, bad_r1=False, uuid="fixture-GPU0"):
    runtime, ticks = FakeRuntime(uuid), itertools.count(100)
    per_round = 2 * cell["endpoint_replays"] + 1
    samples = []
    def sample(case):
        assert not runtime.profiled
        runtime.events.append("sample:" + case.name)
        ms = .6 if bad_r1 and len(samples) % per_round == per_round - 1 else .5
        stamps = {name: next(ticks) for name in pilot.PROTOCOL["required_timestamps_ns"]}
        result = {**stamps, "numeric_ok": True, "calls": 50, "batch_ms": ms,
                  "device": "cuda:0", "start_device": "cuda:0", "end_device": "cuda:0", "stream": 123,
                  "invariants": {"all_50_outputs": True}, "path_evidence_state": "pending_post_timing_trace"}
        samples.append(result)
        return result
    result = pilot.execute_cell(runtime, cell, directory, sample=sample,
        observe=lambda d: {"fixture": True}, clock=lambda: next(ticks))
    return result, runtime


def test_frozen_plan_has_no_gpu_side_effects_and_keeps_historical_protocol():
    plan = pilot.pilot_plan()
    assert plan == pilot.pilot_plan()
    assert plan["protocol"] == pilot.read(pilot.ROOT / "data/ffn_b4_cpu_preparation_v1/anchor_protocol.json")
    assert plan["collector_implemented"] and not plan["gpu_validated"]
    assert plan["default_phase"] == "plan" and not plan["followup_matrix_open"]


@pytest.mark.parametrize("repeats", [1, 3])
def test_seven_attempts_warmup_position_and_trace_after_all_samples(tmp_path, repeats):
    cell = next(c for c in pilot.PROTOCOL["cells"] if c["endpoint_replays"] == repeats)
    result, runtime = synthetic_cell(tmp_path, cell)
    assert result["quality"]["status"] == "quality_passed"
    events = runtime.events
    assert events[:11] == ["capture:anchor"] + ["warmup:anchor"] * 10
    assert events[11:22] == ["capture:" + cell["row"]] + ["warmup:" + cell["row"]] * 10
    assert sum(e.startswith("sample:") for e in events) == 7 * (2 * repeats + 1)
    assert events[-2:] == ["trace:" + cell["row"], "trace:anchor"]
    raw = (tmp_path / "samples.json").read_bytes()
    checked = pilot.review_cell(tmp_path, cell)
    assert checked["pipeline_complete"] and raw == (tmp_path / "samples.json").read_bytes()
    assert b"pending_post_timing_trace" in raw
    assert not result["formal_profile"] and not result["followup_matrix_open"]


def test_r1_quality_failure_does_not_stop_r3_or_select_a_replacement(tmp_path):
    calls = []
    def run(command, **kwargs):
        name = command[command.index("--worker") + 1]
        cell = next(c for c in pilot.PROTOCOL["cells"] if c["id"] == name)
        calls.append(name)
        synthetic_cell(command[command.index("--output") + 1], cell, bad_r1=name.endswith("r1"))
        return subprocess.CompletedProcess(command, 0)
    result = pilot.collect(tmp_path, "fixture-unused", pilot.pilot_plan(), run_worker=run)
    assert calls == [c["id"] for c in pilot.PROTOCOL["cells"]]
    assert result["cells"]["down_r1"]["status"] == "quality_failed"
    assert result["cells"]["up_r1"]["status"] == "quality_failed"
    assert result["pilot_gate_passed"] and not result["followup_matrix_open"]
    assert all(c["worker_attempts"] == 1 for c in result["cells"].values())


@pytest.mark.parametrize("failure", ["crash", "uuid_change"])
def test_pipeline_crash_or_identity_change_stops_without_retry(tmp_path, failure):
    calls = []
    def run(command, **kwargs):
        name = command[command.index("--worker") + 1]
        calls.append(name)
        if failure == "crash": return subprocess.CompletedProcess(command, -11)
        cell = next(c for c in pilot.PROTOCOL["cells"] if c["id"] == name)
        synthetic_cell(command[command.index("--output") + 1], cell, uuid="changed" if len(calls) == 2 else "first")
        return subprocess.CompletedProcess(command, 0)
    result = pilot.collect(tmp_path, "fixture-unused", pilot.pilot_plan(), run_worker=run)
    assert len(calls) == (1 if failure == "crash" else 2)
    assert result["cells"][calls[-1]]["status"] == "pipeline_failed"
    assert result["cells"]["down_r3"]["worker_attempts"] == 0 and not result["pilot_gate_passed"]


@pytest.mark.parametrize("mutation", ["timestamps", "numeric", "contract", "missing_round", "device"])
def test_parent_recomputes_and_rejects_missing_or_inconsistent_evidence(tmp_path, mutation):
    cell = pilot.PROTOCOL["cells"][0]
    synthetic_cell(tmp_path, cell)
    if mutation == "contract":
        record = pilot.read(tmp_path / "identity.json")
        record["measurement_contract"]["mode"] = "eager"
        pilot.save(tmp_path / "identity.json", record)
    else:
        rounds = pilot.read(tmp_path / "samples.json")
        if mutation == "timestamps": rounds[0]["target"]["reset_begin"] = 0
        elif mutation == "numeric": rounds[0]["target"]["numeric_ok"] = False
        elif mutation == "device": rounds[0]["target"]["end_device"] = "cuda:1"
        else: rounds.pop()
        pilot.save(tmp_path / "samples.json", rounds)
        result = pilot.read(tmp_path / "result.json")
        result["samples_sha256"] = pilot.digest(tmp_path / "samples.json")
        pilot.save(tmp_path / "result.json", result)
    with pytest.raises(ValueError): pilot.review_cell(tmp_path, cell)


def test_timed_sample_boundaries_exclude_restore_sync_and_validation(monkeypatch):
    events = []
    class Event:
        device = "cuda:0"
        def __init__(self, **kwargs): pass
        def record(self, stream): events.append("record")
        def synchronize(self): events.append("wait")
        def elapsed_time(self, end): return .5
    case = SimpleNamespace(device="cuda:0", devices=(0,), stream=SimpleNamespace(cuda_stream=123),
        graph=SimpleNamespace(replay=lambda: events.append("replay")), outputs=[],
        reset=lambda i: events.append("restore"),
        check=lambda *a: events.append("check") or {"all_50_outputs": True})
    monkeypatch.setattr(torch.cuda, "device", lambda *a: nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", lambda *a: nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr("calibration.ffn_b4_runtime.sync", lambda ds: events.append("sync"))
    result = pilot.timed_sample(case)
    assert events == ["restore", "sync", "record", "replay", "record", "wait", "sync", "check"]
    assert result["reset_begin"] <= result["timing_begin"] <= result["event_wait_end"] <= result["all_device_sync_end"] <= result["validation_end"]


def test_missing_telemetry_is_null_with_source_and_error(monkeypatch):
    def fail(*a): raise RuntimeError("fixture unavailable")
    for method in ("clock_rate", "utilization", "temperature"):
        monkeypatch.setattr(torch.cuda, method, fail, raising=False)
    result = pilot.telemetry()
    assert all(v["value"] is None and v["source"] and v["error"] for v in result.values())


def test_single_device_initialization_does_not_query_peer_or_start_gpu1(tmp_path, monkeypatch):
    devices = []
    class Adapter:
        module, registry = object(), object()
        def __init__(self, *a): pass
        def bind(self, placed, index, mapping):
            assert mapping == {0: "cuda:0"}
            return object()
    def observe(module, device):
        devices.append(device.index)
        return {"uuid": "fixture", "architecture": "fixture"}, {"device_uuid": "fixture"}
    def forbidden(*a): raise AssertionError("unexpected GPU1 or peer operation")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "fixture")
    monkeypatch.setattr(torch.cuda, "set_device", lambda d: devices.append(d))
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", forbidden)
    monkeypatch.setattr(runtime_module, "observe_environment", observe)
    monkeypatch.setattr(runtime_module, "KervSiluMulAdapter", Adapter)
    monkeypatch.setattr(runtime_module, "loaded_source", lambda *a: {"fixture": True})
    monkeypatch.setattr(runtime_module, "native_binding", lambda: {"fixture": True})
    monkeypatch.setattr(runtime_module, "digest", lambda p: "fixture")
    monkeypatch.setattr(runtime_module, "prepare", lambda *a, **k: 0)
    monkeypatch.setattr(runtime_module, "load_placed_op_graph", lambda *a, **k: object())
    runtime = ControlRuntime(tmp_path, {}, "fixture")
    assert devices == [0, 0] and set(runtime.observations) == {0} and runtime.peer == {}


def test_actual_mul_operand_metadata_and_fused_invariants_on_cpu(monkeypatch):
    # Small CPU tensors and a fake in-place operator exercise closures/metadata,
    # not HIP capture, dispatch, performance, or Profile admission.
    monkeypatch.setattr(runtime_module, "E", (1, 2, 3))
    monkeypatch.setattr(runtime_module, "sync", lambda d: None)
    monkeypatch.setattr(torch.cuda, "device", lambda *a: nullcontext())
    class Case:
        def __init__(self, row, device, devices, body, reset, check, identity):
            self.body, self.reset, self.check, self.identity = body, reset, check, identity
    monkeypatch.setattr(runtime_module, "GraphCase", Case)
    class Bound:
        gate_node, up_node = "gate", "up"
        def __call__(self, values, observe_dispatch):
            gate, up = values["gate"], values["up"]
            gate.copy_(F.silu(gate) * up)
            return SimpleNamespace(output=gate, observed_jit_launches=1)
    runtime = ControlRuntime.__new__(ControlRuntime)
    runtime.cache, runtime.bound = {}, {0: Bound()}
    runtime.adapter = SimpleNamespace(manifest={"binding": {"fixture": True}})
    runtime.identity = lambda *a: {"fixture_only": True}
    runtime.tensor = lambda name, shape, gpu: torch.randn(shape, dtype=torch.bfloat16) * .1
    cases = runtime.activation(0)
    mul, fused = cases["gated_mul@gpu0"], cases["silu_mul/fused@gpu0"]
    left = mul.identity["inputs"]["left"]
    gates = mul.identity["inputs"]["gates"]
    assert len(left) == 50 and {v["data_ptr"] for v in left}.isdisjoint(v["data_ptr"] for v in gates)
    observed_ptrs = []
    def observe_mul(tensor, other):
        observed_ptrs.append(tensor.data_ptr())
        return torch.mul(tensor, other)
    monkeypatch.setattr(torch.Tensor, "__mul__", observe_mul)
    mul.reset(0)
    outputs = mul.body()
    assert observed_ptrs == [v["data_ptr"] for v in left]
    assert all(v is True for v in mul.check(outputs, 0).values())
    for _ in range(2):
        fused.reset(0)
        outputs = fused.body()
        assert all(v is True for v in fused.check(outputs, 0).values())
    assert runtime.cache["activation0"] is cases
    with pytest.raises(ValueError): runtime.make("gated_mul@gpu0")
    with pytest.raises(ValueError): runtime.activation(1)
