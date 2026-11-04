"""CPU fixtures only; neither test durations nor accepted fixtures are GPU data."""
import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from calibration import lightop_profile_calibration as formal
from cost_model.fusion_kernel_profile import key_id
from cost_model.lightop_profile import LightopKernelKeys, measurement_contract
from cost_model.measurement_labels import MeasurementContractError, measurement_contract_id
from cost_model.tests.test_lightop_profile import fixture_case
from simulator.placement_artifacts import save_placed_op_graph


def rounds(values=None):
    values = values or [10, 10.1, 10.2, 10.3, 10.2, 10.1, 10]
    return [{"round": i, "started_monotonic_ns": i * 100 + 1, "finished_monotonic_ns": i * 100 + 50,
             "timing": {"elapsed_ms_per_50_calls": v * 50 / 1000, "error": None,
                        "event_boundary": {"device": "cuda:0", "start_device": "cuda:0",
                            "end_device": "cuda:0", "stream": 123, "calls_per_replay": 50}},
             "checks": copy.deepcopy(formal.CHECKS)} for i, v in enumerate(values)]


@pytest.fixture
def receipt(tmp_path, monkeypatch):
    placed, index, observation, runtime = fixture_case()
    keys = LightopKernelKeys(placed, index, observation, runtime)
    key, gaps = keys.describe(placed.plan.groups[index])
    contract = measurement_contract(runtime["environment"], purpose="calibration")
    smoke_id = measurement_contract_id(measurement_contract(runtime["environment"]))
    frozen = {"expected_kernel_key_id": key_id(key), "expected_smoke_contract_id": smoke_id,
              "collector": {"git_revision": "CPU-fixture", "source_sha256": {"fixture": "fixture"}},
              "dispatch_evidence_sha256": {"report.json": "CPU-fixture"}}
    monkeypatch.setattr(formal, "plan", lambda: copy.deepcopy(frozen))
    records = rounds()
    result = {"scope": formal.SCOPE, "complete": True, "attempts": 7, "capture_completed": True,
              "eager_warmup_calls": 50, "warmup_replays": 10, "warmup_checks": formal.CHECKS,
              "instrumentation_guard": {"profiler_environment": {}, "python_profile_or_trace": False,
                                        "profiler_launched": False},
              "rounds": records, "quality": formal.quality(records), "admission": "conditional_dev",
              "collector": frozen["collector"], "automatic_admission": False}
    contents = {"plan.json": frozen, "result.json": result, "runtime.json": runtime,
                "preparation.json": {"one_call": observation, "call_count": 50},
                "identity.json": {"group_index": index, "kernel_key": key, "kernel_key_id": key_id(key),
                    "identity_gaps": list(gaps), "measurement_contract": contract,
                    "measurement_contract_id": measurement_contract_id(contract), "smoke_contract_id": smoke_id},
                "process.json": {"python_returncode": 0, "log_returncode": 0, "automatic_admission": False}}
    contents.update({f"round_{i}.json": row for i, row in enumerate(records)})
    for name, value in contents.items():
        formal.save(tmp_path / name, value)
    save_placed_op_graph(tmp_path / "placed.json", placed)
    return tmp_path


def edit(receipt, name, mutate):
    value = formal.read(receipt / name)
    mutate(value)
    formal.save(receipt / name, value)


def test_actual_plan_uses_verified_dispatch_and_frozen_sources_without_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("plan touched GPU"))
    assert formal.plan() == formal.plan()
    frozen = formal.plan()
    assert frozen["quality_policy"]["attempts"] == 7
    assert not frozen["quality_policy"]["performance_gain_required"]
    assert frozen["expected_kernel_key_id"].startswith("sha256:")
    formal.main(["plan", "--output", str(tmp_path)])
    with pytest.raises(SystemExit):
        formal.main(["plan", "--output", str(tmp_path)])


def test_fixture_acceptance_reaches_strict_cost_query_and_charges_once(receipt):
    decision, profile = formal.review_receipt(receipt)
    assert decision["status"] == "accepted" and profile is not None
    assert decision["cost_query"]["duration_s"] == pytest.approx(10.1e-6)
    assert decision["cost_query"]["source"] == "measured_group_kernel"
    assert decision["cost_query"]["launch_time"] == decision["cost_query"]["memory_time"] == 0
    assert decision["latency"]["statistics"]["n_calls"] == 350
    assert formal.load_reviewed_profile(receipt) is not None


@pytest.mark.parametrize("change", ["source", "contract", "build", "missing_round", "extra_round",
                                   "summary", "process", "checks", "event_device", "failure"])
def test_invalid_receipts_cannot_be_admitted(receipt, change):
    if change == "source":
        edit(receipt, "plan.json", lambda v: v["collector"].update(source_sha256={"wrong": "wrong"}))
    elif change == "contract":
        edit(receipt, "identity.json", lambda v: v["measurement_contract"].update(purpose="smoke"))
    elif change == "build":
        edit(receipt, "runtime.json", lambda v: v["runtime"].update(torch_extension_sha256="wrong"))
    elif change == "missing_round":
        (receipt / "round_4.json").unlink()
    elif change == "extra_round":
        formal.save(receipt / "round_7.json", rounds()[0])
    elif change == "summary":
        edit(receipt, "result.json", lambda v: v["quality"]["statistics"].update(median_us=1))
    elif change == "process":
        edit(receipt, "process.json", lambda v: v.update(python_returncode=1))
    elif change == "failure":
        formal.save(receipt / "failure.json", {"fixture": "failed"})
    else:
        data = formal.read(receipt / "round_0.json")
        if change == "checks":
            data["checks"]["input_bytes_unchanged"] = False
        else:
            data["timing"]["event_boundary"]["end_device"] = "cuda:1"
        formal.save(receipt / "round_0.json", data)
        edit(receipt, "result.json", lambda v: v["rounds"].__setitem__(0, data))
    with pytest.raises((MeasurementContractError, FileNotFoundError)):
        formal.review_receipt(receipt)


def test_uuid_reboot_change_does_not_reject_fixture(receipt):
    edit(receipt, "runtime.json", lambda v: v["device"].update(uuid="SCNet-after-reboot"))
    assert formal.review_receipt(receipt)[0]["status"] == "accepted"


def test_slow_valid_measurement_is_accepted_without_a_speedup_test(receipt):
    data = rounds([1000] * 7)  # CPU fixture, no Native comparison.
    for i, row in enumerate(data):
        formal.save(receipt / f"round_{i}.json", row)
    edit(receipt, "result.json", lambda v: v.update(rounds=data, quality=formal.quality(data)))
    decision, _ = formal.review_receipt(receipt)
    assert decision["status"] == "accepted" and not decision["performance_gain_checked"]


def test_high_dispersion_preserves_every_sample_and_fails_quality(receipt):
    data = rounds([1, 2, 3, 4, 5, 6, 7])
    for i, row in enumerate(data):
        formal.save(receipt / f"round_{i}.json", row)
    edit(receipt, "result.json", lambda v: v.update(rounds=data, quality=formal.quality(data)))
    edit(receipt, "process.json", lambda v: v.update(python_returncode=1))
    decision, profile = formal.review_receipt(receipt)
    assert profile is None and decision["status"] == "quality_failed"
    assert decision["quality"]["samples_us"] == pytest.approx([1, 2, 3, 4, 5, 6, 7])
    with pytest.raises(MeasurementContractError):
        formal.load_reviewed_profile(receipt)


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), None])
def test_invalid_timing_is_not_silently_dropped(bad):
    data = rounds()
    data[3]["timing"]["elapsed_ms_per_50_calls"] = bad
    result = formal.quality(data)
    assert result["status"] == "quality_failed" and result["statistics"] is None
    assert result["reasons"] == ["invalid_timing_round_3"]


def test_quality_threshold_is_preregistered_and_inclusive():
    assert formal.quality(rounds([9, 9, 10, 10, 10, 11, 11]))["status"] == "quality_passed"
    assert formal.quality(rounds([8, 8, 10, 10, 10, 12, 12]))["status"] == "quality_failed"


@pytest.mark.parametrize("change", ["round_order", "timestamps", "stream"])
def test_reordered_or_inconsistent_samples_are_rejected(change):
    data = rounds()
    if change == "round_order":
        data[0], data[1] = data[1], data[0]
    elif change == "timestamps":
        data[0]["finished_monotonic_ns"] = data[1]["finished_monotonic_ns"]
    else:
        data[3]["timing"]["event_boundary"]["stream"] = 999
    with pytest.raises(MeasurementContractError):
        formal.quality(data)


@pytest.mark.parametrize("mode", ["ok", "event_create", "event_readout", "gpu_failure"])
def test_one_attempt_never_replays_twice_and_gpu_errors_propagate(mode):
    stream = SimpleNamespace(cuda_stream=123)
    event = SimpleNamespace(record=Mock(), synchronize=Mock(), device="cuda:0",
                            elapsed_time=Mock(return_value=.5))
    if mode == "event_readout":
        event.elapsed_time.side_effect = RuntimeError("fixture timer error")
    factory = Mock(return_value=event)
    if mode == "event_create":
        factory.side_effect = RuntimeError("fixture event error")
    fake = SimpleNamespace(cuda=SimpleNamespace(Event=factory, synchronize=Mock()))
    graph = SimpleNamespace(replay=Mock())
    if mode == "gpu_failure":
        graph.replay.side_effect = RuntimeError("fixture GPU error")
        with pytest.raises(RuntimeError, match="GPU error"):
            formal.timed_replay(fake, graph, stream, "cuda:0")
    else:
        result = formal.timed_replay(fake, graph, stream, "cuda:0")
        assert (result["error"] is None) == (mode == "ok")
    graph.replay.assert_called_once()


def test_run_failure_keeps_traceback_pool_and_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    with pytest.raises(RuntimeError, match="HIP GPU0"):
        formal.main(["run", "--output", str(tmp_path)])
    failure = formal.read(tmp_path / "failure.json")
    assert failure["traceback"] and failure["pool"] == formal.POOL
    assert failure["context"]["stage"] == "runtime_guard"


def test_known_instrumentation_stops_before_runtime_preparation(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "CPU fixture")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("ROCP_TOOL_LIB", "/fixture/profiler.so")
    with pytest.raises(RuntimeError, match="instrumentation detected"):
        formal.main(["run", "--output", str(tmp_path)])
    assert not (tmp_path / "preparation.json").exists()


def test_cli_quality_failure_keeps_raw_result_without_claiming_a_crash(tmp_path, monkeypatch):
    def failed(output, frozen, context):
        result = {"quality": {"status": "quality_failed"}}
        formal.save(output / "result.json", result)
        return result
    monkeypatch.setattr(formal, "run", failed)
    with pytest.raises(SystemExit):
        formal.main(["run", "--output", str(tmp_path)])
    assert (tmp_path / "result.json").is_file() and not (tmp_path / "failure.json").exists()


def test_review_cli_is_cpu_only_and_writes_a_separate_decision(receipt, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("review touched GPU"))
    output = tmp_path / "cpu-review"
    formal.main(["review", "--input", str(receipt), "--output", str(output)])
    assert formal.read(output / "admission.json")["status"] == "accepted"
