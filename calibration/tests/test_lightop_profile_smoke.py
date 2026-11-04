"""CPU smoke orchestration checks; these results are not GPU evidence."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from calibration import lightop_profile_smoke as smoke
from simulator.placement_artifacts import load_placed_op_graph, save_placed_op_graph, validate_placed_op_graph
from integration.lightop.silu_mul import adapter_registry, group_inputs


def test_plan_is_repeatable_and_does_not_load_vendor_or_use_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("plan used the GPU"))
    for name in ("a", "b"):
        smoke.main(["plan", "--output", str(tmp_path / name)])
    assert (tmp_path / "a/plan.json").read_bytes() == (tmp_path / "b/plan.json").read_bytes()
    data = json.loads((tmp_path / "a/plan.json").read_text())
    assert not data["automatic_admission"] and not data["timing_used_for_pass_fail"]
    assert data["calls_per_replay"] == 50 and data["diagnostic_replays"] == 3


@pytest.mark.parametrize("device", [0, 1])
def test_reference_graph_selected_group_and_serialization(tmp_path, device):
    placed, index = smoke.placed_case(device)
    validate_placed_op_graph(placed, adapter_registry())
    group = placed.plan.groups[index]
    assert group.device == device and len(group.node_ids) == 2
    assert placed.graph.nodes[group.node_ids[0]].extra["layer"] == 0
    assert len(group_inputs(placed.graph, group.node_ids)) == 2
    save_placed_op_graph(tmp_path / "placed.json", placed)
    restored = load_placed_op_graph(tmp_path / "placed.json", registry=adapter_registry())
    assert restored.graph_hash == placed.graph_hash
    assert not restored.extra["deployable"]


def test_runtime_failure_retains_traceback_pool_and_no_admission(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    with pytest.raises(RuntimeError, match="no CPU/native fallback"):
        smoke.main(["run", "--output", str(tmp_path)])
    failed = json.loads((tmp_path / "failure.json").read_text())
    assert failed["traceback"] and failed["pool"] == smoke.POOL
    assert failed["context"]["stage"] == "runtime_guard" and not failed["automatic_admission"]
    assert not (tmp_path / "result.json").exists()
    before = (tmp_path / "failure.json").read_bytes()
    with pytest.raises(SystemExit):
        smoke.main(["run", "--output", str(tmp_path)])
    assert (tmp_path / "failure.json").read_bytes() == before


@pytest.mark.parametrize("event_failure", [False, True])
def test_diagnostic_event_failure_does_not_skip_or_repeat_replay(event_failure):
    graph = SimpleNamespace(replay=Mock())
    event = SimpleNamespace(record=Mock(), elapsed_time=Mock(return_value=999999))
    factory = Mock(side_effect=ValueError("fixture event failure")) if event_failure else Mock(return_value=event)
    fake = SimpleNamespace(cuda=SimpleNamespace(Event=factory, synchronize=Mock()))
    result = smoke.diagnostic_replay(fake, graph, "fixture-stream", "cuda:0")
    graph.replay.assert_called_once()
    fake.cuda.synchronize.assert_called_once_with("cuda:0")
    assert result["used_for_pass_fail"] is False
    assert bool(result["error"]) == event_failure


def test_gpu_execution_errors_remain_smoke_failures():
    event = SimpleNamespace(record=Mock())
    fake = SimpleNamespace(cuda=SimpleNamespace(Event=Mock(return_value=event), synchronize=Mock()))
    graph = SimpleNamespace(replay=Mock(side_effect=RuntimeError("fixture capture failure")))
    with pytest.raises(RuntimeError, match="fixture capture failure"):
        smoke.diagnostic_replay(fake, graph, None, None)


def test_noop_replay_cannot_pass_using_eager_warmup_values():
    gate, up = torch.ones((2, 2)), torch.ones((2, 2))
    snapshots = [x.view(torch.uint8).clone() for x in (gate, up)]
    call = SimpleNamespace(output=gate.clone(), _packed=torch.ones((2, 4)))
    smoke.invalidate_buffers([call])
    assert torch.isnan(call._packed).all() and torch.isnan(call.output).all()
    assert torch.equal(gate.view(torch.uint8), snapshots[0])
    with pytest.raises(AssertionError):
        smoke.check_outputs(torch, [call], gate, up, snapshots, [gate], {"rtol": .01, "atol": .001})


@pytest.mark.parametrize("bad", ["input_mutation", "wrong_output", "shared_output", "input_alias"])
def test_numeric_bytewise_and_buffer_checks_detect_pollution(bad):
    gate, up = torch.ones((2, 2)), torch.ones((2, 2))
    snapshots = [x.view(torch.uint8).clone() for x in (gate, up)]
    calls = [SimpleNamespace(output=gate.clone(), _packed=torch.empty((2, 4))) for _ in range(2)]
    reference = gate.clone()
    if bad == "input_mutation":
        gate[0, 0] = 2
    elif bad == "wrong_output":
        calls[0].output.zero_()
    elif bad == "shared_output":
        calls[1].output = calls[0].output
    else:
        calls[0].output = gate
    with pytest.raises(AssertionError):
        smoke.check_outputs(torch, calls, gate, up, snapshots, [reference], {"rtol": .01, "atol": .001})
