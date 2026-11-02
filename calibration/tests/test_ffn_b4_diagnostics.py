"""CPU checks for diagnostic boundaries; no GPU compatibility claim."""
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch

from calibration.ffn_b4 import save
from calibration.ffn_b4_diagnostics import collect, paired_sample, projection, CASES


def test_preallocated_projection_preserves_inputs_and_output_storage():
    left, weight = torch.randn(2,3, dtype=torch.bfloat16), torch.randn(3,4, dtype=torch.bfloat16)
    left_before, weight_before = left.clone(), weight.clone()
    out = torch.empty((2,4), dtype=torch.bfloat16)
    result = projection(left, weight, out)
    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(result, projection(left, weight), rtol=0, atol=0)
    assert torch.equal(left, left_before) and torch.equal(weight, weight_before)


def test_failed_one_call_chain_blocks_fifty_calls_without_worker_retry():
    calls = []
    with tempfile.TemporaryDirectory() as directory:
        def run(command, **kwargs):
            name = command[command.index("--worker")+1]
            calls.append(name)
            if name == "chain_preallocated_1":
                return subprocess.CompletedProcess(command, -11)
            save(Path(command[command.index("--output")+1])/"result.json", {"status": "passed"})
            return subprocess.CompletedProcess(command, 0)
        result = collect(Path(directory), "fixture-unused", run_worker=run)
        summary = json.loads((Path(directory)/"summary.json").read_text())
    assert calls == list(CASES[:-1])
    assert result["chain_preallocated_50"]["status"] == "blocked"
    assert result["chain_preallocated_50"]["worker_attempts"] == 0
    assert result["chain_preallocated_1"]["returncode"] == -11
    assert not summary["formal_measurements"] and not summary["automatic_admission"]


def test_wall_completion_boundary_precedes_numerical_checks():
    calls = []
    class Event:
        device = "cuda:0"
        def record(self, stream): calls.append("record")
        def synchronize(self): calls.append("event_wait")
        def elapsed_time(self, end): return .5
    case = SimpleNamespace(device="cuda:0", devices=(0,), stream=SimpleNamespace(cuda_stream=123),
                           mark=lambda *a,**k: None, reset=lambda i: calls.append("reset"),
                           graph=SimpleNamespace(replay=lambda: calls.append("replay")), outputs=[])
    case.check = lambda *a: (calls.append("numeric_check") or {"correct": True})
    with patch("torch.cuda.device", return_value=nullcontext()), patch("torch.cuda.stream", return_value=nullcontext()), \
         patch("torch.cuda.Event", side_effect=[Event(), Event()]), \
         patch("calibration.ffn_b4_diagnostics.sync", side_effect=lambda d: calls.append("all_device_sync")):
        result = paired_sample(case, 50)
    assert calls == ["reset", "all_device_sync", "record", "replay", "record", "event_wait", "all_device_sync", "numeric_check"]
    assert result["event_us_per_call"] == 10 and not result["formal_sample"]
    assert result["all_device_completion_wall_us"] >= result["event_wait_wall_us"]
