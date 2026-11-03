"""CPU trial semantics and orchestration; no assertion about DCU performance."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from calibration import hygon_silu_trial as trial


def test_packed_and_split_boundaries_keep_packing_in_the_correct_arm(monkeypatch):
    gate, up = torch.randn(3, 8), torch.randn(3, 8)
    def op(out, value):
        out.copy_(torch.nn.functional.silu(value[..., :8]) * value[..., 8:])
    arms, unchanged = trial.functions(torch, op, gate, up)
    original_cat, calls = torch.cat, []
    def cat(*args, **kwargs):
        calls.append(True)
        return original_cat(*args, **kwargs)
    monkeypatch.setattr(torch, "cat", cat)
    for name in trial.ARMS:
        before = len(calls)
        result = arms[name]()
        torch.testing.assert_close(result, torch.nn.functional.silu(gate) * up)
        assert len(calls) - before == int(name == "lightop_split_with_pack")
    assert unchanged()


@pytest.mark.parametrize("arm", ["lightop_packed", "lightop_split_with_pack"])
def test_mutating_operator_is_detected_for_both_actual_inputs(arm):
    def op(out, value):
        out.zero_()
        value.zero_()
    arms, unchanged = trial.functions(torch, op, torch.ones(3, 8), torch.ones(3, 8))
    arms[arm]()
    assert not unchanged()


@pytest.mark.parametrize("failure", ["missing_export", "benchmark_crash", "profiler_missing", "trace_missing", "none"])
def test_run_inspects_first_times_without_profiler_then_profiles_without_retries(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(trial.shutil, "which", lambda _: None if failure == "profiler_missing" else "/fixture/hipprof")
    calls = []
    def command(args, **kwargs):
        assert kwargs["check"] is False
        directory = Path(kwargs["cwd"])
        calls.append(directory.name)
        if directory.name == "inspection":
            trial.save(directory / "inspection.json", {"lightop": {"required_export_callable": failure != "missing_export"}, "torch": {"hip": "fixture"}})
        elif directory.name == "benchmark":
            assert args[0] == "/fixture/launcher" and "hipprof" not in args
            if failure == "benchmark_crash":
                return subprocess.CompletedProcess(args, -11)
        else:
            assert args[0] == "/fixture/hipprof" and args[1] == "--hip-trace"
            trial.save(directory / "profile_worker.json", {"complete": True})
            if failure != "trace_missing":
                (directory / "trace.csv").write_text("fixture only\n", encoding="utf-8")
        return subprocess.CompletedProcess(args, 0)
    result = trial.run(tmp_path, "/fixture/launcher", command=command)
    expected = ["inspection"] if failure == "missing_export" else ["inspection", "benchmark"]
    if failure in ("trace_missing", "none"):
        expected.append("hipprof_native_packed")
    if failure == "none":
        expected.append("hipprof_lightop_packed")
    assert calls == expected
    assert result["pipeline_passed"] is (failure == "none")
    assert not result["automatic_admission"] and not result["followup_matrix_open"]
    assert all(j["attempts"] == 1 for j in result["jobs"].values())
    assert json.loads((tmp_path / "summary.json").read_text()) == result


def test_plan_is_cpu_only_and_does_not_claim_a_library_benchmark_or_dispatch_certification():
    frozen = trial.plan()
    assert frozen == trial.plan()
    assert frozen["timed_calls_per_arm"] == 100 and frozen["warmup_calls_per_arm"] == 10
    assert not frozen["library_original_benchmark_executed"]
    assert not frozen["cross_batch_uuid_equality_required"]
    assert not frozen["kernel_dispatch_certified"] and not frozen["automatic_admission"]
    assert frozen["binding"] == "lightop.op.silu_and_mul_opt(out, packed_input)"
    assert frozen["collector_version"] == 2


@pytest.mark.parametrize("nested", [True, False])
def test_inspection_requires_the_optimized_export_in_the_actual_operator_module(tmp_path, monkeypatch, nested):
    callback = lambda out, value: None
    module = SimpleNamespace(__file__=None, silu_and_mul_opt=callback,
        op=SimpleNamespace(**({"silu_and_mul_opt": callback} if nested else {})))
    monkeypatch.setattr(trial.importlib, "import_module", lambda _: module)
    monkeypatch.setattr(trial.metadata, "packages_distributions", lambda: {})
    report = trial.inspect_stack(tmp_path)
    assert report["lightop"]["required_export_callable"] is nested
    assert report["lightop"]["binding"] == "lightop.op.silu_and_mul_opt"


def test_vendor_style_outputs_are_preallocated_per_arm():
    def op(out, value):
        out.copy_(torch.nn.functional.silu(value[..., :8]) * value[..., 8:])
    arms, unchanged = trial.functions(torch, op, torch.randn(3, 8), torch.randn(3, 8))
    outputs = []
    for function in arms.values():
        first, second = function(), function()
        assert first.data_ptr() == second.data_ptr()
        outputs.append(first)
    assert len({v.data_ptr() for v in outputs}) == 4 and unchanged()
