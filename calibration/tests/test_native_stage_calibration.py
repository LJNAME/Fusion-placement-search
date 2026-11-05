"""CPU checks and synthetic receipts; no new GPU timing evidence."""
import copy

import pytest
import torch

from calibration import native_stage_calibration as native
from cost_model.measurement_labels import MeasurementContractError


def fixture_records(frozen):
    records = []
    for attempt, order in enumerate(frozen["order"]):
        for stage in order:
            tick = len(records) * 10 + 1
            records.append({"stage": stage, "round": attempt,
                "started_monotonic_ns": tick, "finished_monotonic_ns": tick + 5,
                "timing": {"elapsed_ms_per_50_calls": .3 if stage == "silu" else .2,
                    "error": None, "event_boundary": {"device": "cuda:0", "start_device": "cuda:0",
                        "end_device": "cuda:0", "stream": 123, "calls_per_replay": 50}},
                "checks": copy.deepcopy(native.CHECKS), "stage_checks": copy.deepcopy(native.STAGE_CHECKS)})
    return records


def metadata_fixture():
    tensor = {"shape": list(native.SHAPE), "dtype": "bf16", "layout": "contiguous",
        "strides": [786432, 3072, 1], "layout_provenance": {"kind": "runtime_tensor",
            "device": "cuda:0", "observer": "Tensor.shape/dtype/stride/is_contiguous"}}
    prepared = {s: {"recipe": copy.deepcopy(native.RECIPES[s]), "schema": native.SCHEMAS[s],
        "source_sha256": native.digest(native.Path(native.__file__)),
        "tensors": {k: copy.deepcopy(tensor) for k in ("gate", "up", "left", "output")}} for s in native.STAGES}
    runtime = {"runtime": {"accelerator": "HIP", "device": "cuda:0", "hip": "CPU fixture",
        "driver": "fixture", "grad_enabled": False, "autocast_enabled": False,
        "deterministic_algorithms": False,
        "torch_distribution": "fixture", "torch_git_revision": "fixture",
        "torch_extension_sha256": "fixture", "torch_build_sha256": "fixture"},
        "device": {"architecture": "gfx936:sramecc+:xnack-", "name": "fixture", "ordinal": 0},
        "provenance": {"uuid": "fixture-0"}}
    return prepared, runtime


def test_cpu_plan_is_deterministic_and_does_not_require_old_latency_labels():
    before = {p: native.digest(native.ROOT / p) for p in (
        "data/lightop_local_pair_20261104_01/review/evaluation.json",
        "data/lightop_profile_calibration_20261104_01/review/admission.json")}
    a = native.plan()
    assert a == native.plan()
    assert a["seed"] not in (20261104, 20261105)
    assert a["contract"]["extra_attempts"] == 0
    assert a["order"] == [["silu", "mul"] if i % 2 == 0 else ["mul", "silu"] for i in range(7)]
    assert a["existing_measurements_preserved"]
    assert not a["model_parameters_updated"]
    assert before == {p: native.digest(native.ROOT / p) for p in before}


@pytest.mark.parametrize("stage", native.STAGES)
def test_actual_cpu_stage_numerics_readonly_and_buffers(stage):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(17)
        gate, up = [torch.randn(native.SHAPE, dtype=torch.bfloat16) for _ in range(2)]
        left = torch.nn.functional.silu(gate)
        inputs = {"gate": gate, "up": up, "left": left}
        snapshots = {k: v.view(torch.uint8).clone() for k, v in inputs.items()}
        calls = [native.StageCall(stage, gate, up, left) for _ in range(2)]
        pointers = [c.output.data_ptr() for c in calls]
        for c in calls:
            c(); c()
        refs = (torch.nn.functional.silu(gate), torch.nn.functional.silu(gate.float())) if stage == "silu" else (
            left * up, left.float() * up.float())
        checked = native.check(torch, calls, inputs, snapshots, refs, {"rtol": .01, "atol": .001})
        assert checked["outputs_checked"] == 2
        assert pointers == [c.output.data_ptr() for c in calls]
        assert calls[0].observation["schema"] == native.SCHEMAS[stage]
        left.zero_()
        with pytest.raises(AssertionError, match="input changed"):
            native.check(torch, calls, inputs, snapshots, refs, {"rtol": .01, "atol": .001})
    finally:
        torch.set_num_threads(old_threads)


def test_stage_costs_do_not_claim_chain_additivity_or_parameter_identification():
    frozen = native.plan()
    result = native.evaluate(frozen, fixture_records(frozen))
    assert result["status"] == "calibration_candidates_ready"
    assert result["stages"]["silu"]["statistics"]["median_us"] == 6
    assert result["stages"]["mul"]["statistics"]["median_us"] == 4
    assert result["whole_chain_cost_us"] is None
    assert result["pure_graph_overhead_us"] is None
    assert not result["automatic_admission"]


@pytest.mark.parametrize("bad", ["missing", "extra", "reorder", "overlap", "wrong_stream", "bad_checks", "left_origin_check"])
def test_bad_receipts_rejected(bad):
    frozen = native.plan()
    rows = fixture_records(frozen)
    if bad == "missing": rows.pop()
    elif bad == "extra": rows.append(copy.deepcopy(rows[-1]))
    elif bad == "reorder": rows[:2] = rows[1::-1]
    elif bad == "overlap": rows[0]["finished_monotonic_ns"] = 100
    elif bad == "wrong_stream": rows[0]["timing"]["event_boundary"]["stream"] = 456
    elif bad == "left_origin_check": rows[0]["stage_checks"]["left_matches_bf16_silu_gate"] = False
    else: rows[0]["checks"]["input_bytes_unchanged"] = False
    with pytest.raises(MeasurementContractError):
        native.evaluate(frozen, rows)


@pytest.mark.parametrize("bad", ["timer", "dispersion"])
def test_quality_failure_preserves_each_stage_and_all_attempts(bad):
    frozen = native.plan()
    rows = fixture_records(frozen)
    if bad == "timer": rows[0]["timing"]["error"] = "CPU fixture timer failure"
    else:
        for r in rows:
            if r["stage"] == "silu" and r["round"] > 3:
                r["timing"]["elapsed_ms_per_50_calls"] *= 2
    snapshot = copy.deepcopy(rows)
    result = native.evaluate(frozen, rows)
    assert result["status"] == "quality_failed"
    assert result["stages"]["mul"]["status"] == "quality_passed"
    assert rows == snapshot


def test_uuid_is_provenance_but_build_and_recipe_change_identity():
    frozen = native.plan()
    prep, actual = metadata_fixture()
    original = native.identity(frozen, prep, actual)
    actual["provenance"]["uuid"] = "different-instance"
    assert native.identity(frozen, prep, actual) == original
    actual["runtime"]["torch_build_sha256"] = "changed-build"
    assert native.identity(frozen, prep, actual) != original


@pytest.mark.parametrize("bad", ["left_origin", "schema", "unknown_layout", "cpu", "grad", "deterministic", "wrong_device"])
def test_actual_metadata_must_match_stage_contract(bad):
    prep, actual = metadata_fixture()
    if bad == "left_origin": prep["mul"]["recipe"]["left_origin"] = "random input"
    elif bad == "schema": prep["mul"]["schema"] = "tensor.mul"
    elif bad == "unknown_layout": prep["mul"]["tensors"]["left"]["layout"] = "unknown"
    elif bad == "cpu": actual["runtime"]["accelerator"] = "CPU"
    elif bad == "grad": actual["runtime"]["grad_enabled"] = True
    elif bad == "deterministic": actual["runtime"]["deterministic_algorithms"] = True
    else: actual["device"]["ordinal"] = 1
    with pytest.raises(MeasurementContractError):
        native.identity(native.plan(), prep, actual)


def write_fixture(directory):
    frozen = native.plan()
    rows = fixture_records(frozen)
    prep, actual = metadata_fixture()
    content = {"plan.json": frozen, "preparation.json": prep, "runtime.json": actual,
        "identity.json": native.identity(frozen, prep, actual), "warmup.json": {s: native.CHECKS for s in native.STAGES},
        "process.json": {"python_returncode": 0, "log_returncode": 0, "automatic_admission": False},
        "result.json": {"scope": native.SCOPE, "complete": True, "records": rows,
            "automatic_admission": False, "known_instrumentation_guard_passed": True}}
    content.update({f"attempt_{i:02d}.json": r for i, r in enumerate(rows)})
    for name, value in content.items(): native.save(directory / name, value)


def test_cpu_review_recomputes_and_rejects_contract_tampering(tmp_path):
    write_fixture(tmp_path)
    assert native.review(tmp_path)["status"] == "calibration_candidates_ready"
    observed = native.read(tmp_path / "identity.json")
    observed["stages"]["mul"]["contract_id"] = "tampered"
    native.save(tmp_path / "identity.json", observed)
    with pytest.raises(MeasurementContractError, match="identity or contract"):
        native.review(tmp_path)


def test_no_gpu_failure_retains_plan_and_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    with pytest.raises(RuntimeError, match="HIP GPU0"):
        native.main(["run", "--output", str(tmp_path)])
    assert (tmp_path / "plan.json").exists()
    failure = native.read(tmp_path / "failure.json")
    assert failure["traceback"] and failure["pool"] == native.POOL
