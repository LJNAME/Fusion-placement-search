"""CPU tests; fixture latencies are not measured GPU evidence."""
import copy

import pytest
import torch

from calibration import lightop_local_pair as local
from calibration.tests.test_lightop_profile_calibration import rounds
from cost_model.measurement_labels import MeasurementContractError


def fixture_evaluation():
    frozen = {"predictions": {"analytic": {"cost_us": {"native": 10, "fused": 12}, "chosen": "native"}},
              "order": [["native", "fused"] if i % 2 == 0 else ["fused", "native"] for i in range(7)]}
    prediction = {"predictions": copy.deepcopy(frozen["predictions"]),
                  "prediction_id": local.key_id(frozen["predictions"]), "frozen_monotonic_ns": 0}
    records = []
    for i, order in enumerate(frozen["order"]):
        for arm in order:
            row = rounds([12] * 7 if arm == "native" else [8] * 7)[i]
            row.update(arm=arm, started_monotonic_ns=len(records)*100+1,
                       finished_monotonic_ns=len(records)*100+50)
            records.append(row)
    return frozen, prediction, records


def test_plan_cpu_only_replays_real_l2_admission(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("plan touched GPU"))
    first = local.plan()
    assert first == local.plan()
    assert first["seed"] != local.read(local.RAW / "plan.json")["seed"]
    assert first["predictions"]["L1_analytic"]["chosen"] == "native"
    assert first["predictions"]["L2_fused_calibrated"]["chosen"] == "fused"
    assert sum(s["logical_io_bytes"] for s in first["analytic_stages"]["native"]) == 7864320
    assert sum(s["logical_io_bytes"] for s in first["analytic_stages"]["fused"]) == 11010048


def test_native_preallocated_cpu_numeric_and_storage():
    generator = torch.Generator().manual_seed(40)
    gate, up = [torch.randn(local.SHAPE, dtype=torch.bfloat16, generator=generator) for _ in range(2)]
    before = [t.view(torch.uint8).clone() for t in (gate, up)]
    calls = [local.NativeCall(gate, up) for _ in range(2)]
    references = (torch.nn.functional.silu(gate) * up,
                  torch.nn.functional.silu(gate.float()) * up.float())
    for call in calls:
        pointers = (call._packed.data_ptr(), call.output.data_ptr())
        call()
        call()
        assert pointers == (call._packed.data_ptr(), call.output.data_ptr())
    assert local.check_outputs(torch, calls, gate, up, before, references, {"rtol": .01, "atol": .001})["outputs_checked"] == 2
    assert calls[0].observation["schemas"] == [
        "aten::silu.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
        "aten::mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"]


def test_reference_evaluates_but_does_not_change_frozen_choice():
    frozen, prediction, records = fixture_evaluation()
    snapshot = copy.deepcopy((frozen, prediction))
    result = local.evaluate(frozen, prediction, records)
    assert result["comparisons"]["analytic"]["local_absolute_regret_us"] == 4
    assert result["comparisons"]["analytic"]["frozen_choice"] == "native"
    assert result["comparisons"]["analytic"]["reference_choice"] == "fused"
    assert (frozen, prediction) == snapshot


@pytest.mark.parametrize("bad", ["prediction", "late_freeze", "reordered", "missing", "extra"])
def test_invalid_freeze_or_attempts_rejected(bad):
    frozen, prediction, records = fixture_evaluation()
    if bad == "prediction":
        prediction["predictions"]["analytic"]["chosen"] = "fused"
    elif bad == "late_freeze":
        prediction["frozen_monotonic_ns"] = 10000
    elif bad == "reordered":
        records[:2] = records[1::-1]
    elif bad == "missing":
        records.pop()
    else:
        records.append(copy.deepcopy(records[-1]))
    with pytest.raises(MeasurementContractError):
        local.evaluate(frozen, prediction, records)


@pytest.mark.parametrize("bad", ["event", "unstable"])
def test_quality_failure_does_not_publish_reference_ranking(bad):
    frozen, prediction, records = fixture_evaluation()
    if bad == "event":
        records[0]["timing"]["error"] = "fixture timer failure"
    else:
        for row in records:
            if row["arm"] == "native" and row["round"] > 3:
                row["timing"]["elapsed_ms_per_50_calls"] *= 2
    result = local.evaluate(frozen, prediction, records)
    assert result["status"] == "quality_failed" and result["comparisons"] is None


def test_no_gpu_failure_preserves_freeze_and_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    with pytest.raises(RuntimeError, match="HIP GPU0"):
        local.main(["run", "--output", str(tmp_path)])
    assert local.read(tmp_path / "prediction.json")["prediction_id"]
    failure = local.read(tmp_path / "failure.json")
    assert failure["traceback"] and failure["context"]["pool"] == local.POOL


def test_cpu_review_replays_identity_and_rejects_edited_contract(tmp_path):
    # Actual old implementation metadata, synthetic new round values: CPU fixture only.
    frozen = local.plan()
    _, _, records = fixture_evaluation()
    prediction = {"predictions": frozen["predictions"],
                  "prediction_id": local.key_id(frozen["predictions"]), "frozen_monotonic_ns": 0}
    fused = local.read(local.RAW / "preparation.json")["one_call"]
    native = {"recipe": local.NATIVE_RECIPE,
              "implementation_source_sha256": local.digest(local.Path(local.__file__)),
              "schemas": ["aten::silu.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
                          "aten::mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"],
              "tensors": {k: copy.deepcopy(fused["tensors"]["gate"]) for k in
                          ("gate", "up", "intermediate", "output")}}
    prep = {"native": native, "fused": fused}
    actual = local.read(local.RAW / "runtime.json")
    placed = local.load_placed_op_graph(local.RAW / "placed.json", registry=local.adapter_registry())
    index = local.read(local.RAW / "identity.json")["group_index"]
    identity = local.verify_identity(frozen, placed, index, prep, actual)
    contents = {"plan.json": frozen, "prediction.json": prediction, "preparation.json": prep,
                "runtime.json": actual, "identity.json": identity,
                "warmup.json": {a: local.CHECKS for a in local.ARMS},
                "process.json": {"python_returncode": 0, "log_returncode": 0, "automatic_admission": False},
                "result.json": {"scope": local.SCOPE, "complete": True, "records": records,
                                "automatic_admission": False, "known_instrumentation_guard_passed": True}}
    contents.update({f"attempt_{i:02d}.json": row for i, row in enumerate(records)})
    for name, value in contents.items():
        local.save(tmp_path / name, value)
    local.save_placed_op_graph(tmp_path / "placed.json", placed)
    assert local.review(tmp_path)["status"] == "reference_ready"
    identity["shared_measurement_contract"]["warmup_replays"] = 0
    local.save(tmp_path / "identity.json", identity)
    with pytest.raises(MeasurementContractError, match="identity/contract"):
        local.review(tmp_path)
