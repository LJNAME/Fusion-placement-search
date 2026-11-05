"""CPU checks for batch-03; synthetic receipts only."""
import pytest

from calibration import stage_size_batch03 as b3
from calibration import stage_size_calibration as base
from calibration import stage_size_followup as b02
from calibration.tests.test_stage_size_calibration import metadata_fixture
from calibration.tests.test_stage_size_followup import records as b02_records
from cost_model.measurement_labels import MeasurementContractError


def records(frozen):
    rows = b02_records(frozen)
    for r in rows:
        r["host_load_1min"] = 1.0
    return rows


def write(directory):
    frozen = b3.plan()
    rows = records(frozen)
    prep, runtime = metadata_fixture(frozen)
    status = b3.evaluate(frozen, rows)["status"]
    content = {"plan.json": frozen, "preparation.json": prep, "runtime.json": runtime,
        "identity.json": base.identity(frozen, prep, runtime), "load_gate.json": {"passed": True},
        "warmup.json": {c: base.expected_checks(k) for c, k in frozen["cases"].items()},
        "process.json": {"python_returncode": 0 if status == "calibration_candidates_ready" else 1,
                         "log_returncode": 0, "automatic_admission": False},
        "result.json": {"scope": b3.SCOPE, "complete": True, "records": rows, "automatic_admission": False}}
    content.update({f"attempt_{i:03d}.json": r for i, r in enumerate(rows)})
    for name, value in content.items():
        base.save(directory / name, value)


def test_plan_sizes_roles_and_gate():
    p = b3.plan()
    assert p == b3.plan() and p["sizes"] == [256, 384, 448, 512]
    assert p["cases"]["mul/S448"]["role"] == "diagnostic"
    assert p["cases"]["mul/S512"]["role"] == "drift_repeat"
    assert p["contract"] == base.contract() and p["load_gate"]["max_load_1min"] == 4.0
    assert p["protocol"]["piecewise_fit"] is False


def test_load_gate_waits_then_fails_without_relaxing():
    t = [0.]
    loads = iter([9., 6., 3.])
    gate = {"max_load_1min": 4., "poll_seconds": 30, "max_wait_seconds": 3600}
    result = b3.wait_for_quiet_host(gate, clock=lambda: t[0],
        sleep=lambda s: t.__setitem__(0, t[0] + s), load=lambda: (next(loads), 0, 0))
    assert result["passed"] and result["waited_seconds"] == 60 and result["polls"] == [9., 6., 3.]
    t[0] = 0.
    with pytest.raises(RuntimeError, match="no GPU execution"):
        b3.wait_for_quiet_host({**gate, "max_wait_seconds": 60}, clock=lambda: t[0],
            sleep=lambda s: t.__setitem__(0, t[0] + s), load=lambda: (5., 0, 0))


def test_review_and_diagnose_cover_three_batches(tmp_path):
    write(tmp_path)
    evaluation = b3.review(tmp_path)
    batches = (("batch01", base.read(b3.ROOT / b02.BATCH01_REVIEW)["cases"]),
               ("batch02", base.read(b3.ROOT / b3.BATCH02_REVIEW)["cases"]))
    report = b3.diagnose(evaluation, base.read(b3.ROOT / b02.BATCH01_FIT), batches)
    assert report["refit"] is False
    assert {r["batch"] for r in report["stages"]["mul"]["rows"]} == {"batch01", "batch02", "batch03"}
    assert "S512:batch01->batch03" in report["stages"]["mul"]["drift"]
    assert report["same_batch_mul_jump"]["element_ratio_512_over_384"] == pytest.approx(4 / 3)
    gate = base.read(tmp_path / "load_gate.json"); gate["passed"] = False
    base.save(tmp_path / "load_gate.json", gate)
    with pytest.raises(MeasurementContractError, match="Load gate"):
        b3.review(tmp_path)
