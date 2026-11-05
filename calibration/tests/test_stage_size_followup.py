"""CPU checks for the batch-02 Mul diagnostic; synthetic receipts only."""
import copy

import pytest

from calibration import stage_size_calibration as base
from calibration import stage_size_followup as fu
from calibration.tests.test_stage_size_calibration import metadata_fixture
from cost_model.measurement_labels import MeasurementContractError


def records(frozen, value=lambda case: 2.0 + 4e-6 * case["elements"]):
    rows = []
    for attempt, order in enumerate(frozen["order"]):
        for case_id in order:
            case = frozen["cases"][case_id]
            tick = len(rows) * 10 + 1
            us = value(case) * (1 + .001 * (attempt - 3))
            rows.append({"case": case_id, "round": attempt, "started_monotonic_ns": tick,
                "finished_monotonic_ns": tick + 5,
                "timing": {"elapsed_ms_per_50_calls": us * 50 / 1000, "error": None,
                    "event_boundary": {"device": "cuda:0", "start_device": "cuda:0", "end_device": "cuda:0",
                                       "stream": 7, "calls_per_replay": 50}},
                "checks": base.expected_checks(case)})
    return rows


def write(directory, frozen, rows):
    prep, runtime = metadata_fixture(frozen)
    status = fu.evaluate(frozen, rows)["status"]
    content = {"plan.json": frozen, "preparation.json": prep, "runtime.json": runtime,
        "identity.json": base.identity(frozen, prep, runtime),
        "warmup.json": {c: base.expected_checks(k) for c, k in frozen["cases"].items()},
        "process.json": {"python_returncode": 0 if status == "calibration_candidates_ready" else 1,
                         "log_returncode": 0, "automatic_admission": False},
        "result.json": {"scope": fu.SCOPE, "complete": True, "records": rows,
                        "automatic_admission": False, "known_instrumentation_guard_passed": True}}
    content.update({f"attempt_{i:03d}.json": r for i, r in enumerate(rows)})
    for name, value in content.items():
        base.save(directory / name, value)


def test_plan_is_diagnostic_and_pins_batch01_fit():
    p = fu.plan()
    assert p == fu.plan()
    assert p["sizes"] == [192, 256, 384, 768] and len(p["cases"]) == 8
    assert sum(len(o) for o in p["order"]) == 56
    assert {c["role"] for c in p["cases"].values()} == {"anchor", "diagnostic"}
    assert p["cases"]["mul/S256"]["role"] == "anchor"
    assert p["batch01_fit_sha256"] == base.digest(fu.ROOT / fu.BATCH01_FIT)
    assert p["contract"] == base.contract() and p["recipes"]["mul"] == base.RECIPES["mul"]
    assert not p["model_parameters_updated"]


def test_review_recomputes_and_rejects_tampering(tmp_path):
    frozen = fu.plan()
    write(tmp_path, frozen, records(frozen))
    assert fu.review(tmp_path)["status"] == "calibration_candidates_ready"
    row = base.read(tmp_path / "attempt_003.json")
    row["timing"]["elapsed_ms_per_50_calls"] *= 2
    base.save(tmp_path / "attempt_003.json", row)
    with pytest.raises(MeasurementContractError):
        fu.review(tmp_path)


def test_diagnose_uses_frozen_line_and_never_refits(tmp_path):
    frozen = fu.plan()
    write(tmp_path, frozen, records(frozen))
    evaluation = fu.review(tmp_path)
    fit = base.read(fu.ROOT / fu.BATCH01_FIT)
    batch01 = base.read(fu.ROOT / fu.BATCH01_REVIEW)
    snapshot = copy.deepcopy(fit)
    report = fu.diagnose(evaluation, fit, batch01)
    assert fit == snapshot and report["refit"] is False
    mul = {e["stage"]: e for e in fit["entries"]}["mul"]
    row = next(r for r in report["stages"]["mul"]["rows"] if r["batch"] == "batch02" and r["S"] == 384)
    line = mul["a_us_composite_for_prediction"] + mul["b_us_per_element"] * 384 * 3072
    assert row["line_us"] == pytest.approx(line)
    assert row["residual_us"] == pytest.approx(2.0 + 4e-6 * 384 * 3072 - line)
    assert len(report["stages"]["mul"]["rows"]) == 9  # 5 batch-01 + 4 batch-02
    assert report["stages"]["mul"]["anchor_S256"]["batch01_us"] == batch01["cases"]["mul/S256"]["statistics"]["median_us"]
