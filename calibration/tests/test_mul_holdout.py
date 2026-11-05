"""CPU checks for the Mul holdout collector; synthetic receipts only."""
import pytest

from calibration import mul_holdout as mh
from calibration import stage_size_calibration as base
from calibration.tests.test_stage_size_calibration import metadata_fixture
from cost_model.measurement_labels import MeasurementContractError


def records(frozen, us_by_case):
    rows = []
    for attempt, order in enumerate(frozen["order"]):
        for case_id in order:
            tick = len(rows) * 10 + 1
            us = us_by_case[case_id] * (1 + .001 * (attempt - 3))
            rows.append({"case": case_id, "round": attempt, "started_monotonic_ns": tick,
                "finished_monotonic_ns": tick + 5, "host_load_1min": 1.0,
                "timing": {"elapsed_ms_per_50_calls": us * 50 / 1000, "error": None,
                    "event_boundary": {"device": "cuda:0", "start_device": "cuda:0", "end_device": "cuda:0",
                                       "stream": 7, "calls_per_replay": 50}},
                "checks": base.expected_checks(frozen["cases"][case_id])})
    return rows


def write(directory, frozen, rows):
    prep, runtime = metadata_fixture(frozen)
    status = mh.evaluate(frozen, rows)["status"]
    content = {"plan.json": frozen, "preparation.json": prep, "runtime.json": runtime,
        "identity.json": base.identity(frozen, prep, runtime), "load_gate.json": {"passed": True},
        "warmup.json": {c: base.expected_checks(k) for c, k in frozen["cases"].items()},
        "process.json": {"python_returncode": 0 if status == "calibration_candidates_ready" else 1,
                         "log_returncode": 0, "automatic_admission": False},
        "result.json": {"scope": mh.SCOPE, "complete": True, "records": rows, "automatic_admission": False}}
    content.update({f"attempt_{i:03d}.json": r for i, r in enumerate(rows)})
    for name, value in content.items():
        base.save(directory / name, value)


def test_plan_pins_frozen_predictions():
    p = mh.plan()
    assert p["sizes"] == [320, 416, 640, 896] and len(p["cases"]) == 4
    assert set(p["frozen_predictions"]) == set(p["cases"])
    assert p["frozen_prediction_id"].startswith("sha256:b875ac16")
    assert p["comparison"]["refit"] is False


def test_comparison_uses_frozen_predictions_only(tmp_path):
    frozen = mh.plan()
    truth = {c: v["piecewise_us"] for c, v in frozen["frozen_predictions"].items()}
    write(tmp_path, frozen, records(frozen, truth))
    comparison = mh.review(tmp_path)["holdout_comparison"]
    assert comparison["summary"]["piecewise_us"]["mape"] == pytest.approx(0, abs=1e-9)
    assert comparison["summary"]["baseline_linear_us"]["coverage"] == "4/4"
    plan = base.read(tmp_path / "plan.json")
    plan["frozen_predictions"]["mul/S320"]["piecewise_us"] += 1
    base.save(tmp_path / "plan.json", plan)
    with pytest.raises(MeasurementContractError):
        mh.review(tmp_path)
