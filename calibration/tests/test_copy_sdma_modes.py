"""CPU checks for the dual SDMA-mode Copy collector; synthetic receipts only."""
import pytest

from calibration import copy_sdma_modes as cm
from calibration import stage_size_calibration as base
from calibration.tests.test_stage_size_calibration import metadata_fixture
from cost_model.fusion_kernel_profile import key_id
from cost_model.measurement_labels import MeasurementContractError


def records(frozen, spikes=()):
    rows = []
    for attempt, order in enumerate(frozen["order"]):
        for case_id in order:
            case = frozen["cases"][case_id]
            tick = len(rows) * 10 + 1
            us = 1500. if (case_id, attempt) in spikes else 30. * (1 + .001 * (attempt - 3))
            rows.append({"case": case_id, "round": attempt, "started_monotonic_ns": tick,
                "finished_monotonic_ns": tick + 5,
                "timing": {"elapsed_ms_per_50_calls": us * 50 / 1000, "error": None,
                    "event_boundary": {"device": case["timing_device"], "start_device": case["timing_device"],
                                       "end_device": case["timing_device"], "stream": 7, "calls_per_replay": 50}},
                "checks": base.expected_checks(case)})
    return rows


def write(directory, mode, spikes=()):
    frozen = cm.plan(mode)
    rows = records(frozen, spikes)
    prep, runtime = metadata_fixture(frozen)
    runtime["environment"] = {"HSA_ENABLE_SDMA": cm.MODES[mode]}
    status = cm.evaluate(frozen, rows)["status"]
    content = {"plan.json": frozen, "preparation.json": prep, "runtime.json": runtime,
        "identity.json": base.identity(frozen, prep, runtime),
        "warmup.json": {c: base.expected_checks(k) for c, k in frozen["cases"].items()},
        "process.json": {"python_returncode": 0 if status == "calibration_candidates_ready" else 1,
                         "log_returncode": 0, "automatic_admission": False},
        "result.json": {"scope": cm.SCOPE, "mode": mode, "complete": True, "records": rows,
                        "automatic_admission": False}}
    content.update({f"attempt_{i:03d}.json": r for i, r in enumerate(rows)})
    for name, value in content.items():
        base.save(directory / name, value)


def test_modes_are_distinct_contracts_with_four_cases_each():
    a, b = cm.plan("sdma_default"), cm.plan("sdma_disabled")
    assert key_id(a["contract"]) != key_id(b["contract"])
    assert a["role"] == "primary" and b["role"] == "control"
    assert "never replaces" in b["cost_model_use"]
    for p in (a, b):
        assert len(p["cases"]) == 4 and sum(len(o) for o in p["order"]) == 28
        assert {(c["bytes"], c["source"], c["destination"]) for c in p["cases"].values()} == set(base.COPIES)


def test_environment_must_match_mode(monkeypatch):
    monkeypatch.delenv("HSA_ENABLE_SDMA", raising=False)
    cm.check_environment("sdma_default")
    with pytest.raises(RuntimeError):
        cm.check_environment("sdma_disabled")
    monkeypatch.setenv("HSA_ENABLE_SDMA", "0")
    cm.check_environment("sdma_disabled")
    with pytest.raises(RuntimeError):
        cm.check_environment("sdma_default")


def test_outliers_counted_not_removed(tmp_path):
    spikes = {("copy/524288/1->0", 0), ("copy/524288/1->0", 4)}
    write(tmp_path, "sdma_default", spikes)
    result = cm.review(tmp_path)
    q = result["cases"]["copy/524288/1->0"]
    assert len(q["samples_us"]) == 7 and q["outliers"]["over_1ms"] == 2
    assert result["mode_outliers"] == {"over_3x_case_median": 2, "over_1ms": 2, "samples": 28}
    assert q["status"] == "quality_failed"


def test_review_rejects_mode_environment_mismatch(tmp_path):
    write(tmp_path, "sdma_disabled")
    assert cm.review(tmp_path)["mode"] == "sdma_disabled"
    runtime = base.read(tmp_path / "runtime.json")
    runtime["environment"] = {"HSA_ENABLE_SDMA": None}
    base.save(tmp_path / "runtime.json", runtime)
    with pytest.raises(MeasurementContractError, match="SDMA"):
        cm.review(tmp_path)
