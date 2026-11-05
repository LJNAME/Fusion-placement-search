import pytest

from cost_model import ffn_comm_instability as ci
from cost_model.ffn_decision_sensitivity import FFNSensitivity


def test_describe_keeps_outliers_and_counts_them():
    d = ci.describe([30., 31., 32., 33., 34., 35., 1400.])
    assert d["n"] == 7 and d["max"] == 1400. and d["over_1ms"] == 1 and d["over_3x_median"] == 1
    assert d["mean_including_outliers"] > d["median"]


def test_status_marks_failed_default_sdma_unstable_and_keeps_batches_separate():
    status, control = ci.communication_status()
    assert set(status) == set(ci.BATCHES)
    assert all(v["status"] == "unstable" for v in status["copy_sdma_default"].values())
    assert status["batch01_default_sdma"]["copy/524288/0->1"]["status"] == "quality_passed_single_batch"
    assert all(v["quality"] == "quality_passed" for v in control.values())


def test_scenario_profile_only_overrides_named_transfers_and_gpu0_projection():
    audit = FFNSensitivity()
    rows, local, _ = ci.rank(audit, {})
    assert len(rows) == 48 and local["cost_us"] == pytest.approx(78.83045049003108)
    profile = ci.ScenarioProfile(audit, {(0, 1, 524288.): 10.}, 1.)
    assert profile.estimate_transfer(0, 1, 524288.).duration == pytest.approx(1e-5)
    assert profile.estimate_transfer(1, 0, 524288.) is audit.transfers[1, 0, 524288.]
