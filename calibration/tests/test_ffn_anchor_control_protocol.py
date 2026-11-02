"""Frozen CPU pilot semantics; these fixtures are never GPU evidence."""
import pytest
from calibration.ffn_anchor_control_protocol import PROTOCOL, endpoint_summary, frozen_protocol


def sample(value):
    return {"reset_begin":0,"timing_begin":1,"event_wait_end":2,"all_device_sync_end":3,"validation_end":4,
            "numeric_ok":True,"calls":50,"batch_ms":value*.05}


def test_pilot_is_frozen_single_factor_and_gpu_closed():
    p=frozen_protocol()
    assert p==frozen_protocol() and not p["gpu_entry_open"] and not p["gpu_collector_ready"]
    assert [(c["row"],c["endpoint_replays"]) for c in p["cells"]]==[
        ("down_proj@gpu0",1),("up_proj@gpu0",3),("up_proj@gpu0",1),("down_proj@gpu0",3)]
    assert p["fixed"]["attempts"]==7 and p["round"]["drift_limit"]==.1


def test_all_repeats_retained_median_and_dispersion_defined():
    result=endpoint_summary([sample(6.),sample(7.),sample(8.)],3)
    assert result["median_us"]==pytest.approx(7.) and result["iqr_over_median"]==pytest.approx(1/7)
    assert not result["endpoint_valid"]
    assert endpoint_summary([sample(7.)],1)["iqr_over_median"] is None


def test_no_extra_repeats_or_missing_timestamps():
    with pytest.raises(ValueError): endpoint_summary([sample(7.)]*4,3)
    s=sample(7.);del s["validation_end"]
    with pytest.raises(ValueError): endpoint_summary([s],1)
    s=sample(7.);s["event_wait_end"]=9
    with pytest.raises(ValueError): endpoint_summary([s],1)
