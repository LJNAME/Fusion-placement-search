"""Reject inconsistent raw units and identity joins in returned B4 evidence."""
from types import SimpleNamespace
import pytest

from calibration.ffn_b4 import contract
from cost_model.fusion_kernel_profile import key_id
from cost_model.measurement_labels import measurement_contract_id
from tools.review_ffn_b4_run import verify_sample_binding


def fixture():
    identity = {"devices": {"0": {"environment": {"torch": "fixture", "hip": "fixture", "triton": "fixture", "device_model": "fixture"}}}}
    measured = contract({"candidate_id": "fixture"}, SimpleNamespace(device=SimpleNamespace(index=0), identity=identity),
                        {"active_recipe": {"recipe_id": "fixture"}})
    samples = [{"attempt": i, "numeric_ok": True, "capture_ok": True, "duration_us": 10.,
                "invariants": {"fixture_only": True}, "path_evidence_state": "pending_post_timing_trace",
                "raw_timing": {"elapsed_ms": .5, "event_boundary": {
                    "device": "cuda:0", "start_device": "cuda:0", "end_device": "cuda:0", "calls_per_replay": 50}}} for i in range(7)]
    row = {"identity": identity, "identity_hash": key_id(identity), "measurement_contract": measured,
           "measurement_contract_id": measurement_contract_id(measured), "samples": samples,
           "path_evidence_phase": "after_all_group_timing", "path_verified": True}
    return row, samples, {"identity_hash": key_id(identity)}


def test_valid_binding():
    verify_sample_binding(*fixture())


def test_wrong_ms_to_us_conversion_rejected_even_when_both_sample_copies_match():
    row, samples, identity = fixture()
    samples[0]["duration_us"] = 20.
    with pytest.raises(AssertionError): verify_sample_binding(row, samples, identity)


def test_contract_tamper_rejected():
    row, samples, identity = fixture()
    row["measurement_contract"]["calls_per_round"] = 100
    with pytest.raises(AssertionError): verify_sample_binding(row, samples, identity)


def test_mismatched_event_device_rejected():
    row, samples, identity = fixture()
    samples[0]["raw_timing"]["event_boundary"]["end_device"] = "cuda:1"
    with pytest.raises(AssertionError): verify_sample_binding(row, samples, identity)
