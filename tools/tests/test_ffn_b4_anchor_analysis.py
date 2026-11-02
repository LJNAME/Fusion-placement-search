"""Synthetic CPU fixtures for shared-window counting; no measurement labels."""
import copy
import pytest

from tools.analyze_ffn_b4_anchor_failures import coverage_diagnostic, describe, windows_and_operators


def fixture(devices=("gpu0",)):
    samples = [{"attempt": i, "duration_us": 10., "anchors": {
        gpu: {"before_us": 100., "after_us": 111. if i == 2 else 100.} for gpu in devices}} for i in range(7)]
    grid = {"cells": {name: {"status": "quality_failed", "samples": copy.deepcopy(samples)} for name in ("a", "b")}}
    plan = {"active_recipe": {"anchor_drift_limit": .1}, "matrix": {"round_groups": [
        {"group_id": "shared", "members": ["a", "b"]}]}}
    return grid, plan


def test_shared_anchors_count_once_not_once_per_operator():
    windows, operators, groups = windows_and_operators(*fixture())
    assert len(windows) == 7 and sum(w["rejected"] for w in windows) == 1
    assert len(operators) == 14 and groups[0]["failed_cell_count"] == 2


def test_two_rejected_devices_are_one_rejected_group_window():
    windows, _, _ = windows_and_operators(*fixture(("gpu0", "gpu1")))
    bad = [w for w in windows if w["rejected"]]
    assert len(bad) == 2 and len({(w["group_id"], w["attempt"]) for w in bad}) == 1


def test_disagreeing_shared_window_records_rejected():
    grid, plan = fixture()
    grid["cells"]["b"]["samples"][2]["anchors"]["gpu0"]["after_us"] = 112.
    with pytest.raises(AssertionError): windows_and_operators(grid, plan)


def test_same_median_does_not_imply_small_pointwise_range():
    measured = describe([10., 10., 10., 10., 10., 10., 12.5])
    assert measured["median"] == 10. and measured["range_over_median"] == .25
    assert measured["iqr_over_median"] == 0.


def test_missing_device_alternatives_are_not_exact_term_intersection():
    result = coverage_diagnostic({"plans": [
        {"missing_quality_passed_rows": ["down_proj@gpu0"]},
        {"missing_quality_passed_rows": ["down_proj@gpu1"]}]})
    assert result["priceable_plans"] == 0 and result["legal_plans"] == 2
    assert result["exact_missing_term_intersection"] == []
    assert result["counterfactual_only_both_down_available"] == 2
