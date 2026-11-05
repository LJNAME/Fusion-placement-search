"""Held-out labels, contract exclusions, error metrics and FFN break-even tests."""
import copy

import pytest

from calibration.e2_dataset import make_dataset_view
from calibration.e2_records import load_jsonl_records
from cost_model import profile_prediction_validation as prediction
from cost_model import ffn_decision_sensitivity as sensitivity


@pytest.fixture(scope="module")
def prediction_report():
    return prediction.build_report()


@pytest.fixture(scope="module")
def sensitivity_report():
    return sensitivity.build_report()


def test_metrics_keep_coverage_denominator_and_undefined_values():
    rows = [{"measured_s": 2., "prediction": 3.}, {"measured_s": 100., "prediction": None}]
    result = prediction.metrics(rows, "prediction")
    assert result["coverage"] == .5
    assert result["mape"] == .5 and result["signed_relative_bias"] == .5
    assert result["spearman"] is None
    assert prediction.metrics(rows, "unavailable")["mape"] is None


def split_inputs():
    rows = load_jsonl_records(prediction.ROOT / "data/bw1000_e2/profiling_dataset.jsonl")
    contract = prediction.read("data/bw1000_e2/split_contract.json")
    split = contract["splits"][0]
    view = make_dataset_view(rows, contract, split_id=split["split_id"], mode="zero_shot", role="predictor")
    return ([r for r in view if r["object_id"] in split["train_object_ids"]],
            [r for r in view if r["object_id"] in split["hidden_object_ids"]], contract)


def test_predictor_rejects_unmasked_and_exact_training_key_queries():
    train, queries, contract = split_inputs()
    query = copy.deepcopy(queries[0])
    query["measurement"] = {"duration_s": 0.00001}
    with pytest.raises(ValueError, match="labels must be hidden"):
        prediction.predict_masked(train, [query], contract["prediction_protocol"])
    overlap = copy.deepcopy(train[0])
    overlap.pop("measurement")
    overlap.pop("target")
    with pytest.raises(ValueError, match="overlaps visible Profile"):
        prediction.predict_masked(train, [overlap], contract["prediction_protocol"])


def test_changing_hidden_labels_cannot_change_predictions():
    rows = load_jsonl_records(prediction.ROOT / "data/bw1000_e2/profiling_dataset.jsonl")
    train, queries, contract = split_inputs()
    expected = prediction.predict_masked(train, queries, contract["prediction_protocol"])
    split = contract["splits"][0]
    for row in rows:
        if row["object_id"] in split["hidden_object_ids"]:
            row["measurement"]["duration_s"] *= 1000
            row["target"] = {"log_residual": 99., "ratio": 1e20}
    view = make_dataset_view(rows, contract, split_id=split["split_id"], mode="zero_shot", role="predictor")
    hidden = [r for r in view if r["object_id"] in split["hidden_object_ids"]]
    assert prediction.predict_masked(train, hidden, contract["prediction_protocol"]) == expected


def test_nearest_does_not_fall_back_to_incompatible_implementation():
    train, queries, contract = split_inputs()
    query = copy.deepcopy(queries[0])
    query["kernel_key"]["implementation"]["version"] = "different implementation build"
    result = prediction.predict_masked(train, [query], contract["prediction_protocol"])[query["object_id"]]
    assert result["nearest_s"] is None
    assert not result["hybrid_exact_hit"]
    assert result["hybrid_s"] == result["analytic_s"]


def test_112_eager_keys_are_not_relabelled_as_current_graph_test_data(prediction_report):
    audit = prediction_report["profile_audit"]
    assert audit["legacy_keys"] == sum(audit["operators"].values()) == 112
    assert audit["raw_reconstruction_passed"]
    assert audit["current_graph_compatible_legacy_keys"] == 0
    assert audit["current_graph_independent_test_keys"] == 0
    assert all("eager_event_loop_not_graph_replay" in r["reasons"] for r in audit["legacy_records"])
    for split in prediction_report["legacy_eager_leave_out"]["splits"]:
        assert split["hidden_exact_hits"] == 0
        assert all(not r["hybrid_exact_hit"] for r in split["rows"])


def test_graph_diagnostic_is_assumed_not_independent_or_nearest_coverage(prediction_report):
    graph = prediction_report["graph_retrospective"]
    assert graph["independent_test_keys"] == 0
    assert graph["unresolved_model"]["numeric_prediction_coverage"] == 0
    assert graph["metrics"]["all"]["nearest"]["coverage"] == 0
    assert graph["metrics"]["all"]["nearest"]["mape"] is None
    assert all(r["analytic_s"] > r["measured_s"] for r in graph["rows"])


def test_break_even_brackets_and_witnesses_over_all_48_plans(sensitivity_report):
    assert len(sensitivity_report["baseline_plans"]) == 48
    axes = sensitivity_report["sensitivity_axes"]
    assert axes[0]["switch_bracket"][0] == pytest.approx(.242501061587, abs=1e-10)
    assert axes[1]["switch_bracket"][0] == pytest.approx(.161031723905, abs=1e-10)
    for axis in axes[:2]:
        low, high = axis["switch_bracket"]
        assert 0 < high - low < 1e-11
        left, right = axis["signed_margin_us_at_bracket"]
        assert left < 0 <= right
        assert axis["witnesses"]["below"]["best"]["placement"][sensitivity.ffn.COMPUTE[1]] == 1
        assert set(axis["witnesses"]["above"]["best"]["placement"].values()) == {0}


def test_non_switch_axes_are_limited_to_exploration_interval(sensitivity_report):
    for axis in sensitivity_report["sensitivity_axes"][2:]:
        assert axis["switch_bracket"] is None
        assert axis["exploration_interval_not_confidence_bounds"] == [0, 1]
        assert min(axis["cross_minus_best_zero_communication_us_at_endpoints"].values()) > 0


def test_old_communication_is_not_promoted_to_bounds_or_regret(sensitivity_report):
    evidence = sensitivity_report["communication_evidence"]
    assert evidence["supported_current_graph_scale_interval"] is None
    assert evidence["ranking_robustness_under_credible_bounds"] == "unresolved"
    assert sensitivity_report["independent_reference_regret"] is None
    assert not sensitivity_report["new_gpu_execution"]
    assert not sensitivity_report["minimum_conditional_reference"]["automatic_gpu_collection"]
    assert not any(r["admitted"] for r in evidence["historical_graph_copy_review"]["rows"])
