"""Regression checks for the bounded FFN scheduling audit, without GPU work."""
import copy

import pytest

from cost_model.ffn_schedule_audit import build_report, check_resources
from cost_model.structural import StructuralCostModel
from cost_model.tests.test_structural_cost import _FixedProfile, _graph, _plan


@pytest.fixture(scope="module")
def report():
    return build_report()


def test_all_legal_topological_orders_and_graph_reversal_have_identical_traces(report):
    audits = report["scheduling_audit"]["per_scenario"]
    assert len(audits) == 5
    assert sum(a["representations_checked_including_graph_reversal"] for a in audits) == 1280
    for audit in audits:
        assert len(audit["plans"]) == 48
        assert sum(p["legal_topological_orders"] for p in audit["plans"]) == 128
        assert all(p["trace_invariant"] for p in audit["plans"])
        assert audit["trace_and_resource_checks_passed"]
        assert audit["cost_inputs_unchanged"]


def test_rankings_change_but_optimal_decisions_and_mcts_agreement_are_preserved(report):
    for audit, scenario in zip(report["scheduling_audit"]["per_scenario"], report["scenario_results"]):
        assert audit["scores_changed_vs_legacy"] in (8, 12)
        assert audit["ranking_changed"]
        assert not audit["optimal_plan_changed"]
        assert scenario["same_placement_fusion_pairs"]["fused_lower_under_fixed_policy"] == 0
        assert all(run["cost_model_gap_us"] == 0 for run in scenario["mcts"])
        assert all(run["unique_candidates_scored"] == 48 for run in scenario["mcts"])
    assert not report["new_gpu_execution"]


def test_resource_audit_rejects_a_dependency_violation():
    graph, plan, profile = _graph(), _plan(), _FixedProfile()
    estimate = StructuralCostModel().predict(graph, plan, profile)
    check_resources(graph, plan, estimate, profile)
    corrupt = copy.deepcopy(estimate)
    corrupt.details["groups"][1]["start"] = 0.
    with pytest.raises(AssertionError):
        check_resources(graph, plan, corrupt, profile)


def test_legacy_is_explicit_and_unknown_policy_is_rejected():
    model = StructuralCostModel()
    assert model.scheduling_policy == "plan_order_v1"
    assert model.predict(_graph(), _plan(), _FixedProfile()).makespan == 3.5
    with pytest.raises(ValueError, match="unknown scheduling policy"):
        StructuralCostModel(scheduling_policy="unregistered")
