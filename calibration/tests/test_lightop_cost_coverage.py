"""Replay the archived real Profile on CPU; no new GPU data or search scores."""
import pytest
import torch

from tools.audit_lightop_cost_coverage import build, COMPUTE, FUSED, query_id


@pytest.fixture(scope="module")
def audit():
    return build()


def test_audit_replays_without_touching_a_gpu(monkeypatch, audit):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CPU audit touched GPU"))
    assert build() == audit


def test_no_plan_is_priceable_even_though_eight_have_a_measured_fused_term(audit):
    summary = audit["summary"]
    assert summary["accepted_selected_group_profiles"] == 1
    assert summary["priceable_plans"] == 0 and summary["total_plans"] == 48
    assert summary["plans_with_the_local_fused_cost"] == 8
    assert audit["action_space"]["native_plans"] == 32 and audit["action_space"]["fused_plans"] == 16
    assert audit["action_space"]["structural_classes"] == 24
    assert not audit["action_space"]["merged_for_cost_scoring"]
    assert all(p["formal_plan_cost_us"] is None and not p["scoring_performed"] for p in audit["plan_coverage"])


def test_only_the_exact_gpu0_fusion_can_be_queried(audit):
    accepted = [q for q in audit["compute_queries"] if q["accepted_query"]]
    assert len(accepted) == 1 and accepted[0]["query_id"] == query_id(FUSED, 0)
    assert accepted[0]["measured_us"] == pytest.approx(17.811180353164673)
    assert accepted[0]["candidate_plan_uses"] == 8
    assert len(audit["compute_queries"]) == 12
    assert all(q["measured_us"] is None and q["rejection"] for q in audit["compute_queries"] if not q["accepted_query"])
    assert audit["action_space"]["implementation"]["entrypoint"] == "lightop.op.silu_and_mul_opt"


def test_analytic_transfer_is_not_counted_as_accepted_even_with_strict_group_profile(audit):
    links = audit["communication_queries"]
    assert len(links) == 4 and all(not q["accepted_query"] for q in links)
    assert all(q["existing_query_source"] == "analytic_link" and q["measured_us"] is None for q in links)
    assert {(q["tensor_bytes"], q["source_gpu"], q["destination_gpu"]) for q in links} == {
        (size, src, dst) for size in (524288, 1572864) for src, dst in ((0, 1), (1, 0))}
    assert sum(len(q["witnesses"]) for q in links) == 160
    assert audit["summary"]["logical_edge_rows"] == 336


def test_universal_missing_families_mean_one_device_alternative_per_plan(audit):
    families = audit["summary"]["uncovered_in_every_plan_families"]
    assert {f["node_id"] for f in families} == {COMPUTE[0], COMPUTE[1], COMPUTE[-1]}
    for plan in audit["plan_coverage"]:
        for family in families:
            assert sum(q in plan["missing_compute_queries"] for q in family["device_alternatives"]) == 1


def test_minimal_next_scope_has_no_communication_and_no_unproven_matmul_reuse(audit):
    zero = [p for p in audit["plan_coverage"] if p["cross_device_edges"] == 0]
    assert len(zero) == 2 and {p["fusion_enabled"] for p in zero} == {True, False}
    assert all(not p["missing_link_queries"] for p in zero)
    assert set(audit["measurement_next_steps"]["first_scope_missing_queries"]) == {query_id([n], 0) for n in COMPUTE}
    assert audit["measurement_next_steps"]["task_count"] is None
    assert all(not r["reuse_authorized"] for r in audit["measurement_next_steps"]["same_shape_reuse_candidates"])


def test_wrong_boundary_analytic_number_does_not_become_an_error_or_calibration_factor(audit):
    comparison = audit["same_contract_analytic_audit"]
    assert comparison["status"] == "not_comparable_under_target_contract"
    assert comparison["target_contract_analytic_prediction_us"] is None
    assert comparison["prediction_error_percent"] is comparison["calibration_factor"] is None
    io = comparison["logical_tensor_io_bytes"]
    assert io["ideal_fused_boundary"] == 4718592
    assert io["pack_stage"] == 6291456 and io["lightop_stage"] == 4718592
    assert io["complete_adapter"] == 11010048
    assert io["not_a_measured_hbm_traffic_count"]
