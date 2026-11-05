"""FFN boundary costs, provenance and search consistency; CPU development only."""
import copy

import pytest

from cost_model import ffn_profile_search as ffn
from simulator import FusionGroup, Plan
from integration.lightop.silu_mul import adapter_registry


@pytest.fixture(scope="module")
def report():
    return ffn.build_report()


def test_all_48_plans_have_sources_but_not_complete_measured_coverage(report):
    assert len(report["scenario_results"]) == 5
    for scenario in report["scenario_results"]:
        assert len({p["plan_id"] for p in scenario["plans"]}) == 48
        assert sum(p["fused"] for p in scenario["plans"]) == 16
        assert scenario["scoreable_plans"] == 48
        assert scenario["profile_only_plans"] == 0
        assert scenario["coverage"]["measured_slots_under_declared_execution_contract"] == 3
        assert scenario["coverage"]["analytic_compute_slots"] == 9
        assert scenario["coverage"]["communication_types"] == 4
        for q in scenario["compute_queries"].values():
            assert bool(q["profile_identity_id"]) != bool(q["assumption_id"])
            if q["device"] == 1:
                assert q["source"].startswith("analytic_")
        for plan in scenario["plans"]:
            groups = plan["scoring_groups"]
            assert groups[0]["nodes"] == [ffn.INPUT] and groups[0]["device"] == 0
            assert groups[-1]["nodes"] == [ffn.OUTPUT] and groups[-1]["device"] == 0
    assert report["trusted_end_to_end_reference"] is None
    assert report["independent_regret"] is None
    assert not report["srop_enabled"]


def test_down_on_gpu1_includes_both_ingress_and_egress_costs():
    config = ffn.json.loads(ffn.CONFIG.read_text(encoding="utf-8"))
    _, full, hardware, measured, _ = ffn.build_case(config)
    base = ffn.FFNProxyBase(hardware, measured, config, config["scenarios"][0])
    profile = ffn.NativeMeasuredFirstProfile(full, base, measured.store, measured.bindings)
    plan = Plan(groups=[FusionGroup([node], int(node == ffn.COMPUTE[-1]), {"pattern_id": "unfused"}) for node in ffn.COMPUTE])
    estimate = ffn.StructuralCostModel(registry=adapter_registry()).predict(full, ffn.with_boundaries(plan), profile)
    transfers = estimate.details["transfers"]
    assert sorted((t["src_device"], t["dst_device"], t["bytes"]) for t in transfers) == [
        (0, 1, 1572864), (1, 0, 524288)]


def test_common_gpu0_projection_cost_cancels_in_local_native_fused_difference(report):
    results = report["scenario_results"]
    baseline = results[0]["all_gpu0_comparison"]
    stressed = results[1]["all_gpu0_comparison"]
    assert stressed["native_us"] > baseline["native_us"]
    assert stressed["fused_minus_native_us"] == pytest.approx(baseline["fused_minus_native_us"])


def test_unmeasured_link_assumption_can_change_placement(report):
    assert report["sensitivity"][2]["placement_changed"]
    assert not report["sensitivity"][2]["fusion_changed"]
    assert report["scenario_results"][2]["best"]["placement"][ffn.COMPUTE[1]] == 1


def test_existing_mcts_reaches_current_model_exact_score_without_exact_cache(report):
    for scenario in report["scenario_results"]:
        assert all(r["cost_model_gap_us"] == pytest.approx(0, abs=1e-9) for r in scenario["mcts"])
        assert all(0 < r["unique_candidates_scored"] <= 48 for r in scenario["mcts"])


def test_scheduling_priority_is_not_confused_with_intrinsic_fusion_gain(report):
    scenario = report["scenario_results"][0]
    assert scenario["same_placement_fusion_pairs"]["count"] == 16
    diagnostic = scenario["priority_order_diagnostic"]
    assert diagnostic["fused_canonical_us"] < diagnostic["native_canonical_us"]
    assert diagnostic["native_gate_up_before_activation_us"] < diagnostic["fused_canonical_us"]
    assert diagnostic["same_cost_sources_and_placement"]
    assert not diagnostic["added_to_search_space"]


def test_layout_is_a_planning_declaration_not_a_projection_runtime_observation():
    config = ffn.json.loads(ffn.CONFIG.read_text(encoding="utf-8"))
    _, full, hardware, _, _ = ffn.build_case(config)
    assert full.nodes[ffn.COMPUTE[0]].extra["layout_provenance"]["kind"] == "graph_declaration"
    assert hardware.launch_overhead == config["projection_graph_overhead_us"] / 1e6


def test_replay_is_deterministic():
    assert ffn.build_report(seeds=()) == ffn.build_report(seeds=())
