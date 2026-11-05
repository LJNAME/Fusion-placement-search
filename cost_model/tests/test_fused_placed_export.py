import copy

import pytest

from cost_model.hardware_profile import AnalyticHardwareProfile
from cost_model.search_evaluation import build_evaluation_graph
from cost_model.structural import StructuralCostModel
from simulator import FusionGroup, Plan, make_reference_plan
from simulator.fused_placed_export import export_fused_placed_graph, validate_fused_placed_graph
from simulator.fusion_rules import default_fusion_registry, enumerate_fusion_candidates
from simulator.hardware import load_hardware
from simulator.placement_artifacts import PlacedOpGraph, load_placed_op_graph, save_placed_op_graph


@pytest.fixture(scope="module")
def placed():
    graph = build_evaluation_graph("layer")
    hardware = load_hardware("data/bw1000_2card/bw1000_2card.calibrated.json")
    fused = next(c for c in enumerate_fusion_candidates(graph, default_fusion_registry()) if len(c.node_ids) > 1)
    groups = []
    for node in graph.topological_order():
        if node in fused.node_ids[:-1]:
            continue
        if node == fused.node_ids[-1]:
            groups.append(fused.to_group(1))
        else:
            groups.append(FusionGroup([node], int("mlp" in node), {"pattern_id": "unfused"}))
    plan = Plan(groups=groups)
    estimate = StructuralCostModel().predict(graph, plan, AnalyticHardwareProfile(hardware))
    return PlacedOpGraph(graph, hardware, plan, predicted_makespan=estimate.makespan,
                         predicted_peak_memory=estimate.peak_memory), estimate


def test_export_is_consistent_and_honest_about_deployability(placed):
    p, estimate = placed
    artifact = export_fused_placed_graph(p, estimate)
    validate_fused_placed_graph(artifact, p.graph)
    s = artifact["summary"]
    assert s["nodes"] == len(p.graph.nodes) and s["fused_groups"] == 1 and s["groups"] == s["nodes"] - 1
    assert s["cross_device_dependencies"] == len(artifact["communication"]) > 0
    assert artifact["deployable"] is False and len(artifact["non_executable_groups"]) == 1  # silu_mul is "planned"
    assert artifact["execution_order"]["source"].startswith("cost_model_schedule:")
    assert artifact["predicted"]["makespan_s"] == pytest.approx(estimate.makespan)


def test_export_without_estimate_uses_topological_order(placed):
    p, _ = placed
    artifact = export_fused_placed_graph(p)
    validate_fused_placed_graph(artifact, p.graph)
    assert artifact["execution_order"]["source"] == "group_topological_order"


def test_unfused_single_device_plan_is_deployable(placed):
    p, _ = placed
    plan = make_reference_plan(p.graph, device=0)
    estimate = StructuralCostModel().predict(p.graph, plan, AnalyticHardwareProfile(p.hardware))
    artifact = export_fused_placed_graph(PlacedOpGraph(p.graph, p.hardware, plan), estimate)
    assert artifact["deployable"] is True and artifact["communication"] == []


@pytest.mark.parametrize("tamper", ["order", "comm", "duplicate", "deployable"])
def test_validator_rejects_tampering(placed, tamper):
    p, estimate = placed
    artifact = copy.deepcopy(export_fused_placed_graph(p, estimate))
    if tamper == "order":
        order = artifact["execution_order"]["per_device"]["0"]
        order[0], order[1] = order[1], order[0]
    elif tamper == "comm":
        artifact["communication"][0]["bytes"] += 1
    elif tamper == "duplicate":
        artifact["groups"][1]["node_ids"].append(artifact["groups"][0]["node_ids"][0])
    else:
        artifact["deployable"] = True
    with pytest.raises(ValueError):
        validate_fused_placed_graph(artifact, p.graph)


def test_round_trip_through_existing_placed_artifact(placed, tmp_path):
    p, _ = placed
    save_placed_op_graph(tmp_path / "placed.json", p)
    loaded = load_placed_op_graph(tmp_path / "placed.json")
    assert export_fused_placed_graph(loaded) == export_fused_placed_graph(p)
