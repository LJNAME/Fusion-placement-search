"""Place-then-fuse baseline: same budget split, frozen placement, cross-device fusion rejected."""
import pytest

from cost_model.hardware_profile import AnalyticHardwareProfile
from cost_model.search_evaluation import (
    _CachedScorer, _freeze_placement, _modes, _run_search, build_evaluation_graph, placement_then_fusion_mode)
from simulator import FusionGroup, Plan
from simulator.hardware import load_hardware


@pytest.fixture(scope="module")
def setup():
    graph = build_evaluation_graph("small")
    profile = AnalyticHardwareProfile(load_hardware("data/bw1000_2card/bw1000_2card.calibrated.json"))
    return graph, profile


def test_existing_four_modes_unchanged():
    assert [m.name for m in _modes(2)] == ["placement_only", "fusion_only", "sequential", "joint"]


def test_freeze_rejects_fusion_across_devices(setup):
    graph, _ = setup
    order = list(graph.topological_order())
    placement = Plan(groups=[FusionGroup([n], i % 2, {"pattern_id": "unfused"}) for i, n in enumerate(order)])
    expand, devices = _freeze_placement(placement)
    a, b = order[0], order[1]
    fused = Plan(groups=[FusionGroup([a, b], 0, {"pattern_id": "x"}),
                         *[FusionGroup([n], 0, {"pattern_id": "unfused"}) for n in order[2:]]])
    assert expand(fused) is None
    single = Plan(groups=[FusionGroup([n], 0, {"pattern_id": "unfused"}) for n in order])
    assert [g.device for g in expand(single).groups] == [devices[n] for n in order]


@pytest.mark.parametrize("method", ["mcts", "exact"])
def test_staged_run_keeps_frozen_placement_and_splits_budget(setup, method):
    graph, profile = setup
    scorer = _CachedScorer(graph, profile)
    plans, _, budget = _run_search(method, graph, scorer, placement_then_fusion_mode(2), seed=0, beam=4,
                                   rollouts=2, mcts_iterations=16, top_k=4)
    assert plans and [s["stage"] for s in budget["stages"]] == ["placement", "fusion"]
    frozen = budget["frozen_placement"]
    assert all(g.device == frozen[n] for p in plans for g in p.groups for n in g.node_ids)
    if method == "mcts":
        assert [s["iterations"] for s in budget["stages"]] == [8, 8]
