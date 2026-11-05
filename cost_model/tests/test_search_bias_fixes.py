"""Optional unfused guided rollout and the staged-search stage-one incumbent."""
import pytest

from cost_model.hardware_profile import AnalyticHardwareProfile
from cost_model.legal_search import LegalFusionSearchSpace, _unfused_guided_rollout_legal_plan, legal_mcts_search
from cost_model.search_evaluation import (
    _CachedScorer, _modes, _plan_key, _run_search, build_evaluation_graph, placement_then_fusion_mode)
from simulator import validate_fusion_plan
from simulator.fusion_rules import default_fusion_registry
from simulator.hardware import load_hardware


@pytest.fixture(scope="module")
def layer():
    return build_evaluation_graph("layer")


def fused_count(plan):
    return float(sum(len(g.node_ids) > 1 for g in plan.groups) + sum(g.device for g in plan.groups))


def test_unfused_completion_is_legal_and_colocated(layer):
    reg = default_fusion_registry()
    space = LegalFusionSearchSpace(layer, 2, reg, require_executable=False)
    plan = _unfused_guided_rollout_legal_plan(space, space.initial_state())
    validate_fusion_plan(layer, plan, 2, registry=reg)
    assert all(len(g.node_ids) == 1 and g.device == 0 for g in plan.groups)


def test_flag_default_off_reproduces_previous_results(layer):
    kwargs = dict(graph=layer, scorer=fused_count, n_devs=2, iterations=16, rollouts=2, seed=3,
                  registry=default_fusion_registry(), require_executable=False)
    assert [_plan_key(p) for p in legal_mcts_search(**kwargs)] == \
           [_plan_key(p) for p in legal_mcts_search(**kwargs, unfused_guided_rollout=False)]


def test_flag_lets_mcts_reach_no_fusion_when_fusion_is_penalised(layer):
    kwargs = dict(graph=layer, scorer=fused_count, n_devs=2, iterations=4, rollouts=1, seed=0,
                  registry=default_fusion_registry(), require_executable=False)
    assert fused_count(legal_mcts_search(**kwargs)[0]) >= 1  # fuse-max completion only
    assert fused_count(legal_mcts_search(**kwargs, unfused_guided_rollout=True)[0]) == 0


@pytest.mark.parametrize("mode", ["sequential", "placement_then_fusion"])
def test_staged_result_never_worse_than_feasible_stage_one_plan(layer, mode):
    profile = AnalyticHardwareProfile(load_hardware("data/bw1000_2card/bw1000_2card.calibrated.json"))
    scorer = _CachedScorer(layer, profile)
    m = placement_then_fusion_mode(2) if mode == "placement_then_fusion" else next(x for x in _modes(2) if x.name == mode)
    plans, _, budget = _run_search("mcts", layer, scorer, m, seed=0, beam=4, rollouts=2, mcts_iterations=8, top_k=2)
    log = budget["stage_one_incumbent"]
    best = scorer.estimate(plans[0]).makespan
    if log["feasible"]:
        assert best <= log["makespan_s"] + 1e-15
    assert [s["stage"] for s in budget["stages"]] == (["fusion", "placement"] if mode == "sequential"
                                                      else ["placement", "fusion"])
