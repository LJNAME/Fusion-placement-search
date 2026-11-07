import pytest

from cost_model.hardware_profile import AnalyticHardwareProfile
from cost_model.search_evaluation import _CachedScorer, build_evaluation_graph
from cost_model.structural import StructuralCostModel
from integration.lightop.add_rmsnorm import executable_registry
from simulator.fusion_rules import make_reference_plan
from simulator.hardware import load_hardware
from tools.evaluate_search_quality import BudgetScorer, EvaluationBudgetReached, run_one, summarize


@pytest.fixture
def case():
    graph = build_evaluation_graph("layer")
    profile = AnalyticHardwareProfile(load_hardware("data/bw1000_2card/bw1000_2card.calibrated.json"))
    scorer = _CachedScorer(graph, profile)
    registry = executable_registry()
    scorer.model = StructuralCostModel(registry=registry)
    return graph, scorer, registry


def test_cap_counts_repeated_scores_and_stops_before_extra_evaluation(case):
    graph, base, _ = case
    scorer = BudgetScorer(graph, base.profile, base.model, 2)
    scorer.start_run()
    plan = make_reference_plan(graph)
    scorer(plan)
    scorer(plan)
    with pytest.raises(EvaluationBudgetReached):
        scorer(plan)
    assert scorer.run_model_evaluations == 2
    assert scorer.run_requests == 2
    assert len(scorer.trace) == 2
    assert len(scorer.run_keys) == 1


def test_infeasible_scores_do_not_become_incumbents(case):
    graph, base, _ = case
    scorer = BudgetScorer(graph, base.profile, base.model, 2)
    scorer.capacity = (1., 1.)
    scorer.start_run()
    scorer(make_reference_plan(graph))
    assert scorer.best_plan is None
    assert scorer.trace[0]["best_ms"] is None


@pytest.mark.parametrize("mode", ("joint", "sequential", "placement_then_fusion"))
def test_modes_share_one_actual_score_cap(case, mode):
    graph, base, registry = case
    row, trace = run_one(graph, base, registry, mode, seed=0, budget=16)
    assert row["cost_model_evaluations"] == row["scorer_requests"] == len(trace) == 16
    assert row["final_best_ms"] == min(p["candidate_ms"] for p in trace if p["feasible"])


def test_initial_block_and_first_attainment_are_not_final_only_metrics():
    values = (10., 9., 8., 7., 6., 6.)
    trace = [{"evaluation": i, "candidate_ms": v, "feasible": True} for i, v in enumerate(values, 1)]
    row = summarize(trace)
    assert row["initial_best_ms"] == 7.
    assert row["final_best_ms"] == 6.
    assert row["first_final_best_evaluation"] == 5
    assert row["improvement_after_initial_percent"] == pytest.approx(100 / 7)
