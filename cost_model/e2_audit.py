"""Exact prepared candidate scoring and explicit E2 ranking metrics."""

from __future__ import annotations

import copy
import math
import time
from typing import Any, Sequence

from simulator import OpGraph, Plan, default_fusion_registry

from .search_regret_profiling import _ranking_metrics
from .search_sensitivity import _RankedPlan, plan_id
from .structural import StructuralCostModel


class PreparedCandidateRanker:
    """Own frozen candidate snapshots; rerun the full scheduler for each profile."""

    def __init__(self, graph: OpGraph, plans: Sequence[Plan], n_devs: int, exchange, registry=None):
        started = time.perf_counter()
        self.graph = copy.deepcopy(graph)
        self.model = StructuralCostModel(registry=registry or default_fusion_registry())
        self.exchange = exchange
        self.prepared = {}
        for plan in plans:
            identifier = plan_id(plan, exchange)
            self.prepared[identifier] = self.model.prepare(self.graph, copy.deepcopy(plan), n_devs)
        self.preparation_wall_time_s = time.perf_counter() - started
        self.calls = 0
        self.scoring_wall_time_s = 0.0

    def rank(self, profile):
        started = time.perf_counter()
        ranked = []
        for identifier, prepared in self.prepared.items():
            estimate = self.model.predict_prepared(prepared, profile)
            ranked.append(_RankedPlan(identifier, prepared.plan, estimate.makespan,
                                      estimate.critical_path, estimate.confidence))
        ranked.sort(key=lambda item: (item.makespan, item.plan_id))
        self.calls += 1
        self.scoring_wall_time_s += time.perf_counter() - started
        return ranked


def ranking_metrics(oracle_ranked, predicted_ranked, *, top_k=5, tie_tolerance_s=1e-12):
    result = _ranking_metrics(oracle_ranked, predicted_ranked, top_k=top_k,
                              tie_tolerance_s=tie_tolerance_s)
    k = min(max(int(top_k), 1), len(oracle_ranked))
    oracle_ids = {item.plan_id for item in oracle_ranked[:k]}
    predicted_ids = {item.plan_id for item in predicted_ranked[:k]}
    by_id = {item.plan_id: item.makespan for item in oracle_ranked}
    best = oracle_ranked[0].makespan
    best_of_k = min(by_id[item] for item in predicted_ids)
    result.update({
        "strict_top_k_recall": len(oracle_ids & predicted_ids) / k,
        "tie_aware_top_k_recall": result["top_k_recall"],
        "best_of_k_regret_ms": max(0.0, best_of_k - best) * 1e3,
        "normalized_top1_regret": result["top1_regret_ms"] / max(best * 1e3, 1e-12),
        "normalized_best_of_k_regret": max(0.0, best_of_k - best) / max(best, 1e-12),
        "strict_top_k": k,
    })
    return result


def selection_diagnostics(scores: dict[str, float], remaining: Sequence[str], threshold=0.0):
    if not scores:
        return {"max_selection_score": None, "score_tie_count": None,
                "score_tie_fraction": None, "all_scores_zero": None,
                "stop_recommended": False}
    maximum = max(scores.values())
    ties = sum(math.isclose(value, maximum, rel_tol=1e-12, abs_tol=0.0)
               for value in scores.values())
    return {"max_selection_score": maximum, "score_tie_count": ties,
            "score_tie_fraction": ties / len(remaining),
            "all_scores_zero": all(value == 0.0 for value in scores.values()),
            "stop_recommended": maximum <= threshold}


def summarize_metrics(runs: Sequence[dict[str, Any]], by_budget):
    names = ("strict_top_k_recall", "tie_aware_top_k_recall", "best_of_k_regret_ms",
             "normalized_top1_regret", "normalized_best_of_k_regret",
             "profiling_wall_time_s", "predictor_fit_wall_time_s", "selection_wall_time_s",
             "ranking_wall_time_s", "algorithm_wall_time_s")
    for aggregate in by_budget:
        points = [point for point in runs if point["budget"] == aggregate["budget"]]
        for name in names:
            aggregate[f"{name}_mean"] = sum(point[name] for point in points) / len(points)
