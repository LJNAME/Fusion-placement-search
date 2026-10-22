"""Exhaustive oracle for legal fusion and placement on small graphs."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional, Sequence, Tuple

from simulator import FusionPatternRegistry, OpGraph, Plan, validate_fusion_plan

from .estimate import CostEstimate
from .legal_search import LegalFusionSearchSpace, LegalPartialState


class ExactSearchLimitExceeded(ValueError):
    """Raised before enumeration when the full space exceeds a hard limit."""


def _plan_key(plan: Plan) -> tuple:
    return tuple(
        (
            group.extra.get("pattern_id"),
            tuple(group.node_ids),
            int(group.device),
        )
        for group in plan.groups
    )


@dataclass(frozen=True)
class ExactSearchResult:
    plans: Tuple[Plan, ...]
    scores: Tuple[float, ...]
    search_space_size: int
    evaluated_plans: int
    feasible_plans: int
    elapsed_seconds: float

    def __post_init__(self) -> None:
        if not self.plans or len(self.plans) != len(self.scores):
            raise ValueError("exact result must contain aligned plans and scores")
        if tuple(sorted(self.scores)) != self.scores:
            raise ValueError("exact result scores must be ordered")
        if self.evaluated_plans != self.search_space_size:
            raise ValueError("an exact result must evaluate the full search space")
        if not 0 < self.feasible_plans <= self.evaluated_plans:
            raise ValueError("feasible plan count is invalid")

    @property
    def best_plan(self) -> Plan:
        return self.plans[0]

    @property
    def optimal_score(self) -> float:
        return self.scores[0]


@dataclass(frozen=True)
class OptimalityGap:
    optimal_score: float
    candidate_score: float
    absolute_gap: float
    relative_gap: float


def count_exact_search_space(space: LegalFusionSearchSpace) -> int:
    """Count all complete legal candidate covers and device assignments."""

    node_bit = {node_id: 1 << index for index, node_id in enumerate(space.order)}
    candidate_masks = {
        candidate: sum(node_bit[node_id] for node_id in candidate.node_ids)
        for candidate in space.candidates
    }
    full_mask = (1 << len(space.order)) - 1

    @lru_cache(maxsize=None)
    def count(assigned_mask: int) -> int:
        if assigned_mask == full_mask:
            return 1
        anchor = next(
            node_id
            for node_id in space.order
            if not assigned_mask & node_bit[node_id]
        )
        total = 0
        for candidate in space.by_node[anchor]:
            candidate_mask = candidate_masks[candidate]
            if assigned_mask & candidate_mask:
                continue
            total += len(space.allowed_devices) * count(assigned_mask | candidate_mask)
        return total

    return count(0)


def _score_value(value: float | CostEstimate) -> tuple[float, bool]:
    if isinstance(value, CostEstimate):
        overflow = bool(value.details.get("overflow_devices", []))
        return float(value.makespan), not overflow
    score = float(value)
    if not math.isfinite(score):
        raise ValueError("exact-search scorer must return a finite value")
    return score, True


def exact_legal_search(
    graph: OpGraph,
    scorer: Callable[[Plan], float | CostEstimate],
    n_devs: int,
    *,
    top_k: int = 8,
    max_nodes: int = 15,
    max_evaluations: int = 1_000_000,
    registry: Optional[FusionPatternRegistry] = None,
    require_executable: bool = True,
    allowed_devices: Optional[Sequence[int]] = None,
    is_feasible: Optional[Callable[[Plan], bool]] = None,
) -> ExactSearchResult:
    """Enumerate every legal plan and return the globally best ``top_k``.

    The function performs a complete-space count first.  It raises rather than
    returning a partial result when the configured limit would be exceeded.
    ``CostEstimate`` scorers automatically reject plans with memory overflow.
    """

    if top_k <= 0 or max_nodes <= 0 or max_evaluations <= 0:
        raise ValueError("top_k, max_nodes, and max_evaluations must be positive")
    if len(graph.nodes) > max_nodes:
        raise ExactSearchLimitExceeded(
            f"exact search supports at most {max_nodes} nodes, got {len(graph.nodes)}"
        )

    space = LegalFusionSearchSpace(
        graph,
        n_devs,
        registry,
        require_executable=require_executable,
        allowed_devices=allowed_devices,
    )
    search_space_size = count_exact_search_space(space)
    if search_space_size > max_evaluations:
        raise ExactSearchLimitExceeded(
            f"exact search requires {search_space_size} evaluations, "
            f"limit is {max_evaluations}"
        )

    started = time.perf_counter()
    evaluated = 0
    feasible = 0
    best: list[tuple[float, tuple, Plan]] = []

    def visit(state: LegalPartialState) -> None:
        nonlocal evaluated, feasible
        if space.is_terminal(state):
            evaluated += 1
            plan = space.to_plan(state, name="legal-exact", validate=False)
            if is_feasible is not None and not is_feasible(plan):
                return
            score, estimate_feasible = _score_value(scorer(plan))
            if not estimate_feasible:
                return
            feasible += 1
            key = _plan_key(plan)
            best.append((score, key, plan))
            best.sort(key=lambda item: (item[0], item[1]))
            if len(best) > top_k:
                del best[top_k:]
            return
        for action in space.actions(state):
            visit(space.apply(state, action))

    visit(space.initial_state())
    elapsed = time.perf_counter() - started
    if evaluated != search_space_size:
        raise RuntimeError(
            f"exact enumeration mismatch: counted {search_space_size}, evaluated {evaluated}"
        )
    if not best:
        raise ValueError("exact search found no feasible plan")

    plans = tuple(item[2] for item in best)
    scores = tuple(float(item[0]) for item in best)
    for rank, plan in enumerate(plans, start=1):
        plan.name = f"legal-exact-rank-{rank}"
        plan.extra.update(
            {
                "search_method": "exact",
                "search_space_size": search_space_size,
                "rank": rank,
            }
        )
        validate_fusion_plan(
            graph,
            plan,
            n_devs,
            registry=space.registry,
            require_executable=space.require_executable,
        )
    return ExactSearchResult(
        plans=plans,
        scores=scores,
        search_space_size=search_space_size,
        evaluated_plans=evaluated,
        feasible_plans=feasible,
        elapsed_seconds=float(elapsed),
    )


def measure_optimality_gap(
    exact: ExactSearchResult,
    candidates: Sequence[Plan],
    scorer: Callable[[Plan], float | CostEstimate],
) -> OptimalityGap:
    """Measure the best candidate set against an exact reference result."""

    if not candidates:
        raise ValueError("candidates must not be empty")
    candidate_scores = []
    for plan in candidates:
        score, feasible = _score_value(scorer(plan))
        if feasible:
            candidate_scores.append(score)
    if not candidate_scores:
        raise ValueError("candidate set contains no feasible plan")
    candidate_score = min(candidate_scores)
    optimal = exact.optimal_score
    absolute = max(candidate_score - optimal, 0.0)
    relative = absolute / max(abs(optimal), 1e-12)
    return OptimalityGap(
        optimal_score=float(optimal),
        candidate_score=float(candidate_score),
        absolute_gap=float(absolute),
        relative_gap=float(relative),
    )


__all__ = [
    "ExactSearchLimitExceeded",
    "ExactSearchResult",
    "OptimalityGap",
    "count_exact_search_space",
    "exact_legal_search",
    "measure_optimality_gap",
]
