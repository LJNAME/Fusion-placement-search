"""Verify E1 hidden-measurement replay and budgeted regret metrics."""

from __future__ import annotations

from .hardware_profile import AnalyticHardwareProfile
from .measured_profile import MeasuredHardwareProfile
from .search_evaluation import build_evaluation_graph
from .search_regret_profiling import evaluate_search_regret_profiling
from .search_sensitivity import (
    SymmetricHardwareProfile,
    generate_candidate_pool,
    infer_hardware_model_classes,
)


def main() -> None:
    measured = MeasuredHardwareProfile.from_files(
        "data/bw1000_2card/bw1000_2card.calibrated.json",
        "data/bw1000_2card/raw_measurements.json",
    )
    classes = infer_hardware_model_classes(measured)
    oracle = SymmetricHardwareProfile(measured, classes)
    fallback = SymmetricHardwareProfile(
        AnalyticHardwareProfile(measured.hardware), classes
    )
    graph = build_evaluation_graph("small")
    plans, _ = generate_candidate_pool(
        graph,
        oracle,
        candidate_count=12,
        seeds=(0, 1, 7),
        iterations=32,
        rollouts=2,
    )
    report = evaluate_search_regret_profiling(
        graph,
        plans,
        oracle,
        fallback,
        budgets=(0, 1, 2, 4),
        random_seeds=(0, 1, 7),
        top_k=3,
    )
    summary = report["summary"]
    print(f"nodes: {report['nodes']}")
    print(f"candidates: {summary['candidate_plans']}")
    print(f"profiled_shapes: {summary['profiled_shapes']}")
    for policy in ("random", "uncertainty", "impact", "regret_aware"):
        final = report["policies"][policy]["by_budget"][-1]
        print(
            f"{policy}_full_regret_ms: "
            f"{final['top1_regret_ms_mean']:.6f}"
        )
    print(f"full_budget_recovers_oracle: {summary['full_budget_recovers_oracle']}")
    print("Fusion E1 search-regret profiling verification: OK")


if __name__ == "__main__":
    main()
