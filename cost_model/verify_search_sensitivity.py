"""CPU smoke verification for the E0 counterfactual sensitivity report."""

from __future__ import annotations

from pathlib import Path

from .measured_profile import MeasuredHardwareProfile
from .search_evaluation import build_evaluation_graph
from .search_sensitivity import (
    evaluate_counterfactual_sensitivity,
    generate_candidate_pool,
)


ROOT = Path(__file__).resolve().parents[1]


def verify_search_sensitivity() -> dict[str, object]:
    profile = MeasuredHardwareProfile.from_files(
        ROOT / "data" / "bw1000_2card" / "bw1000_2card.calibrated.json",
        ROOT / "data" / "bw1000_2card" / "raw_measurements.json",
    )
    graph = build_evaluation_graph("small")
    plans, pool = generate_candidate_pool(
        graph,
        profile,
        candidate_count=6,
        seeds=(0, 23),
        iterations=16,
        rollouts=2,
        include_planned=False,
    )
    report = evaluate_counterfactual_sensitivity(
        graph,
        plans,
        profile,
        perturbations=(-0.20, 0.20),
        top_k=2,
    )
    summary = report["summary"]
    if report["format"] != "fusion-placement-search.counterfactual-sensitivity":
        raise AssertionError("unexpected E0 report format")
    if summary["candidate_plans"] < 2:
        raise AssertionError("E0 candidate pool is not competitive")
    if summary["shape_keys"] <= 0:
        raise AssertionError("E0 found no perturbable ShapeKey")
    if pool["include_planned"] or not pool["require_executable"]:
        raise AssertionError("E0 smoke verification must be executable-only")
    if any(
        group.extra.get("implementation_status") == "planned"
        for plan in plans
        for group in plan.groups
    ):
        raise AssertionError("executable-only candidate pool contains a placeholder")
    return {
        "nodes": report["nodes"],
        "candidates": summary["candidate_plans"],
        "shape_keys": summary["shape_keys"],
        "top1_flip_shapes": summary["top1_flip_shapes"],
        "top_k_flip_shapes": summary["top_k_flip_shapes"],
        "max_rank_shift": summary["max_rank_shift"],
        "ranking_signal": summary["ranking_signal"],
    }


def main() -> None:
    for key, value in verify_search_sensitivity().items():
        print(f"{key}: {value}")
    print("Fusion E0 counterfactual sensitivity verification: OK")


if __name__ == "__main__":
    main()
