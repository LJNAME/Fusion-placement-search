"""CPU smoke verification for repeated MCTS stability evaluation."""

from __future__ import annotations

from pathlib import Path

from .measured_profile import MeasuredHardwareProfile
from .search_stability import evaluate_stability


ROOT = Path(__file__).resolve().parents[1]


def verify_search_stability() -> dict[str, object]:
    profile = MeasuredHardwareProfile.from_files(
        ROOT / "data" / "bw1000_2card" / "bw1000_2card.calibrated.json",
        ROOT / "data" / "bw1000_2card" / "raw_measurements.json",
    )
    report = evaluate_stability(
        profile,
        scale="small",
        modes=("placement_only", "fusion_only", "joint"),
        seeds=(0, 23),
        iteration_budgets=(16, 32),
        rollouts=2,
        top_k=2,
    )
    if len(report["records"]) != 12:
        raise AssertionError("expected 12 seed/budget/mode records")
    if len(report["aggregates"]) != 6:
        raise AssertionError("expected six budget/mode aggregates")
    if any(aggregate["runs"] != 2 for aggregate in report["aggregates"]):
        raise AssertionError("every aggregate must contain two seed runs")
    if not all(
        "scorer_requests" in record["metrics"] for record in report["records"]
    ):
        raise AssertionError("missing scorer request telemetry")
    return {
        "nodes": report["nodes"],
        "records": len(report["records"]),
        "aggregates": len(report["aggregates"]),
        "all_runs_per_aggregate": 2,
    }


def main() -> None:
    for key, value in verify_search_stability().items():
        print(f"{key}: {value}")
    print("Fusion M6b search stability verification: OK")


if __name__ == "__main__":
    main()
