"""CPU smoke verification for the M6 search-evaluation report."""

from __future__ import annotations

from pathlib import Path

from .measured_profile import MeasuredHardwareProfile
from .search_evaluation import evaluate_searches


ROOT = Path(__file__).resolve().parents[1]


def verify_search_evaluation(*, allow_unbound=False) -> dict[str, object]:
    profile = MeasuredHardwareProfile.from_files(
        ROOT / "data" / "bw1000_2card" / "bw1000_2card.calibrated.json",
        ROOT / "data" / "bw1000_2card" / "raw_measurements.json",
        allow_unbound=allow_unbound,
    )
    report = evaluate_searches(
        profile,
        ("small",),
        seed=23,
        beam=4,
        rollouts=2,
        mcts_iterations=32,
        top_k=2,
    )
    scale = report["scales"][0]
    searches = {
        (item["mode"], item["method"]): item for item in scale["searches"]
    }
    if len(searches) != 12:
        raise AssertionError(f"expected twelve mode/method results, got {len(searches)}")
    modes = ("placement_only", "fusion_only", "sequential", "joint")
    for mode in modes:
        exact = searches[(mode, "exact")]
        gap_field = "reference_gap" if mode == "sequential" else "optimality_gap"
        if searches[(mode, "beam")][gap_field] != 0.0:
            raise AssertionError(f"beam did not close exact gap for {mode}")
        if searches[(mode, "mcts")][gap_field] != 0.0:
            raise AssertionError(f"MCTS did not close exact gap for {mode}")
        if exact["candidates"][0]["makespan_s"] <= 0.0:
            raise AssertionError(f"invalid exact score for {mode}")
        for method in ("beam", "mcts"):
            metrics = searches[(mode, method)]["budget"]
            if metrics["scorer_requests"] <= 0:
                raise AssertionError(f"missing scorer requests for {mode}/{method}")
            if metrics["unique_requested_plans"] <= 0:
                raise AssertionError(f"missing requested plans for {mode}/{method}")
    if searches[("placement_only", "exact")]["candidates"][0]["placeholder_groups"]:
        raise AssertionError("placement-only result contains a placeholder")
    if searches[("joint", "exact")]["candidates"][0]["placeholder_groups"] != 1:
        raise AssertionError("joint result did not select the planned silu_mul pattern")
    return {
        "nodes": scale["nodes"],
        "edges": scale["edges"],
        "searches": len(searches),
        "placement_space": searches[("placement_only", "exact")]["search_space_size"],
        "fusion_space": searches[("fusion_only", "exact")]["search_space_size"],
        "joint_space": searches[("joint", "exact")]["search_space_size"],
        "beam_gaps": ",".join(
            f"{searches[(mode, 'beam')]['reference_gap']:.6f}"
            for mode in modes
        ),
        "mcts_gaps": ",".join(
            f"{searches[(mode, 'mcts')]['reference_gap']:.6f}"
            for mode in modes
        ),
        "telemetry": "requests,unique_requested_plans,cost_model_evaluations",
    }


def main() -> None:
    for key, value in verify_search_evaluation().items():
        print(f"{key}: {value}")
    print("Fusion M6 search evaluation verification: OK")


if __name__ == "__main__":
    main()
