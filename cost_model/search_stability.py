"""Repeat MCTS across seeds and budgets and summarize search stability."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Sequence

from simulator import default_fusion_registry, hardware_fingerprint, make_reference_plan

from .measured_profile import MeasuredHardwareProfile
from .search_evaluation import (
    _CachedScorer,
    _candidate_summary,
    _modes,
    _run_search,
    build_evaluation_graph,
)


REPORT_FORMAT = "fusion-placement-search.search-stability"
REPORT_VERSION = "1.0"


def evaluate_stability(
    profile: MeasuredHardwareProfile,
    *,
    scale: str,
    modes: Sequence[str],
    seeds: Sequence[int],
    iteration_budgets: Sequence[int],
    rollouts: int = 3,
    top_k: int = 4,
) -> dict[str, object]:
    if not modes or not seeds or not iteration_budgets:
        raise ValueError("modes, seeds, and iteration_budgets must not be empty")
    graph = build_evaluation_graph(scale)
    available_modes = {mode.name: mode for mode in _modes(profile.n_devs)}
    unknown = sorted(set(modes) - set(available_modes))
    if unknown:
        raise ValueError(f"unknown modes: {unknown}")
    if any(iterations <= 0 for iterations in iteration_budgets):
        raise ValueError("iteration budgets must be positive")

    baseline_scorer = _CachedScorer(graph, profile)
    baseline = make_reference_plan(
        graph,
        device=0,
        registry=default_fusion_registry(),
    )
    baseline_summary = _candidate_summary(
        baseline,
        baseline_scorer,
        default_fusion_registry(),
        rank=1,
    )
    baseline_makespan = float(baseline_summary["makespan_s"])
    records: list[dict[str, object]] = []
    for iterations in iteration_budgets:
        for seed in seeds:
            for mode_name in modes:
                mode = available_modes[mode_name]
                scorer = _CachedScorer(graph, profile)
                plans, elapsed, metrics = _run_search(
                    "mcts",
                    graph,
                    scorer,
                    mode,
                    seed=int(seed),
                    beam=1,
                    rollouts=rollouts,
                    mcts_iterations=int(iterations),
                    top_k=top_k,
                )
                candidates = [
                    _candidate_summary(plan, scorer, mode.registry, rank)
                    for rank, plan in enumerate(plans, start=1)
                ]
                for candidate in candidates:
                    makespan = float(candidate["makespan_s"])
                    candidate["improvement_percent"] = (
                        (baseline_makespan - makespan) / baseline_makespan * 100.0
                    )
                records.append(
                    {
                        "mode": mode_name,
                        "seed": int(seed),
                        "iterations": int(iterations),
                        "rollouts": rollouts,
                        "elapsed_s": elapsed,
                        "metrics": metrics,
                        "candidates": candidates,
                    }
                )

    aggregates: list[dict[str, object]] = []
    for iterations in iteration_budgets:
        for mode_name in modes:
            selected = [
                record
                for record in records
                if record["mode"] == mode_name
                and record["iterations"] == int(iterations)
            ]
            top1 = [record["candidates"][0] for record in selected]
            makespans = [float(candidate["makespan_s"]) for candidate in top1]
            elapsed = [float(record["elapsed_s"]) for record in selected]
            aggregates.append(
                {
                    "mode": mode_name,
                    "iterations": int(iterations),
                    "runs": len(selected),
                    "best_makespan_s": min(makespans),
                    "mean_makespan_s": statistics.fmean(makespans),
                    "worst_makespan_s": max(makespans),
                    "stdev_makespan_s": statistics.pstdev(makespans),
                    "mean_elapsed_s": statistics.fmean(elapsed),
                    "unique_top1_plans": len(
                        {candidate["plan_signature"] for candidate in top1}
                    ),
                    "placeholder_groups_range": [
                        min(int(candidate["placeholder_groups"]) for candidate in top1),
                        max(int(candidate["placeholder_groups"]) for candidate in top1),
                    ],
                    "all_out_of_distribution": all(
                        bool(candidate["out_of_distribution"]) for candidate in top1
                    ),
                }
            )
    return {
        "format": REPORT_FORMAT,
        "version": REPORT_VERSION,
        "hardware_hash": hardware_fingerprint(profile.hardware),
        "hardware_name": profile.hardware.name,
        "placeholder_policy": (
            "fusion_only and joint include built-in planned patterns for search "
            "evaluation; results are not deployable"
        ),
        "scale": scale,
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "baseline": baseline_summary,
        "configuration": {
            "modes": list(modes),
            "seeds": [int(seed) for seed in seeds],
            "iteration_budgets": [int(value) for value in iteration_budgets],
            "rollouts": rollouts,
            "top_k": top_k,
        },
        "records": records,
        "aggregates": aggregates,
    }


def _csv_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list")
    return result


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hardware",
        type=Path,
        default=root / "data" / "bw1000_2card" / "bw1000_2card.calibrated.json",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=root / "data" / "bw1000_2card" / "raw_measurements.json",
    )
    parser.add_argument("--scale", choices=("small", "layer", "model"), default="model")
    parser.add_argument("--modes", default="joint")
    parser.add_argument("--seeds", type=_csv_ints, default=(0, 1, 7, 23))
    parser.add_argument("--iteration-budgets", type=_csv_ints, default=(32, 64, 128))
    parser.add_argument("--rollouts", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    modes = tuple(item.strip() for item in args.modes.split(",") if item.strip())
    profile = MeasuredHardwareProfile.from_files(args.hardware, args.raw)
    report = evaluate_stability(
        profile,
        scale=args.scale,
        modes=modes,
        seeds=args.seeds,
        iteration_budgets=args.iteration_budgets,
        rollouts=args.rollouts,
        top_k=args.top_k,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for aggregate in report["aggregates"]:
        print(
            f"{aggregate['mode']} iterations={aggregate['iterations']} "
            f"runs={aggregate['runs']} "
            f"best_ms={aggregate['best_makespan_s'] * 1e3:.6f} "
            f"mean_ms={aggregate['mean_makespan_s'] * 1e3:.6f} "
            f"worst_ms={aggregate['worst_makespan_s'] * 1e3:.6f} "
            f"unique_top1={aggregate['unique_top1_plans']} "
            f"mean_elapsed_s={aggregate['mean_elapsed_s']:.3f}"
        )
    print(f"report: {args.output}")
    print("Fusion M6b search stability evaluation: OK")


if __name__ == "__main__":
    main()
