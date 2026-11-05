"""Phase 1: existing MCTS/beam on the full Qwen3 operator graph (591 nodes, 2 GPUs).

Reuses search_evaluation._run_search/_modes/_CachedScorer unchanged, with the
analytic profile used by the earlier mainline/medium experiments. Records
time, scorer requests, cost-model evaluations, best cost and legality.
No Exact reference exists at this size; single-card unfused plans are the
reference points.
"""
import argparse
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cost_model.hardware_profile import AnalyticHardwareProfile
from cost_model.search_evaluation import _CachedScorer, _modes, _partition_key, _run_search, build_evaluation_graph
from simulator import make_reference_plan, validate_fusion_plan
from simulator.hardware import load_hardware

HARDWARE = "data/bw1000_2card/bw1000_2card.calibrated.json"


def describe(graph, scorer, plan, registry):
    estimate = scorer.estimate(plan)
    legal, reason = True, None
    try:
        validate_fusion_plan(graph, plan, scorer.profile.n_devs, registry=registry)
    except Exception as error:  # report, never hide
        legal, reason = False, repr(error)
    devices = {}
    for g in plan.groups:
        devices[g.device] = devices.get(g.device, 0) + len(g.node_ids)
    return {"makespan_ms": estimate.makespan * 1e3, "memory_feasible": scorer.is_feasible(estimate),
            "peak_memory_gib": [m / 2**30 for m in estimate.peak_memory],
            "legal": legal, "illegal_reason": reason,
            "fused_groups": sum(len(g.node_ids) > 1 for g in plan.groups),
            "nodes_per_device": {str(k): v for k, v in sorted(devices.items())},
            "cross_device_bytes": estimate.cross_device_bytes}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", default="model")
    parser.add_argument("--modes", default="joint,sequential,fusion_only,placement_only")
    parser.add_argument("--budgets", default="16,64,256")
    parser.add_argument("--seeds", default="0,7,23")
    parser.add_argument("--reference-iterations", type=int, default=1024)
    parser.add_argument("--beam", default="2:1,4:1", help="width:rollouts pairs for joint beam")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    graph = build_evaluation_graph(args.scale)
    profile = AnalyticHardwareProfile(load_hardware(ROOT / HARDWARE))
    modes = {m.name: m for m in _modes(profile.n_devs)}
    scorer = _CachedScorer(graph, profile)
    references = {}
    for device in range(profile.n_devs):
        plan = make_reference_plan(graph, device=device, registry=modes["joint"].registry)
        references[f"unfused_gpu{device}"] = describe(graph, scorer, plan, modes["joint"].registry)
    runs = []

    def record(method, mode, seed, iterations=None, beam=None, rollouts=3):
        plans, elapsed, budget = _run_search(method, graph, scorer, modes[mode], seed=seed,
            beam=beam or 8, rollouts=rollouts, mcts_iterations=iterations or 2, top_k=4)
        best = describe(graph, scorer, plans[0], modes[mode].registry) if plans else None
        row = {"method": method, "mode": mode, "seed": seed, "iterations": iterations, "beam": beam,
               "rollouts": rollouts, "elapsed_s": elapsed, "scorer_requests": budget["scorer_requests"],
               "cost_model_evaluations": budget["cost_model_evaluations"],
               "unique_requested_plans": budget["unique_requested_plans"], "returned_plans": len(plans),
               "best": best, "stages": budget["stages"],
               "frozen_partition_size": len(budget.get("frozen_partition", [])) or None}
        runs.append(row)
        print(json.dumps({k: row[k] for k in ("method", "mode", "seed", "iterations", "beam", "elapsed_s",
                                              "cost_model_evaluations")} |
                         {"best_ms": best and best["makespan_ms"], "legal": best and best["legal"]}), flush=True)
        (args.output / "runs.json").write_text(json.dumps(runs, indent=1), encoding="utf-8")

    for mode in args.modes.split(","):
        for seed in map(int, args.seeds.split(",")):
            for iterations in map(int, args.budgets.split(",")):
                record("mcts", mode, seed, iterations)
    record("mcts", "joint", 0, args.reference_iterations)
    for pair in filter(None, args.beam.split(",")):
        width, rollouts = map(int, pair.split(":"))
        record("beam", "joint", 0, beam=width, rollouts=rollouts)
    report = {"scope": "phase 1 search scale; analytic profile; no Exact reference at this size",
              "graph": {"scale": args.scale, "nodes": len(graph.nodes), "edges": len(graph.edges)},
              "hardware": HARDWARE, "profile": "AnalyticHardwareProfile", "devices": profile.n_devs,
              "scorer_cache_enabled": scorer.cache_enabled, "references": references, "runs": runs,
              "search_code_changed": False, "host": platform.node()}
    (args.output / "report.json").write_text(json.dumps(report, indent=1, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
