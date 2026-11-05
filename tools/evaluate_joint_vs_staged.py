"""Phases 2-3: joint vs fuse-then-place vs place-then-fuse at equal total budget.

Reuses _run_search, _CachedScorer and the mcts_medium subgraphs and capacity
convention (per-device cap = fraction of total parameter bytes) unchanged.
Medium subgraphs get an Exact reference per mode; the 591-node model has none.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cost_model.hardware_profile import AnalyticHardwareProfile
from cost_model.search_evaluation import (
    _CachedScorer, _modes, _plan_key, _run_search, build_evaluation_graph, placement_then_fusion_mode)
from simulator import validate_fusion_plan
from simulator.fusion_rules import enumerate_fusion_candidates
from simulator.hardware import load_hardware
from tools.evaluate_mcts_medium import SELECTIONS, extract_graph

HARDWARE = ROOT / "data/bw1000_2card/bw1000_2card.calibrated.json"
CAPS = {"actual_capacity": None, "parameter_cap_90": .90, "parameter_cap_75": .75}


def all_modes():
    return [*_modes(2), placement_then_fusion_mode(2)]


def tasks(scope):
    hardware = load_hardware(HARDWARE)
    graphs = {}
    if scope in ("medium", "all"):
        full = build_evaluation_graph("layer")
        graphs.update({name: extract_graph(full, name) for name in SELECTIONS})
    if scope in ("model", "all"):
        graphs["qwen3_model"] = build_evaluation_graph("model")
    for name, graph in graphs.items():
        for variant, fraction in CAPS.items():
            hw = copy.deepcopy(hardware)
            if fraction is not None:
                cap = fraction * sum(n.param_bytes for n in graph.nodes.values())
                for device in hw.devices:
                    device.max_mem = cap
            yield name + ":" + variant, graph, hw


def summarize(graph, scorer, plans, registry):
    if not plans:
        return None
    plan = plans[0]
    estimate = scorer.estimate(plan)
    validate_fusion_plan(graph, plan, scorer.profile.n_devs, registry=registry)
    return {"makespan_ms": estimate.makespan * 1e3, "memory_feasible": scorer.is_feasible(estimate),
            "fused_groups": sum(len(g.node_ids) > 1 for g in plan.groups),
            "used_devices": sorted({g.device for g in plan.groups}),
            "cross_device_bytes": estimate.cross_device_bytes, "plan_key": repr(_plan_key(plan))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("medium", "model", "all"), default="all")
    parser.add_argument("--budgets", default="16,64,256")
    parser.add_argument("--seeds", default="0,7,23")
    parser.add_argument("--model-caps", default="actual_capacity",
                        help="capacity variants for the full model (no Exact gate there)")
    parser.add_argument("--model-modes", default=None,
                        help="restrict modes on the full model, e.g. placement_then_fusion")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    rows, task_info = [], {}
    for task_id, graph, hw in tasks(args.scope):
        if len(graph.nodes) > 15 and task_id.split(":")[1] not in args.model_caps.split(","):
            continue
        profile = AnalyticHardwareProfile(hw)
        scorer = _CachedScorer(graph, profile)
        exact_allowed = len(graph.nodes) <= 15
        multi = [c for c in enumerate_fusion_candidates(graph, all_modes()[-1].registry) if len(c.node_ids) > 1]
        task_info[task_id] = {"nodes": len(graph.nodes), "multi_node_candidates": len(multi),
                              "capacity_gib": [d.max_mem / 2**30 for d in hw.devices]}
        if exact_allowed:
            joint = next(m for m in all_modes() if m.name == "joint")
            gate, _, gate_info = _run_search("exact", graph, scorer, joint, seed=0, beam=8, rollouts=3,
                                             mcts_iterations=2, top_k=1)
            task_info[task_id]["joint_space_size"] = gate_info["stages"][0]["search_space_size"]
            if not gate:
                # Joint space contains every mode's space, so nothing below can be feasible.
                task_info[task_id]["status"] = "infeasible_whole_joint_space"
                print(json.dumps({"task": task_id, "status": "infeasible_whole_joint_space"}), flush=True)
                continue
        task_info[task_id]["status"] = "searched"
        selected = [m for m in all_modes() if args.model_modes is None or exact_allowed
                    or m.name in args.model_modes.split(",")]
        for mode in selected:
            runs = ([("exact", 0, None)] if exact_allowed else []) + [
                ("mcts", seed, budget) for seed in map(int, args.seeds.split(","))
                for budget in map(int, args.budgets.split(","))]
            for method, seed, budget in runs:
                plans, elapsed, info = _run_search(method, graph, scorer, mode, seed=seed, beam=8, rollouts=3,
                                                   mcts_iterations=budget or 2, top_k=4)
                row = {"task": task_id, "mode": mode.name, "method": method, "seed": seed, "iterations": budget,
                       "elapsed_s": elapsed, "cost_model_evaluations": info["cost_model_evaluations"],
                       "scorer_requests": info["scorer_requests"],
                       "evaluated_plans": info.get("evaluated_plans"),
                       "search_space_size": sum(s.get("search_space_size", 0) for s in info["stages"]) or None,
                       "frozen_device_conflicts": sum(s.get("frozen_device_conflicts", 0) for s in info["stages"]),
                       "best": summarize(graph, scorer, plans, mode.registry), "reason": info["reason"]}
                rows.append(row)
                print(json.dumps({k: row[k] for k in ("task", "mode", "method", "seed", "iterations")} |
                                 {"ms": row["best"] and round(row["best"]["makespan_ms"], 5),
                                  "feasible": row["best"] and row["best"]["memory_feasible"],
                                  "s": round(elapsed, 2)}), flush=True)
                (args.output / "rows.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    report = {"scope": "phases 2-3 joint vs staged; analytic profile; equal total budget per mode",
              "capacity_convention": "per-device cap = fraction x total parameter bytes (mcts_medium convention)",
              "modes": [m.name for m in all_modes()], "tasks": task_info, "rows": rows,
              "search_code_changed": "placement_then_fusion branch only", "cost_model_changed": False}
    (args.output / "report.json").write_text(json.dumps(report, indent=1, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
