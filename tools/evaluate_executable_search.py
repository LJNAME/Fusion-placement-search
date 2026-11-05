"""Full-model executable search with cross-layer measured Profiles (existing MCTS, unchanged).

Modes: joint, fuse-then-place, place-then-fuse. Known reference plans
(all GPU0) are scored with the same scorer: no fusion, add_rmsnorm only,
silu_mul only, both.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cost_model.hardware_profile import AnalyticHardwareProfile
from cost_model.layer_invariant_profile import LayerInvariantMeasuredProfile, LayerInvariantStore
from cost_model.search_evaluation import _CachedScorer, _Mode, _run_search, build_evaluation_graph
from cost_model.structural import StructuralCostModel
from integration.lightop.add_rmsnorm import executable_registry
from simulator import FusionGroup, Plan, validate_fusion_plan
from simulator.fusion_rules import enumerate_fusion_candidates
from simulator.fused_placed_export import export_fused_placed_graph, save_fused_placed_graph, validate_fused_placed_graph
from simulator.hardware import load_hardware
from simulator.placement_artifacts import PlacedOpGraph, load_placed_op_graph, save_placed_op_graph

HARDWARE = ROOT / "data/bw1000_2card/bw1000_2card.calibrated.json"
REFERENCES = {"none": (), "add_rmsnorm_only": ("add_rmsnorm",), "silu_mul_only": ("silu_mul",),
              "both": ("add_rmsnorm", "silu_mul")}


def setup():
    reg = executable_registry()
    hw = load_hardware(HARDWARE)
    graph = build_evaluation_graph("model")
    profile = LayerInvariantMeasuredProfile(AnalyticHardwareProfile(hw), LayerInvariantStore())
    scorer = _CachedScorer(graph, profile)
    scorer.model = StructuralCostModel(registry=reg)
    return reg, hw, graph, scorer


def reference_plan(graph, reg, patterns, device=0):
    candidates = enumerate_fusion_candidates(graph, reg, require_executable=True)
    singles = {c.node_ids[0]: c for c in candidates if len(c.node_ids) == 1}
    chosen = {n: c for c in candidates if len(c.node_ids) > 1 and c.pattern_id in patterns for n in c.node_ids}
    groups = []
    for n in graph.topological_order():
        c = chosen.get(n)
        if c is None:
            groups.append(singles[n].to_group(device))
        elif n == c.node_ids[-1]:
            groups.append(c.to_group(device))
    plan = Plan(groups=groups)
    validate_fusion_plan(graph, plan, 2, registry=reg, require_executable=True)
    return plan


def describe(scorer, plan):
    est = scorer.estimate(plan)
    sources = Counter((g.extra["pattern_id"], d["source"]) for g, d in zip(plan.groups, est.details["groups"]))
    return {"makespan_ms": est.makespan * 1e3, "feasible": scorer.is_feasible(est),
            "fused": dict(Counter(g.extra["pattern_id"] for g in plan.groups if len(g.node_ids) > 1)),
            "devices": sorted({g.device for g in plan.groups}),
            "nodes_per_device": dict(Counter(str(g.device) for g in plan.groups for _ in g.node_ids)),
            "cost_sources": {f"{p}|{s}": n for (p, s), n in sorted(sources.items())}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=256)
    parser.add_argument("--seeds", default="0,7,23")
    parser.add_argument("--unfused-guided-rollout", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    reg, hw, graph, scorer = setup()
    references = {name: describe(scorer, reference_plan(graph, reg, pats)) for name, pats in REFERENCES.items()}
    best_ref = min(v["makespan_ms"] for v in references.values())
    print(json.dumps({k: round(v["makespan_ms"], 5) for k, v in references.items()}), flush=True)
    runs = []
    for mode_name in ("joint", "sequential", "placement_then_fusion"):
        mode = _Mode(mode_name, reg, True, (0, 1))
        for seed in map(int, args.seeds.split(",")):
            plans, elapsed, budget = _run_search("mcts", graph, scorer, mode, seed=seed, beam=8, rollouts=3,
                                                 mcts_iterations=args.iterations, top_k=1,
                                                 unfused_guided_rollout=args.unfused_guided_rollout)
            validate_fusion_plan(graph, plans[0], 2, registry=reg, require_executable=True)
            row = {"mode": mode_name, "seed": seed, "iterations": args.iterations, "elapsed_s": elapsed,
                   "cost_model_evaluations": budget["cost_model_evaluations"], **describe(scorer, plans[0])}
            row["gap_to_best_known_reference"] = (row["makespan_ms"] - best_ref) / best_ref
            row["scorer_requests"] = budget["scorer_requests"]
            row["stage_one_incumbent"] = budget.get("stage_one_incumbent")
            if seed == int(args.seeds.split(",")[0]):
                est = scorer.estimate(plans[0])
                placed = PlacedOpGraph(graph, hw, plans[0], rank=1, predicted_makespan=est.makespan,
                                       predicted_peak_memory=est.peak_memory, extra={"mode": mode_name, "seed": seed})
                save_placed_op_graph(args.output / f"{mode_name}.placed.json", placed, registry=reg)
                artifact = export_fused_placed_graph(
                    load_placed_op_graph(args.output / f"{mode_name}.placed.json", registry=reg), est, registry=reg)
                validate_fused_placed_graph(artifact, graph)
                save_fused_placed_graph(args.output / f"{mode_name}.fused_placed.json", artifact)
                row["export"] = artifact["summary"] | {"deployable": artifact["deployable"]}
            runs.append(row)
            print(json.dumps({k: row[k] for k in ("mode", "seed", "makespan_ms", "fused", "devices", "elapsed_s")}
                             | {"gap_%": round(row["gap_to_best_known_reference"] * 100, 3),
                                "evals": row["cost_model_evaluations"]}), flush=True)
            (args.output / "runs.json").write_text(json.dumps(runs, indent=1), encoding="utf-8")
    report = {"scope": "full model; executable registry; cross-layer measured Profiles over analytic base; existing MCTS",
              "store_records": len(scorer.profile.store.records), "references": references, "runs": runs,
              "best_known_reference_ms": best_ref, "unfused_guided_rollout": args.unfused_guided_rollout,
              "note": "best known reference is the best of the listed plans, not a proven global optimum",
              "search_code_changed": False, "cost_model_formula_changed": False}
    (args.output / "report.json").write_text(json.dumps(report, indent=1, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
