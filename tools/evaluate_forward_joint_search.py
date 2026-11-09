"""Does joint fusion+placement MCTS find better forward plans? (layer-2 = HTAE)

Forward makespan from the Proteus-style executor (cost_model/htae.py) over the
layer-1 analytic profile with the LightOP Fusion Library (full scope, calibrated).
Compared at equal budget: joint MCTS vs staged searches (sequential = fuse then
place, placement_then_fusion), single-dimension searches, and random sampling
of complete legal joint plans with the same number of distinct evaluated plans.
Small graphs also get Exact per mode, which shows whether the joint space holds a
strictly better plan than any staged space at all.

Capacity variants: ``free`` (device max_mem) and ``cap60`` (each device holds
0.6 x the unfused single-device forward peak, so a split is forced).
Executor variants: ``blocking`` (sends block the sender's compute, the
hardware snapshot) and ``async`` (copy overlaps compute, gamma = 0; NOT measured
on BW1000).
"""
import argparse
import copy
import json
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cost_model.fusion_library import CalibratedFusionProfile, fit_corrections, load_calibration, load_library
from cost_model.hardware_profile import AnalyticHardwareProfile
from cost_model.htae import HTAECostModel
from cost_model.legal_search import LegalFusionSearchSpace, _unfused_guided_rollout_legal_plan, rollout_legal_plan
from cost_model.search_evaluation import (
    _CachedScorer, _modes, _plan_key, _run_search, build_evaluation_graph, placement_then_fusion_mode)
from dataclasses import replace
from simulator import FusionGroup, Plan, make_reference_plan
from simulator.fusion_rules import enumerate_fusion_candidates
from simulator.hardware import load_hardware
from tools.evaluate_mcts_medium import SELECTIONS, extract_graph

HARDWARE = ROOT / "data/bw1000_2card/bw1000_2card.calibrated.json"
EXECUTORS = {"blocking": dict(feature_comm_blocks_compute=True, overlap_factor=0.0),
             "async": dict(feature_comm_blocks_compute=False, overlap_factor=0.0)}
CAPS = {"free": None, "cap75": 0.75, "cap60": 0.6}
FUSION_PROBABILITY = .7
EXACT_MAX_NODES = 15


def graphs(scope):
    if scope in ("medium", "all"):
        layer = build_evaluation_graph("layer")
        for name in SELECTIONS:
            yield "medium_" + name, extract_graph(layer, name)
    if scope in ("layer", "all"):
        yield "layer", build_evaluation_graph("layer")
    if scope in ("model_3b", "all"):
        yield "model_3b", build_evaluation_graph("model_3b")


def search_modes(library):
    registry = library.registry("full")
    out = []
    for mode in (*_modes(2), placement_then_fusion_mode(2)):
        out.append(mode if mode.name == "placement_only" else replace(mode, registry=registry))
    return out


VARIANTS = {"raw": {}, "fixed": {"normalize_q": True, "skip_exhausted": True}}


def make_scorer(graph, profile, library, executor, capacity):
    # Hard constraint: infeasible plans never enter Q or results (all modes alike).
    scorer = _CachedScorer(graph, profile, capacity=capacity, hard_constraint=True)
    scorer.model = HTAECostModel(registry=library.registry("full"), **EXECUTORS[executor])
    return scorer


def describe(scorer, plan):
    if plan is None:
        return None
    est = scorer.estimate(plan)
    view = scorer.memory_view(plan)
    return {"makespan_ms": view.makespan * 1e3, "feasible": view.feasible,
            "fused_groups": sum(len(g.node_ids) > 1 for g in plan.groups),
            "patterns": sorted({g.extra["pattern_id"] for g in plan.groups if len(g.node_ids) > 1}),
            "used_devices": sorted({g.device for g in plan.groups}),
            "cross_device_bytes": est.cross_device_bytes,
            "peak_memory_bytes": list(view.peak_memory), "plan_key": repr(_plan_key(plan))}


def contiguous_split(graph, library, cut, fuse):
    """Reference, not a search: topological prefix on device 0, suffix on device 1
    (cut=len -> single device). Library fusions in ``fuse`` are applied unless
    they straddle the cut."""
    order = graph.topological_order()
    pos = {n: i for i, n in enumerate(order)}
    dev = lambda n: 0 if pos[n] < cut else 1  # noqa: E731
    chosen = {c.node_ids[-1]: c for c in enumerate_fusion_candidates(graph, library.registry())
              if len(c.node_ids) > 1 and c.pattern_id in fuse and len({dev(x) for x in c.node_ids}) == 1}
    skip = {n for c in chosen.values() for n in c.node_ids[:-1]}
    groups = [chosen[n].to_group(dev(n)) if n in chosen else FusionGroup([n], dev(n), {"pattern_id": "unfused"})
              for n in order if n not in skip]
    return Plan(groups=groups)


def best_contiguous(graph, scorer, library, fuse):
    n = len(graph.nodes)
    step = 1 if n < 100 else 3
    best = None
    for cut in [*range(1, n, step), n]:
        view = scorer.memory_view(contiguous_split(graph, library, cut, fuse))
        if view.feasible and (best is None or view.makespan < best[0]):
            best = (view.makespan, cut)
    return None if best is None else {"makespan_ms": best[0] * 1e3, "cut": best[1], "cut_step": step,
                                      "fused_patterns": sorted(fuse)}


def random_search(graph, scorer, mode, unique_budget, seed):
    """Unfused plan first (same incumbent as MCTS), then uniform legal rollouts."""
    space = LegalFusionSearchSpace(graph, scorer.profile.n_devs, mode.registry,
                                   require_executable=mode.require_executable,
                                   allowed_devices=mode.allowed_devices)
    rng = random.Random(seed)
    seen, draws = {}, 0
    first = _unfused_guided_rollout_legal_plan(space, space.initial_state())
    scorer.start_run()
    while len(seen) < unique_budget and draws < 50 * unique_budget:
        plan = first if draws == 0 else rollout_legal_plan(
            space, space.initial_state(), rng, fusion_probability=FUSION_PROBABILITY, validate=False)
        draws += 1
        key = _plan_key(plan)
        if key in seen:
            continue
        scorer(plan)
        seen[key] = (scorer.last_makespan, scorer.last_feasible, plan)
    feasible = sorted((v for v in seen.values() if v[1]), key=lambda v: v[0])
    return (feasible[0][2] if feasible else None), {"cost_model_evaluations": len(seen), "draws": draws}


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    library = load_library()
    observations, _ = load_calibration()
    base = AnalyticHardwareProfile(load_hardware(HARDWARE))
    corrections, _ = fit_corrections(library, base, observations, build_evaluation_graph)
    profile = CalibratedFusionProfile(base, library, corrections)
    modes = search_modes(library)
    budgets = [int(x) for x in args.budgets.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    rows, tasks = [], {}
    for name, graph in graphs(args.scope):
        reference = make_reference_plan(graph, 0)
        for executor in args.executors.split(","):
            free = make_scorer(graph, profile, library, executor, None)
            free.cache_enabled = True
            ref = describe(free, reference)
            for cap_name in args.caps.split(","):
                fraction = CAPS[cap_name]
                capacity = None if fraction is None else tuple(fraction * max(ref["peak_memory_bytes"])
                                                               for _ in range(2))
                task = f"{name}:{executor}:{cap_name}"
                scorer = make_scorer(graph, profile, library, executor, capacity)
                scorer.cache_enabled = True  # evaluations = distinct plans for every graph size
                ref_view = describe(scorer, reference)
                beneficial = {e.fusion_id for e in library.eligible("full")} - {"lightop_silu_mul"}
                tasks[task] = {"nodes": len(graph.nodes), "reference_unfused_dev0": ref_view,
                               "capacity_bytes": list(scorer.capacity), "executor": EXECUTORS[executor],
                               "reference_contiguous": {
                                   "unfused": best_contiguous(graph, scorer, library, set()),
                                   "fused": best_contiguous(graph, scorer, library, beneficial),
                                   "note": "best single topological cut; silu_mul left unfused (calibrated loss)"}}
                if args.references_only:
                    print(json.dumps({"task": task, **tasks[task]["reference_contiguous"]}), flush=True)
                    continue
                print(json.dumps({"task": task, "reference_ms": round(ref_view["makespan_ms"], 5),
                                  "reference_feasible": ref_view["feasible"]}), flush=True)
                runs = ([("exact", 0, None)] if len(graph.nodes) <= EXACT_MAX_NODES else []) + [
                    ("mcts", s, b) for b in budgets for s in seeds]
                for mode in modes:
                    if args.modes and mode.name not in args.modes.split(","):
                        continue
                    for method, seed, budget in runs:
                        # Fresh cache per run: evaluations = distinct plans this search scored.
                        scorer = make_scorer(graph, profile, library, executor, capacity)
                        scorer.cache_enabled = True
                        started = time.perf_counter()
                        plans, _, info = _run_search(method, graph, scorer, mode, seed=seed, beam=8,
                                                     rollouts=args.rollouts, mcts_iterations=budget or 2,
                                                     top_k=1, unfused_incumbent=method == "mcts",
                                                     **(VARIANTS[args.variant] if method == "mcts" else {}))
                        best = describe(scorer, plans[0]) if plans else None
                        rows.append({"task": task, "mode": mode.name, "method": method, "seed": seed,
                                     "iterations": budget, "elapsed_s": time.perf_counter() - started,
                                     "cost_model_evaluations": info["cost_model_evaluations"],
                                     "best": best, "reason": info["reason"]})
                        if method == "mcts" and mode.name == "joint":
                            started = time.perf_counter()
                            measure = scorer
                            scorer = make_scorer(graph, profile, library, executor, capacity)
                            scorer.cache_enabled = True
                            plan, rinfo = random_search(graph, scorer, mode, info["cost_model_evaluations"], seed)
                            rows.append({"task": task, "mode": "joint", "method": "random", "seed": seed,
                                         "iterations": budget, "elapsed_s": time.perf_counter() - started,
                                         "cost_model_evaluations": rinfo["cost_model_evaluations"],
                                         "draws": rinfo["draws"], "best": describe(measure, plan),
                                         "reason": None if plan else "evaluated_no_feasible"})
                (args.output / "rows.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    summary = summarize(rows, tasks)
    report = {"scope": __doc__.strip().splitlines()[0], "objective": "forward makespan (HTAE, analytic L1)",
              "mcts_variant": {"name": args.variant, **VARIANTS[args.variant]}, "constraint": "hard",
              "fusion_library": {"scope": "full", "corrections": corrections},
              "budgets": budgets, "seeds": seeds, "rollouts": args.rollouts,
              "fusion_probability": FUSION_PROBABILITY, "unfused_incumbent": True,
              "assumptions": {"async": "copy overlaps compute; gamma=0; not measured on BW1000",
                              "l1": "analytic + one LightOP calibration; see data/fusion_benefit_validation_v1"},
              "tasks": tasks, "summary": summary, "rows": rows}
    (args.output / "report.json").write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    return report


def summarize(rows, tasks):
    out = {}
    for task, info in tasks.items():
        ref = info["reference_unfused_dev0"]
        entry = {"reference_ms": ref["makespan_ms"] if ref["feasible"] else None}
        sel = [r for r in rows if r["task"] == task]
        for r in sel:
            if r["method"] == "exact":
                entry.setdefault("exact", {})[r["mode"]] = r["best"] and r["best"]["makespan_ms"]
        for r in sel:
            if r["method"] in ("mcts", "random"):
                key = f"{r['mode']}/{r['method']}@{r['iterations']}"
                entry.setdefault("search", {}).setdefault(key, []).append(
                    r["best"]["makespan_ms"] if r["best"] else None)
        for key, values in entry.get("search", {}).items():
            finite = [v for v in values if v is not None]
            entry["search"][key] = {"median_ms": statistics.median(finite) if finite else None,
                                    "best_ms": min(finite) if finite else None,
                                    "worst_ms": max(finite) if finite else None,
                                    "no_feasible_runs": len(values) - len(finite)}
        out[task] = entry
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("medium", "layer", "model_3b", "all"), default="all")
    parser.add_argument("--executors", default="blocking,async")
    parser.add_argument("--caps", default="free,cap60")
    parser.add_argument("--modes", default=None)
    parser.add_argument("--budgets", default="16,64,256")
    parser.add_argument("--seeds", default="0,7,23")
    parser.add_argument("--rollouts", type=int, default=3)
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="fixed",
                        help="raw: historical MCTS selection; fixed: normalize_q + skip_exhausted")
    parser.add_argument("--references-only", action="store_true",
                        help="only the unfused and best-contiguous-cut references, no searches")
    parser.add_argument("--output", type=Path, required=True)
    report = run(parser.parse_args(argv))
    for task, e in report["summary"].items():
        print(task, "ref", e["reference_ms"] and round(e["reference_ms"], 5),
              "exact", {k: v and round(v, 5) for k, v in e.get("exact", {}).items()})
        for k, v in sorted(e.get("search", {}).items()):
            print("   ", k, v["median_ms"] and round(v["median_ms"], 5), "worst", v["worst_ms"] and round(v["worst_ms"], 5),
                  "nofeas", v["no_feasible_runs"])


if __name__ == "__main__":
    main()
