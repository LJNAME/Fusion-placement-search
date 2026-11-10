"""Full Qwen forward-search experiment; fixed real evaluation budget, no GPU execution."""
import argparse
from collections import Counter
import csv
import hashlib
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
from cost_model.legal_search import LegalFusionSearchSpace, branch_regions, rollout_legal_plan
from cost_model.search_evaluation import _plan_key, build_evaluation_graph
from cost_model.training_memory import parameter_id
from simulator import Plan, make_reference_plan
from simulator.dataset import graph_to_dict, plan_to_dict
from simulator.fusion_rules import enumerate_fusion_candidates
from simulator.hardware import load_hardware
from simulator.placement_artifacts import graph_fingerprint
from tools.evaluate_branch_search import ObservedLedger
from tools.evaluate_stage_search import HARDWARE, run_mcts, save

METHODS = ("joint_node", "joint_hybrid", "random_node", "random_hybrid")
MODEL_NAMES = {"model": "Qwen3-0.6B", "model_3b": "Qwen2.5-3B"}


def assert_shared_colocated(graph, plan):
    placed = {}
    for group in plan.groups:
        for n in group.node_ids:
            pid = parameter_id(graph.nodes[n])
            if placed.setdefault(pid, group.device) != group.device:
                raise AssertionError(f"shared parameter split without a replica policy: {pid}")


def reference_plans(graph, registry):
    """Eight predeclared references, never injected into heuristic search.

    Uniform or half-layer split, both card orientations, Native or AddRMS
    fusion. A tied lm_head returns to the embedding owner rather than silently
    treating a remote shared weight as zero bytes.
    """
    candidates = [c for c in enumerate_fusion_candidates(graph, registry, require_executable=True)
                  if c.pattern_id == "lightop_add_rmsnorm"]
    cut = int(graph.extra["num_layers"]) // 2
    refs = {}
    for kind in ("uniform", "half_layers"):
        for owner in (0, 1):
            devices = {n: owner if kind == "uniform" or node.extra["layer"] < cut else 1 - owner
                       for n, node in graph.nodes.items()}
            for n, node in graph.nodes.items():
                if node.extra.get("shared_param_with"):
                    devices[n] = devices[node.extra["shared_param_with"]]
            for fused in (False, True):
                base = make_reference_plan(graph, 0)
                selected = {c.node_ids[-1]: c for c in candidates
                            if fused and len({devices[n] for n in c.node_ids}) == 1}
                skipped = {n for c in selected.values() for n in c.node_ids[:-1]}
                groups = []
                for group in base.groups:
                    n = group.node_ids[0]
                    if n in skipped:
                        continue
                    if n in selected:
                        group = selected[n].to_group(devices[n])
                    else:
                        group.device = devices[n]
                    groups.append(group)
                plan = Plan(groups=groups, name=f"{kind}_owner{owner}_{'addnorm' if fused else 'native'}")
                assert_shared_colocated(graph, plan)
                refs[plan.name] = plan
    return refs


class FullLedger(ObservedLedger):
    def __init__(self, *args, label="", progress=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph_id = graph_fingerprint(self.graph)
        self.label, self.progress = label, progress
        self.search_context = {}

    def evaluate(self, plan):
        annotate = getattr(self.profile, "annotate_plan", None)
        if annotate is not None:
            annotate(self.graph, plan)
        return super().evaluate(plan)

    def observe(self, plan, estimate):
        assert_shared_colocated(self.graph, plan)
        stats = super().observe(plan, estimate)
        key = json.dumps((self.graph_id, _plan_key(plan)), separators=(",", ":"))
        stats.update(self.search_context)
        bindings = [g.extra["measured_complete_call"] for g in plan.groups if "measured_complete_call" in g.extra]
        stats.update(measured_call_count=len(bindings),
                     measured_execution_keys=dict(Counter(b["execution_key_id"] for b in bindings)),
                     candidate_id=hashlib.sha256(key.encode()).hexdigest(),
                     cost_sources=dict(Counter(g["source"] for g in estimate.details["groups"])),
                     fused_patterns=dict(Counter(g.extra["pattern_id"] for g in plan.groups if len(g.node_ids) > 1)),
                     nodes_per_device=dict(Counter(str(g.device) for g in plan.groups for _ in g.node_ids)))
        count = len(self.cache) + 1
        if self.progress is not None and (count % 16 == 0 or count == self.budget):
            row = {"run": self.label, "evaluation": count,
                   "best_ms": self.best_s * 1e3 if self.best else None}
            save(self.progress, row)
            print(json.dumps({"phase": "progress", **row}), flush=True)
        return stats


def run_one(graph, profile, registry, capacity, method, seed, budget, label, progress=None):
    started = time.perf_counter()
    ledger = FullLedger(graph, profile, registry, capacity, budget, label=label, progress=progress)
    for device in range(profile.n_devs):
        ledger.evaluate(make_reference_plan(graph, device))
    initial = ledger.best_s * 1e3 if ledger.best else None
    policy = "hybrid" if method.endswith("hybrid") else "node"
    ledger.phase = method
    if method.startswith("joint"):
        def observe_rollout(space, state, plan):
            ledger.search_context = {"tree_choices": len(state.choices), "tree_assigned_nodes": len(state.assigned),
                "rollout_completed_nodes": len(space.order) - len(state.assigned),
                "tree_prefix_id": hashlib.sha256(repr(state.key()).encode()).hexdigest()}
        _, stopped = run_mcts(ledger, graph, registry, policy=policy, seed=seed,
                              iterations=200 * budget, tie_shared_params=True, rollout_observer=observe_rollout)
        draws = None
    else:
        space = LegalFusionSearchSpace(graph, profile.n_devs, registry, placement_policy=policy,
                                       tie_shared_params=True)
        rng, draws = random.Random(seed), 0
        while len(ledger.cache) < budget and draws < 1000 * budget:
            ledger.evaluate(rollout_legal_plan(space, space.initial_state(), rng, fusion_probability=.7, validate=False))
            draws += 1
        stopped = "evaluation_budget" if len(ledger.cache) == budget else "draw_limit"
    if len(ledger.cache) != budget:
        raise AssertionError(f"{label}: only {len(ledger.cache)}/{budget} actual evaluations")
    best = ledger.describe(ledger.best)
    first_best = next((r["evaluation"] for r in ledger.trace if best and r["best_ms"] == best["makespan_ms"]), None)
    return {"method": method, "seed": seed, "budget": budget, "budget_used": len(ledger.cache),
            "unique_candidates": len(ledger.cache), "budget_remaining": budget - len(ledger.cache),
            "scorer_requests": ledger.requests, "feasible_evaluations": sum(r["feasible"] for r in ledger.trace),
            "initial_best_ms": initial, "best": best,
            "first_final_best_evaluation": first_best,
            "first_feasible_evaluation": next((r["evaluation"] for r in ledger.trace if r["feasible"]), None),
            "elapsed_s": time.perf_counter() - started, "stop": stopped, "random_draws": draws,
            "reason": None if best else "evaluated_no_feasible_plan", "trace": ledger.trace}, ledger.best


def source_hashes():
    files = [Path(__file__), HARDWARE, ROOT / "tools/evaluate_stage_search.py", ROOT / "tools/evaluate_branch_search.py",
             ROOT / "cost_model/legal_search.py", ROOT / "cost_model/htae.py", ROOT / "cost_model/structural.py",
             ROOT / "cost_model/fusion_library.py", ROOT / "cost_model/hardware_profile.py", ROOT / "cost_model/search_evaluation.py",
             ROOT / "simulator/model_graphs/transformer.py", ROOT / "simulator/model_graphs/dense_layer.py",
             ROOT / "simulator/schema.py", ROOT / "simulator/shape_model.py",
             ROOT / "data/fusion_library_lightop_v1/manifest.json", ROOT / "data/fusion_library_lightop_v1/calibration.json"]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def run(output, scale, budget=64, seeds=(0, 7, 23), caps=("free", "cap75"), resume=False,
        measurements=None, methods=METHODS):
    if budget < 2 or scale not in MODEL_NAMES or any(c not in ("free", "cap75") for c in caps):
        raise ValueError("invalid full-model experiment configuration")
    if not methods or any(m not in METHODS for m in methods) or len(set(methods)) != len(methods):
        raise ValueError("unknown or duplicate methods")
    output = Path(output)
    config = {"model": MODEL_NAMES[scale], "scale": scale, "budget": budget, "seeds": list(seeds),
              "caps": list(caps), "methods": list(methods), "source_sha256": source_hashes(),
              "objective": "forward makespan; analytic L1 plus existing LightOP calibration; blocking HTAE",
              "shared_parameters": "tie_shared_params=True in every method; no unmodeled remote tied weights",
              "budget_contract": "64 by default actual unique complete-plan model evaluations, including both native initializations and infeasible calls",
              "references": "eight predefined legal uniform/half-layer plans, never search seeds; no Exact oracle",
              "capacity_contract": "free=physical total; cap75=75% of unfused single-card forward peak; synthetic, not training",
              "cpu_wall_time_scope": "diagnostic; two model processes may run concurrently; not an algorithm speed benchmark",
              "gpu_measurements": 0, "full_model_gpu_execution": False}
    if measurements is not None:
        from cost_model.measured_fusion_calls import sha
        folder = Path(measurements).resolve()
        config.update(objective="forward makespan; exact local LightOP complete-call costs where matching, existing analytic/calibrated fallback; blocking HTAE",
                      measurement_directory=folder.relative_to(ROOT).as_posix(),
                      measurement_source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in folder.rglob('*.json')})
        for name in ("cost_model/measured_fusion_calls.py", "simulator/fusion_rules.py"):
            config["source_sha256"][name] = sha(ROOT / name)
    if resume:
        if json.loads((output / "plan.json").read_text(encoding="utf-8")) != config:
            raise ValueError("resume configuration/source differs from frozen plan")
    else:
        output.mkdir(parents=True, exist_ok=False)
        save(output / "plan.json", config)
    (output / "runs").mkdir(exist_ok=True)
    (output / "plans").mkdir(exist_ok=True)
    graph = build_evaluation_graph(scale)
    graph_id = graph_fingerprint(graph)
    save(output / "graph.json", graph_to_dict(graph))
    library = load_library()
    observations, _ = load_calibration()
    base = AnalyticHardwareProfile(load_hardware(HARDWARE))
    corrections, _ = fit_corrections(library, base, observations, build_evaluation_graph)
    profile, registry = CalibratedFusionProfile(base, library, corrections), library.registry("full")
    if measurements is not None:
        from cost_model.measured_fusion_calls import MeasuredFusionCalls
        profile = MeasuredFusionCalls(profile, library, measurements)
        save(output / "fusion_cost_audit.json", profile.audit_graph(graph, registry))
    setup = FullLedger(graph, profile, registry, [d.max_mem for d in profile.hardware.devices])
    references = reference_plans(graph, registry)
    ref_rows = {name: setup.evaluate(p) for name, p in references.items()}
    peak = max(ref_rows["uniform_owner0_native"]["peak_memory_bytes"])
    report = {"scope": config, "graph_fingerprint": graph_id, "nodes": len(graph.nodes), "edges": len(graph.edges),
              "branch_regions": len(branch_regions(graph, graph.topological_order())), "corrections": corrections,
              "fusion_candidates": dict(Counter(c.pattern_id for c in enumerate_fusion_candidates(graph, registry, require_executable=True) if len(c.node_ids) > 1)),
              "tasks": {}, "status": "running", "global_optimum_proven": False}
    for cap in caps:
        capacity = [d.max_mem for d in profile.hardware.devices] if cap == "free" else [peak * .75] * profile.n_devs
        refs = {n: {**r, "feasible": all(p <= c for p, c in zip(r["peak_memory_bytes"], capacity))} for n, r in ref_rows.items()}
        feasible_refs = [(r["makespan_ms"], n) for n, r in refs.items() if r["feasible"]]
        best_ref = min(feasible_refs) if feasible_refs else None
        task = {"capacity_bytes": capacity, "references": refs,
                "best_known_reference": {"makespan_ms": best_ref[0], "name": best_ref[1]} if best_ref else None,
                "runs": []}
        report["tasks"][cap] = task
        for method in methods:
            for seed in seeds:
                label = f"{scale}:{cap}:{method}:seed{seed}"
                path = output / "runs" / f"{cap}_{method}_seed{seed}.json"
                if resume and path.exists():
                    row = json.loads(path.read_text(encoding="utf-8"))
                    if row["graph_fingerprint"] != graph_id or row["budget_used"] != budget:
                        raise ValueError("completed run is not compatible with resume")
                else:
                    print(json.dumps({"phase": "start", "run": label}), flush=True)
                    row, plan = run_one(graph, profile, registry, capacity, method, seed, budget, label, output / "last_progress.json")
                    row.update(graph_fingerprint=graph_id)
                    if plan:
                        key = hashlib.sha256(json.dumps((graph_id, _plan_key(plan)), separators=(",", ":")).encode()).hexdigest()
                        plan_path = output / "plans" / f"{key}.json"
                        if not plan_path.exists():
                            save(plan_path, plan_to_dict(plan))
                        row["plan_path"] = plan_path.relative_to(output).as_posix()
                    else:
                        row["plan_path"] = None
                    if row["best"] and best_ref:
                        row["known_reference_gap_percent"] = (row["best"]["makespan_ms"] / best_ref[0] - 1) * 100
                        row["first_within_1pct_of_reference"] = next((r["evaluation"] for r in row["trace"] if r["best_ms"] is not None and r["best_ms"] <= best_ref[0] * 1.01 + 1e-9), None)
                    else:
                        row["known_reference_gap_percent"] = row["first_within_1pct_of_reference"] = None
                    save(path, row)
                    print(json.dumps({"phase": "finished", "run": label, "best_ms": row["best"]["makespan_ms"] if row["best"] else None,
                                      "evaluations": row["budget_used"], "elapsed_s": row["elapsed_s"]}), flush=True)
                task["runs"].append(row)
                save(output / "report.json", report)
        task["summary"] = {}
        for method in methods:
            rows = [r for r in task["runs"] if r["method"] == method]
            feasible = [r for r in rows if r["best"]]
            task["summary"][method] = {"runs": len(rows), "feasible_runs": len(feasible),
                "median_ms": statistics.median([r["best"]["makespan_ms"] for r in feasible]) if feasible else None,
                "median_reference_gap_percent": statistics.median([r["known_reference_gap_percent"] for r in feasible]) if feasible and best_ref else None,
                "median_cross_device_edges": statistics.median([r["best"]["cross_device_edges"] for r in feasible]) if feasible else None}
    report["status"] = "complete"
    save(output / "report.json", report)
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(("model", "capacity", "method", "seed", "makespan_ms", "known_reference_gap_percent", "unique_candidates", "first_final_best_evaluation", "cross_device_edges", "branch_overlap_us", "fused_groups", "elapsed_s"))
        for cap, task in report["tasks"].items():
            for row in task["runs"]:
                best = row["best"] or {}
                writer.writerow((MODEL_NAMES[scale], cap, row["method"], row["seed"], best.get("makespan_ms"), row["known_reference_gap_percent"], row["unique_candidates"], row["first_final_best_evaluation"], best.get("cross_device_edges"), best.get("branch_overlap_us"), best.get("fused_groups"), row["elapsed_s"]))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODEL_NAMES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--seeds", default="0,7,23")
    parser.add_argument("--caps", default="free,cap75")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fusion-measurements", type=Path)
    parser.add_argument("--methods", default=",".join(METHODS))
    args = parser.parse_args()
    run(args.output, args.model, args.budget, tuple(int(x) for x in args.seeds.split(",")), tuple(args.caps.split(",")), args.resume,
        args.fusion_measurements, tuple(args.methods.split(",")))
