"""Nested fixed-cost Qwen3/cap75/decision-centric MCTS budget scaling."""
import argparse
import copy
import csv
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cost_model.fusion_library import CalibratedFusionProfile, fit_corrections, load_calibration, load_library
from cost_model.hardware_profile import AnalyticHardwareProfile
from cost_model.legal_search import legal_mcts_search
from cost_model.measured_fusion_calls import MeasuredFusionCalls, sha
from cost_model.search_evaluation import build_evaluation_graph
from simulator import make_reference_plan, validate_fusion_plan
from simulator.dataset import plan_to_dict
from simulator.hardware import load_hardware
from simulator.placement_artifacts import graph_fingerprint
from tools.evaluate_full_joint import FullLedger, source_hashes
from tools.evaluate_stage_search import BudgetReached, HARDWARE, save

PARENT = ROOT / "data/decision_mcts_qwen3_20261110_v1"
BUDGETS = (64, 128, 256, 512, 1024)
SEEDS = (0, 7, 23)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prefix_metrics(trace, budget, stats):
    rows = trace[:budget]
    if len(rows) != budget or len({r["candidate_id"] for r in rows}) != budget:
        raise ValueError("budget must count different fully scored plans")
    valid = [r for r in rows if r["feasible"]]
    return {"budget": budget, "found_feasible": bool(valid),
            "first_feasible_evaluation": valid[0]["evaluation"] if valid else None,
            "distinct_feasible_plans": len(valid),
            "best_feasible_ms": min((r["makespan_ms"] for r in valid), default=None),
            "expanded_decision_nodes": stats.get("branch_expansions", 0),
            "created_tree_nodes": stats.get("created_nodes", 0),
            "max_decision_depth": stats.get("max_decision_depth", 0),
            "max_scored_assigned_nodes": max((r.get("tree_assigned_nodes", 0) for r in rows), default=0),
            "wall_s": rows[-1]["elapsed_s"], "tree_stats": copy.deepcopy(stats)}


def run_seed(graph, profile, registry, capacity, seed, budgets, output, expected_prefix=None):
    budgets = tuple(budgets)
    if not budgets or sorted(set(budgets)) != list(budgets) or budgets[0] < 2:
        raise ValueError("budgets must be unique increasing actual-call limits >=2")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    stats, points = {}, []
    maximum = max(budgets)
    ledger = FullLedger(graph, profile, registry, capacity, maximum)
    with (output / "trace.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        def evaluate(plan):
            ledger.search_context = {"tree_assigned_nodes": stats.get("selected_assigned_nodes", 0),
                                    "decision_depth": stats.get("selected_decision_depth", 0)}
            before = len(ledger.cache)
            row = ledger.evaluate(plan)
            if len(ledger.cache) > before:
                row.update(elapsed_s=time.perf_counter() - started,
                           expanded_decision_nodes=stats.get("branch_expansions", 0))
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                if len(ledger.cache) in budgets:
                    stream.flush()
                    if expected_prefix is not None and len(ledger.cache) == len(expected_prefix):
                        fields = ("candidate_id", "makespan_ms", "peak_memory_bytes", "feasible", "fused_patterns")
                        if any(any(a[k] != b[k] for k in fields) for a, b in zip(ledger.trace, expected_prefix)):
                            raise AssertionError("independent 64-run prefix differs")
                    point = prefix_metrics(ledger.trace, len(ledger.cache), stats)
                    point["seed"] = seed
                    if ledger.best:
                        save(output / f"best_at_{point['budget']}.json", plan_to_dict(ledger.best))
                        point["best_plan"] = f"best_at_{point['budget']}.json"
                    save(output / f"budget_{point['budget']}.json", point)
                    points.append(point)
                    print(json.dumps({"phase": "checkpoint", **{k: point[k] for k in (
                        "seed", "budget", "found_feasible", "first_feasible_evaluation", "distinct_feasible_plans",
                        "best_feasible_ms", "expanded_decision_nodes", "max_decision_depth", "wall_s")}}), flush=True)
            if len(ledger.cache) == maximum:
                raise BudgetReached
            return SimpleNamespace(feasible=row["feasible"], latency=row["makespan_ms"] * 1e-3 if row["feasible"] else None)
        try:
            for device in range(profile.n_devs):
                evaluate(make_reference_plan(graph, device))
            ledger.phase = "decision_joint"
            legal_mcts_search(graph, None, profile.n_devs, iterations=200 * maximum, rollouts=3,
                fusion_probability=.7, seed=seed, n_best=1, registry=registry, evaluator=evaluate,
                unfused_incumbent=True, normalize_q=True, skip_exhausted=True,
                placement_policy="hybrid", tie_shared_params=True, forced_action_closure=True,
                search_stats=stats)
            stop = "iteration_limit_or_space_exhausted"
        except BudgetReached:
            stop = "actual_evaluation_budget"
    if len(ledger.cache) != maximum or len(points) != len(budgets):
        raise AssertionError("requested scaling budgets not reached; partial checkpoints retained")
    if stats.get("unary_expansions", 0) != 0:
        raise AssertionError("unary node entered decision-only tree")
    if ledger.best:
        validate_fusion_plan(graph, ledger.best, profile.n_devs, registry=registry, require_executable=True)
    result = {"seed": seed, "actual_unique_evaluations": len(ledger.cache), "stop": stop,
              "checkpoints": points, "tree_stats": stats, "independent_64_prefix_checked": expected_prefix is not None,
              "wall_scope": "through each scored call; includes prior trace/checkpoint writes; excludes current checkpoint write and model setup"}
    save(output / "result.json", result)
    return result


def prepare(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    parent = read(PARENT / "plan.json")
    hashes = source_hashes()
    for name in ("cost_model/measured_fusion_calls.py", "tools/evaluate_mcts_budget_scaling.py"):
        hashes[name] = sha(ROOT / name)
    fixed = {p: h for p, h in parent["source_sha256"].items() if p != "tools/evaluate_decision_mcts.py"}
    for name, digest in fixed.items():
        if sha(ROOT / name) != digest:
            raise AssertionError("fixed experiment input changed: " + name)
    graph = build_evaluation_graph("model")
    if graph_fingerprint(graph) != parent["graph_fingerprint"]:
        raise AssertionError("fixed graph changed")
    config = {"budgets": list(BUDGETS), "seeds": list(SEEDS), "graph_fingerprint": graph_fingerprint(graph),
        "capacity": parent["capacities"]["cap75"], "capacity_kind": "cap75: 75% of single-card Native forward peak, not physical memory",
        "algorithm": "decision-centric hybrid Joint MCTS", "forced_action_closure": True,
        "fixed_parent_sha256": sha(PARENT / "plan.json"), "fixed_parent_sources": fixed,
        "source_sha256": hashes, "cost_and_admission": "unchanged analytic/calibrated L1 + matched local measurements + blocking HTAE; unchanged manifest",
        "initialization": "same two Native single-card plans; charged; no witness or feasibility guide",
        "evaluation_contract": "different complete-plan actual model evaluations; infeasible and initialization calls count; cache hits free",
        "trajectory_contract": "three continuous 1024-call trajectories; lower budgets are nested prefixes, not independent observations",
        "decision_node_metric": "distinct expanded nonterminal branch states, including root; excludes unvisited children and unary closure states",
        "diagnostics": "existing tree counters; previous per-action classification observer disabled, no search policy change",
        "wall_scope": "parallel seed processes; descriptive local CPU wall time, includes preceding checkpoint I/O",
        "objective": parent["objective"], "new_gpu_measurements": 0}
    save(directory / "plan.json", config)
    return config


def execute_seed(directory, seed):
    directory = Path(directory)
    config = read(directory / "plan.json")
    if seed not in config["seeds"]:
        raise ValueError("unregistered seed")
    for path, digest in config["source_sha256"].items():
        if sha(ROOT / path) != digest:
            raise ValueError("source changed since freeze: " + path)
    library = load_library()
    base = AnalyticHardwareProfile(load_hardware(HARDWARE))
    correction, _ = fit_corrections(library, base, load_calibration()[0], build_evaluation_graph)
    profile = MeasuredFusionCalls(CalibratedFusionProfile(base, library, correction), library)
    graph = build_evaluation_graph("model")
    previous = read(PARENT / "runs" / f"cap75_decision_joint_seed{seed}.json")
    return run_seed(graph, profile, library.registry(), config["capacity"], seed, config["budgets"],
                    directory / f"seed_{seed}", previous["trace"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "seed"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.output)
    else:
        execute_seed(args.output, args.seed)
