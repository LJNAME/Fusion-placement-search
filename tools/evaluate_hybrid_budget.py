"""P0: 8/16 actual-call budgets, before any hybrid domain is exhausted."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cost_model.fusion_library import CalibratedFusionProfile, fit_corrections, load_calibration, load_library
from cost_model.hardware_profile import AnalyticHardwareProfile
from cost_model.legal_search import LegalFusionSearchSpace, rollout_legal_plan
from cost_model.search_evaluation import build_evaluation_graph
from simulator import make_reference_plan
from simulator.dataset import plan_to_dict
from simulator.hardware import load_hardware
from simulator.placement_artifacts import graph_fingerprint
from tools.evaluate_branch_search import ObservedLedger, exact, search
from tools.evaluate_mcts_medium import SELECTIONS, extract_graph
from tools.evaluate_stage_search import HARDWARE, save

METHODS = ("joint_node", "joint_hybrid", "random_node", "random_hybrid")
TARGET_GAP_PERCENT = 1.0
TOL_MS = 1e-9


def assess(trace, reference_ms):
    def first(target):
        return next((r["evaluation"] for r in trace if r["best_ms"] is not None and r["best_ms"] <= target), None)
    best = trace[-1]["best_ms"] if trace else None
    return {"exact_gap_percent": None if best is None else max(0., (best / reference_ms - 1) * 100),
            "first_exact_evaluation": first(reference_ms + TOL_MS),
            "first_1pct_evaluation": first(reference_ms * (1 + TARGET_GAP_PERCENT / 100) + TOL_MS)}


def trial(graph, profile, registry, capacity, method, budget, seed, domain_size):
    policy = "hybrid" if method.endswith("hybrid") else "node"
    if method.startswith("joint"):
        result = search(graph, profile, registry, capacity, policy, budget, seed, domain_size)
    else:
        ledger = ObservedLedger(graph, profile, registry, capacity, budget)
        for device in range(profile.n_devs):
            ledger.evaluate(make_reference_plan(graph, device))
        ledger.phase = policy
        space = LegalFusionSearchSpace(graph, profile.n_devs, registry, placement_policy=policy)
        rng, draws = random.Random(seed), 0
        while len(ledger.cache) < min(budget, domain_size) and draws < 1000 * budget:
            ledger.evaluate(rollout_legal_plan(space, space.initial_state(), rng, fusion_probability=.7, validate=False))
            draws += 1
        used = len(ledger.cache)
        if used != min(budget, domain_size):
            raise AssertionError("Random did not spend the same unique-call budget")
        result = {"policy": policy, "seed": seed, "budget": budget, "budget_used": used,
                  "budget_remaining": budget - used, "unique_candidates": used,
                  "draws": draws, "scorer_requests": ledger.requests, "best": ledger.describe(ledger.best),
                  "plan": plan_to_dict(ledger.best) if ledger.best else None, "trace": ledger.trace}
    result["method"] = method
    return result


def summarize(rows):
    costs = [r["best"]["makespan_ms"] for r in rows if r["best"]]
    gaps = [r["exact_gap_percent"] for r in rows if r["best"]]
    return {"runs": len(rows), "feasible_runs": len(costs),
            "median_ms_among_feasible": statistics.median(costs) if costs else None,
            "median_gap_percent_among_feasible": statistics.median(gaps) if gaps else None,
            "exact_hits": sum(r["first_exact_evaluation"] is not None for r in rows),
            "within_1pct_hits": sum(r["first_1pct_evaluation"] is not None for r in rows),
            "first_1pct_evaluations": [r["first_1pct_evaluation"] for r in rows],
            "budget_used": [r["budget_used"] for r in rows]}


def followup_curves(task_name, task, graph, profile, registry, seeds, previous, previous_random):
    """Exploratory follow-up: reuse frozen 64-call controls; add only hybrid Random.

    This is explicitly separate from the preregistered 8/16-call comparison.
    Exact targets and metrics are unchanged. All earlier exposed scores must
    match the independently run 8/16 prefixes before a frozen trace is reused.
    """
    prior = previous["tasks"][task_name]
    prior_random = previous_random["tasks"][task_name]
    if any(p["graph_fingerprint"] != task["graph_fingerprint"] or p["capacity_bytes"] != task["capacity_bytes"]
           for p in (prior, prior_random)):
        raise AssertionError("frozen trace graph/capacity mismatch")
    reference = task["oracles"]["node"]["best"]["makespan_ms"]
    if prior["oracles"]["node"]["best"]["makespan_ms"] != reference or prior_random["exact"]["best"]["makespan_ms"] != reference:
        raise AssertionError("frozen cost reference differs")
    columns = ("evaluation", "makespan_ms", "feasible", "peak_memory_bytes", "cross_device_edges", "cross_device_bytes", "best_ms", "fused_groups")
    curves = []
    for method in METHODS:
        for seed in seeds:
            if method == "random_hybrid":
                row = trial(graph, profile, registry, task["capacity_bytes"], method, 64, seed, task["oracles"]["hybrid"]["space_size"])
                source = "new cold-cache same-domain Random follow-up"
            else:
                if method == "random_node":
                    old = next(x for x in prior_random["runs"] if x["method"] == "random" and x["seed"] == seed)
                else:
                    policy = "node" if method == "joint_node" else "hybrid"
                    old = next(x for x in prior["runs"] if x["policy"] == policy and x["seed"] == seed)
                row = {"method": method, "seed": seed, "budget": 64, "budget_used": old["unique_candidates"], "trace": old["trace"], "best": old["best"]}
                source = "frozen 64-budget trace; no new MCTS or full-domain Random run"
            for early in (x for x in task["runs"] if x["method"] == method and x["seed"] == seed):
                for a, b in zip(early["trace"], row["trace"]):
                    if any(a[c] != b[c] for c in columns):
                        raise AssertionError("frozen/follow-up curve does not match the independent early prefix")
            row.update(assess(row["trace"], reference), source=source,
                       failure_censor_at_evaluation=row["budget_used"])
            curves.append(row)
    return {"scope": "secondary exploratory follow-up after primary 8/16 results; never replaces them",
            "reason": "disentangle action-space coverage from tree selection using existing 64-budget curves and one missing same-domain Random control",
            "curves": curves}


def run(output, budgets=(8, 16), seeds=(0, 7, 23), followup=False):
    if not budgets or any(b < 2 for b in budgets):
        raise ValueError("each budget must include the two initialization plans")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    library = load_library()
    observations, _ = load_calibration()
    base = AnalyticHardwareProfile(load_hardware(HARDWARE))
    corrections, _ = fit_corrections(library, base, observations, build_evaluation_graph)
    profile = CalibratedFusionProfile(base, library, corrections)
    registry, full = library.registry("full"), build_evaluation_graph("model_3b")
    report = {"scope": "P0 budget advantage; unchanged three 3B medium templates and cost model",
              "budgets": list(budgets), "seeds": list(seeds), "methods": list(METHODS),
              "quality_targets": {"Exact_tolerance_ms": TOL_MS, "gap_percent": TARGET_GAP_PERCENT},
              "budget_contract": "actual unique complete-plan model calls, shared two-card native initialization; all infeasible calls charged; no future labels or Exact plans in search",
              "random_contract": "same fusion_probability=.7 legal random completions; both original and hybrid domains; not uniform over complete plans",
              "exclusions": "retain both known Exact-infeasible tasks without rerunning meaningless success-rate trials",
              "statistical_scope": "three paired seeds; descriptive, no significance or general superiority claim",
              "tasks": {}}
    if followup:
        previous = json.loads((ROOT / "data/branch_search_medium_v1/report.json").read_text(encoding="utf-8"))
        previous_random = json.loads((ROOT / "data/stage_search_medium_v1/report.json").read_text(encoding="utf-8"))
    for name in SELECTIONS:
        graph = extract_graph(full, name)
        probe = ObservedLedger(graph, profile, registry, [d.max_mem for d in profile.hardware.devices])
        peak = max(probe.model.predict(graph, make_reference_plan(graph, 0), profile).peak_memory)
        for cap in ("free", "cap75"):
            capacity = [d.max_mem for d in profile.hardware.devices] if cap == "free" else [peak * .75] * profile.n_devs
            oracles = {p: exact(graph, profile, registry, capacity, p)[0] for p in ("node", "hybrid")}
            if max(budgets) >= oracles["hybrid"]["space_size"]:
                raise ValueError("P0 budgets must stay strictly below the hybrid domain size")
            task = {"graph_fingerprint": graph_fingerprint(graph), "capacity_bytes": capacity, "oracles": oracles,
                    "status": "feasible" if oracles["node"]["best"] else "Exact_infeasible", "runs": [], "summary": {}}
            if task["status"] == "feasible":
                for budget in budgets:
                    for method in METHODS:
                        policy = "hybrid" if method.endswith("hybrid") else "node"
                        for seed in seeds:
                            row = trial(graph, profile, registry, capacity, method, budget, seed, oracles[policy]["space_size"])
                            row.update(assess(row["trace"], oracles["node"]["best"]["makespan_ms"]))
                            task["runs"].append(row)
                    task["summary"][str(budget)] = {m: summarize([r for r in task["runs"] if r["budget"] == budget and r["method"] == m]) for m in METHODS}
                # Earlier choices must not depend on a future budget or labels.
                for method in METHODS:
                    for seed in seeds:
                        ordered = sorted((r for r in task["runs"] if r["method"] == method and r["seed"] == seed), key=lambda r: r["budget"])
                        for smaller, larger in zip(ordered, ordered[1:]):
                            if smaller["trace"] != larger["trace"][:len(smaller["trace"])]:
                                raise AssertionError("budget-dependent prefix or leaked future information")
                if followup:
                    task["followup_existing64"] = followup_curves(f"{name}:{cap}", task, graph, profile, registry, seeds, previous, previous_random)
            report["tasks"][f"{name}:{cap}"] = task
            print(json.dumps({"task": f"{name}:{cap}", "status": task["status"], "summary": task["summary"]}), flush=True)
            save(output / "report.json", report)
    files = [Path(__file__), HARDWARE, ROOT / "tools/evaluate_branch_search.py", ROOT / "tools/evaluate_stage_search.py",
             ROOT / "cost_model/legal_search.py", ROOT / "cost_model/htae.py", ROOT / "cost_model/hardware_profile.py",
             ROOT / "cost_model/fusion_library.py", ROOT / "cost_model/search_evaluation.py", ROOT / "tools/evaluate_mcts_medium.py",
             ROOT / "cost_model/exact_search.py",
             ROOT / "data/fusion_library_lightop_v1/manifest.json", ROOT / "data/fusion_library_lightop_v1/calibration.json"]
    if followup:
        files += [ROOT / "data/branch_search_medium_v1/report.json", ROOT / "data/stage_search_medium_v1/report.json"]
    report["source_sha256"] = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    save(output / "report.json", report)
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(("task", "budget", "method", "seed", "makespan_ms", "exact_gap_percent", "first_exact_evaluation", "first_1pct_evaluation", "unique_candidates", "cross_device_edges"))
        for name, task in report["tasks"].items():
            for row in task["runs"]:
                best = row["best"] or {}
                writer.writerow((name, row["budget"], row["method"], row["seed"], best.get("makespan_ms"), row["exact_gap_percent"], row["first_exact_evaluation"], row["first_1pct_evaluation"], row["unique_candidates"], best.get("cross_device_edges")))
    if followup:
        with (output / "curve_summary.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(("task", "method", "seed", "source", "budget_used", "first_exact_evaluation", "first_1pct_evaluation", "end_gap_percent"))
            for name, task in report["tasks"].items():
                for row in task.get("followup_existing64", {}).get("curves", []):
                    writer.writerow((name, row["method"], row["seed"], row["source"], row["budget_used"], row["first_exact_evaluation"], row["first_1pct_evaluation"], row["exact_gap_percent"]))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", default="8,16")
    parser.add_argument("--seeds", default="0,7,23")
    parser.add_argument("--followup-existing64", action="store_true", help="separate exploratory curve follow-up; adds only same-domain Random")
    args = parser.parse_args()
    run(args.output, tuple(int(x) for x in args.budgets.split(",")), tuple(int(x) for x in args.seeds.split(",")), args.followup_existing64)
