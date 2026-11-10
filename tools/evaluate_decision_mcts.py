"""Forced-action diagnosis, small Exact equivalence, fixed-64 Qwen3 ablation."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cost_model.decision_diagnostics import ActionAudit, explain_actions
from cost_model.exact_search import count_exact_search_space
from cost_model.fusion_library import CalibratedFusionProfile, fit_corrections, load_calibration, load_library
from cost_model.hardware_profile import AnalyticHardwareProfile
from cost_model.legal_search import LegalFusionSearchSpace, _LegalMCTSNode, legal_mcts_search
from cost_model.measured_fusion_calls import MeasuredFusionCalls, sha
from cost_model.search_evaluation import _plan_key, build_evaluation_graph
from simulator import make_reference_plan, validate_fusion_plan
from simulator.dataset import graph_to_dict, plan_to_dict
from simulator.hardware import load_hardware
from simulator.placement_artifacts import graph_fingerprint
from tools.evaluate_full_joint import FullLedger, source_hashes
from tools.evaluate_mcts_medium import extract_graph, SELECTIONS
from tools.evaluate_stage_search import BudgetReached, HARDWARE, Ledger, save, walk

PARENT = ROOT / "data/fusion_refresh_20261110_v2/qwen3_06b"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def reference_diagnosis(graph, registry, device):
    space = LegalFusionSearchSpace(graph, 2, registry, placement_policy="hybrid", tie_shared_params=True)
    state, rows = space.initial_state(), []
    while not space.is_terminal(state):
        actions = space.actions(state)
        selected = next(a for a in actions if len(a.candidate.node_ids) == 1 and a.device == device)
        rows.append({**explain_actions(space, state, actions), "chosen": {
            "pattern_id": selected.candidate.pattern_id, "nodes": list(selected.candidate.node_ids), "device": device}})
        state = space.apply(state, selected)
    return {"path": f"all Native/GPU{device}", "state_rows": rows,
            "states": len(rows), "kinds": dict(Counter(r["kind"] for r in rows)),
            "forced_primary_causes": dict(Counter(r["primary_forced_cause"] for r in rows if r["legal_actions"] == 1)),
            "classification": "primary causes have fixed priority; all simultaneous constraints are retained in each row"}


def compact_covers(space):
    counts = {"tree_nodes": 0, "unary_tree_nodes": 0}
    def visit(node):
        counts["tree_nodes"] += 1
        if space.is_terminal(node.state):
            yield space.to_plan(node.state)
            return
        width = len(space.actions(node.state))
        counts["unary_tree_nodes"] += width == 1
        node.expand(space, forced_action_closure=True)
        for child in node.children:
            yield from visit(child)
    plans = list(visit(_LegalMCTSNode(space.forced_closure(space.initial_state()))))
    return plans, counts


def exact_validation(profile, registry):
    rows, evaluations = [], 0
    full = build_evaluation_graph("model_3b")
    for name in SELECTIONS:
        graph = extract_graph(full, name)
        for policy in ("node", "hybrid"):
            space = LegalFusionSearchSpace(graph, 2, registry, placement_policy=policy)
            count = count_exact_search_space(space)
            if count > 100_000:
                raise ValueError("small Exact guard exceeded")
            raw_widths = Counter()
            space.action_observer = lambda _space, _state, actions: raw_widths.update((len(actions),))
            raw = list(walk(space))
            space.action_observer = None
            compact, stats = compact_covers(space)
            original = {_plan_key(p): p for p in raw}
            compressed = {_plan_key(p): p for p in compact}
            if set(original) != set(compressed) or len(raw) != len(compact) or len(raw) != count:
                raise AssertionError("compression changed the exact plan space")
            if stats["unary_tree_nodes"]:
                raise AssertionError("unary node survived compression")
            for key in original:
                a, b = plan_to_dict(original[key]), plan_to_dict(compressed[key])
                a.pop("name", None); b.pop("name", None)
                if a != b:
                    raise AssertionError("compression changed an exported plan")
            capacity = [d.max_mem for d in profile.hardware.devices]
            ledgers = [Ledger(graph, profile, registry, capacity), Ledger(graph, profile, registry, capacity)]
            for ledger, plans in zip(ledgers, (raw, compact)):
                for p in plans:
                    ledger.evaluate(p)
                evaluations += len(ledger.cache)
            native = next(k for k in original if all(len(ids) == 1 and device == 0 for _, ids, device in k))
            limit = max(ledgers[0].cache[native]["peak_memory_bytes"]) * .75
            caps = {"free": capacity, "cap75": [limit] * 2}
            cases = {}
            for cap, limits in caps.items():
                feasible, optima, best_sets = [], [], []
                for ledger in ledgers:
                    valid = {k: r for k, r in ledger.cache.items() if all(p <= c for p, c in zip(r["peak_memory_bytes"], limits))}
                    best = min((r["makespan_ms"] for r in valid.values()), default=None)
                    feasible.append(set(valid)); optima.append(best)
                    best_sets.append({k for k, r in valid.items() if r["makespan_ms"] == best})
                if feasible[0] != feasible[1] or optima[0] != optima[1] or best_sets[0] != best_sets[1]:
                    raise AssertionError("compression changed feasibility or the Exact optimum")
                cases[cap] = {"feasible_plans": len(feasible[0]), "raw_optimum_ms": optima[0],
                              "compressed_optimum_ms": optima[1], "optimal_plan_sets_equal": True}
            rows.append({"subgraph": name, "policy": policy, "nodes": len(graph.nodes),
                         "raw_plans": len(raw), "compressed_plans": len(compact), "compressed_tree": stats,
                         "raw_tree_nodes": sum(raw_widths.values()) + len(raw),
                         "raw_unary_tree_nodes": raw_widths[1],
                         "complete_exported_plan_sets_equal": True, "cases": cases})
    return {"passed": True, "tasks": rows, "independent_small_graph_cost_evaluations": evaluations,
            "full_search_reads_exact_labels": False}


def run_one(graph, profile, registry, capacity, seed, closure, budget):
    started = time.perf_counter()
    ledger = FullLedger(graph, profile, registry, capacity, budget)
    for device in range(2):
        ledger.evaluate(make_reference_plan(graph, device))
    initial = ledger.best_s * 1e3 if ledger.best else None
    stats, audit = {}, ActionAudit()
    ledger.phase = "decision_joint" if closure else "action_joint"
    def evaluate(plan):
        ledger.search_context = {"tree_assigned_nodes": stats.get("selected_assigned_nodes", 0),
                                "tree_depth": stats.get("selected_tree_depth", 0),
                                "decision_depth": stats.get("selected_decision_depth", 0)}
        row = ledger.evaluate(plan)
        if len(ledger.cache) == budget:
            raise BudgetReached
        return SimpleNamespace(feasible=row["feasible"], latency=row["makespan_ms"] * 1e-3 if row["feasible"] else None)
    try:
        legal_mcts_search(graph, None, 2, iterations=200 * budget, rollouts=3, fusion_probability=.7,
            seed=seed, n_best=1, registry=registry, evaluator=evaluate, unfused_incumbent=True,
            normalize_q=True, skip_exhausted=True, placement_policy="hybrid", tie_shared_params=True,
            forced_action_closure=closure, action_observer=audit, search_stats=stats)
        stop = "iteration_limit_or_exhausted"
    except BudgetReached:
        stop = "actual_evaluation_budget"
    if len(ledger.cache) != budget:
        raise AssertionError("fixed budget was not consumed")
    best = ledger.describe(ledger.best)
    return {"method": ledger.phase, "seed": seed, "budget_used": len(ledger.cache), "stop": stop,
            "initial_best_ms": initial, "best": best, "elapsed_s": time.perf_counter() - started,
            "first_best_evaluation": next((r["evaluation"] for r in ledger.trace if best and r["best_ms"] == best["makespan_ms"]), None),
            "feasible_evaluations": sum(r["feasible"] for r in ledger.trace), "tree_stats": stats,
            "action_audit": audit.to_dict(), "trace": ledger.trace,
            "best_plan": plan_to_dict(ledger.best) if ledger.best else None}, ledger.best


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "runs").mkdir()
    library = load_library()
    base = AnalyticHardwareProfile(load_hardware(HARDWARE))
    corrections, _ = fit_corrections(library, base, load_calibration()[0], build_evaluation_graph)
    profile = MeasuredFusionCalls(CalibratedFusionProfile(base, library, corrections), library)
    registry = library.registry("full")
    graph = build_evaluation_graph("model")
    parent = read(PARENT / "report.json")
    if graph_fingerprint(graph) != parent["graph_fingerprint"]:
        raise ValueError("Qwen3 graph differs from the preceding experiment")
    hashes = source_hashes()
    for path in ("cost_model/decision_diagnostics.py", "cost_model/measured_fusion_calls.py", "tools/evaluate_decision_mcts.py"):
        hashes[path] = sha(ROOT / path)
    config = {"graph_fingerprint": graph_fingerprint(graph), "methods": ["action_joint", "decision_joint"],
        "budget": 64, "seeds": [0, 7, 23], "capacities": {cap: task["capacity_bytes"] for cap, task in parent["tasks"].items()},
        "objective": "unchanged mixed L1 + blocking HTAE forward makespan", "source_sha256": hashes,
        "only_search_change": "forced-action closure; uniform priors, Q/backup, rollout probability and constraints unchanged",
        "random_rng_contract": "random completions preserve previous single-option randrange(1) draws",
        "initialization": "same two Native single-card plans, both charged; no witness or feasibility guide",
        "parent_report": (PARENT / "report.json").relative_to(ROOT).as_posix(),
        "parent_report_sha256": sha(PARENT / "report.json"),
        "measurement_review_sha256": sha(ROOT / "data/fusion_timing_expansion_20261109_01_review/verification.json"),
        "new_gpu_measurements": 0, "diagnostics_scope": "all queried states incl. repeats; all states on two Native paths; small Exact enumerated separately"}
    save(output / "plan.json", config)
    for device in (0, 1):
        save(output / f"native_gpu{device}_action_diagnosis.json", reference_diagnosis(graph, registry, device))
    save(output / "exact_validation.json", exact_validation(profile, registry))
    report = {"config": config, "status": "running", "tasks": {}}
    checks = []
    for cap, capacity in config["capacities"].items():
        rows = []
        report["tasks"][cap] = {"runs": rows}
        for closure in (False, True):
            for seed in config["seeds"]:
                print(json.dumps({"phase": "start", "capacity": cap, "closure": closure, "seed": seed}), flush=True)
                row, best_plan = run_one(graph, profile, registry, capacity, seed, closure, config["budget"])
                if not closure:
                    old = next(r for r in parent["tasks"][cap]["runs"] if r["method"] == "joint_hybrid" and r["seed"] == seed)
                    fields = ("candidate_id", "makespan_ms", "feasible", "peak_memory_bytes", "cross_device_edges", "fused_patterns")
                    if any(any(a[k] != b[k] for k in fields) for a, b in zip(row["trace"], old["trace"])):
                        raise AssertionError("default path no longer reproduces the preceding 64 proposals/scores")
                    checks.append({"capacity": cap, "seed": seed, "preceding_64_proposals_and_scores_equal": True})
                if closure and row["tree_stats"]["unary_expansions"] != 0:
                    raise AssertionError("unary tree state survived forced closure")
                if best_plan:
                    validate_fusion_plan(graph, best_plan, 2, registry=registry, require_executable=True)
                save(output / "runs" / f"{cap}_{row['method']}_seed{seed}.json", row)
                rows.append(row)
                save(output / "report.json", report)
                print(json.dumps({"phase": "finished", "capacity": cap, "method": row["method"], "seed": seed,
                    "best_ms": row["best"]["makespan_ms"] if row["best"] else None,
                    "max_assigned": max(r.get("tree_assigned_nodes", 0) for r in row["trace"]),
                    "max_decision_depth": row["tree_stats"]["max_decision_depth"]}), flush=True)
    summary = []
    for cap, task in report["tasks"].items():
        for method in config["methods"]:
            rows = [r for r in task["runs"] if r["method"] == method]
            bests = [r["best"]["makespan_ms"] for r in rows if r["best"]]
            summary.append({"capacity": cap, "method": method, "feasible_runs": len(bests),
                "median_ms": statistics.median(bests) if bests else None,
                "median_search_s": statistics.median(r["elapsed_s"] for r in rows),
                "first_best": [r["first_best_evaluation"] for r in rows],
                "feasible_evaluations": [r["feasible_evaluations"] for r in rows],
                "max_scored_assigned_nodes": max(r.get("tree_assigned_nodes", 0) for row in rows for r in row["trace"]),
                "max_decision_depth": max(r["tree_stats"]["max_decision_depth"] for r in rows),
                "unary_expansions": [r["tree_stats"]["unary_expansions"] for r in rows]})
    report.update(status="complete", summary=summary)
    save(output / "report.json", report)
    save(output / "summary.json", summary)
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(summary)
    for path, expected in hashes.items():
        if sha(ROOT / path) != expected:
            raise ValueError("source changed while running: " + path)
    save(output / "verification.json", {"completed_runs": 12, "actual_full_search_evaluations": 768,
        "default_path_control": checks, "source_hashes_match": True, "compressed_unary_expansions": 0,
        "same_initialization_and_capacities": True, "exact_labels_not_used_by_full_search": True,
        "new_gpu_measurements": 0, "artifact_sha256": {p.relative_to(output).as_posix(): sha(p)
            for p in sorted(output.rglob('*.json'))}})
    print(json.dumps(summary, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
