"""Exact-referenced stage-placement ablation on existing 3B medium subgraphs.

Heuristics share one cold-cache budget of distinct complete-plan evaluations,
including initialization and both Sequential stages. Exact is a separate oracle.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cost_model.exact_search import count_exact_search_space
from cost_model.fusion_library import CalibratedFusionProfile, fit_corrections, load_calibration, load_library
from cost_model.hardware_profile import AnalyticHardwareProfile
from cost_model.htae import HTAECostModel
from cost_model.legal_search import LegalFusionSearchSpace, legal_mcts_search, rollout_legal_plan
from cost_model.search_evaluation import _contract_partition, _plan_key, _singleton_registry, build_evaluation_graph
from simulator import make_reference_plan
from simulator.dataset import graph_to_dict, plan_to_dict
from simulator.hardware import load_hardware
from simulator.placement_artifacts import graph_fingerprint
from tools.evaluate_mcts_medium import SELECTIONS, extract_graph

HARDWARE = ROOT / "data/bw1000_2card/bw1000_2card.calibrated.json"
METHODS = ("random", "sequential", "joint_mcts", "stage_joint_mcts")


class BudgetReached(Exception):
    pass


class Ledger:
    """Own all real model calls and retain every feasible evaluated incumbent."""
    def __init__(self, graph, profile, registry, capacity, budget=None):
        self.graph, self.profile, self.registry = graph, profile, registry
        self.capacity, self.budget = tuple(capacity), budget
        self.limit = budget
        self.model = HTAECostModel(registry=registry, feature_comm_blocks_compute=True, overlap_factor=0.0)
        self.cache, self.trace = {}, []
        self.requests = 0
        self.best, self.best_s = None, float("inf")
        self.phase = "initialization"

    def observe(self, plan, estimate):
        """Optional diagnostics from the same charged model evaluation."""
        return {}

    def evaluate(self, plan):
        self.requests += 1
        key = _plan_key(plan)
        if key not in self.cache:
            if self.limit is not None and len(self.cache) >= self.limit:
                raise BudgetReached
            est = self.model.predict(self.graph, plan, self.profile)
            feasible = all(p <= c for p, c in zip(est.peak_memory, self.capacity))
            if feasible and est.makespan < self.best_s:
                self.best, self.best_s = copy.deepcopy(plan), est.makespan
            devices = {n: g.device for g in plan.groups for n in g.node_ids}
            row = {"evaluation": len(self.cache) + 1, "phase": self.phase,
                   "makespan_ms": est.makespan * 1e3, "feasible": feasible,
                   "peak_memory_bytes": list(est.peak_memory),
                   "cross_device_edges": sum(devices[e.src] != devices[e.dst] for e in self.graph.edges),
                   "cross_device_bytes": est.cross_device_bytes,
                   "fused_groups": sum(len(g.node_ids) > 1 for g in plan.groups),
                   "best_ms": None if self.best is None else self.best_s * 1e3}
            row.update(self.observe(plan, est))
            self.cache[key] = row
            self.trace.append(row)
        return self.cache[key]

    def describe(self, plan):
        if plan is None:
            return None
        # Reporting is cache-only: no uncharged evaluation after search.
        return dict(self.cache[_plan_key(plan)])


def walk(space):
    def visit(state):
        if space.is_terminal(state):
            yield space.to_plan(state, validate=False)
        else:
            for action in space.actions(state):
                yield from visit(space.apply(state, action))
    yield from visit(space.initial_state())


def oracle(graph, profile, registry, capacity, policy):
    space = LegalFusionSearchSpace(graph, profile.n_devs, registry, placement_policy=policy)
    count = count_exact_search_space(space)
    if count > 100_000:
        raise ValueError(f"medium Exact space too large: {count}")
    ledger = Ledger(graph, profile, registry, capacity)
    ledger.phase = "exact"
    started = time.perf_counter()
    for plan in walk(space):
        ledger.evaluate(plan)
    if len(ledger.cache) != count:
        raise AssertionError("duplicate or missing exact candidates")
    return {"placement_policy": policy, "space_size": count,
            "feasible_candidates": sum(x["feasible"] for x in ledger.trace),
            "elapsed_s": time.perf_counter() - started,
            "best": ledger.describe(ledger.best),
            "plan": None if ledger.best is None else plan_to_dict(ledger.best)}


def run_mcts(ledger, graph, registry, *, policy="node", seed=0, devices=(0, 1), expand=lambda p: p,
             proposal=False, iterations=12_800, tie_shared_params=False):
    best_proposal = [float("inf"), None]

    def evaluate(plan):
        original = expand(plan)
        row = ledger.evaluate(original)
        if row["makespan_ms"] < best_proposal[0]:
            best_proposal[:] = [row["makespan_ms"], copy.deepcopy(original)]
        feasible = proposal or row["feasible"]
        return SimpleNamespace(feasible=feasible,
                               latency=row["makespan_ms"] / 1e3 if feasible else None)

    stopped = "space_exhausted_or_iteration_limit"
    try:
        legal_mcts_search(graph, None, ledger.profile.n_devs, iterations=iterations, rollouts=3,
                          fusion_probability=.7, seed=seed, n_best=1, registry=registry,
                          allowed_devices=devices, evaluator=evaluate,
                          unfused_incumbent=True, normalize_q=True, skip_exhausted=True,
                          placement_policy=policy, tie_shared_params=tie_shared_params)
    except BudgetReached:
        stopped = "evaluation_budget"
    return best_proposal[1], stopped


def heuristic(graph, profile, registry, capacity, method, budget, seed, global_exact, stage_exact):
    ledger = Ledger(graph, profile, registry, capacity, budget)
    started = time.perf_counter()
    # Identical feasible incumbents for all four methods, charged to the budget.
    for device in range(profile.n_devs):
        ledger.evaluate(make_reference_plan(graph, device))
    stages = []
    if method == "random":
        ledger.phase = "random"
        space = LegalFusionSearchSpace(graph, profile.n_devs, registry)
        size = global_exact["space_size"]
        rng = random.Random(seed)
        draws = 0
        while len(ledger.cache) < min(budget, size) and draws < 1000 * budget:
            ledger.evaluate(rollout_legal_plan(space, space.initial_state(), rng, fusion_probability=.7, validate=False))
            draws += 1
        stages.append({"phase": "random", "draws": draws})
        domain_size = size
    elif method == "sequential":
        # Fusion proposal first, then frozen-partition placement. The cache and
        # budget remain shared; proposal candidates are not made memory-feasible.
        ledger.phase, ledger.limit = "fusion_proposal", budget // 2
        before = len(ledger.cache)
        template, stop = run_mcts(ledger, graph, registry, seed=seed, devices=(0,), proposal=True,
                                  iterations=200 * budget)
        stages.append({"phase": ledger.phase, "evaluations": len(ledger.cache) - before, "stop": stop})
        if template is None:
            raise AssertionError("fusion proposal did not evaluate any candidate")
        macro, expand = _contract_partition(graph, template, registry, profile.n_devs)
        ledger.phase, ledger.limit = "frozen_partition_placement", budget
        before = len(ledger.cache)
        _, stop = run_mcts(ledger, macro, _singleton_registry(), seed=seed, expand=expand,
                           iterations=200 * budget)
        stages.append({"phase": ledger.phase, "evaluations": len(ledger.cache) - before, "stop": stop})
        # A sequential domain can exhaust early after freezing a partition;
        # unused budget is reported rather than spent on oracle or random plans.
        domain_size = None
    else:
        policy = "contiguous_stages" if method == "stage_joint_mcts" else "node"
        ledger.phase = method
        _, stop = run_mcts(ledger, graph, registry, seed=seed, policy=policy, iterations=200 * budget)
        stages.append({"phase": method, "stop": stop})
        domain_size = (stage_exact if policy != "node" else global_exact)["space_size"]
    elapsed = time.perf_counter() - started
    best = ledger.describe(ledger.best)
    exact = global_exact["best"]
    scoped = stage_exact["best"] if method == "stage_joint_mcts" else exact
    if best is not None and exact is None:
        raise AssertionError("heuristic found a plan in an Exact-infeasible space")
    def gap(reference):
        if best is None or reference is None:
            return None
        value = (best["makespan_ms"] / reference["makespan_ms"] - 1) * 100
        if value < -1e-8:
            raise AssertionError("heuristic beat the corresponding Exact oracle")
        return max(0., value)
    evaluated = len(ledger.cache)
    if evaluated > budget:
        raise AssertionError("evaluation budget exceeded")
    if domain_size is not None and evaluated < min(budget, domain_size):
        raise AssertionError(f"{method} stopped before using its budget or exhausting its space")
    return {"method": method, "seed": seed, "budget": budget, "unique_candidates": evaluated,
            "feasible_evaluations": sum(x["feasible"] for x in ledger.trace),
            "budget_used": evaluated, "budget_remaining": budget - evaluated,
            "scorer_requests": ledger.requests, "elapsed_s": elapsed, "stages": stages,
            "best": best, "exact_gap_percent": gap(exact), "in_domain_gap_percent": gap(scoped),
            "best_plan": None if ledger.best is None else plan_to_dict(ledger.best), "trace": ledger.trace}


def save(path, data):
    path.write_bytes((json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def run(output, budget=64, seeds=(0, 7, 23)):
    if budget < 4:
        raise ValueError("budget must be at least four")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    library = load_library()
    observations, _ = load_calibration()
    base = AnalyticHardwareProfile(load_hardware(HARDWARE))
    corrections, _ = fit_corrections(library, base, observations, build_evaluation_graph)
    profile = CalibratedFusionProfile(base, library, corrections)
    registry = library.registry("full")
    full = build_evaluation_graph("model_3b")
    report = {"scope": "3B existing medium templates; forward makespan, blocking HTAE; no GPU",
              "budget": budget, "seeds": list(seeds),
              "budget_contract": "maximum distinct complete-plan model calls; initialization/infeasible/proposal calls count; cache hits free; Exact separate and invisible to search",
              "placement_contract": "canonical topological contiguous stages, each card at most once; both card orders; cuts cannot split fusion spans",
              "sequential": "fusion then placement, half-budget proposal cap; frozen partition; retain every actually feasible evaluated plan",
              "capacity_contract": "free=physical total; cap75=75% of unfused single-card forward peak (synthetic, not training memory)",
              "cost_contract": "same analytic+LightOP calibration as forward_joint_search_v1; no new fitting; layout/async/model errors unchanged",
              "corrections": corrections, "tasks": {}}
    for name in SELECTIONS:
        graph = extract_graph(full, name)
        probe = Ledger(graph, profile, registry, [d.max_mem for d in profile.hardware.devices])
        single_peak = max(probe.model.predict(graph, make_reference_plan(graph, 0), profile).peak_memory)
        for cap in ("free", "cap75"):
            capacity = ([d.max_mem for d in profile.hardware.devices] if cap == "free"
                        else [single_peak * .75] * profile.n_devs)
            exact = oracle(graph, profile, registry, capacity, "node")
            stage_exact = oracle(graph, profile, registry, capacity, "contiguous_stages")
            restriction_gap = None
            if exact["best"] and stage_exact["best"]:
                restriction_gap = (stage_exact["best"]["makespan_ms"] / exact["best"]["makespan_ms"] - 1) * 100
            rows = [heuristic(graph, profile, registry, capacity, method, budget, seed, exact, stage_exact)
                    for method in METHODS for seed in seeds]
            task = {"graph": graph_to_dict(graph), "graph_fingerprint": graph_fingerprint(graph),
                    "nodes": len(graph.nodes), "edges": len(graph.edges), "capacity_bytes": capacity,
                    "exact": exact, "stage_exact": stage_exact,
                    "restriction_gap_percent": restriction_gap, "runs": rows}
            report["tasks"][f"{name}:{cap}"] = task
            print(json.dumps({"task": f"{name}:{cap}", "node_space": exact["space_size"],
                              "stage_space": stage_exact["space_size"], "restriction_gap_percent": restriction_gap,
                              "median_gap_percent": {m: statistics.median([x["exact_gap_percent"] for x in rows
                                  if x["method"] == m and x["exact_gap_percent"] is not None])
                                  if any(x["method"] == m and x["exact_gap_percent"] is not None for x in rows) else None
                                  for m in METHODS}}), flush=True)
            save(output / "report.json", report)
    source_paths = [Path(__file__), HARDWARE, ROOT / "cost_model/legal_search.py", ROOT / "cost_model/exact_search.py",
                    ROOT / "cost_model/search_evaluation.py", ROOT / "cost_model/hardware_profile.py",
                    ROOT / "tools/evaluate_mcts_medium.py",
                    ROOT / "cost_model/htae.py", ROOT / "cost_model/fusion_library.py",
                    ROOT / "data/fusion_library_lightop_v1/manifest.json", ROOT / "data/fusion_library_lightop_v1/calibration.json"]
    report["source_sha256"] = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    save(output / "report.json", report)
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(("task", "method", "seed", "makespan_ms", "exact_gap_percent", "in_domain_gap_percent",
                         "unique_candidates", "budget_remaining", "cross_device_edges", "elapsed_s"))
        for name, task in report["tasks"].items():
            for label in ("exact", "stage_exact"):
                row, best = task[label], task[label]["best"] or {}
                global_gap = (task["restriction_gap_percent"] if label == "stage_exact" else 0) if best else None
                writer.writerow((name, label, "", best.get("makespan_ms"), global_gap,
                                 0 if best else None, row["space_size"], "", best.get("cross_device_edges"), row["elapsed_s"]))
            for row in task["runs"]:
                best = row["best"] or {}
                writer.writerow((name, row["method"], row["seed"], best.get("makespan_ms"), row["exact_gap_percent"],
                                 row["in_domain_gap_percent"], row["unique_candidates"], row["budget_remaining"],
                                 best.get("cross_device_edges"), row["elapsed_s"]))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--seeds", default="0,7,23")
    args = parser.parse_args()
    run(args.output, args.budget, tuple(int(x) for x in args.seeds.split(",")))
