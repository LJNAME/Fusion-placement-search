"""Four Joint action spaces on unchanged 3B medium graphs, at one real-call budget."""
import argparse
from collections import defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cost_model.exact_search import count_exact_search_space
from cost_model.fusion_library import CalibratedFusionProfile, fit_corrections, load_calibration, load_library
from cost_model.hardware_profile import AnalyticHardwareProfile
from cost_model.legal_search import LegalFusionSearchSpace, branch_regions
from cost_model.search_evaluation import build_evaluation_graph
from simulator import make_reference_plan
from simulator.dataset import graph_to_dict, plan_to_dict
from simulator.hardware import load_hardware
from simulator.placement_artifacts import graph_fingerprint
from tools.evaluate_mcts_medium import SELECTIONS, extract_graph
from tools.evaluate_stage_search import HARDWARE, Ledger, run_mcts, save, walk

POLICIES = ("node", "contiguous_stages", "branch_aware", "hybrid")
TOL_MS = 1e-9


def parallelism(plan, estimate, unit_of):
    """Compute overlap from the same charged HTAE timeline; no second prediction."""
    events = defaultdict(list)
    for i, (group, timing) in enumerate(zip(plan.groups, estimate.details["groups"])):
        if timing["end"] > timing["start"]:
            tags = {unit_of[n] for n in group.node_ids if n in unit_of}
            events[timing["start"]].append((i, (group.device, tags)))
            events[timing["end"]].append((i, None))
    active, previous = {}, 0.
    compute_overlap = branch_overlap = 0.
    max_devices = max_branch_devices = 0
    for moment in sorted(events):
        devices = {d for d, _ in active.values()}
        distinct_branches = any(d0 != d1 and any(a != b for a in u0 for b in u1)
                                for (d0, u0), (d1, u1) in itertools.combinations(active.values(), 2))
        if len(devices) > 1:
            compute_overlap += moment - previous
        if distinct_branches:
            branch_overlap += moment - previous
        if moment > previous:
            max_devices = max(max_devices, len(devices))
            if any(u for _, u in active.values()):
                max_branch_devices = max(max_branch_devices, 1)
            if distinct_branches:
                max_branch_devices = max(max_branch_devices, len({d for d, u in active.values() if u}))
        for index, value in events[moment]:
            if value is None:
                active.pop(index, None)
            else:
                active[index] = value
        previous = moment
    return {"compute_overlap_us": compute_overlap * 1e6,
            "branch_overlap_us": branch_overlap * 1e6,
            "branch_overlap_fraction": branch_overlap / estimate.makespan if estimate.makespan else 0.,
            "max_compute_devices": max_devices, "max_parallel_branch_devices": max_branch_devices}


class ObservedLedger(Ledger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        regions = branch_regions(self.graph, self.graph.topological_order())
        self.unit_of = {n: f"r{r}:b{b}" for r, region in enumerate(regions)
                        for b, path in enumerate(region["branches"]) for n in path}

    def observe(self, plan, estimate):
        return parallelism(plan, estimate, self.unit_of)


def partition(plan):
    return tuple(sorted((g.extra["pattern_id"], tuple(sorted(g.node_ids))) for g in plan.groups))


def describe_partition(key):
    return [{"pattern_id": p, "nodes": list(ns)} for p, ns in key]


def coupling(catalog, node_order):
    """Offline factorial analysis over Exact labels, never exposed to search."""
    natives = [p for p in catalog if all(len(nodes) == 1 for _, nodes in p)]
    if len(natives) != 1:
        return {"status": "requires_one_native_partition", "pairs": []}
    native = catalog[natives[0]]
    pairs = []
    for fused_key, fused in catalog.items():
        if fused_key == natives[0]:
            continue
        shared = sorted(p for p in native.keys() & fused.keys() if native[p]["feasible"] and fused[p]["feasible"])
        pair = {"fusion_partition": describe_partition(fused_key),
                "native_legal_placements": len(native), "fused_legal_placements": len(fused),
                "shared_feasible_placements": len(shared),
                "legality_changes": native.keys() != fused.keys()}
        if shared:
            deltas = {p: fused[p]["makespan_ms"] - native[p]["makespan_ms"] for p in shared}
            lo, hi = min(deltas, key=deltas.get), max(deltas, key=deltas.get)
            def witness(p):
                return {"placement": dict(zip(node_order, p)), "native_ms": native[p]["makespan_ms"],
                        "fused_ms": fused[p]["makespan_ms"], "delta_ms": deltas[p]}
            mn = min(native[p]["makespan_ms"] for p in shared)
            mf = min(fused[p]["makespan_ms"] for p in shared)
            native_opts = {p for p in shared if native[p]["makespan_ms"] <= mn + TOL_MS}
            fused_opts = {p for p in shared if fused[p]["makespan_ms"] <= mf + TOL_MS}
            reversal = None
            for p, q in itertools.combinations(shared, 2):
                dn = native[p]["makespan_ms"] - native[q]["makespan_ms"]
                df = fused[p]["makespan_ms"] - fused[q]["makespan_ms"]
                if (dn < -TOL_MS and df > TOL_MS) or (dn > TOL_MS and df < -TOL_MS):
                    reversal = {"P": witness(p), "Q": witness(q)}
                    break
            pair.update(delta_min_ms=deltas[lo], delta_max_ms=deltas[hi],
                        interaction_range_ms=deltas[hi] - deltas[lo],
                        cost_interaction=deltas[hi] - deltas[lo] > TOL_MS,
                        fusion_preference_flip=deltas[lo] < -TOL_MS and deltas[hi] > TOL_MS,
                        best_placement_changes_on_shared_domain=native_opts.isdisjoint(fused_opts),
                        placement_ranking_reversal=reversal,
                        delta_witnesses={"min": witness(lo), "max": witness(hi)})
            pair["decision_coupling_observed"] = pair["fusion_preference_flip"] or pair["best_placement_changes_on_shared_domain"]
        pairs.append(pair)
    return {"status": "evaluated" if pairs else "no_fusion_alternative",
            "scope": "current cost model on common feasible placements; not GPU evidence or proof Joint beats Sequential",
            "decision_coupling_status": ("observed" if any(p.get("decision_coupling_observed") for p in pairs)
                                         else "not_observed" if any(p.get("shared_feasible_placements") for p in pairs)
                                         else "not_evaluable"),
            "tolerance_ms": TOL_MS, "pairs": pairs}


def exact(graph, profile, registry, capacity, policy, keep_catalog=False):
    space = LegalFusionSearchSpace(graph, profile.n_devs, registry, placement_policy=policy)
    size = count_exact_search_space(space)
    if size > 100_000:
        raise ValueError("medium Exact limit exceeded")
    ledger = ObservedLedger(graph, profile, registry, capacity)
    ledger.phase = "exact"
    catalog = defaultdict(dict)
    started = time.perf_counter()
    for plan in walk(space):
        row = ledger.evaluate(plan)
        if keep_catalog:
            devices = {n: g.device for g in plan.groups for n in g.node_ids}
            catalog[partition(plan)][tuple(devices[n] for n in space.order)] = row
    if len(ledger.cache) != size:
        raise AssertionError("Exact space has duplicate or missing plans")
    return {"space_size": size, "feasible_candidates": sum(r["feasible"] for r in ledger.trace),
            "elapsed_s": time.perf_counter() - started, "best": ledger.describe(ledger.best),
            "plan": plan_to_dict(ledger.best) if ledger.best else None}, catalog


def search(graph, profile, registry, capacity, policy, budget, seed, domain_size):
    ledger = ObservedLedger(graph, profile, registry, capacity, budget)
    started = time.perf_counter()
    for device in range(profile.n_devs):
        ledger.evaluate(make_reference_plan(graph, device))
    ledger.phase = policy
    _, stopped = run_mcts(ledger, graph, registry, policy=policy, seed=seed, iterations=200 * budget)
    used = len(ledger.cache)
    if used != min(domain_size, budget):
        raise AssertionError(f"{policy} did not spend the shared budget or exhaust its space: {used}")
    return {"policy": policy, "seed": seed, "budget": budget, "budget_used": used,
            "budget_remaining": budget - used, "unique_candidates": used, "scorer_requests": ledger.requests,
            "stop": "space_exhausted" if used == domain_size else stopped,
            "elapsed_s": time.perf_counter() - started, "best": ledger.describe(ledger.best),
            "plan": plan_to_dict(ledger.best) if ledger.best else None, "trace": ledger.trace}


def run(output, budget=64, seeds=(0, 7, 23)):
    if budget < 2:
        raise ValueError("budget must cover both shared initialization plans")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    library = load_library()
    observations, _ = load_calibration()
    base = AnalyticHardwareProfile(load_hardware(HARDWARE))
    corrections, _ = fit_corrections(library, base, observations, build_evaluation_graph)
    profile = CalibratedFusionProfile(base, library, corrections)
    registry = library.registry("full")
    full = build_evaluation_graph("model_3b")
    report = {"scope": "same three 3B medium templates; blocking forward HTAE; CPU only",
              "budget": budget, "seeds": list(seeds), "corrections": corrections,
              "budget_contract": "distinct actual model calls including both native initializations and infeasible candidates; Exact labels never enter search; exhausted spaces stop early",
              "action_contract": {"node": "independent node/fusion-group device choices",
                                  "contiguous_stages": "one contiguous stage per card in canonical topological order",
                                  "branch_aware": "co-locate each detected linear fork-to-join branch; other nodes independent",
                                  "hybrid": "branch units independent; contiguous stages on remaining trunk nodes; fusion must satisfy both"},
              "parallelism_contract": "simulated time with different branch units active on different devices; not merely devices used",
              "coupling_contract": "offline Exact counterfactuals on identical node placements, separating legality, numerical interaction, preference flips and best-placement changes",
              "capacity_contract": "same free/cap75 synthetic forward budgets as stage_search_medium_v1; not training",
              "tasks": {}}
    for name in SELECTIONS:
        graph = extract_graph(full, name)
        regions = branch_regions(graph, graph.topological_order())
        probe = ObservedLedger(graph, profile, registry, [d.max_mem for d in profile.hardware.devices])
        peak = max(probe.model.predict(graph, make_reference_plan(graph, 0), profile).peak_memory)
        for cap in ("free", "cap75"):
            capacity = [d.max_mem for d in profile.hardware.devices] if cap == "free" else [peak * .75] * profile.n_devs
            oracles = {}
            catalog = None
            for policy in POLICIES:
                oracles[policy], labels = exact(graph, profile, registry, capacity, policy, keep_catalog=policy == "node")
                if policy == "node":
                    catalog = labels
            # Analysis is offline; no cost labels or oracle-selected plans are passed to search.
            analysis = coupling(catalog, graph.topological_order())
            runs = [search(graph, profile, registry, capacity, p, budget, seed, oracles[p]["space_size"])
                    for p in POLICIES for seed in seeds]
            reference = oracles["node"]["best"]
            for row in runs:
                local = oracles[row["policy"]]["best"]
                for field, ref in (("exact_gap_percent", reference), ("in_domain_gap_percent", local)):
                    value = None if row["best"] is None or ref is None else (row["best"]["makespan_ms"] / ref["makespan_ms"] - 1) * 100
                    if value is not None and value < -1e-8:
                        raise AssertionError("search beat Exact")
                    row[field] = None if value is None else max(0., value)
            for policy, oracle in oracles.items():
                best = oracle["best"]
                oracle["restriction_gap_percent"] = None if reference is None or best is None else max(0., (best["makespan_ms"] / reference["makespan_ms"] - 1) * 100)
            report["tasks"][f"{name}:{cap}"] = {
                "graph": graph_to_dict(graph), "graph_fingerprint": graph_fingerprint(graph),
                "nodes": len(graph.nodes), "branch_regions": regions, "capacity_bytes": capacity,
                "oracles": oracles, "coupling": analysis, "runs": runs}
            print(json.dumps({"task": f"{name}:{cap}", "branch_regions": len(regions),
                              "spaces": {p: x["space_size"] for p, x in oracles.items()},
                              "median_gaps": {p: statistics.median([r["exact_gap_percent"] for r in runs
                                 if r["policy"] == p and r["exact_gap_percent"] is not None])
                                 if any(r["policy"] == p and r["exact_gap_percent"] is not None for r in runs) else None for p in POLICIES}}), flush=True)
            save(output / "report.json", report)
    files = [Path(__file__), HARDWARE, ROOT / "cost_model/legal_search.py", ROOT / "cost_model/exact_search.py",
             ROOT / "cost_model/htae.py", ROOT / "cost_model/search_evaluation.py", ROOT / "cost_model/hardware_profile.py",
             ROOT / "cost_model/fusion_library.py", ROOT / "tools/evaluate_stage_search.py", ROOT / "tools/evaluate_mcts_medium.py",
             ROOT / "data/fusion_library_lightop_v1/manifest.json", ROOT / "data/fusion_library_lightop_v1/calibration.json"]
    report["source_sha256"] = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    save(output / "report.json", report)
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(("task", "policy", "seed", "makespan_ms", "exact_gap_percent", "in_domain_gap_percent",
                         "unique_candidates", "budget_used", "budget_remaining", "cross_device_edges",
                         "branch_overlap_us", "branch_overlap_fraction", "max_parallel_branch_devices"))
        for name, task in report["tasks"].items():
            for row in task["runs"]:
                best = row["best"] or {}
                writer.writerow((name, row["policy"], row["seed"], best.get("makespan_ms"), row["exact_gap_percent"],
                                 row["in_domain_gap_percent"], row["unique_candidates"], row["budget_used"], row["budget_remaining"],
                                 best.get("cross_device_edges"), best.get("branch_overlap_us"), best.get("branch_overlap_fraction"),
                                 best.get("max_parallel_branch_devices")))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--seeds", default="0,7,23")
    args = parser.parse_args()
    run(args.output, args.budget, tuple(int(x) for x in args.seeds.split(",")))
