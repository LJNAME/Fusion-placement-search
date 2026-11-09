"""Qwen3 cap75 feasibility audit; equal-prior fixed-budget CPU comparison."""
import argparse
from collections import Counter
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

from cost_model.forward_feasibility import ForwardResidentBound, replay_plan
from cost_model.fusion_library import CalibratedFusionProfile, fit_corrections, load_calibration, load_library
from cost_model.hardware_profile import AnalyticHardwareProfile
from cost_model.legal_search import (LegalChoice, LegalPartialState, LegalFusionSearchSpace,
    NoLegalCompletion, legal_mcts_search, rollout_legal_plan)
from cost_model.search_evaluation import _plan_key, build_evaluation_graph
from simulator import make_reference_plan, validate_fusion_plan
from simulator.dataset import plan_from_dict, plan_to_dict
from simulator.hardware import load_hardware
from simulator.placement_artifacts import graph_fingerprint
from tools.evaluate_full_joint import FullLedger, METHODS, assert_shared_colocated, source_hashes
from tools.evaluate_stage_search import BudgetReached, HARDWARE, save

PARENT = ROOT / "data/full_joint_models_v1"


class AuditLedger(FullLedger):
    def __init__(self, *args, space, **kwargs):
        super().__init__(*args, **kwargs)
        self.space = space
        self.bound = ForwardResidentBound(self.capacity)
        self.context = None
        self.rollouts = self.dead_leaf_rollouts = self.invalid_shared = 0
        self.first_certain_dead_prefix = None
        self.lookup = {(c.pattern_id, frozenset(c.node_ids)): c for c in space.candidates}

    def rollout_observer(self, space, state, plan):
        self.context = self.bound.explain(space, state)
        self.rollouts += 1
        self.dead_leaf_rollouts += bool(self.context["overflow_devices"])

    def observe(self, plan, estimate):
        validate_fusion_plan(self.graph, plan, self.profile.n_devs, registry=self.registry, require_executable=True)
        assert_shared_colocated(self.graph, plan)
        stats = super().observe(plan, estimate)
        terminal = LegalPartialState(choices=tuple(
            LegalChoice(self.lookup[(g.extra["pattern_id"], frozenset(g.node_ids))], g.device)
            for g in plan.groups), assigned=frozenset(self.graph.nodes), anchor_index=len(self.space.order))
        detail = self.bound.explain(self.space, terminal)
        if any(lo > peak + 1e-6 for lo, peak in zip(detail["peak_lower_bound_bytes"], estimate.peak_memory)):
            raise AssertionError("unsound forward lower bound")
        stats.update(lower_bound=detail, selected_prefix=self.context,
                     tied_weight_violation=False, fusion_plan_valid=True)
        if (self.phase not in ("initialization", "shared_witness") and detail["overflow_devices"]
                and self.first_certain_dead_prefix is None):
            for state in replay_plan(self.space, plan):
                bound = self.bound.explain(self.space, state)
                if bound["overflow_devices"]:
                    self.first_certain_dead_prefix = {**bound,
                        "last_choice": {"nodes": list(state.choices[-1].candidate.node_ids),
                                        "device": state.choices[-1].device},
                        "note": "all completions of this prefix exceed a proved memory lower bound"}
                    break
        return stats


def run_one(graph, profile, registry, capacity, witness, method, seed, guided, budget=64):
    policy = "hybrid" if method.endswith("hybrid") else "node"
    space = LegalFusionSearchSpace(graph, 2, registry, placement_policy=policy, tie_shared_params=True)
    ledger = AuditLedger(graph, profile, registry, capacity, budget, space=space)
    for d in range(2):
        ledger.evaluate(make_reference_plan(graph, d))
    ledger.phase = "shared_witness"
    ledger.evaluate(witness)
    initial = ledger.best_s * 1e3
    bound = ForwardResidentBound(capacity) if guided else None
    started = time.perf_counter()
    ledger.phase = "guided" if guided else "seed_only_original"
    draws = dead_rollouts = 0
    if method.startswith("joint"):
        def evaluate(plan):
            row = ledger.evaluate(plan)
            return SimpleNamespace(feasible=row["feasible"],
                                   latency=row["makespan_ms"] / 1e3 if row["feasible"] else None)
        try:
            legal_mcts_search(graph, None, 2, iterations=200 * budget, rollouts=3,
                fusion_probability=.7, seed=seed, n_best=1, registry=registry,
                evaluator=evaluate, unfused_incumbent=True, normalize_q=True,
                skip_exhausted=True, tie_shared_params=True, placement_policy=policy,
                partial_prune=bound, rollout_observer=ledger.rollout_observer)
        except BudgetReached:
            pass
    else:
        sampling = LegalFusionSearchSpace(graph, 2, registry, placement_policy=policy,
                                         tie_shared_params=True, partial_prune=bound)
        rng = random.Random(seed)
        while len(ledger.cache) < budget and draws < budget * 1000:
            draws += 1
            state = sampling.initial_state()
            try:
                plan = rollout_legal_plan(sampling, state, rng, fusion_probability=.7, validate=False)
            except NoLegalCompletion:
                dead_rollouts += 1
                continue
            ledger.rollout_observer(sampling, state, plan)
            ledger.evaluate(plan)
    if len(ledger.cache) != budget:
        raise AssertionError(f"incomplete scoring budget {method}/{seed}/{guided}: {len(ledger.cache)}")
    best = ledger.describe(ledger.best)
    return {"method": method, "seed": seed, "guided": guided, "budget_used": len(ledger.cache),
        "unique_candidates": len({r["candidate_id"] for r in ledger.trace}),
        "first_feasible_evaluation": next(r["evaluation"] for r in ledger.trace if r["feasible"]),
        "initial_best_ms": initial, "best": best,
        "first_final_best_evaluation": next(r["evaluation"] for r in ledger.trace if r["best_ms"] == best["makespan_ms"]),
        "feasible_evaluations": sum(r["feasible"] for r in ledger.trace),
        "feasible_after_initialization": sum(r["feasible"] for r in ledger.trace[3:]),
        "lower_bound_rejectable_evaluations": sum(bool(r["lower_bound"]["overflow_devices"]) for r in ledger.trace[3:]),
        "rollout_observations": ledger.rollouts, "rollouts_from_proved_dead_tree_prefix": ledger.dead_leaf_rollouts,
        "first_certain_dead_prefix": ledger.first_certain_dead_prefix,
        "invalid_shared_weight_plans": ledger.invalid_shared,
        "all_64_fusion_plans_valid": True,
        "pruning": bound.summary() if bound else None,
        "random_draws": draws, "dead_end_rollouts": dead_rollouts,
        "elapsed_s": time.perf_counter() - started, "trace": ledger.trace}, ledger.best


def run(output, budget=64, seeds=(0, 7, 23)):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    graph = build_evaluation_graph("model")
    parent = json.loads((PARENT / "qwen3_06b/report.json").read_text(encoding="utf-8"))
    diagnostic = json.loads((PARENT / "hybrid_feasibility_diagnostic.json").read_text(encoding="utf-8"))
    if graph_fingerprint(graph) != parent["graph_fingerprint"]:
        raise AssertionError("different graph")
    # Only search implementation may change; original hardware/cost inputs stay frozen.
    frozen = parent["scope"]["source_sha256"]
    for name, digest in frozen.items():
        if name.startswith(("cost_model/htae", "cost_model/structural", "cost_model/fusion_library",
                            "cost_model/hardware_profile", "data/")):
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
                raise AssertionError(f"changed original cost input: {name}")
    library = load_library()
    base = AnalyticHardwareProfile(load_hardware(HARDWARE))
    observations, _ = load_calibration()
    corrections, _ = fit_corrections(library, base, observations, build_evaluation_graph)
    profile, registry = CalibratedFusionProfile(base, library, corrections), library.registry("full")
    capacity = parent["tasks"]["cap75"]["capacity_bytes"]
    witness = plan_from_dict(diagnostic["witness_plan"])
    space = LegalFusionSearchSpace(graph, 2, registry, placement_policy="hybrid", tie_shared_params=True)
    path = list(replay_plan(space, witness))
    witness_ledger = AuditLedger(graph, profile, registry, capacity, space=space)
    witness_row = witness_ledger.evaluate(witness)
    if not witness_row["feasible"] or witness_row["makespan_ms"] != diagnostic["witness"]["makespan_ms"]:
        raise AssertionError("witness drift")
    save(output / "witness_plan.json", plan_to_dict(witness))
    dev = {n: g.device for g in witness.groups for n in g.node_ids}
    save(output / "witness.json", {"estimate": witness_row, "capacity_bytes": capacity,
        "device_assignment": dev, "fusion_groups": [list(g.node_ids) for g in witness.groups if len(g.node_ids) > 1],
        "legal_action_count": len(path) - 1, "graph_fingerprint": graph_fingerprint(graph),
        "lower_bound_along_path": [witness_ledger.bound.explain(space, s) for s in path]})
    hashes = source_hashes()
    for name in ("cost_model/forward_feasibility.py", "tools/audit_qwen3_feasibility.py"):
        hashes[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    scope = {"graph_fingerprint": graph_fingerprint(graph), "capacity_bytes": capacity,
        "budget": budget, "seeds": list(seeds), "methods": list(METHODS),
        "arms": ["seed_only_original", "guided"], "source_sha256": hashes,
        "shared_prior": "same saved all-native feasible witness for every method and arm, scored at evaluation 3",
        "preregistered_difference": "guided arm only adds sound partial lower-bound action filter; no cost/UCT/rollout probability changes",
        "budget_contract": "two Native initializations + shared witness + remaining distinct actual model calls; no free timing queries",
        "parent_unseeded_reference": str(PARENT.relative_to(ROOT)),
        "witness_generation": "reused fixed constructor, one extra CPU validation; never selects best of new probes",
        "objective": "existing forward predicted makespan, not training or GPU deployment"}
    save(output / "plan.json", scope)
    report = {"scope": scope, "status": "running", "runs": [], "summary": {}}
    (output / "runs").mkdir()
    (output / "plans").mkdir()
    for guided in (False, True):
        for method in METHODS:
            for seed in seeds:
                label = f"{'guided' if guided else 'seed_only_original'}_{method}_seed{seed}"
                print(json.dumps({"phase": "start", "run": label}), flush=True)
                row, best = run_one(graph, profile, registry, capacity, witness, method, seed, guided, budget)
                save(output / "runs" / f"{label}.json", row)
                save(output / "plans" / f"{label}.json", plan_to_dict(best))
                report["runs"].append(row)
                save(output / "report.json", report)
                print(json.dumps({"phase": "finished", "run": label,
                    "best_ms": row["best"]["makespan_ms"], "feasible": row["feasible_evaluations"],
                    "elapsed_s": row["elapsed_s"]}), flush=True)
    for guided in (False, True):
        arm = "guided" if guided else "seed_only_original"
        report["summary"][arm] = {}
        for method in METHODS:
            rows = [r for r in report["runs"] if r["guided"] == guided and r["method"] == method]
            report["summary"][arm][method] = {
                "runs": len(rows), "feasible_runs": sum(r["best"] is not None for r in rows),
                "median_ms": statistics.median(r["best"]["makespan_ms"] for r in rows),
                "median_feasible_after_initialization": statistics.median(r["feasible_after_initialization"] for r in rows),
                "median_cross_device_edges": statistics.median(r["best"]["cross_device_edges"] for r in rows)}
    report["status"] = "complete"
    save(output / "report.json", report)
    save(output / "summary.json", report["summary"])
    print(json.dumps(report["summary"], indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--seeds", default="0,7,23")
    args = parser.parse_args()
    run(args.output, args.budget, tuple(int(s) for s in args.seeds.split(",")))
