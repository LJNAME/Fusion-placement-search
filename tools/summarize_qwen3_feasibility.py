"""Verify the equal-prior audit and summarize attribution, without predictions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cost_model.forward_feasibility import replay_plan
from cost_model.fusion_library import load_library
from cost_model.legal_search import LegalFusionSearchSpace
from cost_model.search_evaluation import _plan_key, build_evaluation_graph
from simulator import validate_fusion_plan
from simulator.dataset import plan_from_dict
from simulator.placement_artifacts import graph_fingerprint
from tools.audit_qwen3_feasibility import PARENT
from tools.evaluate_full_joint import assert_shared_colocated
from tools.evaluate_stage_search import save


def run(folder):
    folder = Path(folder)
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    config = json.loads((folder / "plan.json").read_text(encoding="utf-8"))
    if report["status"] != "complete" or report["scope"] != config:
        raise AssertionError("unfinished audit or changed configuration")
    for path, digest in config["source_sha256"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise AssertionError(f"changed audit input: {path}")
    graph, registry = build_evaluation_graph("model"), load_library().registry("full")
    witness = json.loads((folder / "witness.json").read_text(encoding="utf-8"))
    witness_id = witness["estimate"]["candidate_id"]
    parent = json.loads((PARENT / "qwen3_06b/report.json").read_text(encoding="utf-8"))["tasks"]["cap75"]["runs"]
    old = {(r["method"], r["seed"]): r for r in parent}
    seen, source_matches, evaluations, prefix_stats = set(), 0, 0, []
    for row in report["runs"]:
        identity = (row["guided"], row["method"], row["seed"])
        if identity in seen:
            raise AssertionError("duplicate run")
        seen.add(identity)
        trace = row["trace"]
        if not (len(trace) == row["budget_used"] == row["unique_candidates"] == config["budget"]):
            raise AssertionError("different actual scoring budgets")
        if len({e["candidate_id"] for e in trace}) != len(trace):
            raise AssertionError("duplicate charged candidate")
        if trace[2]["candidate_id"] != witness_id or trace[2]["phase"] != "shared_witness":
            raise AssertionError("unequal initial prior")
        best = None
        for i, entry in enumerate(trace, 1):
            feasible = all(p <= c for p, c in zip(entry["peak_memory_bytes"], config["capacity_bytes"]))
            if entry["evaluation"] != i or entry["feasible"] != feasible:
                raise AssertionError("budget/feasibility trace mismatch")
            if any(lo > peak + 1e-6 for lo, peak in zip(entry["lower_bound"]["peak_lower_bound_bytes"], entry["peak_memory_bytes"])):
                raise AssertionError("unsound bound recorded")
            if entry["tied_weight_violation"] or not entry["fusion_plan_valid"]:
                raise AssertionError("illegal scored candidate")
            if feasible and (best is None or entry["makespan_ms"] < best["makespan_ms"]):
                best = entry
            if entry["best_ms"] != (best["makespan_ms"] if best else None):
                raise AssertionError("lost feasible incumbent")
        if best != row["best"]:
            raise AssertionError("final incumbent mismatch")
        arm = "guided" if row["guided"] else "seed_only_original"
        label = f"{arm}_{row['method']}_seed{row['seed']}"
        plan = plan_from_dict(json.loads((folder / "plans" / f"{label}.json").read_text(encoding="utf-8")))
        validate_fusion_plan(graph, plan, 2, registry=registry, require_executable=True)
        assert_shared_colocated(graph, plan)
        space = LegalFusionSearchSpace(graph, 2, registry, placement_policy="hybrid" if row["method"].endswith("hybrid") else "node", tie_shared_params=True)
        list(replay_plan(space, plan))
        key = hashlib.sha256(json.dumps((graph_fingerprint(graph), _plan_key(plan)), separators=(",", ":")).encode()).hexdigest()
        if key != best["candidate_id"]:
            raise AssertionError("exported plan was not scored incumbent")
        if not row["guided"]:
            # Inserting the common witness takes one score from the old 64-call
            # run. Apart from that insertion, candidates/costs remain identical.
            actual = trace[:2] + trace[3:]
            expected = old[(row["method"], row["seed"])]["trace"][:len(actual)]
            fields = ("candidate_id", "makespan_ms", "peak_memory_bytes", "feasible")
            if [[e[k] for k in fields] for e in actual] != [[e[k] for k in fields] for e in expected]:
                raise AssertionError("original control changed beyond charged witness insertion")
            source_matches += 1
        if row["method"] == "joint_hybrid" and not row["guided"]:
            contexts = [e["selected_prefix"] for e in trace if e["selected_prefix"]]
            prefix_stats.append({"seed": row["seed"],
                "max_selected_depth": max(c["depth"] for c in contexts),
                "max_selected_assigned_nodes": max(c["assigned_nodes"] for c in contexts),
                "max_selected_bound_branch_units": max(c["branch_units_bound"] for c in contexts),
                "rollouts_from_proved_dead_tree_prefix": row["rollouts_from_proved_dead_tree_prefix"],
                "rejectable_completed_rollouts": row["lower_bound_rejectable_evaluations"],
                "first_certain_dead_prefix": row["first_certain_dead_prefix"]})
        evaluations += row["budget_used"]
    expected = {(g, m, s) for g in (False, True) for m in config["methods"] for s in config["seeds"]}
    if seen != expected:
        raise AssertionError("missing comparison run")
    space = LegalFusionSearchSpace(graph, 2, registry, placement_policy="hybrid", tie_shared_params=True)
    trunk = sum(graph.nodes[n].param_bytes for n in space.trunk_nodes)
    head = graph.nodes[space.order[-1]].out_bytes
    weights = {}
    for n, unit in space.branch_unit.items():
        weights[unit] = weights.get(unit, 0.) + graph.nodes[n].param_bytes
    if len(set(weights.values())) != 1:
        raise AssertionError("equal-unit interpretation no longer applies")
    per_unit = next(iter(weights.values()))
    max_on_owner = int((config["capacity_bytes"][0] - trunk - head) // per_unit)
    paired = {}
    for guided in (False, True):
        rows = [r for r in report["runs"] if r["guided"] == guided and r["method"].endswith("hybrid")]
        joint = {r["seed"]: r for r in rows if r["method"] == "joint_hybrid"}
        rand = {r["seed"]: r for r in rows if r["method"] == "random_hybrid"}
        paired[str(guided)] = {"joint_wins_over_same_domain_random": sum(joint[s]["best"]["makespan_ms"] < rand[s]["best"]["makespan_ms"] - 1e-9 for s in joint),
                              "seeds": len(joint)}
    findings = {"capacity_bytes": config["capacity_bytes"], "trunk_parameter_bytes": trunk,
        "terminal_output_bytes": head, "necessary_owner_peak_bytes": trunk + head,
        "headroom_for_branch_weights_bytes": config["capacity_bytes"][0] - trunk - head,
        "branch_units": len(weights), "parameter_bytes_per_unit": per_unit,
        "max_branch_units_on_trunk_owner": max_on_owner,
        "minimum_branch_units_to_offload": len(weights) - max_on_owner,
        "witness_representable_in_hybrid": True,
        "invalid_tied_weight_plans": 0,
        "original_hybrid_prefix_audit": prefix_stats,
        "paired_joint_vs_random": paired,
        "summary": report["summary"],
        "new_gpu_measurements": 0,
        "scope": "necessary static forward bounds and predicted makespan, not real training memory or GPU execution"}
    save(folder / "findings.json", findings)
    with (folder / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(("arm", "method", "runs", "feasible_runs", "median_ms", "median_feasible_after_initialization", "median_cross_device_edges"))
        for arm, methods in report["summary"].items():
            for method, stats in methods.items():
                writer.writerow((arm, method, *(stats[k] for k in ("runs", "feasible_runs", "median_ms", "median_feasible_after_initialization", "median_cross_device_edges"))))
    save(folder / "verification.json", {"completed_new_runs": len(seen), "actual_search_evaluations": evaluations,
        "all_initial_prior_ids": [witness_id], "common_prior_scored_in_budget": True,
        "original_control_prefixes_matching_parent": source_matches,
        "all_sources_match_frozen_hashes": True,
        "all_best_plans_valid_and_in_their_domains": True,
        "all_lower_bounds_below_measured_model_peaks": True,
        "additional_predictions_for_verification": 0,
        "separate_witness_validation_model_calls": 1,
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "artifact_sha256": {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(folder.rglob("*.json")) if p.name != "verification.json"}})
    print(json.dumps(findings, indent=2))
    return findings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    run(parser.parse_args().folder)
