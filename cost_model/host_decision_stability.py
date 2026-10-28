"""Finite host-cost scenarios and epsilon decisions, not a stopping certificate."""

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import statistics

from calibration.calibrate_reference_host_cpu import audit_collection
from calibration.e2_manifest import content_id
from calibration.e2_records import load_jsonl_records
from calibration.gpu_holdout import ROOT, load_freeze
from cost_model.e2_srop_proxy import DatasetLatencyProfile, E2Workload, object_map_for_workload
from cost_model.reference_semantics import HostSubmissionProfile, ReferenceEagerCostModel
from cost_model.search_sensitivity import permute_plan_devices, plan_id
from execution.audit_gpu_holdout import write_json
from execution.diagnose_reference_state import audit as audit_state
from simulator.dataset import graph_from_dict, plan_from_dict
from simulator.hardware import hardware_from_dict


CPU = ROOT / "data/received_reference_host_cpu_v1/bw1000_reference_host_cpu_v1"
STATE = ROOT / "data/received_reference_state_v1/bw1000_reference_state_v1"
EPSILONS = (.001, .005, .01)
TIE_S = 1e-12


def aggregate_host(records):
    """Median per block, then per case/lifetime, then across strata; no GPU tail."""
    strata = defaultdict(lambda: defaultdict(list))
    for row in records:
        values = defaultdict(list)
        for sample in row["probe_samples"]:
            values["start_s"].append(sample["start_s"])
            values["finish_s"].append(sample["idle_sync_s"])
            for operation in sample["operations"]:
                item = row["catalog"][operation["key"]]
                if item["kind"] == "copy":
                    values["copy_submit_s"].append(operation["dispatch_step_s"])
                else:
                    values["op:" + item["operator"]].append(operation["dispatch_step_s"])
                    values["local:" + item["local_name"]].append(operation["dispatch_step_s"])
        for key, items in values.items():
            strata[row["case_id"], row.get("lifecycle", "")][key].append(statistics.median(items))
    combined = defaultdict(list)
    for entries in strata.values():
        for key, values in entries.items():
            combined[key].append(statistics.median(values))
    ops = {key[3:]: statistics.median(values) for key, values in combined.items() if key.startswith("op:")}
    ops.update({key: statistics.median(values) for key, values in combined.items() if key.startswith("local:")})
    return HostSubmissionProfile(start_s=statistics.median(combined["start_s"]),
        finish_s=statistics.median(combined["finish_s"]), copy_submit_s=statistics.median(combined["copy_submit_s"]),
        node_submit_s=statistics.median(v for key, v in ops.items() if not key.startswith("local:")),
        operator_submit_s=ops, calibrated=False, source="descriptive_training_host_aggregate")


def coordinates(host):
    return {"start_s": host.start_s, "finish_s": host.finish_s, "copy_submit_s": host.copy_submit_s,
            **{key: value for key, value in host.operator_submit_s.items() if key.startswith("local:")}}


def with_coordinates(host, values):
    return replace(host, start_s=values["start_s"], finish_s=values["finish_s"], copy_submit_s=values["copy_submit_s"],
        operator_submit_s={**host.operator_submit_s, **{k: v for k, v in values.items() if k.startswith("local:")}}, calibrated=False)


def scenarios(nominal, rounds):
    """Keep correlated round aggregates; OAT/corners are synthetic, zero is separate."""
    result = []
    def add(name, host, role="sensitivity"):
        result.append({"name": name, "role": role, "host": host})
    add("nominal", replace(nominal, calibrated=False))
    for repeat, host in enumerate(rounds):
        add(f"round_{repeat}", host)
    points = [coordinates(x) for x in [nominal, *rounds]]
    bounds = {key: [min(point[key] for point in points), max(point[key] for point in points)] for key in points[0]}
    center = coordinates(nominal)
    for index, label in enumerate(("low", "high")):
        add("all_" + label, with_coordinates(nominal, {k: v[index] for k, v in bounds.items()}))
        for key in bounds:
            if bounds[key][index] != center[key]:
                add(f"oat_{key}_{label}", with_coordinates(nominal, {**center, key: bounds[key][index]}))
    add("zero_host_ablation", HostSubmissionProfile(), role="ablation")
    return result, bounds


def argmin_set(scores):
    best = min(scores.values())
    return sorted(key for key, value in scores.items() if value <= best + TIE_S)


def decision_summary(scored, classes, roles, epsilons=EPSILONS):
    """All losses are predicted excess within this finite candidate/scenario set."""
    included = {key: value for key, value in scored.items() if roles[key] != "ablation"}
    class_scores = {name: {c: min(value[p] for p in value if classes[p] == c) for c in set(classes.values())}
                    for name, value in included.items()}
    nominal = class_scores["nominal"]
    reference = argmin_set(nominal)
    losses = {c: max(values[c] / min(values.values()) - 1 for values in class_scores.values()) for c in nominal}
    physical_losses = {p: max(values[p] / min(values.values()) - 1 for values in included.values()) for p in classes}
    scene_rows = []
    for name, values in class_scores.items():
        pairs = list(nominal)
        flips = sum((nominal[a] - nominal[b]) * (values[a] - values[b]) < 0
                    and abs(nominal[a] - nominal[b]) > TIE_S and abs(values[a] - values[b]) > TIE_S
                    for i, a in enumerate(pairs) for b in pairs[i + 1:])
        scene_rows.append({"scenario": name, "physical_top1_set": argmin_set(included[name]),
            "class_top1_set": argmin_set(values), "class_top1_set_changed": argmin_set(values) != reference,
            "strict_class_pairwise_flips": flips})
    best_loss = min(losses.values())
    return {"sensitivity_scenarios": len(included), "ablation_scenarios_excluded": len(scored) - len(included),
        "nominal_physical_top1_set": argmin_set(included["nominal"]), "nominal_class_top1_set": reference,
        "class_top1_change_scenarios": sum(row["class_top1_set_changed"] for row in scene_rows),
        "max_strict_class_pairwise_flips": max(row["strict_class_pairwise_flips"] for row in scene_rows),
        "nominal_choice_worst_predicted_excess_pct": max(losses[c] for c in reference) * 100,
        "finite_minimax_predicted_excess_pct": best_loss * 100,
        "finite_minimax_class_set": sorted(c for c, value in losses.items() if abs(value - best_loss) <= 1e-12),
        "epsilon_sets": [{"epsilon_pct": epsilon * 100,
            "class_ids_within_epsilon_in_every_scenario": sorted(c for c, loss in losses.items() if loss <= epsilon + 1e-12),
            "physical_ids_within_epsilon_in_every_scenario": sorted(p for p, loss in physical_losses.items() if loss <= epsilon + 1e-12)}
            for epsilon in epsilons], "scenarios": scene_rows,
        "separate_ablations": [{"scenario": name, "physical_top1_set": argmin_set(values),
            "class_top1_set": sorted({classes[p] for p in argmin_set(values)})}
            for name, values in scored.items() if roles[name] == "ablation"],
        "scope": "Finite predicted-loss sensitivity only: not true Search Regret, confidence coverage or a stopping certificate."}


def cohorts():
    sources = []
    for name, directory, state in (("cpu_70255", CPU, False), ("state_68186", STATE, True)):
        checked = audit_state(directory) if state else audit_collection(directory)
        records = load_jsonl_records(directory / "blocks.jsonl")
        records = ([r for r in records if r.get("phase") == "middle" and r.get("middle_kind") == "cpu_probe"]
                   if state else [r for r in records if r["mode"] == "cpu_probe"])
        nominal = aggregate_host(records) if state else HostSubmissionProfile(**json.loads((directory / "host_submission_profile.json").read_text()))
        points = [aggregate_host([r for r in records if r["repeat"] == repeat]) for repeat in range(3)]
        scenes, bounds = scenarios(nominal, points)
        sources.append({"name": name, "directory": directory, "audit": checked, "scenarios": scenes, "bounds": bounds,
            "families": {"qwen3-0.6b"} if state else {"qwen3-0.6b", "qwen2.5-3b"}})
    return sources


def evaluate():
    freeze, _ = load_freeze()
    hardware = hardware_from_dict(freeze["hardware"])
    results, source_info = [], []
    for cohort in cohorts():
        directory = cohort["directory"]
        source_info.append({"name": cohort["name"], "audit": cohort["audit"], "coordinate_bounds_s": cohort["bounds"],
            "scenarios": [{"name": s["name"], "role": s["role"], "host": asdict(s["host"])} for s in cohort["scenarios"]],
            "input_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in ("protocol.json", "blocks.jsonl", "environment.json")}})
        for workload in freeze["workloads"]:
            if workload["model_family"] not in cohort["families"]:
                continue
            graph = graph_from_dict(workload["graph"])
            mapping = object_map_for_workload(freeze["query_features"], E2Workload(workload["model_id"],
                "exposed_development", workload["model_id"], workload["model_family"], workload["sequence_length"]))
            profile = DatasetLatencyProfile(hardware, mapping, {k: v["nearest_s"] for k, v in freeze["zero_shot_predictions"].items()},
                {k: "frozen_predicted_service" for k in freeze["zero_shot_predictions"]})
            plans, classes = {}, {}
            for class_id, raw_plan in workload["candidate_plans"].items():
                for orientation in ((0, 1), (1, 0)):
                    plan = permute_plan_devices(plan_from_dict(raw_plan), orientation)
                    physical_id = plan_id(plan)
                    if plan_id(plan, ((0, 1), (1, 0))) != class_id:
                        raise ValueError("frozen candidate class identity differs")
                    plans[physical_id], classes[physical_id] = plan, class_id
            scores = {s["name"]: {p: ReferenceEagerCostModel(s["host"]).predict(graph, plan, profile).makespan
                                  for p, plan in plans.items()} for s in cohort["scenarios"]}
            summary = decision_summary(scores, classes, {s["name"]: s["role"] for s in cohort["scenarios"]})
            placements = {c: {"devices_used_count": len({g.device for g in plan.groups}), "groups": len(plan.groups),
                             "all_on_one_device": len({g.device for g in plan.groups}) == 1}
                          for p, plan in plans.items() for c in (classes[p],)}
            results.append({"cohort": cohort["name"], "model_id": workload["model_id"], "nodes": len(graph.nodes),
                "physical_candidates": len(plans), "exchange_classes": len(set(classes.values())),
                "candidate_pool_sha256": content_id("frozen-physical-pool", {p: classes[p] for p in sorted(plans)}),
                "summary": summary, "class_placements": placements, "physical_to_class": classes,
                "scores_s": scores})
            print(cohort["name"], workload["model_id"], "classes", len(placements), "scenarios", summary["sensitivity_scenarios"],
                "top1_changes", summary["class_top1_change_scenarios"], "worst_excess_pct", summary["nominal_choice_worst_predicted_excess_pct"], flush=True)
    return {"format": "fusion-placement-search.host-decision-stability", "freeze_id": freeze["freeze_id"],
        "primary_epsilon_pct": 1.0, "epsilon_sensitivity_pct": [.1, .5, 1.0], "tie_tolerance_s": TIE_S,
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
            ("cost_model/host_decision_stability.py", "cost_model/reference_semantics.py")},
        "summary": {"unique_exposed_workloads": len({r["model_id"] for r in results}), "cohort_workload_views": len(results),
            "independent_validation": False, "stop_rule_enabled": False,
            "rows": [{"cohort": r["cohort"], "model_id": r["model_id"], "physical_candidates": r["physical_candidates"],
                "exchange_classes": r["exchange_classes"], "scenarios": r["summary"]["sensitivity_scenarios"],
                "class_top1_changes": r["summary"]["class_top1_change_scenarios"],
                "max_strict_pairwise_flips": r["summary"]["max_strict_class_pairwise_flips"],
                "nominal_physical_ties": len(r["summary"]["nominal_physical_top1_set"]),
                "epsilon_class_counts": {str(e["epsilon_pct"]): len(e["class_ids_within_epsilon_in_every_scenario"])
                    for e in r["summary"]["epsilon_sets"]},
                "nominal_class_is_single_device": all(r["class_placements"][c]["all_on_one_device"]
                    for c in r["summary"]["nominal_class_top1_set"])} for r in results]},
        "cohorts": source_info, "workloads": results,
        "limitations": ["Already exposed development workloads/candidates; no new search, Oracle candidates or independent evaluation labels.",
            "Different hosts are separate scenario cohorts, never pooled into a calibrated profile or a same-host accuracy score.",
            "Round aggregates, synthetic OAT/corners and finite epsilon sets are not probability distributions or confidence intervals.",
            "Zero host is a separate ablation, excluded from observed-range stability and epsilon sets.",
            "GPU service and link predictions remain fixed, symmetric and operator-body coverage unverified. Physical ties cannot identify the truly faster device.",
            "No proof over the full parameter box, true regret bound, N-dependent net-benefit stopping, SROP improvement or calibration promotion."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/host_decision_stability_v1.json")
    args = parser.parse_args()
    write_json(args.output, evaluate())
    print("report:", args.output)


if __name__ == "__main__":
    main()
