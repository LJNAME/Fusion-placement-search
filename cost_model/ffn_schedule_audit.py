"""Bounded CPU audit of FFN representation order and fixed scheduling policy."""
import argparse
import copy
import itertools
import json
from pathlib import Path

from cost_model import ffn_profile_search as ffn
from integration.lightop.silu_mul import adapter_registry
from simulator import FusionGroup, Plan


def topological_representations(graph, plan):
    """Enumerate group orders only; keep fusion, placement and costs unchanged."""
    owner = {n: i for i, g in enumerate(plan.groups) for n in g.node_ids}
    dependencies = {(owner[e.src], owner[e.dst]) for e in graph.edges
                    if owner[e.src] != owner[e.dst]}
    for order in itertools.permutations(range(len(plan.groups))):
        positions = {g: i for i, g in enumerate(order)}
        if all(positions[src] < positions[dst] for src, dst in dependencies):
            yield Plan(groups=[copy.deepcopy(plan.groups[i]) for i in order])


def normalized_trace(plan, estimate):
    """Compare physical traces by node identities, not representation group IDs."""
    names = {f"fg{i:04d}": "+".join(sorted(g.node_ids)) for i, g in enumerate(plan.groups)}
    result = copy.deepcopy(estimate.to_dict())
    result["critical_path"] = [names[g] for g in result["critical_path"]]
    details = result["details"]
    details["schedule_order"] = [names[g] for g in details["schedule_order"]]
    for group in details["groups"]:
        group["group_id"] = names[group["group_id"]]
    details["groups"].sort(key=lambda g: g["group_id"])
    for transfer in details["transfers"]:
        transfer.pop("dependency_index")
        for role in ("src_group", "dst_group"):
            transfer[role] = names[transfer[role]]
    details["transfers"].sort(key=lambda t: (t["src_group"], t["dst_group"], t["bytes"]))
    return result


def check_resources(graph, plan, estimate, profile):
    """Independently check dependency times and exclusive device/link intervals."""
    details = estimate.details
    groups = {g["group_id"]: g for g in details["groups"]}
    owner = {n: f"fg{i:04d}" for i, g in enumerate(plan.groups) for n in g.node_ids}
    for edge in graph.edges:
        src, dst = owner[edge.src], owner[edge.dst]
        if src != dst:
            assert groups[src]["end"] <= groups[dst]["start"]
    device_intervals = {}
    link_intervals = {}
    for g in groups.values():
        assert g["start"] >= g["dependency_ready"]
        assert g["start"] >= g["device_ready"]
        device_intervals.setdefault(g["device"], []).append((g["start"], g["end"]))
    for t in details["transfers"]:
        assert t["start"] >= groups[t["src_group"]]["end"]
        assert groups[t["dst_group"]]["start"] >= t["end"]
        link = profile.link_key(t["src_device"], t["dst_device"])
        link_intervals.setdefault(link, []).append((t["start"], t["end"]))
        if not profile.overlap_send:
            device_intervals.setdefault(t["src_device"], []).append((t["start"], t["end"]))
    for intervals in (*device_intervals.values(), *link_intervals.values()):
        positive = sorted((s, e) for s, e in intervals if e > s)
        assert all(left[1] <= right[0] for left, right in zip(positive, positive[1:]))


def build_report(seeds=(0, 1, 7)):
    result = ffn.build_report(seeds=seeds)
    config = result["configuration"]
    search, full, hardware, measured, _ = ffn.build_case(config)
    model = ffn.StructuralCostModel(registry=adapter_registry(), scheduling_policy=ffn.SCHEDULING_POLICY)
    legacy = json.loads((ffn.ROOT / "data/ffn_profile_search_v1/report.json").read_text(encoding="utf-8"))
    assert result["configuration"] == legacy["configuration"]
    assert result["source_inputs"] == legacy["source_inputs"]
    audits = []
    for scenario_result, old in zip(result["scenario_results"], legacy["scenario_results"]):
        scenario = scenario_result["assumptions"]
        assert scenario == old["assumptions"]
        assert scenario_result["compute_queries"].keys() == old["compute_queries"].keys()
        for key, query in scenario_result["compute_queries"].items():
            previous = old["compute_queries"][key]
            assert {k: v for k, v in query.items() if k != "duration_us"} == {
                k: v for k, v in previous.items() if k != "duration_us"}
            assert abs(query["duration_us"] - previous["duration_us"]) < 1e-9
        for query, previous in zip(scenario_result["communication_union"], old["communication_union"]):
            assert {k: v for k, v in query.items() if k != "duration_us"} == {
                k: v for k, v in previous.items() if k != "duration_us"}
            assert abs(query["duration_us"] - previous["duration_us"]) < 1e-9
        base = ffn.FFNProxyBase(hardware, measured, config, scenario)
        profile = ffn.NativeMeasuredFirstProfile(full, base, measured.store, measured.bindings)
        comparisons = []
        variants = 0
        for scored in scenario_result["plans"]:
            plan = Plan(groups=[FusionGroup(g["nodes"], g["device"], {"pattern_id": g["pattern"]})
                                for g in scored["scoring_groups"][1:-1]])
            expected = normalized_trace(ffn.with_boundaries(plan), model.predict(full, ffn.with_boundaries(plan), profile))
            count = 0
            for representation in topological_representations(search, plan):
                bounded = ffn.with_boundaries(representation)
                for reverse in (False, True):
                    graph = copy.deepcopy(full)
                    if reverse:
                        graph.nodes = dict(reversed(list(graph.nodes.items())))
                        graph.edges.reverse()
                    bound_profile = ffn.NativeMeasuredFirstProfile(graph, base, measured.store, measured.bindings)
                    estimate = model.predict(graph, bounded, bound_profile)
                    check_resources(graph, bounded, estimate, bound_profile)
                    assert normalized_trace(bounded, estimate) == expected
                    variants += 1
                count += 1
            comparisons.append({"plan_id": scored["plan_id"], "legal_topological_orders": count,
                                "trace_invariant": True})
        old_by_id = {p["plan_id"]: p for p in old["plans"]}
        changed = [p["plan_id"] for p in scenario_result["plans"]
                   if abs(p["cost_us"] - old_by_id[p["plan_id"]]["cost_us"]) > 1e-9]
        audits.append({"scenario": scenario["name"], "plans": comparisons,
            "representations_checked_including_graph_reversal": variants,
            "trace_and_resource_checks_passed": True,
            "cost_inputs_unchanged": True,
            "scores_changed_vs_legacy": len(changed), "changed_plan_ids": changed,
            "old_best_us": old["best"]["cost_us"], "new_best_us": scenario_result["best"]["cost_us"],
            "optimal_plan_changed": old["best"]["plan_id"] != scenario_result["best"]["plan_id"],
            "ranking_changed": [p["plan_id"] for p in old["plans"]] != [p["plan_id"] for p in scenario_result["plans"]]})
    result["scheduling_audit"] = {"semantic": "resource-constrained simulated makespan",
        "policy": ffn.SCHEDULING_POLICY, "legacy_policy": "plan_order_v1",
        "group_order_is_search_variable": False, "gap_filling": False,
        "legacy_report_preserved": "data/ffn_profile_search_v1/report.json",
        "per_scenario": audits}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps([{k: v for k, v in row.items() if k not in ("plans", "changed_plan_ids")}
                      for row in report["scheduling_audit"]["per_scenario"]], indent=2))


if __name__ == "__main__":
    main()
