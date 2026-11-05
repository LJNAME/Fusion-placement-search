"""Generic fused-and-placed graph export for downstream consumers.

Input is an existing PlacedOpGraph (search output). Output is a
framework-neutral group-level DAG:

- devices
- fusion groups: original node ids, device, pattern, backend,
  implementation status, executable flag
- data dependencies between groups, carrying tensor bytes
- an explicit communication op for every cross-device dependency
- per-device execution order

If a CostEstimate is supplied, predicted start/end times are attached; they
are predictions, not measurements. The artifact claims deployability only
when every group's pattern is executable; nothing here launches kernels or
targets a specific auto-parallel framework.
"""
from __future__ import annotations

from collections import defaultdict, deque
import copy
import json
from pathlib import Path
from typing import Any, Mapping, Optional

from cost_model.structure import analyze_plan_structure
from simulator.fusion_rules import FusionPatternRegistry, default_fusion_registry
from simulator.placement_artifacts import PlacedOpGraph, validate_placed_op_graph

FORMAT = "fused-placed-graph"
VERSION = 1


def _group_topological_order(groups, dependencies):
    indegree = {g["group_id"]: 0 for g in groups}
    successors = defaultdict(set)
    for dep in dependencies:
        if dep["dst_group"] not in successors[dep["src_group"]]:
            successors[dep["src_group"]].add(dep["dst_group"])
            indegree[dep["dst_group"]] += 1
    queue = deque(sorted(g for g, d in indegree.items() if d == 0))
    order = []
    while queue:
        current = queue.popleft()
        order.append(current)
        for nxt in sorted(successors[current]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    if len(order) != len(groups):
        raise ValueError("fusion-group graph contains a cycle")
    return order


def export_fused_placed_graph(placed: PlacedOpGraph, estimate=None,
                              registry: Optional[FusionPatternRegistry] = None) -> dict[str, Any]:
    rules = registry or default_fusion_registry()
    validate_placed_op_graph(placed, registry=rules)
    structure = analyze_plan_structure(placed.graph, placed.plan, placed.hardware.n_devs, registry=rules)
    timing = {}
    if estimate is not None:
        timing = {g["group_id"]: g for g in estimate.details["groups"]}
    groups = []
    for group in structure.groups:
        pattern = rules.get(group.pattern_id)
        row = {"group_id": group.group_id, "node_ids": list(group.node_ids), "device_id": group.device,
               "pattern_id": group.pattern_id, "backend": group.backend,
               "implementation_status": group.implementation_status, "executable": pattern.executable,
               "op_types": [placed.graph.nodes[n].op_type for n in group.node_ids],
               "flops": group.flops, "parameter_bytes": group.parameter_bytes,
               "input_bytes": group.input_bytes, "output_bytes": group.output_bytes,
               "workspace_bytes": group.workspace_bytes}
        if group.group_id in timing:
            t = timing[group.group_id]
            row["predicted"] = {"start_s": t["start"], "end_s": t["end"], "cost_source": t["source"]}
        groups.append(row)
    names = {g.index: g.group_id for g in structure.groups}
    dependencies, communication = [], []
    transfer_timing = {}
    if estimate is not None:
        transfer_timing = {t["dependency_index"]: t for t in estimate.details["transfers"]}
    for dep in structure.dependencies:
        row = {"dependency_id": f"dep{dep.index:05d}", "src_group": names[dep.src_group],
               "dst_group": names[dep.dst_group], "src_node": dep.src_node, "dst_node": dep.dst_node,
               "bytes": dep.bytes, "cross_device": dep.cross_device}
        if dep.cross_device:
            comm = {"comm_id": f"comm{dep.index:05d}", "dependency_id": row["dependency_id"],
                    "kind": "point_to_point_copy",
                    "src_device": structure.groups[dep.src_group].device,
                    "dst_device": structure.groups[dep.dst_group].device, "bytes": dep.bytes,
                    "after_group": row["src_group"], "before_group": row["dst_group"]}
            if dep.index in transfer_timing:
                t = transfer_timing[dep.index]
                comm["predicted"] = {"start_s": t["start"], "end_s": t["end"], "cost_source": t["source"]}
            row["comm_id"] = comm["comm_id"]
            communication.append(comm)
        dependencies.append(row)
    topological = _group_topological_order(groups, dependencies)
    if estimate is not None:
        order = list(estimate.details["schedule_order"])
        order_source = "cost_model_schedule:" + estimate.details["scheduling_policy"]
    else:
        order, order_source = topological, "group_topological_order"
    device_of = {g["group_id"]: g["device_id"] for g in groups}
    per_device = {str(d): [g for g in order if device_of[g] == d] for d in range(placed.hardware.n_devs)}
    devices = [{"device_id": i, "name": d.name, "max_mem_bytes": d.max_mem,
                "class": d.extra.get("class")} for i, d in enumerate(placed.hardware.devices)]
    peak = list(estimate.peak_memory) if estimate is not None else list(placed.predicted_peak_memory)
    capacity_ok = all(p <= d.max_mem for p, d in zip(peak, placed.hardware.devices)) if peak else None
    non_executable = sorted(g["group_id"] for g in groups if not g["executable"])
    return {"format": FORMAT, "version": VERSION,
        "source": {"graph_hash": placed.graph_hash, "hardware_hash": placed.hardware_hash,
                   "candidate_id": placed.candidate_id, "graph_name": placed.graph.name},
        "devices": devices, "groups": groups, "dependencies": dependencies, "communication": communication,
        "execution_order": {"source": order_source, "per_device": per_device},
        "predicted": {"makespan_s": estimate.makespan if estimate is not None else placed.predicted_makespan,
                      "peak_memory_bytes": peak, "memory_within_capacity": capacity_ok,
                      "is_prediction_not_measurement": True},
        "deployable": not non_executable and capacity_ok is not False,
        "non_executable_groups": non_executable,
        "summary": {"nodes": len(placed.graph.nodes), "groups": len(groups),
                    "fused_groups": sum(len(g["node_ids"]) > 1 for g in groups),
                    "cross_device_dependencies": len(communication),
                    "groups_per_device": {k: len(v) for k, v in per_device.items()}}}


def validate_fused_placed_graph(artifact: Mapping[str, Any], graph=None) -> None:
    """Self-consistency checks a downstream consumer can rely on."""
    if artifact.get("format") != FORMAT or artifact.get("version") != VERSION:
        raise ValueError("unsupported fused-placed-graph format/version")
    groups = {g["group_id"]: g for g in artifact["groups"]}
    covered = [n for g in artifact["groups"] for n in g["node_ids"]]
    if len(covered) != len(set(covered)):
        raise ValueError("a node appears in more than one fusion group")
    if graph is not None and set(covered) != set(graph.nodes):
        raise ValueError("fusion groups do not cover exactly the source graph nodes")
    device_ids = {d["device_id"] for d in artifact["devices"]}
    if any(g["device_id"] not in device_ids for g in groups.values()):
        raise ValueError("group placed on an undeclared device")
    comms = {c["comm_id"]: c for c in artifact["communication"]}
    for dep in artifact["dependencies"]:
        src, dst = groups[dep["src_group"]], groups[dep["dst_group"]]
        crosses = src["device_id"] != dst["device_id"]
        if crosses != dep["cross_device"] or crosses != ("comm_id" in dep):
            raise ValueError("cross-device flag or communication op mismatch: " + dep["dependency_id"])
        if crosses:
            comm = comms[dep["comm_id"]]
            if (comm["src_device"], comm["dst_device"], comm["bytes"]) != (src["device_id"], dst["device_id"], dep["bytes"]):
                raise ValueError("communication op endpoints differ from dependency")
    _group_topological_order(artifact["groups"], artifact["dependencies"])
    position = {}
    for device, order in artifact["execution_order"]["per_device"].items():
        if any(groups[g]["device_id"] != int(device) for g in order):
            raise ValueError("execution order lists a group on the wrong device")
        position.update({g: (int(device), i) for i, g in enumerate(order)})
    if set(position) != set(groups):
        raise ValueError("execution order does not cover every group exactly once")
    for dep in artifact["dependencies"]:
        a, b = position[dep["src_group"]], position[dep["dst_group"]]
        if a[0] == b[0] and a[1] >= b[1]:
            raise ValueError("same-device execution order violates a dependency")
    expected_deployable = (not any(not g["executable"] for g in groups.values())
                           and artifact["predicted"]["memory_within_capacity"] is not False)
    if artifact["deployable"] != expected_deployable:
        raise ValueError("deployable flag inconsistent with groups and memory")


def save_fused_placed_graph(path, artifact) -> None:
    validate_fused_placed_graph(artifact)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main(argv=None):
    import argparse
    from simulator.placement_artifacts import load_placed_op_graph
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--placed", type=Path, required=True, help="PlacedOpGraph JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--with-schedule", action="store_true",
                        help="attach predicted timing from the analytic StructuralCostModel")
    args = parser.parse_args(argv)
    placed = load_placed_op_graph(args.placed)
    estimate = None
    if args.with_schedule:
        from cost_model.hardware_profile import AnalyticHardwareProfile
        from cost_model.structural import StructuralCostModel
        estimate = StructuralCostModel().predict(placed.graph, placed.plan, AnalyticHardwareProfile(placed.hardware))
    artifact = export_fused_placed_graph(placed, estimate)
    validate_fused_placed_graph(artifact, placed.graph)
    save_fused_placed_graph(args.output, artifact)
    print(json.dumps(artifact["summary"] | {"deployable": artifact["deployable"]}))


if __name__ == "__main__":
    main()
