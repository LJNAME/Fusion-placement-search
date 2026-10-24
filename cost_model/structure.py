"""Hardware-independent extraction of a placed fusion plan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from simulator.fusion_rules import FusionPatternRegistry, default_fusion_registry
from simulator.schema import OpGraph, Plan, TensorEdge, edge_bytes
from simulator.validate import validate_fusion_plan, validate_plan


@dataclass(frozen=True)
class GroupStructure:
    index: int
    group_id: str
    node_ids: Tuple[str, ...]
    device: int
    pattern_id: str
    backend: str
    implementation_status: str
    flops: float
    parameter_bytes: float
    input_bytes: float
    output_bytes: float
    workspace_bytes: float


@dataclass(frozen=True)
class DependencyStructure:
    index: int
    src_group: int
    dst_group: int
    src_node: str
    dst_node: str
    bytes: float
    cross_device: bool


@dataclass(frozen=True)
class PlanStructure:
    graph_name: str
    n_devs: int
    groups: Tuple[GroupStructure, ...]
    dependencies: Tuple[DependencyStructure, ...]
    node_to_group: Tuple[Tuple[str, int], ...]

    @property
    def cross_device_bytes(self) -> float:
        return sum(dep.bytes for dep in self.dependencies if dep.cross_device)

    def group_index(self, node_id: str) -> int:
        for stored_node_id, group_index in self.node_to_group:
            if stored_node_id == node_id:
                return group_index
        raise KeyError(node_id)


def analyze_plan_structure(
    graph: OpGraph,
    plan: Plan,
    n_devs: int,
    *,
    strict_fusion: bool = True,
    registry: Optional[FusionPatternRegistry] = None,
) -> PlanStructure:
    """Extract only graph/plan facts; no hardware rate enters this function."""

    if strict_fusion:
        validate_fusion_plan(graph, plan, n_devs, registry=registry)
    else:
        validate_plan(graph, plan, n_devs)

    incoming: dict[str, list[TensorEdge]] = {node_id: [] for node_id in graph.nodes}
    outgoing: dict[str, list[TensorEdge]] = {node_id: [] for node_id in graph.nodes}
    edge_sizes: dict[int, float] = {}
    for edge in graph.edges:
        incoming[edge.dst].append(edge)
        outgoing[edge.src].append(edge)
        edge_sizes[id(edge)] = edge_bytes(edge, graph)

    node_to_group: dict[str, int] = {}
    groups = []
    rules = registry or default_fusion_registry()
    for index, group in enumerate(plan.groups):
        for node_id in group.node_ids:
            node_to_group[node_id] = index
        node_ids = tuple(group.node_ids)
        node_set = set(node_ids)
        input_bytes = sum(float(graph.nodes[node_id].param_bytes) for node_id in node_ids)
        input_bytes += sum(
            edge_sizes[id(edge)]
            for node_id in node_ids
            for edge in incoming[node_id]
            if edge.src not in node_set
        )
        output_nodes = {
            node_id
            for node_id in node_ids
            if not outgoing[node_id]
            or any(edge.dst not in node_set for edge in outgoing[node_id])
        }
        output_bytes = sum(
            float(graph.nodes[node_id].out_bytes) for node_id in output_nodes
        )
        pattern_id = str(group.extra.get("pattern_id", "legacy"))
        pattern = rules.get(pattern_id) if strict_fusion else None
        workspace_bytes = float(group.extra.get("workspace_bytes", 0.0))
        if workspace_bytes < 0.0:
            raise ValueError(f"fusion group {index} workspace_bytes must be non-negative")
        groups.append(
            GroupStructure(
                index=index,
                group_id=f"fg{index:04d}",
                node_ids=node_ids,
                device=int(group.device),
                pattern_id=pattern_id,
                backend=str(
                    group.extra.get(
                        "backend",
                        pattern.backend if pattern is not None else "unknown",
                    )
                ),
                implementation_status=str(
                    group.extra.get(
                        "implementation_status",
                        pattern.implementation_status
                        if pattern is not None
                        else "unknown",
                    )
                ),
                flops=sum(float(graph.nodes[node_id].flops) for node_id in node_ids),
                parameter_bytes=sum(
                    float(graph.nodes[node_id].param_bytes) for node_id in node_ids
                ),
                input_bytes=float(input_bytes),
                output_bytes=float(output_bytes),
                workspace_bytes=workspace_bytes,
            )
        )

    dependencies = []
    for edge in graph.edges:
        src_group = node_to_group[edge.src]
        dst_group = node_to_group[edge.dst]
        if src_group == dst_group:
            continue
        dependencies.append(
            DependencyStructure(
                index=len(dependencies),
                src_group=src_group,
                dst_group=dst_group,
                src_node=edge.src,
                dst_node=edge.dst,
                bytes=float(edge_bytes(edge, graph)),
                cross_device=(
                    plan.groups[src_group].device != plan.groups[dst_group].device
                ),
            )
        )

    return PlanStructure(
        graph_name=graph.name,
        n_devs=int(n_devs),
        groups=tuple(groups),
        dependencies=tuple(dependencies),
        node_to_group=tuple(sorted(node_to_group.items())),
    )


__all__ = [
    "DependencyStructure",
    "GroupStructure",
    "PlanStructure",
    "analyze_plan_structure",
]
