"""Hardware-parameterized scheduling over a hardware-independent plan structure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from simulator.fusion_rules import FusionPatternRegistry
from simulator.hardware import HardwareSpec
from simulator.schema import OpGraph, Plan

from .estimate import CostEstimate
from .hardware_profile import HardwareCostProfile, as_hardware_profile
from .structure import PlanStructure, analyze_plan_structure


@dataclass(frozen=True)
class _ScheduledTransfer:
    dependency_index: int
    src_group: int
    dst_group: int
    src_device: int
    dst_device: int
    bytes: float
    start: float
    end: float
    duration: float
    confidence: float
    source: str
    ood_reasons: tuple[str, ...]
    cause_group: int


@dataclass(frozen=True)
class PreparedPlan:
    """Validated graph/plan structure, bound to unchanged inputs for reuse."""

    graph: OpGraph
    plan: Plan
    structure: PlanStructure
    incoming: tuple[tuple[int, ...], ...]
    outgoing: tuple[tuple[int, ...], ...]


def _peak_memory(
    graph: OpGraph,
    plan: Plan,
    structure: PlanStructure,
    starts: list[float],
    ends: list[float],
    makespan: float,
    transfers: Dict[int, _ScheduledTransfer],
) -> tuple[float, ...]:
    node_to_group = dict(structure.node_to_group)
    events: dict[int, list[tuple[float, float]]] = {
        device: [] for device in range(structure.n_devs)
    }

    for group in structure.groups:
        device = group.device
        if group.parameter_bytes > 0.0:
            events[device].append((0.0, group.parameter_bytes))
            events[device].append((makespan, -group.parameter_bytes))
        if group.workspace_bytes > 0.0:
            events[device].append((starts[group.index], group.workspace_bytes))
            events[device].append((ends[group.index], -group.workspace_bytes))

    dependency_by_edge: dict[tuple[str, str], list] = {}
    outgoing_by_node: dict[str, list] = {node_id: [] for node_id in graph.nodes}
    for dep in structure.dependencies:
        dependency_by_edge.setdefault((dep.src_node, dep.dst_node), []).append(dep)
    for edge in graph.edges:
        outgoing_by_node[edge.src].append(edge)
    dependency_cursor: dict[tuple[str, str], int] = {}
    for node_id, node in graph.nodes.items():
        if node.out_bytes <= 0.0:
            continue
        src_group = node_to_group[node_id]
        src_device = structure.groups[src_group].device
        consumers = outgoing_by_node[node_id]
        if not consumers:
            events[src_device].append((ends[src_group], float(node.out_bytes)))
            events[src_device].append((makespan, -float(node.out_bytes)))
            continue
        external = [
            edge
            for edge in consumers
            if node_to_group[edge.dst] != src_group
        ]
        if not external:
            continue
        release_times = []
        for edge in external:
            key = (edge.src, edge.dst)
            cursor = dependency_cursor.get(key, 0)
            dep = dependency_by_edge[key][cursor]
            dependency_cursor[key] = cursor + 1
            if dep.cross_device:
                release_times.append(transfers[dep.index].end)
            else:
                release_times.append(ends[dep.dst_group])
        release = max(release_times, default=makespan)
        events[src_device].append((ends[src_group], float(node.out_bytes)))
        events[src_device].append((release, -float(node.out_bytes)))

    for transfer in transfers.values():
        events[transfer.dst_device].append((transfer.end, transfer.bytes))
        events[transfer.dst_device].append((ends[transfer.dst_group], -transfer.bytes))

    peaks = []
    for device in range(structure.n_devs):
        current = 0.0
        peak = 0.0
        for _time, delta in sorted(events[device], key=lambda item: (item[0], -item[1])):
            current += delta
            peak = max(peak, current)
        peaks.append(float(peak))
    return tuple(peaks)


class StructuralCostModel:
    """M3a reference model: common structure plus a replaceable hardware profile."""

    def __init__(
        self,
        *,
        strict_fusion: bool = True,
        registry: Optional[FusionPatternRegistry] = None,
        name: str = "structural-v1",
    ) -> None:
        self.strict_fusion = bool(strict_fusion)
        self.registry = registry
        self.name = name

    def predict(
        self,
        graph: OpGraph,
        plan: Plan,
        hardware: HardwareSpec | HardwareCostProfile,
    ) -> CostEstimate:
        profile = as_hardware_profile(hardware)
        return self.predict_prepared(self.prepare(graph, plan, profile.n_devs), profile)

    def prepare(self, graph: OpGraph, plan: Plan, n_devs: int) -> PreparedPlan:
        """Extract static structure once; callers must keep graph/plan unchanged."""

        structure = analyze_plan_structure(
            graph,
            plan,
            n_devs,
            strict_fusion=self.strict_fusion,
            registry=self.registry,
        )

        incoming: list[list[int]] = [[] for _ in structure.groups]
        outgoing: list[list[int]] = [[] for _ in structure.groups]
        for dep in structure.dependencies:
            incoming[dep.dst_group].append(dep.index)
            outgoing[dep.src_group].append(dep.index)

        return PreparedPlan(
            graph, plan, structure,
            tuple(tuple(items) for items in incoming),
            tuple(tuple(items) for items in outgoing),
        )

    def predict_prepared(
        self, prepared: PreparedPlan, hardware: HardwareSpec | HardwareCostProfile
    ) -> CostEstimate:
        """Run the same scheduler with new costs, including memory and critical paths."""

        profile = as_hardware_profile(hardware)
        graph, plan, structure = prepared.graph, prepared.plan, prepared.structure
        if profile.n_devs != structure.n_devs:
            raise ValueError("prepared plan and profile device counts differ")
        incoming, outgoing = prepared.incoming, prepared.outgoing

        starts = [0.0] * len(structure.groups)
        ends = [0.0] * len(structure.groups)
        causes: list[Optional[int]] = [None] * len(structure.groups)
        device_free = [0.0] * profile.n_devs
        device_last: list[Optional[int]] = [None] * profile.n_devs
        device_busy = [0.0] * profile.n_devs
        link_free: dict[tuple[int, int], float] = {}
        link_last_group: dict[tuple[int, int], int] = {}
        transfers: Dict[int, _ScheduledTransfer] = {}
        active_sends: list[list[tuple[float, float]]] = [
            [] for _ in range(profile.n_devs)
        ]
        group_costs = []
        all_confidence = []
        ood_reasons: set[str] = set()

        for group in structure.groups:
            candidates: list[tuple[float, int, Optional[int]]] = [
                (device_free[group.device], 0, device_last[group.device])
            ]
            for dep_index in incoming[group.index]:
                dep = structure.dependencies[dep_index]
                if dep.cross_device:
                    transfer = transfers[dep_index]
                    candidates.append((transfer.end, 2, transfer.cause_group))
                else:
                    candidates.append((ends[dep.src_group], 1, dep.src_group))
            ready_time, _priority, cause = max(
                candidates,
                key=lambda item: (item[0], item[1], -1 if item[2] is None else item[2]),
            )
            starts[group.index] = ready_time
            causes[group.index] = cause

            cost = profile.estimate_group(graph, group)
            group_costs.append(cost)
            all_confidence.append(cost.confidence)
            ood_reasons.update(cost.ood_reasons)
            overlap_count = sum(
                1
                for transfer_start, transfer_end in active_sends[group.device]
                if transfer_start <= ready_time < transfer_end
            )
            duration = cost.duration * (
                1.0 + profile.interference * overlap_count
            )
            ends[group.index] = ready_time + duration
            device_busy[group.device] += duration
            device_free[group.device] = ends[group.index]
            device_last[group.device] = group.index

            for dep_index in outgoing[group.index]:
                dep = structure.dependencies[dep_index]
                if not dep.cross_device:
                    continue
                dst_device = structure.groups[dep.dst_group].device
                transfer_cost = profile.estimate_transfer(
                    group.device,
                    dst_device,
                    dep.bytes,
                )
                all_confidence.append(transfer_cost.confidence)
                ood_reasons.update(transfer_cost.ood_reasons)
                link = profile.link_key(group.device, dst_device)
                previous_link_end = link_free.get(link, 0.0)
                start = max(ends[group.index], previous_link_end)
                cause_group = group.index
                if previous_link_end > ends[group.index]:
                    cause_group = link_last_group[link]
                end = start + transfer_cost.duration
                transfer = _ScheduledTransfer(
                    dependency_index=dep.index,
                    src_group=group.index,
                    dst_group=dep.dst_group,
                    src_device=group.device,
                    dst_device=dst_device,
                    bytes=dep.bytes,
                    start=start,
                    end=end,
                    duration=transfer_cost.duration,
                    confidence=transfer_cost.confidence,
                    source=transfer_cost.source,
                    ood_reasons=transfer_cost.ood_reasons,
                    cause_group=cause_group,
                )
                transfers[dep.index] = transfer
                active_sends[group.device].append((start, end))
                link_free[link] = end
                link_last_group[link] = cause_group
                if not profile.overlap_send:
                    device_free[group.device] = max(device_free[group.device], end)

        makespan = max(ends, default=0.0)
        peak_memory = _peak_memory(
            graph,
            plan,
            structure,
            starts,
            ends,
            makespan,
            transfers,
        )
        overflow_devices = tuple(
            device
            for device, peak in enumerate(peak_memory)
            if peak > profile.max_memory(device)
        )
        for device in overflow_devices:
            ood_reasons.add(f"device:{device}:memory_overflow")

        critical_indices = []
        current: Optional[int] = (
            max(range(len(ends)), key=lambda index: (ends[index], index))
            if ends
            else None
        )
        seen = set()
        while current is not None and current not in seen:
            seen.add(current)
            critical_indices.append(current)
            current = causes[current]
        critical_indices.reverse()

        communication_time = sum(item.duration for item in transfers.values())
        computation_time = sum(item.duration for item in group_costs)
        confidence = min(all_confidence, default=1.0)
        reasons = tuple(sorted(ood_reasons))
        return CostEstimate(
            makespan=float(makespan),
            peak_memory=peak_memory,
            critical_path=tuple(
                structure.groups[index].group_id for index in critical_indices
            ),
            communication_time=float(communication_time),
            computation_time=float(computation_time),
            per_device_time=tuple(float(value) for value in device_busy),
            cross_device_bytes=float(structure.cross_device_bytes),
            confidence=float(confidence),
            out_of_distribution=bool(reasons),
            ood_reasons=reasons,
            model_name=self.name,
            details={
                "structure_version": "plan-structure-v1",
                "overflow_devices": list(overflow_devices),
                "groups": [
                    {
                        "group_id": group.group_id,
                        "device": group.device,
                        "pattern_id": group.pattern_id,
                        "start": starts[group.index],
                        "end": ends[group.index],
                        "duration": ends[group.index] - starts[group.index],
                        "compute_time": group_costs[group.index].compute_time,
                        "memory_time": group_costs[group.index].memory_time,
                        "launch_time": group_costs[group.index].launch_time,
                        "source": group_costs[group.index].source,
                    }
                    for group in structure.groups
                ],
                "transfers": [
                    {
                        "dependency_index": item.dependency_index,
                        "src_group": structure.groups[item.src_group].group_id,
                        "dst_group": structure.groups[item.dst_group].group_id,
                        "src_device": item.src_device,
                        "dst_device": item.dst_device,
                        "bytes": item.bytes,
                        "start": item.start,
                        "end": item.end,
                        "duration": item.duration,
                        "source": item.source,
                    }
                    for item in sorted(
                        transfers.values(), key=lambda value: value.dependency_index
                    )
                ],
            },
        )


__all__ = ["PreparedPlan", "StructuralCostModel"]
