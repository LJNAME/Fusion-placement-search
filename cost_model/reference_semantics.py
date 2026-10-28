"""Experimental cost model for the current singleton Torch reference executor.

Unlike producer-triggered DAG scheduling, copies are submitted at consumers.
One host submission lane and one current stream per device are modeled. Copy
barriers follow upstream PyTorch v2.9.0 Copy.cu:220-278; vendor parity still
requires validation. This module does not replace StructuralCostModel.
"""

from dataclasses import dataclass, field, replace
import math
from typing import Mapping

from cost_model.estimate import CostEstimate
from cost_model.hardware_profile import GroupCost, as_hardware_profile
from cost_model.structural import StructuralCostModel, _peak_memory, _ScheduledTransfer
from simulator.validate import validate_fusion_plan


@dataclass(frozen=True)
class HostSubmissionProfile:
    node_submit_s: float = 0.0
    copy_submit_s: float = 0.0
    finish_s: float = 0.0
    start_s: float = 0.0
    operator_submit_s: Mapping[str, float] = field(default_factory=dict)
    calibrated: bool = False
    source: str = "unmeasured_zero_host_cost"

    def __post_init__(self):
        values = [self.node_submit_s, self.copy_submit_s, self.finish_s, self.start_s, *self.operator_submit_s.values()]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("host submission costs must be finite and non-negative")

    def node_cost(self, operator, local_name=None):
        if local_name is not None and f"local:{local_name}" in self.operator_submit_s:
            return self.operator_submit_s[f"local:{local_name}"]
        return self.operator_submit_s.get(operator, self.node_submit_s)


def _payload_bytes(node):
    shape = node.extra.get("output_shape")
    dtype = node.extra.get("dtype")
    if shape is None or dtype not in {"bf16", "fp16", "fp32"}:
        raise ValueError(f"reference tensor metadata missing for {node.id}")
    if any(int(value) != value or value <= 0 for value in shape):
        raise ValueError(f"invalid reference tensor shape for {node.id}")
    return math.prod(shape) * {"bf16": 2, "fp16": 2, "fp32": 4}[dtype]


class ReferenceEagerCostModel:
    """Queue the actual plan order; do not assume that host submission is free.

    Group duration is a calibrated operator-block service window. Host work
    starts alongside that window, rather than being added again after it.
    Supplying unprofiled zeros is an explicit diagnostic, not calibration.
    """

    def __init__(self, host=None, *, service_matches_executor=False):
        self.host = host or HostSubmissionProfile()
        self.service_matches_executor = bool(service_matches_executor)

    def predict(self, graph, plan, hardware):
        profile = as_hardware_profile(hardware)
        if any(len(group.node_ids) != 1 for group in plan.groups):
            raise ValueError("reference semantics requires singleton groups")
        validate_fusion_plan(graph, plan, profile.n_devs, require_executable=True)
        supported = {"input", "matmul", "rmsnorm", "rope", "softmax", "silu", "mul", "add"}
        if any(node.op_type not in supported for node in graph.nodes.values()):
            raise ValueError("graph contains an operator unsupported by the reference executor")
        for node in graph.nodes.values():
            _payload_bytes(node)
        prepared = StructuralCostModel().prepare(graph, plan, profile.n_devs)
        # The executor copies entire predecessor tensors, including repeated
        # consumer edges. Edge slices and transfer caching are not implemented.
        structure = replace(prepared.structure,
            groups=tuple(replace(group, input_bytes=group.parameter_bytes + sum(
                _payload_bytes(graph.nodes[source]) for source in graph.predecessors(group.node_ids[0])))
                for group in prepared.structure.groups),
            dependencies=tuple(replace(dep, bytes=float(_payload_bytes(graph.nodes[dep.src_node])))
                for dep in prepared.structure.dependencies))
        events, stream_last, link_last = [], [None] * profile.n_devs, {}
        group_events, transfers, group_costs = {}, {}, []
        host_last = None

        def event(kind, duration, dependencies=(), **metadata):
            if not math.isfinite(duration) or duration < 0:
                raise ValueError("event duration must be finite and non-negative")
            deps = list(dict.fromkeys(index for index in dependencies if index is not None))
            cause = max(deps, key=lambda index: (events[index]["end"], events[index]["kind"] not in {"host_submit", "host_release"}, index)) if deps else None
            start = events[cause]["end"] if cause is not None else 0.0
            result = len(events)
            events.append({"event_id": result, "kind": kind, "start": start, "end": start + duration,
                "duration": duration, "dependencies": deps, "cause": cause, **metadata})
            return result

        host_last = event("host_start", self.host.start_s)
        confidence, reasons = [], set()
        if not self.host.calibrated:
            reasons.add("reference:host_submission_uncalibrated")
        if not self.service_matches_executor:
            reasons.add("reference:operator_body_coverage_unverified")
        if profile.interference != 0:
            reasons.add("reference:interference_unmodeled")
        starts, ends, device_busy = [], [], [0.0] * profile.n_devs
        for group in structure.groups:
            node = graph.nodes[group.node_ids[0]]
            for dep_index in prepared.incoming[group.index]:
                dep = structure.dependencies[dep_index]
                if dep.src_group not in group_events:
                    raise ValueError("reference groups must follow dependency order")
                if not dep.cross_device:
                    continue
                source = structure.groups[dep.src_group].device
                target = group.device
                cost = profile.estimate_transfer(source, target, dep.bytes)
                confidence.append(cost.confidence)
                reasons.update(cost.ood_reasons)
                release = event("host_release", 0.0, [host_last], consumer=group.group_id)
                host_last = event("host_submit", self.host.copy_submit_s, [release], operation="copy", consumer=group.group_id)
                link = profile.link_key(source, target)
                copy = event("copy", cost.duration,
                    [release, stream_last[source], stream_last[target], group_events[dep.src_group], link_last.get(link)],
                    consumer=group.group_id, source_node=dep.src_node, destination_node=dep.dst_node,
                    source_device=source, destination_device=target, bytes=dep.bytes)
                stream_last[source] = stream_last[target] = link_last[link] = copy
                transfers[dep_index] = _ScheduledTransfer(dep_index, dep.src_group, dep.dst_group, source, target,
                    dep.bytes, events[copy]["start"], events[copy]["end"], cost.duration,
                    cost.confidence, cost.source, cost.ood_reasons, dep.src_group)
            cost = (GroupCost(0.0, 0.0, 0.0, 0.0, 1.0, "cached_reference_input")
                if node.op_type == "input" else profile.estimate_group(graph, group))
            group_costs.append(cost)
            confidence.append(cost.confidence)
            reasons.update(cost.ood_reasons)
            release = event("host_release", 0.0, [host_last], consumer=group.group_id)
            host_last = event("host_submit", self.host.node_cost(node.op_type, str(node.extra.get("local_name", node.id))), [release],
                operation=node.op_type, consumer=group.group_id)
            dependencies = [release, stream_last[group.device]]
            dependencies.extend(group_events[structure.dependencies[index].src_group]
                for index in prepared.incoming[group.index])
            compute = event("compute", cost.duration, dependencies,
                group_id=group.group_id, node_id=node.id, device=group.device, source=cost.source)
            group_events[group.index] = stream_last[group.device] = compute
            starts.append(events[compute]["start"])
            ends.append(events[compute]["end"])
            device_busy[group.device] += cost.duration
        finish = event("finish", self.host.finish_s, [host_last, *stream_last])
        makespan = events[finish]["end"]
        peak = _peak_memory(graph, plan, structure, starts, ends, makespan, transfers)
        for device, value in enumerate(peak):
            if value > profile.max_memory(device):
                reasons.add(f"device:{device}:memory_overflow")
        critical = []
        current = finish
        while current is not None:
            critical.append(current)
            current = events[current]["cause"]
        critical.reverse()
        critical_groups = tuple(dict.fromkeys(events[index]["group_id"] for index in critical if events[index]["kind"] == "compute"))
        return CostEstimate(makespan=makespan, peak_memory=peak, critical_path=critical_groups,
            communication_time=sum(transfer.duration for transfer in transfers.values()),
            computation_time=sum(cost.duration for cost in group_costs), per_device_time=tuple(device_busy),
            cross_device_bytes=structure.cross_device_bytes,
            confidence=min([*confidence, .45 if reasons else .95]), out_of_distribution=bool(reasons),
            ood_reasons=tuple(sorted(reasons)), model_name="reference-eager-streams-v1",
            details={"execution_semantics": "consumer-copy-two-sided-current-streams",
                "host_profile": {"node_submit_s": self.host.node_submit_s, "copy_submit_s": self.host.copy_submit_s,
                    "finish_s": self.host.finish_s, "start_s": self.host.start_s,
                    "operator_submit_s": dict(self.host.operator_submit_s),
                    "calibrated": self.host.calibrated, "source": self.host.source},
                "events": events, "critical_events": critical,
                "groups": [events[group_events[index]] for index in range(len(structure.groups))],
                "transfers": [row for row in events if row["kind"] == "copy"],
                "assumptions": ["One host lane and one current stream per device; the host does not wait for GPU-copy completion.",
                    "A copy waits for both already-submitted streams and is issued only at its consumer.",
                    "Host submission overlaps the operator service window; finish waits for all streams and host work.",
                    "Cached inputs have zero GPU service cost; initialization is outside steady-state timing.",
                    "Memory is graph-level liveness only; eager temporaries/workspaces need separate calibration.",
                    "Upstream PyTorch v2.9.0 Copy.cu is the stream-barrier reference; BW1000 vendor parity is not certified."]})
