"""Graph-level forward execution simulation after Proteus's HTAE (layer 2).

Reference: Duan et al., "Proteus: Simulating the Performance of Distributed
DNN Training", arXiv 2306.02267 (IEEE TPDS 2024), Sec. VI. Borrowed here for
the forward pass only (the objective stays forward makespan):

* executor per device with a computation queue and a feature-communication
  queue; operators enter a queue when their dependencies are satisfied and
  are popped in ready order (Sec. VI-B "Executor");
* feature communication may block the computation stream of the sending
  device (``feature_comm_blocks_compute``), as Proteus notes for feature
  communication; off = asynchronous copy stream;
* bandwidth sharing: concurrent transfers on one physical link share it fairly
  (Sec. VI-C), simulated as processor sharing between events;
* comp-comm overlap: an operator that runs while an operator of the other kind
  is active on the same device slows by (1 + overlap_factor) (Sec. VI-C, gamma).

Not modeled: gradient communication, backward/optimizer, compute-compute
overlap on one device (Proteus does not model it either), hierarchical NIC
links (two cards, one point-to-point link per direction).

Operator costs come unchanged from layer 1 (``profile.estimate_group`` /
``profile.estimate_transfer``). A transfer's standalone duration (alpha-beta)
is its work; under sharing each of k transfers progresses at 1/k, which also
stretches its latency term (a simplification).
"""
from __future__ import annotations

from typing import Optional

from simulator.fusion_rules import FusionPatternRegistry

from .estimate import CostEstimate
from .hardware_profile import as_hardware_profile
from .structural import PreparedPlan, StructuralCostModel, _peak_memory, _ScheduledTransfer

_EPS = 1e-12


class HTAECostModel(StructuralCostModel):
    """Event-driven forward simulation; same structure/memory code as StructuralCostModel."""

    def __init__(self, *, strict_fusion: bool = True, registry: Optional[FusionPatternRegistry] = None,
                 feature_comm_blocks_compute: Optional[bool] = None,
                 overlap_factor: Optional[float] = None, name: str = "htae-forward-v1") -> None:
        super().__init__(strict_fusion=strict_fusion, registry=registry, name=name)
        if overlap_factor is not None and overlap_factor < 0:
            raise ValueError("overlap_factor must be non-negative")
        self.feature_comm_blocks_compute = feature_comm_blocks_compute
        self.overlap_factor = overlap_factor

    def settings(self, profile) -> tuple[bool, float]:
        """None means: take the hardware snapshot (overlap_send / interference)."""
        blocks = (not profile.overlap_send) if self.feature_comm_blocks_compute is None \
            else bool(self.feature_comm_blocks_compute)
        gamma = float(profile.interference) if self.overlap_factor is None else float(self.overlap_factor)
        return blocks, gamma

    def predict_prepared(self, prepared: PreparedPlan, hardware) -> CostEstimate:
        profile = as_hardware_profile(hardware)
        graph, plan, structure = prepared.graph, prepared.plan, prepared.structure
        if profile.n_devs != structure.n_devs:
            raise ValueError("prepared plan and profile device counts differ")
        blocks, gamma = self.settings(profile)
        groups, deps = structure.groups, structure.dependencies
        n, n_devs = len(groups), profile.n_devs

        costs = [profile.estimate_group(graph, g) for g in groups]
        confidence = [c.confidence for c in costs]
        ood = {r for c in costs for r in c.ood_reasons}
        tcost = {}
        for dep in deps:
            if dep.cross_device:
                c = profile.estimate_transfer(groups[dep.src_group].device, groups[dep.dst_group].device, dep.bytes)
                tcost[dep.index] = c
                confidence.append(c.confidence)
                ood.update(c.ood_reasons)

        pending = [len(prepared.incoming[g.index]) for g in groups]
        ready_at = [0.0] * n
        cause: list[Optional[int]] = [None] * n
        compute_queue: list[list[int]] = [[] for _ in range(n_devs)]
        comm_queue: list[list[int]] = [[] for _ in range(n_devs)]  # dep indices, FIFO per sender
        running_group: list[Optional[int]] = [None] * n_devs
        running_comm: list[Optional[int]] = [None] * n_devs
        remaining: dict[tuple[str, int], float] = {}  # ("g", idx) / ("t", dep) -> standalone seconds left
        starts, ends = [0.0] * n, [0.0] * n
        t_start, t_end, t_cause = {}, {}, {}
        device_busy = [0.0] * n_devs
        now = 0.0

        def make_ready(gi: int, time: float, by: Optional[int]) -> None:
            ready_at[gi] = time
            cause[gi] = by
            compute_queue[groups[gi].device].append(gi)

        for g in groups:
            if pending[g.index] == 0:
                make_ready(g.index, 0.0, None)

        def dispatch() -> None:
            for d in range(n_devs):
                if running_comm[d] is None and comm_queue[d]:
                    dep = comm_queue[d].pop(0)
                    running_comm[d] = dep
                    t_start[dep] = now
                    remaining[("t", dep)] = tcost[dep].duration
                busy_sending = blocks and (running_comm[d] is not None or comm_queue[d])
                if running_group[d] is None and compute_queue[d] and not busy_sending:
                    # Proteus pops in ready order; ties keep plan order.
                    gi = min(compute_queue[d], key=lambda i: (ready_at[i], i))
                    compute_queue[d].remove(gi)
                    running_group[d] = gi
                    starts[gi] = now
                    remaining[("g", gi)] = costs[gi].duration

        def rates() -> dict:
            out = {}
            link_load: dict = {}
            for d in range(n_devs):
                dep = running_comm[d]
                if dep is not None:
                    key = profile.link_key(d, groups[deps[dep].dst_group].device)
                    link_load[key] = link_load.get(key, 0) + 1
            comm_on = [False] * n_devs
            for d in range(n_devs):
                dep = running_comm[d]
                if dep is not None:
                    comm_on[d] = True
                    comm_on[groups[deps[dep].dst_group].device] = True
            for d in range(n_devs):
                gi = running_group[d]
                if gi is not None:
                    out[("g", gi)] = 1.0 / (1.0 + gamma) if comm_on[d] else 1.0
            for d in range(n_devs):
                dep = running_comm[d]
                if dep is not None:
                    dst = groups[deps[dep].dst_group].device
                    share = 1.0 / link_load[profile.link_key(d, dst)]
                    overlapped = running_group[d] is not None or running_group[dst] is not None
                    out[("t", dep)] = share / (1.0 + gamma) if overlapped else share
            return out

        done = 0
        while done < n:
            dispatch()
            r = rates()
            if not r:
                raise RuntimeError("HTAE deadlock: no runnable operator")
            dt = min(remaining[k] / v for k, v in r.items())
            for k, v in r.items():
                remaining[k] -= dt * v
            now += dt
            finished = [k for k in r if remaining[k] <= _EPS * max(1.0, now)]
            for kind, idx in sorted(finished):
                del remaining[(kind, idx)]
                if kind == "g":
                    d = groups[idx].device
                    running_group[d] = None
                    ends[idx] = now
                    device_busy[d] += now - starts[idx]
                    done += 1
                    for dep_index in prepared.outgoing[idx]:
                        dep = deps[dep_index]
                        if dep.cross_device:
                            comm_queue[d].append(dep_index)
                            t_cause[dep_index] = idx
                        else:
                            pending[dep.dst_group] -= 1
                            if pending[dep.dst_group] == 0:
                                make_ready(dep.dst_group, now, idx)
                else:
                    src = groups[deps[idx].src_group].device
                    running_comm[src] = None
                    t_end[idx] = now
                    dst_group = deps[idx].dst_group
                    pending[dst_group] -= 1
                    if pending[dst_group] == 0:
                        make_ready(dst_group, now, t_cause[idx])

        transfers = {}
        for dep_index, c in tcost.items():
            dep = deps[dep_index]
            transfers[dep_index] = _ScheduledTransfer(
                dependency_index=dep_index, src_group=dep.src_group, dst_group=dep.dst_group,
                src_device=groups[dep.src_group].device, dst_device=groups[dep.dst_group].device,
                bytes=dep.bytes, start=t_start[dep_index], end=t_end[dep_index], duration=c.duration,
                confidence=c.confidence, source=c.source, ood_reasons=c.ood_reasons,
                cause_group=t_cause[dep_index])
        makespan = max(ends, default=0.0)
        peak = _peak_memory(graph, plan, structure, starts, ends, makespan, transfers)
        overflow = tuple(d for d, v in enumerate(peak) if v > profile.max_memory(d))
        for d in overflow:
            ood.add(f"device:{d}:memory_overflow")

        critical, current, seen = [], (max(range(n), key=lambda i: (ends[i], i)) if n else None), set()
        while current is not None and current not in seen:
            seen.add(current)
            critical.append(current)
            current = cause[current]
        critical.reverse()
        schedule = sorted(range(n), key=lambda i: (starts[i], i))
        reasons = tuple(sorted(ood))
        incoming = prepared.incoming

        def dep_ready(i):
            return max((transfers[j].end if deps[j].cross_device else ends[deps[j].src_group]
                        for j in incoming[i]), default=0.0)

        def local_ready(i):
            return max((ends[deps[j].src_group] for j in incoming[i] if not deps[j].cross_device), default=0.0)

        def comm_ready(i):
            return max((transfers[j].end for j in incoming[i] if deps[j].cross_device), default=0.0)

        return CostEstimate(
            makespan=float(makespan), peak_memory=peak,
            critical_path=tuple(groups[i].group_id for i in critical),
            communication_time=float(sum(c.duration for c in tcost.values())),
            computation_time=float(sum(c.duration for c in costs)),
            per_device_time=tuple(float(v) for v in device_busy),
            cross_device_bytes=float(structure.cross_device_bytes),
            confidence=float(min(confidence, default=1.0)),
            out_of_distribution=bool(reasons), ood_reasons=reasons, model_name=self.name,
            details={
                "structure_version": "plan-structure-v1",
                "scheduling_policy": "htae_ready_queue_v1",
                "executor": {"feature_comm_blocks_compute": blocks, "overlap_factor": gamma,
                             "bandwidth_sharing": "fair_share_per_link",
                             "reference": "Proteus HTAE (arXiv 2306.02267, Sec. VI), forward only"},
                "schedule_order": [groups[i].group_id for i in schedule],
                "overflow_devices": list(overflow),
                "groups": [{
                    "group_id": g.group_id, "device": g.device, "pattern_id": g.pattern_id,
                    "start": starts[g.index], "end": ends[g.index],
                    "duration": ends[g.index] - starts[g.index],
                    "dependency_ready": dep_ready(g.index), "device_ready": ready_at[g.index],
                    "local_dependency_ready": local_ready(g.index),
                    "communication_ready": comm_ready(g.index),
                    "communication_wait": max(0.0, comm_ready(g.index) - local_ready(g.index)),
                    "compute_time": costs[g.index].compute_time, "memory_time": costs[g.index].memory_time,
                    "launch_time": costs[g.index].launch_time, "source": costs[g.index].source,
                } for g in groups],
                "transfers": [{
                    "dependency_index": t.dependency_index, "src_group": groups[t.src_group].group_id,
                    "dst_group": groups[t.dst_group].group_id, "src_device": t.src_device,
                    "dst_device": t.dst_device, "bytes": t.bytes, "start": t.start, "end": t.end,
                    "duration": t.end - t.start, "source": t.source,
                } for t in sorted(transfers.values(), key=lambda x: x.dependency_index)],
            },
        )


__all__ = ["HTAECostModel"]
