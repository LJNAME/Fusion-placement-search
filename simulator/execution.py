"""模块二：仿真器核心。

执行模型（roofline + 事件队列）：
- 算子/融合组时间 = max(计算量/算力, 读写字节/显存带宽) + 启动开销 +
  融合开销 ×（组内算子数 - 1）；
- 融合组内共享中间张量（不写全局内存、不产生通信）；
- 跨设备边通信走共享总线（FIFO 排队），时间 = 字节/总线带宽 + 延迟；
- 设备竞争：每台设备一个就绪队列，依赖满足后按 FIFO 执行，
  端到端时间（makespan）= 最后一个完成时刻。

内存模型：
- 张量生命周期：产生（组开始）→ 消费（消费者组结束 / 跨设备传输完成）→ 释放；
- 融合组内中间张量不落显存；
- 跨设备张量：源端保留到传输完成，目的端在传输完成后分配接收缓冲，
  消费者组结束时释放；
- 参数张量全程驻留所在设备；
- 逐设备累计峰值显存，超出 max_mem 标记溢出。
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .hardware import HardwareSpec
from .schema import OpGraph, Plan, group_boundary_io
from .shape_model import group_compute_time
from .validate import validate_plan


@dataclass
class GroupStats:
    """融合组的执行统计。"""

    index: int
    node_ids: List[str]
    device: int
    flops: float
    input_bytes: float
    output_bytes: float
    compute_time: float
    io_time: float
    launch_time: float
    duration: float
    start: float = 0.0
    end: float = 0.0


@dataclass
class TransferStats:
    """跨设备传输统计。"""

    index: int
    src_node: str
    dst_node: str
    src_group: int
    dst_group: int
    bytes: float
    duration: float
    start: float = 0.0
    end: float = 0.0


@dataclass
class SimResult:
    """一次仿真的完整结果。"""

    makespan: float
    peak_mem: Dict[int, float]             # 每设备峰值显存
    mem_overflow: bool
    overflow_devices: List[int]
    group_stats: List[GroupStats]
    transfer_stats: List[TransferStats]
    op_intervals: Dict[str, Tuple[float, float, int, int]]  # node -> (start,end,device,group)
    device_busy: Dict[int, float]          # 每设备累计忙碌时间
    bus_busy: float                        # 总线累计忙碌时间
    link_busy: Dict[str, float] = field(default_factory=dict)  # "i-j" -> link busy time
    timeline: List[Tuple[float, str, str]] = field(default_factory=list)


def simulate(
    graph: OpGraph,
    plan: Plan,
    hw: HardwareSpec,
    collect_timeline: bool = True,
) -> SimResult:
    """对给定 (图, 方案, 硬件) 执行事件驱动仿真，返回 SimResult。"""

    validate_plan(graph, plan, hw.n_devs)
    ng = len(plan.groups)
    ndev = hw.n_devs

    node_to_group: Dict[str, int] = {}
    for gi, g in enumerate(plan.groups):
        for nid in g.node_ids:
            node_to_group[nid] = gi

    # ---- 1. 融合组 roofline 统计 ----
    input_edges: List[List] = [[] for _ in range(ng)]
    output_edges: List[List] = [[] for _ in range(ng)]
    for e in graph.edges:
        sg, dg = node_to_group[e.src], node_to_group[e.dst]
        if sg != dg:
            output_edges[sg].append(e)
            input_edges[dg].append(e)

    group_stats: List[GroupStats] = []
    for gi, g in enumerate(plan.groups):
        dev = hw.devices[g.device]
        in_bytes, out_bytes = group_boundary_io(graph, g.node_ids)
        flops = sum(graph.nodes[nid].flops for nid in g.node_ids)
        compute_time = group_compute_time(graph, g.node_ids, dev, hw)
        io_time = (
            (in_bytes + out_bytes) / dev.mem_bw if dev.mem_bw > 0 else float("inf")
        )
        launch_time = hw.launch_overhead + hw.fusion_overhead * max(
            0, len(g.node_ids) - 1
        )
        duration = max(compute_time, io_time) + launch_time
        group_stats.append(
            GroupStats(
                index=gi,
                node_ids=list(g.node_ids),
                device=g.device,
                flops=flops,
                input_bytes=in_bytes,
                output_bytes=out_bytes,
                compute_time=compute_time,
                io_time=io_time,
                launch_time=launch_time,
                duration=duration,
            )
        )

    # ---- 2. 跨设备传输与依赖 ----
    from .schema import edge_bytes

    transfers: List[TransferStats] = []
    output_transfers: List[List[int]] = [[] for _ in range(ng)]
    incoming: List[List[int]] = [[] for _ in range(ng)]
    for e in graph.edges:
        sg, dg = node_to_group[e.src], node_to_group[e.dst]
        # 同组内共享中间张量；同设备跨组只等生产者结束即可，无需走总线
        if sg == dg or plan.groups[sg].device == plan.groups[dg].device:
            continue
        b = edge_bytes(e, graph)
        ti = len(transfers)
        transfers.append(
            TransferStats(ti, e.src, e.dst, sg, dg, b, 0.0)  # duration set at schedule time
        )
        output_transfers[sg].append(ti)
        incoming[dg].append(ti)

    pred_groups: List[set[int]] = [set() for _ in range(ng)]
    for gi in range(ng):
        for e in input_edges[gi]:
            pred_groups[gi].add(node_to_group[e.src])

    # ---- 3. 事件队列仿真 ----
    remaining_incoming = [len(incoming[gi]) for gi in range(ng)]
    group_done = [False] * ng
    group_started = [False] * ng
    ready_pending = [False] * ng
    starts = [0.0] * ng
    ends = [0.0] * ng

    device_queues: Dict[int, List[int]] = {d: [] for d in range(ndev)}
    device_busy_until = [0.0] * ndev
    active_sends = [0] * ndev
    use_links = hw.has_links
    link_busy_until: Dict[tuple, float] = {}
    link_total: Dict[tuple, float] = {}

    events: List[Tuple[float, int, str, int]] = []
    seq = itertools.count()
    timeline: List[Tuple[float, str, str]] = []

    def push(t: float, kind: str, payload: int) -> None:
        heapq.heappush(events, (t, next(seq), kind, payload))
        if collect_timeline:
            timeline.append((t, kind, str(payload)))

    def start_device(dev: int, t: float) -> None:
        q = device_queues[dev]
        while q and device_busy_until[dev] <= t + 1e-12:
            gi = q.pop(0)
            start = max(t, device_busy_until[dev])
            dur = group_stats[gi].duration
            active = active_sends[dev]
            if hw.interference > 0 and active > 0:
                # Echo-style placeholder: in-flight sends slow down new compute
                dur *= 1.0 + hw.interference * active
            end = start + dur
            starts[gi] = start
            ends[gi] = end
            device_busy_until[dev] = end
            group_started[gi] = True
            ready_pending[gi] = False
            push(end, "group_finish", gi)

    def schedule_transfer(ti: int, t: float) -> None:
        """Place transfer on its link FIFO (single-bus mode: one shared logical link)."""

        tr = transfers[ti]
        src_dev = plan.groups[tr.src_group].device
        dst_dev = plan.groups[tr.dst_group].device
        if use_links:
            key = (src_dev, dst_dev) if src_dev <= dst_dev else (dst_dev, src_dev)
            bw = hw.link_bandwidth(src_dev, dst_dev)
        else:
            key = (0, 0)
            bw = hw.comm_bw
        dur = tr.bytes / bw + hw.comm_latency
        tr.duration = dur
        start = max(t, link_busy_until.get(key, 0.0))
        end = start + dur
        link_busy_until[key] = end
        link_total[key] = link_total.get(key, 0.0) + dur
        transfer_start[ti] = start
        transfer_end[ti] = end
        tr.start = start
        tr.end = end
        active_sends[src_dev] += 1
        push(start, "transfer_start", ti)
        push(end, "transfer_finish", ti)

    def schedule_ready(gi: int, t: float) -> None:
        if ready_pending[gi] or group_started[gi]:
            return
        if all(group_done[p] for p in pred_groups[gi]) and remaining_incoming[gi] == 0:
            ready_pending[gi] = True
            push(t, "group_ready", gi)

    for gi in range(ng):
        schedule_ready(gi, 0.0)
    for dev in range(ndev):
        start_device(dev, 0.0)

    transfer_start = [0.0] * len(transfers)
    transfer_end = [0.0] * len(transfers)

    while events:
        t, _, kind, payload = heapq.heappop(events)
        if kind == "group_ready":
            gi = payload
            if not group_started[gi]:
                device_queues[plan.groups[gi].device].append(gi)
                start_device(plan.groups[gi].device, t)
        elif kind == "group_finish":
            gi = payload
            group_done[gi] = True
            dev = plan.groups[gi].device
            for ti in output_transfers[gi]:
                schedule_transfer(ti, t)
            if not hw.overlap_send and output_transfers[gi]:
                # 执行框架限制（如 DTK 主机同步拷贝/流序）：设备要等自己的发送完成
                device_busy_until[dev] = max(
                    device_busy_until[dev],
                    max(transfer_end[ti] for ti in output_transfers[gi]),
                )
            for dg in range(ng):
                if gi in pred_groups[dg]:
                    schedule_ready(dg, t)
            start_device(dev, t)
        elif kind == "transfer_finish":
            ti = payload
            tr = transfers[ti]
            src_dev = plan.groups[tr.src_group].device
            active_sends[src_dev] = max(0, active_sends[src_dev] - 1)
            dst = tr.dst_group
            remaining_incoming[dst] -= 1
            schedule_ready(dst, t)
            # overlap_send=false 时，源设备可能被自己的发送阻塞，传输完成后重新调度
            start_device(src_dev, t)

    for ti, tr in enumerate(transfers):
        tr.start = transfer_start[ti]
        tr.end = transfer_end[ti]
    for gi, gs in enumerate(group_stats):
        gs.start = starts[gi]
        gs.end = ends[gi]

    makespan = max(ends) if ng else 0.0
    if transfers:
        makespan = max(makespan, max(transfer_end))

    # ---- 4. 内存模型：张量生命周期 -> 每设备峰值 ----
    mem_events: Dict[int, List[Tuple[float, float]]] = {d: [] for d in range(ndev)}

    # 参数常驻整步
    for gi, g in enumerate(plan.groups):
        dev = g.device
        for nid in g.node_ids:
            pb = graph.nodes[nid].param_bytes
            if pb > 0:
                mem_events[dev].append((0.0, pb))
                mem_events[dev].append((makespan, -pb))

    # 算子输出张量
    for nid, node in graph.nodes.items():
        if node.out_bytes <= 0:
            continue
        gi = node_to_group[nid]
        dev = plan.groups[gi].device
        consumers = [e for e in graph.edges if e.src == nid]
        if not consumers:
            # 图输出：写全局内存，保留到仿真结束
            mem_events[dev].append((starts[gi], node.out_bytes))
            mem_events[dev].append((makespan, -node.out_bytes))
            continue
        external = [e for e in consumers if node_to_group[e.dst] != gi]
        if not external:
            # 组内全内部消费 -> 不落显存
            continue
        alloc_t = starts[gi]
        release_t = makespan
        for e in external:
            cgi = node_to_group[e.dst]
            if plan.groups[cgi].device == dev:
                release_t = max(release_t, ends[cgi])
            else:
                ti = next(
                    i
                    for i, tr in enumerate(transfers)
                    if tr.src_node == nid and tr.dst_node == e.dst
                )
                release_t = max(release_t, transfer_end[ti])
        mem_events[dev].append((alloc_t, node.out_bytes))
        mem_events[dev].append((release_t, -node.out_bytes))

    # 跨设备接收缓冲：传输完成时分配，消费者组结束时释放
    for tr in transfers:
        dev = plan.groups[tr.dst_group].device
        mem_events[dev].append((transfer_end[tr.index], tr.bytes))
        mem_events[dev].append((ends[tr.dst_group], -tr.bytes))

    peak_mem: Dict[int, float] = {}
    overflow_devices: List[int] = []
    for dev, evs in mem_events.items():
        cur = 0.0
        peak = 0.0
        for _t, delta in sorted(evs, key=lambda x: (x[0], -x[1])):
            cur += delta
            peak = max(peak, cur)
        peak_mem[dev] = peak
        if peak > hw.devices[dev].max_mem + 1e-9:
            overflow_devices.append(dev)

    op_intervals: Dict[str, Tuple[float, float, int, int]] = {}
    for gi, g in enumerate(plan.groups):
        for nid in g.node_ids:
            op_intervals[nid] = (starts[gi], ends[gi], g.device, gi)

    device_busy = {
        d: sum(gs.duration for gs in group_stats if gs.device == d)
        for d in range(ndev)
    }
    bus_busy = sum(tr.duration for tr in transfers)
    link_busy = {
        f"{a}-{b}": float(v) for (a, b), v in sorted(link_total.items())
    }

    return SimResult(
        makespan=makespan,
        peak_mem=peak_mem,
        mem_overflow=len(overflow_devices) > 0,
        overflow_devices=overflow_devices,
        group_stats=group_stats,
        transfer_stats=transfers,
        op_intervals=op_intervals,
        device_busy=device_busy,
        bus_busy=bus_busy,
        link_busy=link_busy,
        timeline=timeline,
    )
