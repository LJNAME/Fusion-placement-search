"""方案生成器：随机方案 + 启发式方案（贪心 / 固定阈值融合）。

用于 Stage 0 训练数据采集，以及搜索算法的初始种群 / 基线。
"""

from __future__ import annotations

import heapq
import random
from typing import List, Optional

from .hardware import HardwareSpec
from .schema import OpGraph, Plan, FusionGroup, edge_bytes, group_boundary_io
from .shape_model import group_compute_time


def random_topological_order(graph: OpGraph, rng: random.Random) -> List[str]:
    """随机拓扑序（Kahn + 随机优先级）。"""

    indeg = {nid: len(graph.in_edges(nid)) for nid in graph.nodes}
    heap = [(rng.random(), nid) for nid in graph.nodes if indeg[nid] == 0]
    heapq.heapify(heap)
    order: List[str] = []
    while heap:
        _, nid = heapq.heappop(heap)
        order.append(nid)
        for e in graph.out_edges(nid):
            indeg[e.dst] -= 1
            if indeg[e.dst] == 0:
                heapq.heappush(heap, (rng.random(), e.dst))
    if len(order) != len(graph.nodes):
        raise ValueError("计算图存在环，无法得到拓扑序")
    return order


def greedy_topological_order(graph: OpGraph) -> List[str]:
    """确定性拓扑序（字典序最小）。"""

    indeg = {nid: len(graph.in_edges(nid)) for nid in graph.nodes}
    heap = [nid for nid in graph.nodes if indeg[nid] == 0]
    heapq.heapify(heap)
    order: List[str] = []
    while heap:
        nid = heapq.heappop(heap)
        order.append(nid)
        for e in graph.out_edges(nid):
            indeg[e.dst] -= 1
            if indeg[e.dst] == 0:
                heapq.heappush(heap, e.dst)
    return order


def random_plan(
    graph: OpGraph,
    hw: HardwareSpec,
    rng: Optional[random.Random] = None,
    merge_prob: float = 0.5,
    name: str = "random",
) -> Plan:
    """随机方案：随机拓扑序 -> 按 merge_prob 切融合组 -> 随机设备。"""

    rng = rng or random.Random()
    order = random_topological_order(graph, rng)
    groups: List[FusionGroup] = []
    cur: List[str] = []
    for i, nid in enumerate(order):
        cur.append(nid)
        is_last = i == len(order) - 1
        if is_last or rng.random() >= merge_prob:
            groups.append(FusionGroup(list(cur), rng.randrange(hw.n_devs)))
            cur = []
    return Plan(groups, name=name)


def _estimate_duration(
    graph: OpGraph, node_ids: List[str], dev, hw: HardwareSpec
) -> float:
    """用单设备规格估算一个融合组的执行时间（供设备分配启发式用）。"""

    in_bytes, out_bytes = group_boundary_io(graph, node_ids)
    compute_time = group_compute_time(graph, node_ids, dev, hw)
    io_time = (in_bytes + out_bytes) / dev.mem_bw if dev.mem_bw > 0 else float("inf")
    launch_time = hw.launch_overhead + hw.fusion_overhead * max(
        0, len(node_ids) - 1
    )
    return max(compute_time, io_time) + launch_time


def heuristic_plan(
    graph: OpGraph,
    hw: HardwareSpec,
    mode: str = "threshold",
    threshold_bytes: float = 0.0,
    name: str = "heuristic",
) -> Plan:
    """启发式方案。

    mode:
      - "all"       : 全部算子融成一个组；
      - "chain"     : 存在依赖边就融合（最大链融合）；
      - "threshold" : 固定阈值融合，依赖边张量 >= threshold_bytes 才融合。
    设备分配：按估计时长做最小负载贪心（支持异构）。
    """

    order = greedy_topological_order(graph)
    partitions: List[List[str]] = []
    cur: List[str] = []
    for i, nid in enumerate(order):
        cur.append(nid)
        if i == len(order) - 1:
            partitions.append(list(cur))
            break
        nxt = order[i + 1]
        edge = next((e for e in graph.edges if e.src == nid and e.dst == nxt), None)
        if mode == "all":
            merge = True
        elif mode == "chain":
            merge = edge is not None
        elif mode == "threshold":
            merge = edge is not None and edge_bytes(edge, graph) >= threshold_bytes
        else:
            raise ValueError(f"未知启发式模式: {mode}")
        if not merge:
            partitions.append(list(cur))
            cur = []

    # 最小负载贪心设备分配
    load = [0.0] * hw.n_devs
    groups: List[FusionGroup] = []
    for part in partitions:
        best_dev = 0
        best_finish = float("inf")
        for d, dev in enumerate(hw.devices):
            finish = load[d] + _estimate_duration(graph, part, dev, hw)
            if finish < best_finish:
                best_finish = finish
                best_dev = d
        load[best_dev] = best_finish
        groups.append(FusionGroup(list(part), best_dev))
    return Plan(groups, name=name)


def alternating_plan(
    graph: OpGraph,
    hw: HardwareSpec,
    name: str = "alternating",
) -> Plan:
    """链式融合 + 设备轮流放置：制造通信频繁、负载分散的方案。

    与启发式（最小负载贪心）形成对照，用来扫描"通信拓扑/干扰是否
    改变排序"：同一图、同一融合划分，仅设备分配不同。
    """

    order = greedy_topological_order(graph)
    partitions: List[List[str]] = []
    cur: List[str] = []
    for i, nid in enumerate(order):
        cur.append(nid)
        if i == len(order) - 1:
            partitions.append(list(cur))
            break
        nxt = order[i + 1]
        edge = next((e for e in graph.edges if e.src == nid and e.dst == nxt), None)
        if edge is None:
            partitions.append(list(cur))
            cur = []
    groups = [
        FusionGroup(list(part), i % hw.n_devs) for i, part in enumerate(partitions)
    ]
    return Plan(groups, name=name)


def generate_plans(
    graph: OpGraph,
    hw: HardwareSpec,
    n_random: int = 8,
    random_merge_prob: float = 0.5,
    heuristic_modes: List[str] = ("threshold", "chain"),
    threshold_bytes: float = 0.0,
    seed: int = 0,
) -> List[Plan]:
    """生成一批随机 + 启发式方案（供 Stage 0 打标签）。"""

    rng = random.Random(seed)
    plans = [
        random_plan(graph, hw, rng, merge_prob=random_merge_prob, name=f"random{i}")
        for i in range(n_random)
    ]
    for mode in heuristic_modes:
        plans.append(
            heuristic_plan(
                graph, hw, mode=mode, threshold_bytes=threshold_bytes, name=mode
            )
        )
    return plans
