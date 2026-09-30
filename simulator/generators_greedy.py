"""贪心拓扑序下的方案生成器（与 beam/MCTS 的决策空间一致）。

背景：generators.random_plan() 用随机拓扑序建组，属于搜索空间之外；
policy/value 头的监督必须来自"贪心序可达"的方案，否则动作前缀与
makespan 对不上（见 cost_model/diagnose_head_labels.py）。

本模块提供两类一致的样本来源：
- random_plan_greedy：随机融合边界（贪心序）+ 随机/均衡设备分配；
- device_variant：固定融合划分，只随机设备分配（专门监督放置决策）。
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence

from .generators import greedy_topological_order, heuristic_plan
from .hardware import HardwareSpec
from .schema import FusionGroup, OpGraph, Plan


def random_plan_greedy(
    graph: OpGraph,
    hw: HardwareSpec,
    rng: Optional[random.Random] = None,
    merge_prob: float = 0.5,
    name: str = "greedy-random",
    device_mode: str = "random",
) -> Plan:
    """随机融合边界（按贪心拓扑序）+ 设备分配。

    device_mode: "random" 每组随机设备；"roundrobin" 轮流放置；
    "balance" 按估计时长的最小负载贪心（复用 heuristic 的估计函数）。
    """

    rng = rng or random.Random()
    order = greedy_topological_order(graph)
    partitions: List[List[str]] = []
    cur: List[str] = []
    for i, nid in enumerate(order):
        cur.append(nid)
        is_last = i == len(order) - 1
        if is_last or rng.random() >= merge_prob:
            partitions.append(list(cur))
            cur = []

    groups: List[FusionGroup] = []
    if device_mode == "random":
        for part in partitions:
            groups.append(FusionGroup(list(part), rng.randrange(hw.n_devs)))
    elif device_mode == "roundrobin":
        for i, part in enumerate(partitions):
            groups.append(FusionGroup(list(part), i % hw.n_devs))
    elif device_mode == "balance":
        from .generators import _estimate_duration

        load = [0.0] * hw.n_devs
        for part in partitions:
            best_dev, best_finish = 0, float("inf")
            for d, dev in enumerate(hw.devices):
                finish = load[d] + _estimate_duration(graph, part, dev, hw)
                if finish < best_finish:
                    best_finish, best_dev = finish, d
            load[best_dev] = best_finish
            groups.append(FusionGroup(list(part), best_dev))
    else:
        raise ValueError(f"未知 device_mode: {device_mode}")
    return Plan(groups, name=name)


def device_variant(
    graph: OpGraph,
    hw: HardwareSpec,
    base_plan: Plan,
    rng: Optional[random.Random] = None,
    name: str = "device-variant",
) -> Plan:
    """保持融合划分不变，只重新随机设备分配（监督放置决策）。"""

    rng = rng or random.Random()
    groups = [
        FusionGroup(list(g.node_ids), rng.randrange(hw.n_devs))
        for g in base_plan.groups
    ]
    return Plan(groups, name=name)


def perturb_plan(
    graph: OpGraph,
    hw: HardwareSpec,
    base_plan: Plan,
    rng: Optional[random.Random] = None,
    name: str = "perturb",
) -> Plan:
    """在基准方案上只改一个融合边界（合并相邻组或拆分一个组）。

    与纯随机方案相比，扰动方案与基准共享大部分动作前缀，能在分叉点
    制造真正的"多动作状态"，给 policy 提供可比较的动作价值标签。
    拆分时第二段的设备随机，顺带覆盖放置决策。
    """

    rng = rng or random.Random()
    groups = [(list(g.node_ids), int(g.device)) for g in base_plan.groups]
    if not groups:
        return Plan([], name=name)
    if len(groups) == 1 and len(groups[0][0]) < 2:
        return Plan([FusionGroup(list(groups[0][0]), groups[0][1])], name=name)

    moves = []
    for i in range(len(groups) - 1):
        moves.append(("merge", i))
    for i, (nids, _dev) in enumerate(groups):
        if len(nids) >= 2:
            moves.append(("split", i))
    if not moves:
        return Plan([FusionGroup(list(n), d) for n, d in groups], name=name)

    kind, i = rng.choice(moves)
    if kind == "merge":
        nids = groups[i][0] + groups[i + 1][0]
        new = groups[:i] + [(nids, groups[i][1])] + groups[i + 2 :]
    else:
        nids, dev = groups[i]
        cut = rng.randrange(1, len(nids))
        new = (
            groups[:i]
            + [(nids[:cut], dev), (nids[cut:], rng.randrange(hw.n_devs))]
            + groups[i + 1 :]
        )
    return Plan(
        [FusionGroup(list(n), d) for n, d in new],
        name=name,
    )


def generate_plans_greedy(
    graph: OpGraph,
    hw: HardwareSpec,
    n_random: int = 8,
    n_device_variants: int = 4,
    n_perturb: int = 8,
    seed: int = 0,
    device_modes: Sequence[str] = ("random", "balance"),
    include_probes: bool = True,
) -> List[Plan]:
    """生成一批"搜索空间内"的方案：随机融合 + 放置变体 + 启发式基线。"""

    rng = random.Random(seed)
    plans: List[Plan] = []
    merge_probs = (0.25, 0.5, 0.75)
    for i in range(n_random):
        mp = merge_probs[i % len(merge_probs)]
        mode = device_modes[i % len(device_modes)]
        plans.append(
            random_plan_greedy(
                graph,
                hw,
                rng,
                merge_prob=mp,
                name=f"greedy-random{i}",
                device_mode=mode,
            )
        )
    base = heuristic_plan(graph, hw, mode="chain", name="chain")
    plans.append(base)
    for i in range(n_device_variants):
        plans.append(
            device_variant(graph, hw, base, rng, name=f"device-variant{i}")
        )
    for i in range(n_perturb):
        plans.append(
            perturb_plan(graph, hw, base, rng, name=f"perturb{i}")
        )
    if include_probes:
        plans.extend(probe_plans(graph, hw, base))
    plans.append(heuristic_plan(graph, hw, mode="threshold", name="threshold"))
    return plans


def probe_plans(
    graph: OpGraph,
    hw: HardwareSpec,
    base_plan: Plan,
) -> List[Plan]:
    """系统性探针：每个决策点上构造"只改一个决定"的对照方案。

    - 设备探针：某一组改到其它设备（监督放置）；
    - 合并探针：相邻两组合并（监督融合边界）；
    - 拆分探针：某组在同一设备上拆开（监督融合边界）。
    这些方案与基准共享前缀，能在每个决策点制造多动作状态。
    """

    groups = [(list(g.node_ids), int(g.device)) for g in base_plan.groups]
    plans: List[Plan] = []

    def _make(new_groups, name):
        return Plan([FusionGroup(list(n), d) for n, d in new_groups], name=name)

    for i, (nids, dev) in enumerate(groups):
        for d in range(hw.n_devs):
            if d == dev:
                continue
            new = groups[:i] + [(list(nids), d)] + groups[i + 1 :]
            plans.append(_make(new, f"probe-dev{i}-{d}"))
    for i in range(len(groups) - 1):
        new = (
            groups[:i]
            + [(groups[i][0] + groups[i + 1][0], groups[i][1])]
            + groups[i + 2 :]
        )
        plans.append(_make(new, f"probe-merge{i}"))
    for i, (nids, dev) in enumerate(groups):
        for cut in range(1, len(nids)):
            new = (
                groups[:i]
                + [(nids[:cut], dev), (nids[cut:], dev)]
                + groups[i + 1 :]
            )
            plans.append(_make(new, f"probe-split{i}-{cut}"))
    return plans
