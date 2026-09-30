"""方案合法性校验：覆盖、不重复、拓扑序、设备范围。"""

from __future__ import annotations

from typing import List

from .schema import OpGraph, Plan


def validate_plan(graph: OpGraph, plan: Plan, n_devs: int) -> None:
    """校验 plan 是否合法，不合法抛出 ValueError。"""

    if not plan.groups:
        raise ValueError("方案不能为空（至少一个融合组）")

    seen: set[str] = set()
    order: List[str] = []
    for gi, g in enumerate(plan.groups):
        if not g.node_ids:
            raise ValueError(f"融合组 {gi} 为空")
        if not (0 <= g.device < n_devs):
            raise ValueError(
                f"融合组 {gi} 设备越界: {g.device} (设备数 {n_devs})"
            )
        for nid in g.node_ids:
            if nid not in graph.nodes:
                raise ValueError(f"方案引用不存在的节点: {nid}")
            if nid in seen:
                raise ValueError(f"节点重复出现: {nid}")
            seen.add(nid)
            order.append(nid)

    missing = sorted(set(graph.nodes) - seen)
    if missing:
        raise ValueError(f"方案遗漏节点: {missing}")

    pos = {nid: i for i, nid in enumerate(order)}
    for e in graph.edges:
        if pos[e.src] >= pos[e.dst]:
            raise ValueError(
                f"方案不满足拓扑序: 边 {e.src} -> {e.dst} 在 {pos[e.src]} / {pos[e.dst]}"
            )
