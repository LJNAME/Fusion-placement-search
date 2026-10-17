"""方案合法性校验：覆盖、不重复、拓扑序、设备范围。"""

from __future__ import annotations

from typing import List, Optional

from .fusion_rules import (
    FusionPatternRegistry,
    FusionRuleError,
    default_fusion_registry,
)
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


def validate_fusion_plan(
    graph: OpGraph,
    plan: Plan,
    n_devs: int,
    registry: Optional[FusionPatternRegistry] = None,
    require_executable: bool = False,
) -> None:
    """Validate topology plus each group's explicitly declared fusion pattern."""

    validate_plan(graph, plan, n_devs)
    rules = registry or default_fusion_registry()
    for group_index, group in enumerate(plan.groups):
        pattern_id = group.extra.get("pattern_id")
        if not pattern_id:
            raise FusionRuleError(
                f"fusion group {group_index} is missing extra.pattern_id"
            )
        pattern = rules.get(str(pattern_id))
        try:
            pattern.validate(graph, group.node_ids)
        except FusionRuleError as exc:
            raise FusionRuleError(
                f"fusion group {group_index} ({pattern_id}) is invalid: {exc}"
            ) from exc
        if require_executable and not pattern.executable:
            raise FusionRuleError(
                f"fusion group {group_index} pattern {pattern_id} has no executable "
                f"backend (status={pattern.implementation_status})"
            )

        declared_backend = group.extra.get("backend")
        if declared_backend is not None and declared_backend != pattern.backend:
            raise FusionRuleError(
                f"fusion group {group_index} pattern {pattern_id} declares backend "
                f"{declared_backend}, expected {pattern.backend}"
            )
