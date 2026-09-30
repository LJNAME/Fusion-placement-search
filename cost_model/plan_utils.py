"""与框架无关的 plan 工具（不依赖 torch，可单独用于诊断脚本）。"""

from __future__ import annotations

from typing import Sequence


def is_greedy_compatible(order: Sequence[str], plan) -> bool:
    """判断 plan 能否被 order 序上的动作回放复现。

    random_plan 用随机拓扑序建组，而搜索（beam/MCTS）用贪心拓扑序；
    若某个组的节点在 order 中不连续（或有交叠），回放出来的就是另一个方案，
    这类 plan 不能用作 policy/value 的监督标签。
    """

    pos = {nid: i for i, nid in enumerate(order)}
    seen = set()
    for group in plan.groups:
        try:
            idx = sorted(pos[nid] for nid in group.node_ids)
        except KeyError:
            return False
        if idx != list(range(idx[0], idx[0] + len(idx))):
            return False  # 组内节点在 order 中不连续
        if seen & set(idx):
            return False  # 组之间交叠
        seen.update(idx)
    return len(seen) == len(order)
