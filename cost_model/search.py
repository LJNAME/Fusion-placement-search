"""Legacy adjacent-topological-order fusion + placement search.

New standalone Fusion code must use ``cost_model.legal_search``.  This module is
kept to reproduce historical datasets and experiments whose action space allowed
arbitrary adjacent nodes to merge.

对应笔记《树搜索——以三算子，两GPU为例》的决策模型：
按拓扑序逐个算子决策，动作 = 并入当前融合组 / 开新组放设备 d；
部分方案用"随机补完 + 代价模型评估取平均"打分（rollout）。

- `beam_search`：树搜索的宽度受限版，每层保留预测最优的 top-k 状态；
- `mcts_search`：AlphaZero 式 MCTS，UCB 选择 → 展开挂孩子 → rollout 模拟 →
  回传更新 Q/N；policy 先验暂为均匀（对应笔记 PUCT 公式里 π=1 的特例）。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from simulator import FusionGroup, Plan, greedy_topological_order


@dataclass
class PartialState:
    groups: List[tuple] = field(default_factory=list)  # [(node_ids, device)] 已关闭的组
    cur: List[str] = field(default_factory=list)       # 当前打开的融合组
    cur_dev: int = 0
    i: int = 0                                          # 下一个待决策的拓扑序下标


def _expand(state: PartialState, order: List[str], n_devs: int) -> List[PartialState]:
    """当前节点的所有合法动作：并入当前组 / 关闭当前组后开新组放设备 d。"""

    if state.i >= len(order):
        return []
    nid = order[state.i]
    actions: List[PartialState] = []
    if state.cur:
        actions.append(
            PartialState(list(state.groups), list(state.cur) + [nid], state.cur_dev, state.i + 1)
        )
    for d in range(n_devs):
        groups = list(state.groups)
        if state.cur:
            groups.append((list(state.cur), state.cur_dev))
        actions.append(PartialState(groups, [nid], d, state.i + 1))
    return actions


def state_to_plan(state: PartialState) -> Plan:
    groups = list(state.groups)
    if state.cur:
        groups.append((list(state.cur), state.cur_dev))
    return Plan([FusionGroup(list(nids), dev) for nids, dev in groups], name="beam")


def rollout_plan(
    graph,
    state: PartialState,
    order: List[str],
    rng: random.Random,
    merge_prob: float,
    n_devs: int,
) -> Plan:
    """把部分方案随机补完为完整方案。"""

    groups = list(state.groups)
    cur = list(state.cur)
    cur_dev = state.cur_dev
    for i in range(state.i, len(order)):
        nid = order[i]
        if cur and rng.random() >= merge_prob:
            groups.append((list(cur), cur_dev))
            cur = []
            cur_dev = rng.randrange(n_devs)
        cur.append(nid)
    if cur:
        groups.append((list(cur), cur_dev))
    return Plan([FusionGroup(list(nids), dev) for nids, dev in groups], name="rollout")


def beam_search(
    graph,
    scorer: Callable[[Plan], float],
    n_devs: int,
    beam: int = 8,
    rollouts: int = 3,
    merge_prob: float = 0.5,
    seed: int = 0,
) -> List[Plan]:
    """beam 搜索，返回预测得分最优的 beam 个完整方案。

    scorer(plan) 返回预测 makespan（越小越好）。
    """

    rng = random.Random(seed)
    order = greedy_topological_order(graph)
    states = [PartialState()]
    for _ in range(len(order)):
        candidates = []
        for st in states:
            for ns in _expand(st, order, n_devs):
                score = 0.0
                for _r in range(rollouts):
                    plan = rollout_plan(graph, ns, order, rng, merge_prob, n_devs)
                    score += scorer(plan)
                score /= rollouts
                candidates.append((score, ns))
        candidates.sort(key=lambda x: x[0])
        states = [ns for _, ns in candidates[:beam]]
    return [state_to_plan(s) for s in states]


@dataclass
class _MCTSNode:
    """MCTS 树节点：状态 = 一个部分方案。

    q = 穿过该节点的所有补完方案分数的平均值（分数 = -预测 makespan）；
    n = 访问次数。首次访问时一次性挂上全部合法孩子（笔记的 AlphaZero 规则）。
    """

    state: PartialState
    parent: Optional["_MCTSNode"] = None
    children: List["_MCTSNode"] = field(default_factory=list)
    n: int = 0
    q: float = 0.0
    expanded: bool = False
    terminal: bool = False

    def update(self, score: float) -> None:
        """回传：增量式平均分。"""

        self.n += 1
        self.q += (score - self.q) / self.n

    def expand(self, order: List[str], n_devs: int) -> None:
        """第一次访问时把合法孩子一次性挂上；没有可决策算子则标记为终端。"""

        if self.expanded:
            return
        self.expanded = True
        if self.state.i >= len(order):
            self.terminal = True
            return
        for ns in _expand(self.state, order, n_devs):
            self.children.append(
                _MCTSNode(ns, parent=self, terminal=ns.i >= len(order))
            )

    def select_child(self, C: float) -> Optional["_MCTSNode"]:
        """未访问的孩子优先级最高（先试一遍）；都访问过后用 UCB 选。

        UCB = Q + C * sqrt(ln(父访问数 + 1) / (1 + 子访问数))
        （policy 先验为均匀，即笔记 PUCT 公式 π=1 的特例）
        """

        for c in self.children:
            if c.n == 0:
                return c
        if not self.children:
            return None
        ln_n = math.log(self.n + 1.0)

        def ucb(c: _MCTSNode) -> float:
            return c.q + C * math.sqrt(ln_n / (1.0 + c.n))

        return max(self.children, key=ucb)

    def best_child(self) -> Optional["_MCTSNode"]:
        """Q 最高（同分取访问更多）的孩子；无孩子返回 None。"""

        if not self.children:
            return None
        return max(self.children, key=lambda c: (c.q, c.n))


def _plan_key(plan: Plan) -> tuple:
    return tuple((tuple(g.node_ids), g.device) for g in plan.groups)


def mcts_search(
    graph,
    scorer: Callable[[Plan], float],
    n_devs: int,
    iterations: int = 500,
    rollouts: int = 3,
    merge_prob: float = 0.5,
    seed: int = 0,
    C: float = 1.5,
    n_best: int = 8,
) -> List[Plan]:
    """AlphaZero 式 MCTS，返回预测最优的若干完整方案。

    scorer(plan) 返回预测 makespan（越小越好）；树内 Q 用 -makespan。
    每轮：选择（未访问优先，否则 UCB）→ 展开挂孩子 → 从选中孩子 rollout
    补完 + 代价模型评估取平均 → 回传更新路径上所有节点 Q/N。
    搜索结束后沿"最佳 Q 路径"贪心补完出一个确定性完整方案，避免最终
    候选只依赖随机 rollout 见过的方案。
    搜索过程中见过的完整方案按预测得分去重排序，取 top-n_best。
    """

    rng = random.Random(seed)
    order = greedy_topological_order(graph)
    root = _MCTSNode(PartialState())
    best: Dict[tuple, tuple] = {}  # plan_key -> (score, plan)

    def remember(plan: Plan, score: float) -> None:
        k = _plan_key(plan)
        if k not in best or score < best[k][0]:
            best[k] = (score, plan)

    for _ in range(iterations):
        node = root
        path: List[_MCTSNode] = [root]
        # 选择：沿已展开且非终端的路径下降
        while node.expanded and not node.terminal:
            child = node.select_child(C)
            if child is None:
                break
            path.append(child)
            node = child
        # 展开 + 模拟
        if node.terminal:
            plan = state_to_plan(node.state)
            score = float(scorer(plan))
            remember(plan, score)
            sim_score = -score
        else:
            node.expand(order, n_devs)
            child = node.select_child(C)  # 刚展开，全部未访问，取第一个
            if child is None:
                break
            path.append(child)
            sim_score = 0.0
            for _r in range(rollouts):
                plan = rollout_plan(graph, child.state, order, rng, merge_prob, n_devs)
                s = float(scorer(plan))
                remember(plan, s)
                sim_score -= s / rollouts
        # 回传
        for nd in path:
            nd.update(sim_score)

    # 沿最佳 Q 路径贪心补完：确定性完整方案（不依赖随机 rollout 运气）
    node = root
    while not node.terminal:
        if not node.expanded:
            node.expand(order, n_devs)
        child = node.best_child()
        if child is None:
            break
        node = child
    if node.terminal:
        plan = state_to_plan(node.state)
        score = float(scorer(plan))
        remember(plan, score)

    if not best:
        return [state_to_plan(root.state)]
    ranked = sorted(best.values(), key=lambda kv: kv[0])
    return [plan for _, plan in ranked[:n_best]]
