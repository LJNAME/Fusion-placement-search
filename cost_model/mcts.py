"""MCTS：policy 先验 + value 叶子估计 + cost 模型终端评估。

与 beam 的区别
--------------
- 选择：PUCT，先验来自 policy 头，Q 来自已访问动作的平均回报；
- 非叶子节点用 value 头估计"从该状态走完"的回报（快速、可批量）；
- 完整方案用独立的 cost 模型打分（保持与训练时同一套代价语义）；
- 每个状态的访问分布可以直接导出自博弈数据，用于下一轮训练。

回报尺度与训练一致：z = log1p(clip(log(mk) - log(mk_ref), 0, clip))，
其中 mk_ref 取启发式方案（chain）的代价模型估计，保证量纲稳定。
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from cost_model.search import PartialState, state_to_plan
from cost_model.search_heads import (
    apply_action,
    build_partial_sample,
    collate,
    state_meta,
)
from simulator import greedy_topological_order, heuristic_plan


@dataclass
class MCTSNode:
    state: object
    prefix: Tuple[int, ...]
    n_actions: int
    N: Optional[np.ndarray] = None
    W: Optional[np.ndarray] = None
    P: Optional[np.ndarray] = None
    value: float = 0.0
    expanded: bool = False
    terminal: bool = False
    children: Dict[int, "MCTSNode"] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.N = np.zeros(self.n_actions, dtype=float)
        self.W = np.zeros(self.n_actions, dtype=float)
        self.P = np.zeros(self.n_actions, dtype=float)


class MCTS:
    def __init__(
        self,
        net,
        cost_model,
        graph,
        hw,
        meta: Dict,
        device: str = "cpu",
        c_puct: float = 1.5,
        clip: float = 2.0,
        seed: int = 0,
    ) -> None:
        self.net = net
        self.cost_model = cost_model
        self.graph = graph
        self.hw = hw
        self.vocab = meta["vocab"]
        self.n_devs = int(meta["n_devs"])
        self.device = device
        self.c_puct = c_puct
        self.clip = clip
        self.rng = np.random.RandomState(seed)
        self.order = greedy_topological_order(graph)
        self.n_cost_calls = 0
        self.terminal_prefix_counts: Counter = Counter()
        self.anchor_counter = 0
        self.ref_log = self._reference_log_cost()

    # ---------- 基础工具 ----------

    def _legal_mask(self, state) -> np.ndarray:
        mask = np.zeros(1 + self.n_devs, dtype=bool)
        if state.cur:
            mask[0] = True
        mask[1:] = True
        return mask

    def _is_terminal(self, state) -> bool:
        return state.i >= len(self.order)

    def _reference_log_cost(self) -> float:
        plan = heuristic_plan(self.graph, self.hw, mode="chain", name="ref")
        self.n_cost_calls += 1
        return float(self.cost_model.predict(self.graph, plan)[0])

    def _terminal_cost(self, state) -> float:
        plan = state_to_plan(state)
        return self._cost_of_plan(plan)

    def _cost_of_plan(self, plan) -> float:
        self.n_cost_calls += 1
        log_mk = float(self.cost_model.predict(self.graph, plan)[0])
        z = min(max(log_mk - self.ref_log, 0.0), self.clip)
        return float(math.log1p(z))

    def _node_meta(self, state) -> Dict:
        meta = state_meta(self.order, state, self.n_devs)
        meta["total_nodes"] = len(self.order)
        return meta

    def _evaluate(self, state) -> Tuple[float, np.ndarray]:
        sample = build_partial_sample(
            self.graph, self.order, state, self.vocab, self.n_devs
        )
        batch = collate([sample])
        batch = {
            k: (v.to(self.device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        mask_np = self._legal_mask(state)
        mask = torch.tensor(mask_np, device=self.device)
        with torch.no_grad():
            enc = self.net.encode(batch)
            value = float(self.net.value_of(enc, batch, self._node_meta(state)))
            logits, _ = self.net.priors(
                enc,
                {**self._node_meta(state), "can_merge": int(bool(state.cur))},
                mask,
            )
            priors = torch.softmax(
                logits.masked_fill(~mask, -1e9), dim=-1
            ).cpu().numpy()
        priors = priors * mask_np
        if priors.sum() <= 0:
            priors = mask_np.astype(float)
        return value, priors / priors.sum()

    # ---------- 搜索 ----------

    def run(
        self,
        n_iters: int = 200,
        anchor_every: int = 0,
        anchor_mix: float = 0.5,
    ) -> Dict:
        n_actions = 1 + self.n_devs
        root = MCTSNode(state=PartialState(), prefix=(), n_actions=n_actions)
        self.root = root
        for _ in range(n_iters):
            node = root
            path: List[Tuple[MCTSNode, int]] = []
            while node.expanded and not self._is_terminal(node.state):
                mask = self._legal_mask(node.state)
                q = np.where(
                    node.N > 0,
                    node.W / np.maximum(node.N, 1.0),
                    node.value,
                )
                u = np.where(
                    mask,
                    self.c_puct
                    * node.P
                    * math.sqrt(node.N.sum() + 1.0)
                    / (1.0 + node.N),
                    -1e9,
                )
                scores = np.where(mask, -q + u, -1e9)
                action = int(np.argmax(scores))
                child = node.children.get(action)
                if child is None:
                    child = MCTSNode(
                        state=apply_action(
                            node.state, action, self.order, self.n_devs
                        ),
                        prefix=node.prefix + (action,),
                        n_actions=n_actions,
                    )
                    node.children[action] = child
                path.append((node, action))
                node = child

            if self._is_terminal(node.state):
                node.terminal = True
                cost = self._terminal_cost(node.state)
                self.terminal_prefix_counts[node.prefix] += 1
            else:
                value, priors = self._evaluate(node.state)
                node.value = value
                node.P = priors
                node.expanded = True
                cost = value
            # 少量真实锚点：每 anchor_every 次迭代，用 cost 模型评一次
            # 补完到终态的方案，把 value 估计往真实代价方向拉。
            if (
                anchor_every
                and not node.terminal
                and self.anchor_counter % anchor_every == 0
            ):
                anchor_plan = self._complete_sample(node.state, node, temperature=1.0)
                anchor_cost = self._cost_of_plan(anchor_plan)
                cost = anchor_mix * anchor_cost + (1.0 - anchor_mix) * cost
            self.anchor_counter += 1
            for parent, action in path:
                parent.N[action] += 1.0
                parent.W[action] += cost

        plan, actions = self._extract_plan(root)
        best_cost = self._terminal_cost_from_actions(actions)
        return {
            "plan": plan,
            "actions": actions,
            "root": root,
            "n_cost_calls": self.n_cost_calls,
            "best_cost": best_cost,
        }

    def _complete_greedy(self, state, node) -> object:
        """从给定状态补完到终态：优先跟随访问最多的子节点，否则用 policy 贪心。"""

        actions: List[int] = []
        while not self._is_terminal(state):
            mask = self._legal_mask(state)
            visits = np.where(mask, node.N, -1.0)
            if visits.max() > 0:
                action = int(np.argmax(visits))
            else:
                _, priors = self._evaluate(state)
                action = int(np.argmax(np.where(mask, priors, -1.0)))
            actions.append(action)
            state = apply_action(state, action, self.order, self.n_devs)
            child = node.children.get(action)
            if child is None:
                child = MCTSNode(
                    state=state,
                    prefix=node.prefix + (action,),
                    n_actions=1 + self.n_devs,
                )
            node = child
        return state_to_plan(state)

    def _complete_sample(self, state, node, temperature: float = 1.0) -> object:
        """采样补完：已访问节点按访问分布采样，未访问节点按 policy 采样。"""

        actions: List[int] = []
        while not self._is_terminal(state):
            mask = self._legal_mask(state)
            weights = np.where(mask, node.N, 0.0)
            if weights.sum() > 0:
                p = weights / weights.sum()
            else:
                _, priors = self._evaluate(state)
                p = np.where(mask, priors, 0.0)
                if p.sum() <= 0:
                    p = mask.astype(float)
                p = np.power(p, 1.0 / max(temperature, 1e-6))
                p = p / p.sum()
            action = int(self.rng.choice(len(p), p=p))
            actions.append(action)
            state = apply_action(state, action, self.order, self.n_devs)
            child = node.children.get(action)
            if child is None:
                child = MCTSNode(
                    state=state,
                    prefix=node.prefix + (action,),
                    n_actions=1 + self.n_devs,
                )
            node = child
        return state_to_plan(state)

    def candidate_plans(
        self,
        k: int = 8,
        samples_per_prefix: int = 2,
        temperature: float = 1.0,
    ) -> List[Dict]:
        """沿"访问最多"的路径逐层生成候选，每个前缀做贪心 + 采样补完。

        贪心补完容易收敛到同一条路，加入按访问/policy 采样的补完可以覆盖
        不同分支；最后按方案结构去重。
        """

        root = getattr(self, "root", None)
        if root is None:
            return []
        out: List[Dict] = []
        seen = set()
        state = PartialState()
        node = root
        while not self._is_terminal(state) and len(out) < k:
            plans = [self._complete_greedy(state, node)]
            for _ in range(max(0, samples_per_prefix)):
                plans.append(self._complete_sample(state, node, temperature))
            for plan in plans:
                sig = tuple(
                    (tuple(g.node_ids), int(g.device)) for g in plan.groups
                )
                if sig in seen:
                    continue
                seen.add(sig)
                out.append(
                    {
                        "prefix": list(node.prefix),
                        "plan": plan,
                        "visits": int(node.N.sum()),
                    }
                )
                if len(out) >= k:
                    break
            mask = self._legal_mask(state)
            visits = np.where(mask, node.N, -1.0)
            if visits.max() > 0:
                action = int(np.argmax(visits))
            else:
                priors = np.where(mask, node.P, -1.0)
                action = (
                    int(np.argmax(priors))
                    if priors.max() > 0
                    else int(np.argmax(mask))
                )
            state = apply_action(state, action, self.order, self.n_devs)
            child = node.children.get(action)
            if child is None:
                break
            node = child
        if not out:  # 兜底：至少给出一个完整方案
            out.append(
                {
                    "prefix": [],
                    "plan": self._complete_greedy(PartialState(), root),
                    "visits": int(root.N.sum()),
                }
            )
        return out

    def _extract_plan(self, root: MCTSNode):
        state = PartialState()
        node = root
        actions: List[int] = []
        while not self._is_terminal(state):
            mask = self._legal_mask(state)
            visits = np.where(mask, node.N, -1.0)
            if visits.max() > 0:
                action = int(np.argmax(visits))
            else:
                priors = np.where(mask, node.P, -1.0)
                if priors.max() > 0:
                    action = int(np.argmax(priors))
                else:
                    action = int(np.argmax(mask))
            actions.append(action)
            state = apply_action(state, action, self.order, self.n_devs)
            child = node.children.get(action)
            if child is None:
                # 未访问过的分支：造一个空节点继续往下走（后续会走贪心兜底）
                child = MCTSNode(
                    state=state,
                    prefix=node.prefix + (action,),
                    n_actions=1 + self.n_devs,
                )
            node = child
        return state_to_plan(state), actions

    def _terminal_cost_from_actions(self, actions: Sequence[int]) -> float:
        state = PartialState()
        for action in actions:
            if self._is_terminal(state):
                break
            state = apply_action(state, int(action), self.order, self.n_devs)
        if not self._is_terminal(state):
            # 兜底：按"新组放设备 0"补完
            while not self._is_terminal(state):
                state = apply_action(state, 1, self.order, self.n_devs)
        return self._terminal_cost(state)

    # ---------- 自博弈导出 ----------

    def export_selfplay(
        self,
        result: Dict,
        graph_id: int,
    ) -> List[Dict]:
        """导出每个访问过节点的 (prefix, visit 分布, 最终回报)。"""

        z = float(result["best_cost"])
        records: List[Dict] = []
        stack = [result["root"]]
        while stack:
            node = stack.pop()
            if node.expanded and node.N.sum() > 0:
                dist = (node.N / node.N.sum()).astype(float).tolist()
                records.append(
                    {
                        "graph_id": int(graph_id),
                        "prefix": list(node.prefix),
                        "visits": dist,
                        "z": z,
                    }
                )
            stack.extend(node.children.values())
        return records
