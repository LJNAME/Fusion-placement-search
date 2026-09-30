"""MCTS 的 value 头与 policy 头实现（cost 头仍走代价模型）。

设计约定
--------
- cost 头：由调用方提供（roofline / simulator / 已训练的 GNNCostModel），
  只负责给"完整方案"打分，本模块不实现它。
- value 头：输入部分状态，输出"从该状态走完"的归一化期望代价（标量）。
- policy 头：输入部分状态，输出合法动作上的分布（pointer 风格打分 + mask）。

共享 trunk 直接复用 CostGNN 的 node_enc / gin / group_mlp：
节点特征在原 (op_type one-hot + log flops/out/param) 后面追加 3 个状态标志
(1,0,0)=未处理 / (0,1,0)=当前打开的组 / (0,0,1)=已关闭的组；
新增输入维度的权重初始化为 0，因此载入原 cost 模型权重后，
cost 头的输出在初始化时完全不变（后续可选冻结或联合训练）。

用法（搜索阶段，单状态）
------------------------
    sample = build_partial_sample(graph, order, state, op_vocab, n_devs)
    batch = collate([sample])
    enc = net.encode(batch)
    v = net.value_of(enc)                     # (1,) 归一化剩余代价，越小越好
    meta = state_meta(order, state, n_devs)
    logits = net.priors(enc, meta, legal_mask)  # (A,)

用法（训练阶段，批量）
----------------------
    # 自博弈一条轨迹得到 (sample, visit_dist, final_cost)
    # cost 头先给完整方案打分 -> z = log(final_cost / ref_cost)
    loss = value_loss(v, z) + policy_loss(logits, visit_dist, action_mask)

本模块只依赖 torch 与 simulator 的数据结构，不依赖 torch_geometric。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from simulator import graph_to_dict, greedy_topological_order

try:  # 兼容包式与脚本式两种运行方式
    from .gnn import CostGNN
except Exception:  # pragma: no cover
    try:
        from gnn import CostGNN
    except Exception:
        CostGNN = None  # 允许在没有 gnn.py 的环境里单独使用本文件的结构

try:  # 无 torch 的诊断脚本也要能用同一个实现
    from .plan_utils import is_greedy_compatible
except Exception:  # pragma: no cover
    from plan_utils import is_greedy_compatible


STATE_FLAG_DIM = 3
ACTION_MERGE = 0  # 动作 0：并入当前组


def _log1p(x) -> float:
    return float(np.log1p(max(float(x or 0.0), 0.0)))


# ---------------------------------------------------------------------------
# 1. 部分状态 -> 样本张量（与 gnn.build_sample 同构，额外带状态标志）
# ---------------------------------------------------------------------------


def build_partial_sample(
    graph,
    order: Sequence[str],
    state,
    op_vocab: Sequence[str],
    n_devs: int,
) -> Dict[str, torch.Tensor]:
    """把一个 PartialState 转成与 gnn.py 兼容的样本（多了 3 维状态标志）。

    state 需要具备 search.PartialState 的字段：
        groups: [(node_ids, device)] 已关闭的组
        cur:    [node_id]             当前打开的组（可能为空）
        cur_dev: int                  当前组所在设备
        i:      int                   下一个待决策的拓扑序下标
    """

    gd = graph_to_dict(graph)
    nodes, edges = gd["nodes"], gd["edges"]
    id2idx = {nd["id"]: i for i, nd in enumerate(nodes)}

    closed = [(list(nids), int(dev)) for nids, dev in state.groups]
    cur = list(state.cur)
    groups: List[Tuple[List[str], int]] = list(closed)
    cur_group_index = -1
    if cur:
        cur_group_index = len(groups)
        groups.append((cur, int(state.cur_dev)))

    node2group: Dict[str, int] = {}
    for gi, (nids, _dev) in enumerate(groups):
        for nid in nids:
            node2group[nid] = gi
    group_assign = [node2group.get(nd["id"], -1) for nd in nodes]

    # 节点特征：原有特征 + 三态 one-hot 标志
    node_feats = []
    for nd, gi in zip(nodes, group_assign):
        oh = [0.0] * len(op_vocab)
        key = nd["op_type"] if nd["op_type"] in op_vocab else "UNK"
        if key not in op_vocab and "UNK" in op_vocab:
            key = "UNK"
        if key in op_vocab:
            oh[list(op_vocab).index(key)] = 1.0
        if gi < 0:
            flag = [1.0, 0.0, 0.0]  # 未处理
        elif gi == cur_group_index:
            flag = [0.0, 1.0, 0.0]  # 当前打开组
        else:
            flag = [0.0, 0.0, 1.0]  # 已关闭组
        node_feats.append(
            oh
            + [
                _log1p(nd.get("flops", 0.0)),
                _log1p(nd.get("out_bytes", 0.0)),
                _log1p(nd.get("param_bytes", 0.0)),
            ]
            + flag
        )

    edge_src, edge_dst, edge_attr = [], [], []
    for e in edges:
        edge_src.append(id2idx[e["src"]])
        edge_dst.append(id2idx[e["dst"]])
        eb = e.get("bytes")
        if eb is None:
            eb = nodes[id2idx[e["src"]]].get("out_bytes", 0.0)
        edge_attr.append([_log1p(eb)])

    # 组特征：设备 one-hot + log 组大小 + 组序号归一化
    group_feats = []
    ngroups = max(len(groups), 1)
    for gi, (nids, dev) in enumerate(groups):
        onehot = [0.0] * n_devs
        if 0 <= dev < n_devs:
            onehot[dev] = 1.0
        group_feats.append(
            onehot + [_log1p(len(nids)), gi / max(ngroups - 1, 1)]
        )
    if not group_feats:  # 空状态：放一个占位组，避免 (G,) 维度为 0
        group_feats.append([0.0] * n_devs + [0.0, 0.0])

    # 每台设备显式字节（只统计已经确定归属的节点/边）
    dev_bytes = [0.0] * n_devs
    for nids, dev in groups:
        if not (0 <= dev < n_devs):
            continue
        dev_bytes[dev] += sum(
            float(nodes[id2idx[nid]].get("out_bytes", 0.0))
            + float(nodes[id2idx[nid]].get("param_bytes", 0.0))
            for nid in nids
        )
    for e in edges:
        sg = node2group.get(e["src"], -1)
        dg = node2group.get(e["dst"], -1)
        if sg < 0 or dg < 0:
            continue
        if groups[sg][1] != groups[dg][1]:
            eb = e.get("bytes")
            if eb is None:
                eb = nodes[id2idx[e["src"]]].get("out_bytes", 0.0)
            dev_bytes[groups[dg][1]] += float(eb or 0.0)

    return {
        "node_feats": torch.tensor(node_feats, dtype=torch.float32),
        "edge_src": torch.tensor(edge_src, dtype=torch.long),
        "edge_dst": torch.tensor(edge_dst, dtype=torch.long),
        "edge_attr": torch.tensor(edge_attr, dtype=torch.float32),
        "group_assign": torch.tensor(group_assign, dtype=torch.long),
        "group_feats": torch.tensor(group_feats, dtype=torch.float32),
        "dev_bytes": torch.tensor(
            [_log1p(b) for b in dev_bytes], dtype=torch.float32
        ),
    }


def state_meta(order: Sequence[str], state, n_devs: int) -> Dict[str, int]:
    """policy/value 头需要的部分状态标量（单状态搜索时用）。"""

    return {
        "next_idx": int(state.i),
        "next_node": order[state.i] if state.i < len(order) else "",
        "can_merge": int(bool(state.cur)),
        "cur_dev": int(state.cur_dev) if state.cur else -1,
        "n_closed": int(len(state.groups)),
        "n_devs": int(n_devs),
        "done": int(state.i >= len(order)),
    }


def collate(samples: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """把一批样本 padding 成 (B, max_N / max_G) 的 batch（与 gnn.collate 一致）。"""

    B = len(samples)
    max_n = max(s["node_feats"].size(0) for s in samples)
    max_g = max(s["group_feats"].size(0) for s in samples)
    node_dim = samples[0]["node_feats"].size(1)
    group_dim = samples[0]["group_feats"].size(1)

    node_feats = torch.zeros(B, max_n, node_dim)
    node_mask = torch.zeros(B, max_n, dtype=torch.bool)
    group_feats = torch.zeros(B, max_g, group_dim)
    group_mask = torch.zeros(B, max_g, dtype=torch.bool)
    group_assign = torch.full((B, max_n), -1, dtype=torch.long)
    dev_bytes = torch.stack([s["dev_bytes"] for s in samples])

    ei_parts, ea_parts = [], []
    for b, s in enumerate(samples):
        n = s["node_feats"].size(0)
        g = s["group_feats"].size(0)
        node_feats[b, :n] = s["node_feats"]
        node_mask[b, :n] = True
        group_feats[b, :g] = s["group_feats"]
        group_mask[b, :g] = True
        group_assign[b, :n] = s["group_assign"]
        off = b * max_n
        ei_parts.append(torch.stack([s["edge_src"] + off, s["edge_dst"] + off]))
        ea_parts.append(s["edge_attr"])

    return {
        "node_feats": node_feats,
        "node_mask": node_mask,
        "edge_index": torch.cat(ei_parts, dim=1),
        "edge_attr": torch.cat(ea_parts, dim=0),
        "group_assign": group_assign,
        "group_feats": group_feats,
        "group_mask": group_mask,
        "dev_bytes": dev_bytes,
    }


# ---------------------------------------------------------------------------
# 2. 共享 trunk：复用 CostGNN 的编码层，只扩展节点输入维度
# ---------------------------------------------------------------------------


class TrunkAdapter(nn.Module):
    """把已训练的 CostGNN 改造成 value/policy 的共享编码器。

    - 替换 node_enc 的第一层，输入维度 +3（状态标志），新增列初始化为 0；
    - 复用原 GIN 层与 group_mlp，输出节点/组/设备/全局嵌入；
    - 初始化后 cost 头的行为不变（新增维度权重为 0）。
    """

    def __init__(self, cost_gnn, state_flag_dim: int = STATE_FLAG_DIM):
        super().__init__()
        first = cost_gnn.node_enc[0]
        old_dim, hidden = first.in_features, first.out_features
        new_first = nn.Linear(old_dim + state_flag_dim, hidden)
        with torch.no_grad():
            new_first.weight.zero_()
            new_first.weight[:, :old_dim].copy_(first.weight)
            new_first.bias.copy_(first.bias)
        cost_gnn.node_enc = nn.Sequential(new_first, *list(cost_gnn.node_enc)[1:])
        self.cost_gnn = cost_gnn
        self.node_enc = cost_gnn.node_enc
        self.gin = cost_gnn.gin
        self.group_mlp = cost_gnn.group_mlp
        self.hidden = hidden

    def freeze_trunk(self, freeze: bool = True) -> None:
        for p in list(self.node_enc.parameters()) + list(self.gin.parameters()) + list(
            self.group_mlp.parameters()
        ):
            p.requires_grad = not freeze

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        node_feats = batch["node_feats"]
        node_mask = batch["node_mask"]
        group_feats = batch["group_feats"]
        group_mask = batch["group_mask"]
        group_assign = batch["group_assign"]
        B, N, _ = node_feats.shape
        G = group_feats.shape[1]
        n_devs = batch["dev_bytes"].shape[1]

        x = self.node_enc(node_feats).reshape(B * N, -1)
        for layer in self.gin:
            x = layer(x, batch["edge_index"], batch["edge_attr"])
        x = x * node_mask.reshape(-1, 1)

        gids = group_assign.reshape(-1)
        batch_idx = torch.arange(B, device=x.device).repeat_interleave(N)
        gids_global = gids.clamp(min=0) + batch_idx * G
        valid = (gids >= 0).float()
        counts = torch.zeros(B * G, device=x.device).scatter_add_(0, gids_global, valid)
        pooled = torch.zeros(B * G, x.size(1), device=x.device)
        pooled.index_add_(0, gids_global, x)
        pooled_mean = pooled / counts.clamp(min=1).unsqueeze(1)
        x_max = torch.where(node_mask.reshape(-1, 1), x, torch.full_like(x, -1e9))
        pooled_max = torch.full((B * G, x.size(1)), -1e9, device=x.device)
        pooled_max.index_reduce_(0, gids_global, x_max, "amax", include_self=False)
        pooled_max = torch.where(
            (counts > 0).unsqueeze(1), pooled_max, torch.zeros_like(pooled_max)
        )

        gf = group_feats.reshape(B * G, -1)
        combined = self.group_mlp(torch.cat([pooled_mean, pooled_max, gf], dim=1))
        combined = combined * group_mask.reshape(-1, 1)
        group_emb = combined.reshape(B, G, -1)
        global_emb = group_emb.sum(1) / group_mask.sum(1, keepdim=True).clamp(min=1)

        w = group_feats[:, :, :n_devs] * group_mask.unsqueeze(-1)
        denom = w.sum(1).clamp(min=1)
        device_emb = (w.transpose(1, 2) @ group_emb) / denom.unsqueeze(-1)

        remain = (group_assign < 0) & node_mask
        rem = x.reshape(B, N, -1)
        rem_mean = _masked_mean(rem, remain)
        rem_max = _masked_max(rem, remain)

        return {
            "node_emb": rem,
            "group_emb": group_emb,
            "device_emb": device_emb,
            "global_emb": global_emb,
            "rem_mean": rem_mean,
            "rem_max": rem_max,
            "group_assign": group_assign,
            "node_mask": node_mask,
        }


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.float().unsqueeze(-1)
    return (x * m).sum(1) / m.sum(1).clamp(min=1)


def _masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1)
    counts = m.sum(1)
    out = torch.where(m, x, torch.full_like(x, -1e9)).amax(dim=1)
    return torch.where(counts > 0, out, torch.zeros_like(out))


# ---------------------------------------------------------------------------
# 3. value 头：部分状态 -> 归一化期望剩余代价
# ---------------------------------------------------------------------------


class ValueHead(nn.Module):
    """输入部分状态摘要，输出归一化期望代价（越小越好）。

    输入拼接：[global, 当前组, 当前设备, 未处理节点 mean, 未处理节点 max, 进度]
    输出：标量，目标是 log(cost / ref_cost)（按图归一化，见 value_target）。
    """

    def __init__(self, hidden: int, mid: int = 128, dropout: float = 0.1):
        super().__init__()
        in_dim = hidden * 5 + 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, mid // 2),
            nn.GELU(),
            nn.Linear(mid // 2, 1),
        )

    def forward(
        self,
        global_emb: torch.Tensor,
        cur_group_emb: torch.Tensor,
        cur_dev_emb: torch.Tensor,
        rem_mean: torch.Tensor,
        rem_max: torch.Tensor,
        progress: torch.Tensor,
    ) -> torch.Tensor:
        feat = torch.cat(
            [global_emb, cur_group_emb, cur_dev_emb, rem_mean, rem_max, progress],
            dim=-1,
        )
        return self.net(feat).squeeze(-1)


def value_target(cost: float, ref_cost: float) -> float:
    """value 的回归目标：相对参考方案的 log 比值（按图归一化）。"""

    cost = max(float(cost), 1e-12)
    ref_cost = max(float(ref_cost), 1e-12)
    return float(math.log(cost / ref_cost))


def value_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(pred, target)


# ---------------------------------------------------------------------------
# 4. policy 头：部分状态 -> 合法动作分布
# ---------------------------------------------------------------------------


class PolicyHead(nn.Module):
    """pointer 风格的动作打分器。

    动作编码约定（与 search._expand 一致）：
        动作 0                = 并入当前组（当前组为空时非法）
        动作 1 + d (d=0..D-1) = 关闭当前组，开新组放到设备 d
    """

    def __init__(self, hidden: int, mid: int = 128, dropout: float = 0.1):
        super().__init__()
        self.type_emb = nn.Embedding(2, hidden)
        self.state_proj = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.GELU()
        )
        self.action_proj = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU()
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden * 2, mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, 1),
        )

    def build_action_embeddings(
        self,
        op_emb: torch.Tensor,
        cur_group_emb: torch.Tensor,
        device_embs: torch.Tensor,
        can_merge: bool,
    ) -> torch.Tensor:
        """返回 (A, hidden) 的动作嵌入；can_merge=False 时不产生动作 0。"""

        hidden = op_emb.size(-1)
        actions = []
        if can_merge:
            t = self.type_emb(torch.tensor(ACTION_MERGE, device=op_emb.device))
            actions.append(torch.cat([t, cur_group_emb + op_emb], dim=-1))
        n_devs = device_embs.size(0)
        for d in range(n_devs):
            t = self.type_emb(torch.tensor(1, device=op_emb.device))
            actions.append(torch.cat([t, device_embs[d] + op_emb], dim=-1))
        return self.action_proj(torch.stack(actions, dim=0))

    def forward(
        self,
        global_emb: torch.Tensor,
        op_emb: torch.Tensor,
        cur_group_emb: torch.Tensor,
        device_embs: torch.Tensor,
        can_merge: bool,
        legal_mask: Optional[torch.Tensor] = None,
        include_merge: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """单状态打分，返回 (logits (A,), action_emb (A, hidden))。

        动作序固定为 [merge, 设备 0..D-1]。include_merge=False 时只输出设备段；
        搜索/训练统一用 include_merge=True + mask（merge 非法时置 False），
        这样动作编号在整条流水线里保持一致。
        """

        state = self.state_proj(
            torch.cat([global_emb, op_emb, cur_group_emb], dim=-1)
        )
        if include_merge is None:
            include_merge = can_merge
        action_emb = self.build_action_embeddings(
            op_emb, cur_group_emb, device_embs, include_merge
        )
        pair = torch.cat(
            [state.unsqueeze(0).expand(action_emb.size(0), -1), action_emb], dim=-1
        )
        logits = self.scorer(pair).squeeze(-1)
        if legal_mask is not None:
            logits = logits.masked_fill(
                ~_align_mask(logits, legal_mask, can_merge), -1e9
            )
        return logits, action_emb

    def score_batch(
        self,
        state_vec: torch.Tensor,
        action_emb: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """批量打分：state_vec (B,h)，action_emb (B,A,h)，mask (B,A)。"""

        B, A, _ = action_emb.shape
        pair = torch.cat(
            [state_vec.unsqueeze(1).expand(B, A, -1), action_emb], dim=-1
        )
        logits = self.scorer(pair).squeeze(-1)
        return logits.masked_fill(~action_mask.bool(), -1e9)


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    logits = logits.masked_fill(~mask, -1e9)
    return F.log_softmax(logits, dim=-1)


def policy_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    entropy_coef: float = 0.01,
) -> torch.Tensor:
    """target 为 MCTS 访问分布（或监督动作 one-hot），mask 为合法动作。"""

    logp = masked_log_softmax(logits, mask)
    ce = -(target * logp).sum(dim=-1)
    p = logp.exp()
    ent = -(p * logp).sum(dim=-1)
    return (ce - entropy_coef * ent).mean()


def _action_mask(can_merge: bool, n_devs: int, device: str = "cpu") -> torch.Tensor:
    mask = [False] * ((1 if can_merge else 0) + n_devs)
    offset = 0
    if can_merge:
        mask[0] = True
        offset = 1
    for d in range(n_devs):
        mask[offset + d] = True
    return torch.tensor(mask, device=device)


def _align_mask(
    logits: torch.Tensor, mask: torch.Tensor, can_merge: bool
) -> torch.Tensor:
    """把外部 mask 对齐到动作 logits 的长度。

    兼容两类输入：
    - 完整长度 (1 + n_devs)：动作序为 merge, dev0, dev1, ...；
    - 仅设备段 (n_devs)：merge 槽自动补上（can_merge 时可合并，否则不可）。
    """

    a = int(logits.numel())
    m = mask.to(logits.device).bool().reshape(-1)
    if m.numel() == a:
        return m
    if m.numel() == a - 1:
        # 外部只给了设备段：前面补 merge 槽（can_merge 时合法）
        head = (
            torch.ones(1, dtype=torch.bool, device=m.device)
            if can_merge
            else torch.zeros(1, dtype=torch.bool, device=m.device)
        )
        return torch.cat([head, m])
    if m.numel() > a:
        return m[:a]
    pad = torch.zeros(a - m.numel(), dtype=torch.bool, device=m.device)
    return torch.cat([m, pad])


def legal_action_mask(state, n_devs: int, device: str = "cpu") -> torch.Tensor:
    """从 PartialState 直接得到合法动作 mask（与 PolicyHead 的动作序一致）。"""

    return _action_mask(bool(state.cur), n_devs, device)


def apply_action(state, action: int, order: Sequence[str], n_devs: int):
    """执行一个动作，返回新的 PartialState（不改原状态）。"""

    if state.i >= len(order):
        raise ValueError("state is already terminal")
    nid = order[state.i]
    if action == ACTION_MERGE:
        if not state.cur:
            raise ValueError("merge action is illegal when current group is empty")
        return type(state)(
            groups=list(state.groups),
            cur=list(state.cur) + [nid],
            cur_dev=state.cur_dev,
            i=state.i + 1,
        )
    d = int(action) - 1
    if not (0 <= d < n_devs):
        raise ValueError(f"illegal device index {d}")
    groups = list(state.groups)
    if state.cur:
        groups.append((list(state.cur), state.cur_dev))
    return type(state)(groups=groups, cur=[nid], cur_dev=d, i=state.i + 1)


def replay_trajectory(
    order: Sequence[str],
    actions: Sequence[int],
    init_state,
    n_devs: int,
):
    """按动作序列回放：返回 [(state, action)] 与最终 state。"""

    traj = []
    st = init_state
    for a in actions:
        traj.append((st, int(a)))
        st = apply_action(st, int(a), order, n_devs)
    return traj, st


def actions_from_plan(
    order: Sequence[str],
    plan,
    n_devs: int,
) -> List[int]:
    """把一个完整 Plan 还原成动作序列（监督 warm start 用）。"""

    node2group = {}
    for gi, group in enumerate(plan.groups):
        for nid in group.node_ids:
            node2group[nid] = gi
    group_dev = [int(g.device) for g in plan.groups]
    actions: List[int] = []
    prev_group = None
    for nid in order:
        gi = node2group[nid]
        if gi == prev_group:
            actions.append(ACTION_MERGE)
        else:
            actions.append(1 + group_dev[gi])
        prev_group = gi
    return actions


# ---------------------------------------------------------------------------
# 5. 组合网络：trunk + value + policy
# ---------------------------------------------------------------------------


class SearchNet(nn.Module):
    """共享 trunk + value/policy 两头；cost 头由外部代价模型提供。"""

    def __init__(self, cost_gnn, mid: int = 128, dropout: float = 0.1):
        super().__init__()
        self.trunk = TrunkAdapter(cost_gnn)
        h = self.trunk.hidden
        self.value = ValueHead(h, mid=mid, dropout=dropout)
        self.policy = PolicyHead(h, mid=mid, dropout=dropout)

    def encode(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.trunk(batch)

    def value_of(
        self,
        enc: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        meta: Optional[Dict[str, int]] = None,
    ) -> torch.Tensor:
        """单状态（B=1）value；meta 提供当前组/设备下标。"""

        B, N, h = enc["node_emb"].shape
        dev_emb = enc["device_emb"].new_zeros(B, h)
        grp_emb = enc["group_emb"].new_zeros(B, h)
        progress = enc["global_emb"].new_zeros(B, 1)
        if meta is not None:
            n_closed = int(meta.get("n_closed", 0))
            cur_dev = int(meta.get("cur_dev", -1))
            cur_group = n_closed if meta.get("can_merge") else -1
            if 0 <= cur_dev < enc["device_emb"].size(1):
                dev_emb = enc["device_emb"][:, cur_dev, :]
            if 0 <= cur_group < enc["group_emb"].size(1):
                grp_emb = enc["group_emb"][:, cur_group, :]
            total = max(int(meta.get("total_nodes", N)), 1)
            progress = torch.full(
                (B, 1), float(meta.get("next_idx", 0)) / total, device=progress.device
            )
        return self.value(
            enc["global_emb"], grp_emb, dev_emb, enc["rem_mean"], enc["rem_max"], progress
        )

    def priors(
        self,
        enc: Dict[str, torch.Tensor],
        meta: Dict[str, int],
        legal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """单状态（B=1）policy logits；返回 (logits, action_emb)。

        动作序固定为 [merge, 设备 0..D-1]：merge 非法时它会被 legal_mask 屏蔽，
        概率为 0；调用方用它算 PUCT 时，动作编号与 _expand 的顺序一一对应
        （merge 对应 _expand 的第 0 项，设备 d 对应第 1+d 项）。
        """

        B, N, h = enc["node_emb"].shape
        idx = int(meta.get("next_idx", 0))
        if idx >= N:  # 已到叶子
            return (
                torch.zeros(0, device=enc["global_emb"].device),
                torch.zeros(0, h, device=enc["global_emb"].device),
            )
        op_emb = enc["node_emb"][0, idx]
        n_closed = int(meta.get("n_closed", 0))
        cur_group = n_closed if meta.get("can_merge") else -1
        if 0 <= cur_group < enc["group_emb"].size(1):
            cur_group_emb = enc["group_emb"][0, cur_group]
        else:
            cur_group_emb = torch.zeros(h, device=op_emb.device)
        global_emb = enc["global_emb"][0]
        logits, action_emb = self.policy(
            global_emb,
            op_emb,
            cur_group_emb,
            enc["device_emb"][0],
            bool(meta.get("can_merge")),
            legal_mask=legal_mask,
            include_merge=True,
        )
        return logits, action_emb

    def value_batch(
        self,
        enc: Dict[str, torch.Tensor],
        metas: Sequence[Dict[str, int]],
    ) -> torch.Tensor:
        """批量 value（训练用）：metas 与 batch 内样本一一对应。"""

        B, N, h = enc["node_emb"].shape
        dev_emb = enc["device_emb"].new_zeros(B, h)
        grp_emb = enc["group_emb"].new_zeros(B, h)
        progress = enc["global_emb"].new_zeros(B, 1)
        for b, meta in enumerate(metas):
            cur_dev = int(meta.get("cur_dev", -1))
            if 0 <= cur_dev < enc["device_emb"].size(1):
                dev_emb[b] = enc["device_emb"][b, cur_dev]
            if meta.get("can_merge"):
                cur_group = int(meta.get("n_closed", 0))
                if 0 <= cur_group < enc["group_emb"].size(1):
                    grp_emb[b] = enc["group_emb"][b, cur_group]
            total = max(int(meta.get("total_nodes", N)), 1)
            progress[b, 0] = float(meta.get("next_idx", 0)) / total
        return self.value(
            enc["global_emb"], grp_emb, dev_emb, enc["rem_mean"], enc["rem_max"], progress
        )

    def policy_batch(
        self,
        enc: Dict[str, torch.Tensor],
        metas: Sequence[Dict[str, int]],
        masks: Optional[Sequence[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """批量 policy（训练用）：返回 (logits (B,Amax), action_mask (B,Amax))。

        每个样本的动作数可以不同（当前组为空时没有 merge 动作），这里 padding 对齐。
        """

        B, N, h = enc["node_emb"].shape
        device = enc["global_emb"].device
        n_devs = int(enc["device_emb"].size(1))
        logits_list, mask_list = [], []
        for b, meta in enumerate(metas):
            idx = int(meta.get("next_idx", 0))
            if idx >= N:
                logits_list.append(torch.full((1 + n_devs,), -1e9, device=device))
                mask_list.append(
                    torch.zeros(1 + n_devs, dtype=torch.bool, device=device)
                )
                continue
            cur_group = int(meta.get("n_closed", 0)) if meta.get("can_merge") else -1
            if 0 <= cur_group < enc["group_emb"].size(1):
                cur_group_emb = enc["group_emb"][b, cur_group]
            else:
                cur_group_emb = torch.zeros(h, device=device)
            can_merge = bool(meta.get("can_merge"))
            # 规范动作空间：长度固定 1 + n_devs（0 = merge，1+d = 设备 d）；
            # merge 非法时仅通过 mask 屏蔽，不改变动作向量的长度。
            legal = torch.ones(1 + n_devs, dtype=torch.bool, device=device)
            if not can_merge:
                legal[0] = False
            if masks is not None:
                m = masks[b].to(device).bool().reshape(-1)
                if m.numel() == 1 + n_devs:
                    legal &= m
                elif m.numel() == n_devs:
                    legal[1:] &= m  # 只给了设备段
                elif m.numel() > 1 + n_devs:
                    legal &= m[: 1 + n_devs]
            logits, _ = self.policy(
                enc["global_emb"][b],
                enc["node_emb"][b, idx],
                cur_group_emb,
                enc["device_emb"][b],
                can_merge,
                legal_mask=legal,
                include_merge=True,
            )
            logits_list.append(logits)
            mask_list.append(legal)
        amax = max((int(x.size(0)) for x in logits_list), default=0)
        logits_pad = torch.full((B, max(amax, 1)), -1e9, device=device)
        mask_pad = torch.zeros(B, max(amax, 1), dtype=torch.bool, device=device)
        for b, (lg, mk) in enumerate(zip(logits_list, mask_list)):
            if lg.numel() == 0:
                continue
            logits_pad[b, : lg.size(0)] = lg
            mask_pad[b, : mk.size(0)] = mk
        return logits_pad, mask_pad


# ---------------------------------------------------------------------------
# 6. MCTS 辅助：PUCT 打分
# ---------------------------------------------------------------------------


def puct_scores(
    q_values: Sequence[float],
    priors: Sequence[float],
    visits: Sequence[int],
    c_puct: float = 1.0,
) -> np.ndarray:
    """PUCT：Q + c * P * sqrt(sum(N)) / (1 + N)。"""

    q = np.asarray(q_values, dtype=float)
    p = np.asarray(priors, dtype=float)
    n = np.asarray(visits, dtype=float)
    return q + c_puct * p * math.sqrt(n.sum() + 1.0) / (1.0 + n)
