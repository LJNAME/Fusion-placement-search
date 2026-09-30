"""GNN 双头代价模型（方向 1 · 模块三正式设计）。

架构：节点特征编码（op_type one-hot + log FLOPs/字节）-> GIN 消息传递
（带边特征）-> 融合组池化（组内 mean + max，拼接设备/大小/顺序特征）
-> 图嵌入 -> 双头回归。

输出：第 0 列 = log makespan（图级头）；第 1..n_devs 列 = 各设备的
log 峰值显存（逐设备头，组嵌入按设备 one-hot 聚合）。
峰值显存是"设备 × 时间重叠"的 max 型目标，必须按设备聚合才能表达。

mean+max 池化的动机：makespan 与峰值显存都是"max 型"目标（最重的那组 /
最紧张的那台设备），平均池化会稀释关键组，max 池化给模型表达这条路的通道。

依赖 PyTorch，在服务器上运行；本地测试环境没有 torch 时不导入本模块。
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn

from simulator import graph_to_dict, plan_to_dict


def _log1p(x) -> float:
    return float(np.log1p(max(float(x), 0.0)))


class GINLayer(nn.Module):
    """带边特征的 GIN 层（无需 torch_geometric，用 index_add 实现聚合）。"""

    def __init__(self, hidden: int, edge_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        self.edge_lin = nn.Linear(edge_dim, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index[0], edge_index[1]
        msg = x[col] + self.edge_lin(edge_attr)
        agg = torch.zeros_like(x)
        agg.index_add_(0, row, msg)
        return self.mlp(x + agg)


class CostGNN(nn.Module):
    """双头回归代价模型。

    node_feats: (B, N, vocab_size+3)
    group_feats: (B, G, n_devs+2) 设备 one-hot + log 组大小 + 组顺序
    """

    def __init__(
        self,
        vocab_size: int,
        n_devs: int,
        hidden: int = 128,
        num_layers: int = 3,
        edge_dim: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_devs = n_devs
        node_dim = vocab_size + 3
        self.node_enc = nn.Sequential(
            nn.Linear(node_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gin = nn.ModuleList(
            [GINLayer(hidden, edge_dim, dropout) for _ in range(num_layers)]
        )
        group_feat_dim = n_devs + 2
        self.group_mlp = nn.Sequential(
            nn.Linear(hidden * 2 + group_feat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head_mk = nn.Linear(hidden, 1)
        # 峰值显存头：额外输入每台设备的显式字节数特征（见 build_sample 的 dev_bytes）
        self.head_pk = nn.Linear(hidden + 1, 1)

    def forward(
        self,
        node_feats,
        node_mask,
        edge_index,
        edge_attr,
        group_assign,
        group_feats,
        group_mask,
        dev_bytes,
    ):
        B, N, _ = node_feats.shape
        G = group_feats.shape[1]
        x = self.node_enc(node_feats).reshape(B * N, -1)
        for layer in self.gin:
            x = layer(x, edge_index, edge_attr)
        x = x * node_mask.reshape(-1, 1)  # 屏蔽 padding 节点

        # 融合组池化：组全局 id = b*G + g（batch 内每个样本要加偏移，
        # 否则不同样本的同名组会池化到同一行，设备/组特征互相串扰）
        gids = group_assign.reshape(-1)  # (B*N,)，padding 为 -1
        batch_idx = torch.arange(B, device=x.device).repeat_interleave(N)
        gids_global = gids.clamp(min=0) + batch_idx * G
        valid = (gids >= 0).float()
        counts = torch.zeros(B * G, device=x.device).scatter_add_(0, gids_global, valid)
        pooled = torch.zeros(B * G, x.size(1), device=x.device)
        pooled.index_add_(0, gids_global, x)
        pooled_mean = pooled / counts.clamp(min=1).unsqueeze(1)
        # 组内 max 池化：makespan / 峰值显存都是 max 型目标，平均池化会
        # 稀释关键组；padding 节点用 -1e9 屏蔽，避免污染 max
        x_max = torch.where(
            node_mask.reshape(-1, 1), x, torch.full_like(x, -1e9)
        )
        pooled_max = torch.full((B * G, x.size(1)), -1e9, device=x.device)
        pooled_max.index_reduce_(0, gids_global, x_max, "amax", include_self=False)
        pooled_max = torch.where(
            (counts > 0).unsqueeze(1), pooled_max, torch.zeros_like(pooled_max)
        )
        gf = group_feats.reshape(B * G, -1)
        combined = self.group_mlp(torch.cat([pooled_mean, pooled_max, gf], dim=1))
        combined = combined * group_mask.reshape(-1, 1)
        combined_g = combined.reshape(B, G, -1)
        graph_emb = combined_g.sum(1) / group_mask.sum(1, keepdim=True).clamp(min=1)
        mk = self.head_mk(graph_emb)  # (B, 1)

        # 逐设备聚合：group_feats 前 n_devs 维是设备 one-hot，
        # 用它把组嵌入加权聚合成"每台设备的嵌入"，再输出逐设备峰值
        w = group_feats[:, :, : self.n_devs] * group_mask.unsqueeze(-1)
        denom = w.sum(1).clamp(min=1)  # (B, n_devs)
        device_emb = (w.transpose(1, 2) @ combined_g) / denom.unsqueeze(-1)
        # 峰值显存是"设备 × 时间重叠"的 max 型目标，量级由显式字节数主导：
        # 直接把每台设备的字节特征拼进设备嵌入，避免模型从 log 特征反推求和
        pk_in = torch.cat([device_emb, dev_bytes.unsqueeze(-1)], dim=-1)
        pk = self.head_pk(pk_in.reshape(B * self.n_devs, -1)).reshape(B, self.n_devs)
        return torch.cat([mk, pk], dim=1)


def build_sample(
    rec: dict,
    op_vocab: List[str],
    n_devs: int,
    include_target: bool = True,
) -> dict:
    """把一条标签记录（或 graph/plan 字典）转成 GNN 样本张量。"""

    nodes = rec["graph"]["nodes"]
    edges = rec["graph"]["edges"]
    id2idx = {nd["id"]: i for i, nd in enumerate(nodes)}

    node_feats = []
    for nd in nodes:
        oh = [0.0] * len(op_vocab)
        idx = op_vocab.index(nd["op_type"]) if nd["op_type"] in op_vocab else op_vocab.index("UNK")
        oh[idx] = 1.0
        node_feats.append(
            oh
            + [
                _log1p(nd["flops"]),
                _log1p(nd["out_bytes"]),
                _log1p(nd["param_bytes"]),
            ]
        )

    edge_src, edge_dst, edge_attr = [], [], []
    for e in edges:
        edge_src.append(id2idx[e["src"]])
        edge_dst.append(id2idx[e["dst"]])
        eb = e.get("bytes")
        if eb is None:
            eb = nodes[id2idx[e["src"]]]["out_bytes"]
        edge_attr.append([_log1p(eb)])

    groups = rec["plan"]["groups"]
    node2group = {}
    for gi, gd in enumerate(groups):
        for nid in gd["node_ids"]:
            node2group[nid] = gi
    group_assign = [node2group[nd["id"]] for nd in nodes]

    group_feats = []
    for gi, gd in enumerate(groups):
        dev = [0.0] * n_devs
        dev[gd["device"]] = 1.0
        group_feats.append(
            dev + [_log1p(len(gd["node_ids"])), gi / max(len(groups) - 1, 1)]
        )

    # 每台设备的显式字节特征：参数（常驻）+ 输出张量 + 跨设备接收缓冲。
    # 峰值显存是 max 型且量级差异大，直接给字节数比让模型从 log 特征里
    # 反推求和可靠得多，也能避免空设备（target=log1）被系统性高估。
    dev_bytes = [0.0] * n_devs
    for gd in groups:
        dev_bytes[gd["device"]] += sum(
            float(nodes[id2idx[nid]].get("out_bytes", 0.0))
            + float(nodes[id2idx[nid]].get("param_bytes", 0.0))
            for nid in gd["node_ids"]
        )
    for e in edges:
        sg = node2group[e["src"]]
        dg = node2group[e["dst"]]
        if groups[sg]["device"] != groups[dg]["device"]:
            eb = e.get("bytes")
            if eb is None:
                eb = nodes[id2idx[e["src"]]].get("out_bytes", 0.0)
            dev_bytes[groups[dg]["device"]] += float(eb or 0.0)

    sample = {
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
    if include_target:
        # 未使用设备的峰值显存是 0：log(1e-12) ≈ -27 的离群目标会把
        # 方差撑爆、饿死 pk 头。下限取 1 字节（log=0），空设备学"≈0"即可
        peak_list = [float(rec["peak_mem"][str(d)]) for d in range(n_devs)]
        sample["target"] = torch.tensor(
            [math.log(float(rec["makespan"]))]
            + [math.log(max(p, 1.0)) for p in peak_list],
            dtype=torch.float32,
        )
    return sample


class LabelGraphDataset(torch.utils.data.Dataset):
    """把标签 JSONL 记录转成 GNN 样本。"""

    def __init__(self, records: List[dict], op_vocab: List[str], n_devs: int):
        if "UNK" not in op_vocab:
            op_vocab = list(op_vocab) + ["UNK"]
        self.op_vocab = op_vocab
        self.n_devs = n_devs
        self.samples = [build_sample(r, op_vocab, n_devs) for r in records]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class GNNCostModel:
    """GNN 代价模型统一接口：predict(graph, plan) -> log [makespan, 峰值显存]。"""

    def __init__(self, model: CostGNN, vocab: List[str], n_devs: int, device="cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.vocab = vocab
        self.n_devs = n_devs

    @torch.no_grad()
    def predict(self, graph, plan) -> np.ndarray:
        rec = {
            "graph": graph_to_dict(graph),
            "plan": plan_to_dict(plan),
            "makespan": 1.0,
            "peak_mem": {"0": 1.0},
        }
        sample = build_sample(rec, self.vocab, self.n_devs, include_target=False)
        batch = collate([sample])
        batch = {k: v.to(self.device) for k, v in batch.items()}
        out = self.model(
            batch["node_feats"],
            batch["node_mask"],
            batch["edge_index"],
            batch["edge_attr"],
            batch["group_assign"],
            batch["group_feats"],
            batch["group_mask"],
            batch["dev_bytes"],
        )
        log_mk = out[0, 0].item()
        log_pk_max = out[0, 1:].max().item()
        return np.asarray([log_mk, log_pk_max], dtype=float)


def load_gnn_model(outdir, device="cpu"):
    """加载 train_gnn.py / compare_models.py 保存的 GNN 模型目录。

    兼容两种命名：train_gnn.py 用 model.pt / meta.json，
    compare_models.py 用 gnn_model.pt / gnn_meta.json。
    """

    import json
    from pathlib import Path

    outdir = Path(outdir)
    meta_path = outdir / "meta.json"
    if not meta_path.exists():
        meta_path = outdir / "gnn_meta.json"
    pt_path = outdir / "model.pt"
    if not pt_path.exists():
        pt_path = outdir / "gnn_model.pt"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model = CostGNN(
        vocab_size=len(meta["vocab"]),
        n_devs=meta["n_devs"],
        hidden=meta["hidden"],
        num_layers=meta["layers"],
    )
    model.load_state_dict(torch.load(pt_path, map_location=device))
    return GNNCostModel(model, meta["vocab"], meta["n_devs"], device=device), meta


def collate(samples: List[dict]) -> dict:
    """把一批样本 padding 成 (B, max_N/max_G) 的 batch。"""

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

    ei_parts, ea_parts, targets = [], [], []
    has_target = all("target" in s for s in samples)
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
        if has_target:
            targets.append(s["target"])

    batch = {
        "node_feats": node_feats,
        "node_mask": node_mask,
        "edge_index": torch.cat(ei_parts, dim=1),
        "edge_attr": torch.cat(ea_parts, dim=0),
        "group_assign": group_assign,
        "group_feats": group_feats,
        "group_mask": group_mask,
        "dev_bytes": dev_bytes,
    }
    if has_target:
        batch["target"] = torch.stack(targets)
    return batch
