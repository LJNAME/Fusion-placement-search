"""加载 SearchNet（cost GNN trunk + value/policy 头）。

注意：SearchNet 的 TrunkAdapter 会就地替换 cost_gnn 的第一层输入维度，
所以这里加载**两份独立的 cost 模型**：
- trunk 用的那份接上头后训练/推理；
- terminal 打分用的那份保持原样，避免 trunk 微调导致代价漂移。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from cost_model.gnn import load_gnn_model
from cost_model.search_heads import SearchNet


def load_search_net(
    cost_model_dir: str,
    heads_dir: str,
    device: str = "cpu",
):
    """返回 (net, cost_model, meta)。

    net: SearchNet（eval 模式）；cost_model: 独立的 GNNCostModel，用于终端打分。
    """

    trunk_model, meta = load_gnn_model(cost_model_dir, device="cpu")
    cost_model, _ = load_gnn_model(cost_model_dir, device=device)

    heads_path = Path(heads_dir)
    heads_meta = {}
    meta_path = heads_path / "heads_meta.json"
    if meta_path.exists():
        heads_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    net = SearchNet(
        trunk_model.model,
        mid=int(heads_meta.get("mid", 128)),
        dropout=0.0,
    )
    state = torch.load(heads_path / "heads_model.pt", map_location="cpu")
    net.load_state_dict(state)
    net.to(device).eval()
    return net, cost_model, meta
