"""数据集划分与特征矩阵构建（按图划分，防止数据泄漏）。"""

from __future__ import annotations

import json
from typing import List, Tuple

import numpy as np

from simulator import graph_from_dict, plan_from_dict

from .features import compute_plan_features, make_targets


def load_records(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_by_graph(
    records: List[dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 0,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """按 graph_id 划分训练/验证/测试（同一张图的所有方案进同一集合）。"""

    unique = sorted({r["graph_id"] for r in records})
    rng = np.random.RandomState(seed)
    rng.shuffle(unique)
    n_train = max(1, int(len(unique) * train_ratio))
    n_val = max(1, int(len(unique) * val_ratio))
    train_ids = set(unique[:n_train])
    val_ids = set(unique[n_train : n_train + n_val])
    test_ids = set(unique[n_train + n_val :])

    def pick(ids) -> List[dict]:
        return [r for r in records if r["graph_id"] in ids]

    return pick(train_ids), pick(val_ids), pick(test_ids)


def build_feature_arrays(
    records: List[dict], hw
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构建手写特征矩阵 X、对数目标 y、graph_id 数组。"""

    X, y, gids = [], [], []
    for r in records:
        g = graph_from_dict(r["graph"])
        p = plan_from_dict(r["plan"])
        X.append(compute_plan_features(g, p, hw))
        y.append(make_targets(r))
        gids.append(r["graph_id"])
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float), np.asarray(gids, dtype=int)


def records_by_graph(records: List[dict]) -> dict:
    """按 graph_id 去重，返回每个图的一条记录（供搜索评估取图）。"""

    by = {}
    for r in records:
        by.setdefault(r["graph_id"], r)
    return by
