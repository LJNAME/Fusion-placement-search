"""代价模型评估指标：MAPE、RMSE(log)、Spearman、top-k 命中率。"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-12)))


def rmse_log(log_true, log_pred) -> float:
    t = np.asarray(log_true, dtype=float)
    p = np.asarray(log_pred, dtype=float)
    return float(np.sqrt(np.mean((t - p) ** 2)))


def spearman(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def evaluate_metrics(log_true, log_pred) -> Dict[str, float]:
    """在 log 空间预测上输出常用指标（MAPE 回真实空间）。"""

    t = np.asarray(log_true, dtype=float)
    p = np.asarray(log_pred, dtype=float)
    return {
        "mape": mape(np.exp(t), np.exp(p)),
        "rmse_log": rmse_log(t, p),
        "spearman": spearman(np.exp(t), np.exp(p)),
    }


def ranking_metrics(records: Iterable[Tuple[int, float, float]]) -> Dict[str, float]:
    """按 graph_id 分组的方案排名指标。

    records: (graph_id, 预测值, 真实值)，值越小越好（时间/显存）。
    返回 top-1 命中率与 top-3 平均重叠率。
    """

    by_graph: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    for gid, pred, true in records:
        by_graph[gid].append((pred, true))

    top1_hits = 0
    top3_overlap_sum = 0.0
    n_graphs = 0
    for gid, items in by_graph.items():
        if len(items) < 2:
            continue
        n_graphs += 1
        order_true = sorted(range(len(items)), key=lambda i: items[i][1])
        order_pred = sorted(range(len(items)), key=lambda i: items[i][0])
        if order_pred[0] == order_true[0]:
            top1_hits += 1
        k = min(3, len(items))
        top3_overlap_sum += len(set(order_pred[:k]) & set(order_true[:k])) / k

    return {
        "top1_acc": top1_hits / max(n_graphs, 1),
        "top3_overlap": top3_overlap_sum / max(n_graphs, 1),
        "n_graphs": float(n_graphs),
    }
