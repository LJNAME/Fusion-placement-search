"""统一代价模型接口（predict(graph, plan) -> log 空间 [makespan, 峰值显存]）。

GNN 包装器在 gnn.py（需要 torch），这里只放不依赖 torch 的模型。
"""

from __future__ import annotations

import numpy as np

from .features import compute_plan_features


class RidgeCostModel:
    """手写特征 Ridge 代价模型包装器。"""

    def __init__(self, model, hw):
        self.model = model
        self.hw = hw

    def predict(self, graph, plan) -> np.ndarray:
        f = compute_plan_features(graph, plan, self.hw)
        return self.model.predict(f[None, :])[0]
