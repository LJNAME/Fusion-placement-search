"""代价模型（方向 1 · 模块三）。

- features.py     手写特征基线（不依赖 torch，本地可跑）
- baseline.py     Ridge 线性回归基线
- metrics.py      评估指标（MAPE / RMSE(log) / Spearman / top-k 命中率）
- gnn.py          GNN 双头回归（PyTorch，服务器运行）
- make_dataset.py 合成图 + 仿真器打标签
- train_baseline.py / train_gnn.py 训练入口
"""

from .features import compute_plan_features, make_targets
from .baseline import RidgeLogRegressor
from .metrics import mape, spearman, evaluate_metrics, ranking_metrics

__all__ = [
    "compute_plan_features",
    "make_targets",
    "RidgeLogRegressor",
    "mape",
    "spearman",
    "evaluate_metrics",
    "ranking_metrics",
]
