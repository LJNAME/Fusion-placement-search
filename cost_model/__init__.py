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
from .legal_search import (
    LegalChoice,
    LegalFusionSearchSpace,
    LegalPartialState,
    legal_beam_search,
    legal_mcts_search,
    rollout_legal_plan,
)
from .estimate import CostEstimate, CostModel
from .hardware_profile import (
    AnalyticHardwareProfile,
    GroupCost,
    HardwareCostProfile,
    TransferCost,
    as_hardware_profile,
)
from .structure import (
    DependencyStructure,
    GroupStructure,
    PlanStructure,
    analyze_plan_structure,
)
from .structural import StructuralCostModel
from .exact_search import (
    ExactSearchLimitExceeded,
    ExactSearchResult,
    OptimalityGap,
    count_exact_search_space,
    exact_legal_search,
    measure_optimality_gap,
)

__all__ = [
    "compute_plan_features",
    "make_targets",
    "RidgeLogRegressor",
    "mape",
    "spearman",
    "evaluate_metrics",
    "ranking_metrics",
    "LegalChoice",
    "LegalFusionSearchSpace",
    "LegalPartialState",
    "legal_beam_search",
    "legal_mcts_search",
    "rollout_legal_plan",
    "CostEstimate",
    "CostModel",
    "AnalyticHardwareProfile",
    "GroupCost",
    "HardwareCostProfile",
    "TransferCost",
    "as_hardware_profile",
    "DependencyStructure",
    "GroupStructure",
    "PlanStructure",
    "analyze_plan_structure",
    "StructuralCostModel",
    "ExactSearchLimitExceeded",
    "ExactSearchResult",
    "OptimalityGap",
    "count_exact_search_space",
    "exact_legal_search",
    "measure_optimality_gap",
]
