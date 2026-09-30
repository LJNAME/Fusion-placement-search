# Fusion Placement Search

自动并行策略搜索项目（融合 + 放置）。对应 Obsidian《方向 1》的 Agentic 方案，
当前进度：仿真器 → 代价模型训练数据 → 代价模型。

## 目录结构

```
configs/           硬件参数配置（异构四卡，标称 × 效率系数，待实测校准）
simulator/         仿真器（roofline + 事件队列 + 显存生命周期 + 方案生成 + 打标签）
cost_model/        代价模型（手写特征基线 + GNN 双头回归 + 训练脚本）
data/              生成的数据集与模型产物（运行时创建）
```

## 流水线

```
合成算子图 (simulator/graphs.py)
   │ 随机 + 启发式方案 (generators.py)
   ▼
仿真器打标签 (dataset.py) ──> data/labels.jsonl   ← 当前的 ground truth
   │
   ├─ 手写特征基线 (train_baseline.py，纯 numpy，本地可跑)
   └─ GNN 双头回归 (train_gnn.py，PyTorch，服务器跑)
   │
   ▼
三模型统一对比 (compare_models.py) + 端到端搜索评估 (eval_search.py)
```

## 常用命令

```bash
# 1. 生成训练数据（默认 80 图 × 12 方案，写 data/labels.jsonl + data/hw.json）
python cost_model/make_dataset.py \
    --out data/labels.jsonl \
    --extrap-out data/labels_extrapolation.jsonl

# 2. 训练基线代价模型（本地即可）
python cost_model/train_baseline.py --labels data/labels.jsonl --hw data/hw.json

# 3. 训练 GNN 代价模型（服务器，需 torch）
python cost_model/train_gnn.py --labels data/labels.jsonl --outdir data/gnn_model

# 4. 三模型统一对比（roofline / Ridge / GNN，同一划分同一 seed；无 torch 时自动跳过 GNN）
python cost_model/compare_models.py \
    --labels data/labels.jsonl \
    --extrap-labels data/labels_extrapolation.jsonl \
    --hw data/hw.json \
    --outdir data/compare

# 5. 端到端搜索评估（代价模型驱动 beam 搜索，最终方案由仿真器精评）
python cost_model/eval_search.py \
    --hw data/hw.json \
    --extrap-labels data/labels_extrapolation.jsonl \
    --ridge-model data/compare/ridge_model.json \
    --gnn-model data/compare \
    --out data/search_results.json

# 测试
python -m unittest discover -s simulator/tests
python -m unittest discover -s cost_model/tests
```

## 当前说明

- 硬件参数用"官方标称 × 效率系数"替代实测（见 `configs/README.md`），
  拿到真实国产卡后用微基准校准，替换配置即可，代码不用改；
- 代价模型的训练标签全部来自仿真器，等真机数据后可微调/重训；
- GNN 代码（`cost_model/gnn.py`）需要 PyTorch，本地没有安装不影响写代码，
  部署到服务器后按 `requirements.txt` 装依赖。

## 评估协议（防泄漏）

- 数据按 `graph_id` 划分训练/验证/测试（同一张图的所有方案只进一个集合）；
- `--extrap-out` 生成**外推测试集**：结构类型相同但尺寸显著更大
  （更长链 / 更大 Transformer），专门检验结构泛化；
- 三模型（roofline 解析式 / Ridge 手写特征 / GNN）同划分、同 seed 对比，
  指标分两层：预测精度（MAPE / RMSE(log) / Spearman）与排名质量
  （每个测试图内 top-1 / top-3）；
- 端到端评估用 beam 搜索 + 代价模型打分，**最终方案一律由仿真器精评**，
  比较各模型的最终 makespan 与搜索耗时。
