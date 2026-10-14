# Fusion Placement Search

面向异构训练的融合与放置策略搜索项目。当前目标是把本项目迁移到 **Hygon BW1000**，并接入 FlagScale 异构 AutoTuner。

FlagScale 本地基线：

- 仓库：`E:/Chatgpt_work/FlagScale-main/FlagScale-main`
- commit：`f8e9a62c499d526a6ffa47f533bcf342330b7326`
- 日期：2026-09-28
- 版本字段：`1.0.0`

## 当前进度

已经完成：

- 算子图、融合组、设备放置方案的数据结构；
- 事件驱动仿真器、显存生命周期和通信开销建模；
- Roofline、Ridge、GNN 代价模型及 Beam/MCTS 搜索；
- 海光历史标定数据与 A6000 + TITAN 两卡异构实验；
- FlagScale 上下文适配层 `integration/flagscale/context.py`；
- 两卡层切分扫描与 FlagScale 异构配置转换；
- BW1000 上的 FlagScale 单卡训练、四卡多 mesh 分组消融。

当前主线：

1. 把仓库、依赖和实验配置迁移到 BW1000；
2. 在 FlagScale 异构搜索的层切分阶段之后、训练参数阶段之前插入本项目候选；
3. 对每份 `layer_split` 生成新的 `tp/cp/ep/dp`，保持 mesh、卡数与 `pp` 不变；
4. 将本项目候选与 FlagScale 原生候选一起交给上游剪枝和真机评估。

尚未完成：

- `FusionGroup.device` 到 mesh/rank 感知 `Placement` 的兼容升级；
- Transformer Layer 到 `OpGraph` 的真实 Graph Builder；
- AutoTuner 候选注入；
- plan 解释器与真实融合 kernel；
- BW1000 上的本项目标定、搜索和端到端插件验证。

## 目录结构

```text
configs/                 硬件参数与异构配置
simulator/               图、方案、硬件模型和事件驱动仿真器
cost_model/              Roofline/Ridge/GNN、Beam/MCTS 与层切分搜索
calibration/             微基准、真机执行器和误差归因
integration/flagscale/   FlagScale/Megatron 适配层
integration/tests/       FlagScale 适配层测试
data/                    数据集、模型与实验输出
```

只有 `integration/flagscale/` 可以直接依赖 FlagScale 或 Megatron；其余模块保持独立。

## 已知限制

- 当前 `FusionGroup` 仍使用局部整数设备编号，尚不能完整表达 mesh、逻辑 rank 与物理 rank。
- 仿真器中的融合组边界 IO 语义尚未由真实融合 kernel 实现。
- 现有离线 HIP 执行器仍是逐算子 launch，不能作为融合收益的最终验证。
- 毫秒级验证集不能覆盖秒级搜索方案的框架和流水线开销。
- `configs/hygon_k100_4cards.calibrated.json` 与 A6000/TITAN 数据是历史实验资产，不代表当前 BW1000 主实验口径。
- BW1000 的正式硬件配置、节点数和卡数在迁移后重新记录。

## 安装

```bash
python -m pip install -r requirements.txt
```

不运行 GNN 时，仿真器与解析式/手工特征模型可在不加载 PyTorch 的路径下使用。

## 测试

```bash
python -m pytest integration/tests/test_context.py -q
python -m unittest discover -s simulator/tests
python -m unittest discover -s cost_model/tests
python -m compileall -q integration cost_model calibration simulator
```

当前本地基线：Context 7/7、Simulator 3/3、Cost Model 4/5（1 项因可选依赖跳过），Python 编译检查通过。

## 常用入口

```bash
# 生成仿真标签
python cost_model/make_dataset.py --out data/labels.jsonl --extrap-out data/labels_extrapolation.jsonl

# 训练手工特征基线
python cost_model/train_baseline.py --labels data/labels.jsonl --hw data/hw.json

# 三模型对比
python cost_model/compare_models.py --labels data/labels.jsonl --extrap-labels data/labels_extrapolation.jsonl --hw data/hw.json --outdir data/compare

# 两卡层切分扫描（历史 A6000 + TITAN 实验）
python cost_model/sweep_layer_split.py --hw configs/hetero_a6000_titan.json --layers 28 --d-model 1024 --seq-len 256 --batch 2 --ffn-dim 3072 --out data/a6000_titan/sweep.json
```

## 评估口径

- 数据集按 `graph_id` 划分，避免同一图的不同方案跨训练集与测试集；
- 预测指标使用 MAPE、log-RMSE、Spearman 和图内 top-k；
- 搜索结果必须同时报告代价模型预测、仿真结果和真机结果；
- FlagScale 接入后的最终排序以真机稳态迭代耗时为准。
