# BW1000 新 Shape 与候选图实测（2026-10-28）

当前状态：预测和候选已冻结，GPU 结果待采集。本地 212 项检查验证准备流程和小型参考执行逻辑；它们不是本轮 GPU 性能结果。

## 服务器执行

分配 **2 张 BW1000**，使用之前成功运行校准脚本的 PyTorch 2.9.0 / HIP 6.3.26093 环境。脚本只依赖原有 Torch/NumPy，不导入 FlagGems、FlagScale 或 Megatron。

```bash
cd ~/private_data/LJ/Fusion-placement-search
git pull
bash tools/run_gpu_holdout.sh
```

脚本先校验冻结 ID、源码和训练/硬件文件，再做双卡算子采集、零样本评价和候选图执行。无 HIP/CUDA GPU 时会报错，不会回退到 CPU。默认物理设备为 `0,1`。

运行中断后，在相同环境、相同目录重新执行同一命令即可续跑。不要在回传数据前改预测器、清单、候选、阈值或覆盖冻结包。旧实验数据位于 `data/bw1000_e2/`，本轮结果写到独立目录。

## 冻结清单

冻结 ID：`sha256:59a32c2eb827b95a395bc95f31e057716b52bfd44f025f913ef60c7c897829b2`。

| 用例 | 单层节点数 | 预测候选池 | GPU 执行候选 |
|---|---:|---:|---:|
| Qwen3-0.6B，seq1536 | 21 | 33 | 6 |
| Qwen3-0.6B，seq2048 | 21 | 33 | 6 |
| Qwen2.5-3B，seq1536 | 19 | 30 | 5 |
| Qwen2.5-3B，seq2048 | 19 | 30 | 5 |

micro batch=1、BF16。清单共 64 个完整 Kernel Key，其中 56 个可测、8 个逻辑节点跳过。56 个可测对象与已使用的 112 个训练标签按完整 Key 无交集；是否在仓库外接触过相同标签未核实，不能据此宣称绝对盲测。


预采集特征核对：52 个对象的当前预测特征与训练集不同；另 4 个完整 Key 在“12 个既定数值特征＋算子/实现”上与旧样本完全一致，但完整 Shape 不同（2 个 RoPE、1 个 add、1 个 RMSNorm）。这是模型表示的别名，不等于物理 kernel 相同，也不等于旧标签重复。`feature_overlap_audit.json` 保留对象映射。结果保留全部 56 项，回传后单独检查这 4 项，不据此改候选、删对象或调整模型。

训练集合固定为原 112 条记录，预测仍是 nearest-shape 均值和只用训练标签校准的 distance-floor 区间。候选来自冻结预测器的 MCTS（seed 0/1/7/23，各 64 次、rollouts=3）和不依赖时延标签的结构探针。新 Shape 标签及 Oracle 搜索结果没有进入候选生成。

冻结文件：`data/bw1000_gpu_holdout_v1/manifest.json`、`freeze.json`。只检查冻结包可运行：

```bash
python -m calibration.gpu_holdout check
```

这个 `check` 是 CPU 可运行的准备检查，会打印 `GPU measurements and execution: PENDING`，不能当作 GPU 实测通过。

## 实际采集内容

1. **新算子性能**：每个对象分别在 GPU0、GPU1 采一次，预计 112 条成功记录。每条预热 10 次，7 组、每组 50 次 Event 计时，另存 profiling wall time。正式评价拒绝 quick/改动后的计时配方。
2. **零样本预测**：采集后读取已封存的预测，比较 Analytic 与 nearest 的 MAPE、中位误差、log MAE、排序及区间覆盖。新数据不参与重训、归一化或候选修改。
3. **真实候选执行**：4 个单层用例共 22 个候选，每个 2 轮、每轮预热 5 次、计时 20 次，共 44 轮。包括预测 Top-4、公开单卡参考和半图跨卡切分，重复方案去重。
4. **数值与运行代价**：输入和权重先在 CPU 按固定 seed 生成，复制到实际运行 GPU，所有放置使用同一数值；相对单卡参考检查输出，记录双卡同步前向墙钟、显存峰值和当前 P2P 可见性。

单算子清单中最大估计工作集为 1,253,048,320 bytes（估计，不是执行峰值）。图执行的峰值由报告给出。

## 验收与失败记录

- 正式覆盖：每个对象恰好两条成功记录，物理设备为 0 和 1，配方一致；稳健样本 CV≤0.1 为质量目标。报告保留所有对象及失败尝试，不按误差好坏删样本。
- 正确性：输出有限且满足 `rtol=0.01, atol=0.001`。这是一开始固定的工程阈值，尚未在本轮 GPU 上证实；失败先检查张量、布局和实现，不事后放宽阈值算通过。
- 排序目标：预测 Top-1 的实测时间不超过**本轮已测候选中最优值**的 1.05 倍。另报相对单卡速度比、实测/预测时间比和候选排序；不代表整个搜索空间的真实最优。
- 性能目标失败时仍生成报告。正确性失败或运行错误会返回非零，已有原始记录保留。不能只根据命令退出成功就声称预测或性能目标达成。

候选执行采用已有 eager 参考后端，包含 Python 调度、张量转换和跨卡复制；它并不按代价模型的抽象调度时间线显式实现所有并行重叠。尤其算子校准的 QK/AV 是预建输入的 `torch.bmm`，图执行还包含 GQA repeat/permute（`calibration/profile_e2_bw1000.py:86`、`execution/reference.py:180`）。两种计时不能直接按整体比例相互校准。

本轮验证新序列 Shape 的泛化和单层可执行放置，不验证完整训练、真实融合 kernel 或 SROP 在线主动测量收益。通信仍用原校准 profile 作预测，当前 P2P 状态只记录、不根据实测重调冻结候选。

## 回传结果

目录：`data/bw1000_gpu_holdout_v1/results/`。

先回传：

- `zero_shot_report.json`
- `execution_report.json`

原始文件先保留：

- `raw_measurements.jsonl`、`environments.json`
- `execution_samples.jsonl`、`execution_environment.json`

若报错，回传终端 traceback 和已生成的 `execution_report.json`；不要覆盖已有记录重做一份有利结果。

## 分步执行

```bash
python -m calibration.profile_e2_bw1000 \
  --manifest data/bw1000_gpu_holdout_v1/manifest.json \
  --out-dir data/bw1000_gpu_holdout_v1/results \
  --devices 0,1 --assignment replicate

python -m calibration.evaluate_gpu_holdout \
  --raw data/bw1000_gpu_holdout_v1/results/raw_measurements.jsonl \
  --environments data/bw1000_gpu_holdout_v1/results/environments.json \
  --output data/bw1000_gpu_holdout_v1/results/zero_shot_report.json

python -m execution.gpu_holdout \
  --devices 0,1 --out-dir data/bw1000_gpu_holdout_v1/results
```
