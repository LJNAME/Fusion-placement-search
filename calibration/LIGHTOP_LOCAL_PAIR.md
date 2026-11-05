# GPU0 两候选局部验证

固定 BF16 `[1,256,3072]`、GPU0，只比较 Native SiLU→Mul 与 pack＋LightOP 完整调用。两种冻结预测都用现有 roofline 思路；L2 模式只替换 Fused 成本，其余参数相同。

| 冻结模式 | Native / μs | Fused / μs | 选择 |
| --- | ---: | ---: | --- |
| L1 完整边界解析估计 | 21.40285 | 23.74767 | Native |
| L2 替换正式融合成本 | 21.40285 | 17.81118 | Fused |

L1 的 launch 使用历史 eager 参数作为 Graph 代理假设，尚未独立标定。逻辑读写量为 Native 7.5 MiB、完整 Fused 10.5 MiB，并非实测显存流量。Fused 的 pack 和 LightOP 分别计算再加和，未把理想融合成本当成完整调用成本。

已有正式 Profile 仅用于 L2 选择。新参考批次两臂都重新测量，seed=20261105（旧批次为20261104），不以旧 Fused 数据充当新参考臂。

## 服务器执行

先更新至包含 `tools/run_lightop_local_pair.sh` 的版本。输出目录必须不存在；失败保留原目录，不自动重跑。

```bash
export FLAGSCALE_ROOT="$HOME/private_data/LJ/FlagScale-main"
export FUSION_PYTHON="$HOME/private_data/LJ/dcu-kernel310-20261031/run_python.sh"
bash tools/run_lightop_local_pair.sh ../lightop_local_pair_20261104_01 run
```

不接触 GPU 的计划预览使用相同命令，将末尾 `run` 改为 `plan`，并指定另一个新目录。

回传整个目录，包括 `process.log`、`process.json`；失败时也回传。CPU 复核：

```bash
python -X utf8 -m calibration.lightop_local_pair review \
  --input /path/to/lightop_local_pair_20261104_01 \
  --output /fresh/local_pair_review
```

## 执行边界

Native 为预分配的 `aten.silu.out → aten.mul.out`；Fused 复用现有 Adapter，计时包含 `cat(out=packed)`。输入只读，重复 Replay 无需恢复 gate。每臂50份独立中间缓冲及输出，合计显式可变缓冲约375 MiB，不含输入、参考值与 Graph pool。

每臂先50次 eager 预热、Capture、10次 Graph 预热，再正式7轮×50次。偶数轮 Native→Fused，奇数轮 Fused→Native。两臂共享输入、同一 stream 与 Event 计时语义；各自保留实现和 Recipe。缓冲填 NaN、同步、全部输出数值校验与输入字节校验位于计时区间之外。

每轮检查50个输出、BF16与FP32两种参考、只读输入和输出独立性。固定7轮，不补轮、不挑样本；IQR/Median 超过10%或 Event 计时无效，不生成参考排名。数值或执行错误立即停止，保存阶段、pool、显存快照和 traceback。没有锚点或跨窗口漂移认证，不声称已排除环境变化。

`plan.json` 与 `prediction.json` 在 GPU 准备前落盘。`identity.json` 保存两臂身份、共享测量契约哈希和各自 Recipe 哈希；复用已有 LightOP dispatch 证据，不另开 profiler。Native 未采集 dispatch trace，不将接口数量宣称为已观测 Kernel 数。

## 结果范围

`evaluation.json` 分别给出两种冻结选择、新参考选择、逐臂预测误差和局部绝对 regret（μs）。新参考不回流修改预测；没有自动 Profile 准入，不覆盖既有正式记录。

共同 Projection 成本 C 仅在加性模型下相消。局部差值不等于完整 FFN 实测收益，不报告完整 FFN 相对 regret；Native 链成本不拆成单算子 Profile。两个候选穷举不构成 MCTS、联合放置或 SROP 的效果证据。单个 Shape、单个新批次也不证明泛化或统计显著性。
## 首批回传：2026-11-04

回执 `lightop_local_pair_20261104_01`，采集版本 `04c6d8c`。冻结 CPU review 返回 `reference_ready`；两臂各7轮×50次全部保留，没有补轮或替换。IQR/Median：Native 0.5073%，Fused 4.4567%，均低于预注册10%阈值。回执中的全部输出双参考、输入字节保持及缓冲独立检查通过；CPU复核记录与冻结执行代码，未重做GPU数值计算。

| 执行路径 | 新参考中位数 / μs |
| --- | ---: |
| Native 预分配 SiLU→Mul 链 | 10.09280 |
| 预分配 pack＋LightOP 完整调用 | 18.84796 |

7/7轮中Native均低于Fused。上述差异包含pack，没有分别测pack与LightOP Kernel，不能据此定位是哪一阶段造成差异，也不能据此评价LightOP独立融合Kernel的速度。

| 测量前冻结模式 | 选择 | 新参考选择 | 局部绝对 regret / μs |
| --- | --- | --- | ---: |
| L1 完整边界解析估计 | Native | Native | 0 |
| L2 仅替换正式融合成本 | Fused | Native | 8.75516 |

Native解析值21.40285 μs，比本批10.09280 μs高112.06%；旧Fused Profile为17.81118 μs，比本批18.84796 μs低5.50%。只替换Fused成本保留了过高的Native估计，导致错误排序。这是本Case的成本替换范围问题，不表示已证明解析模型普遍优于实测校准，也没有修改原始Profile的准入结论。

原始25文件逐字节保存在 `data/lightop_local_pair_20261104_01/raw/`；冻结复核输出为 `review/evaluation.json`，来源、源码校验和独立统计复算见 `report.json`。本批为独立局部评价数据，不新增accepted Profile，不回填后改写测前选择，不改变原完整FFN成本覆盖结论。

用户随后调整顺序：先按 [代价模型选型](../cost_model/COST_MODEL_SELECTION.md) 优先适配不依赖大规模训练集的解析方法；NeuSight为后续备选，暂停继续补测和修补旧模型；完成适配可行性评估后再确定两臂成本接入。若未来把本批作为校准输入，应单独标记该用途；本批不能同时作为更新后模型的独立验证数据。