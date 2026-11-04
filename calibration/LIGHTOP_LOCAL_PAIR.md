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