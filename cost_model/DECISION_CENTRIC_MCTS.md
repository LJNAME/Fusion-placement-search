# Forced-action Closure：诊断、Exact验证与Qwen3消融

2026-11-10。固定已有混合放置域、成本、共享权重与容量约束，只比较单选未压缩和单选压缩的MCTS。无新GPU测量。

## P0-1：动作空间诊断

逐状态保存全Native/GPU0、全Native/GPU1两条完整路径；完整搜索另统计每次合法动作查询及按anchor分组的动作数。后者包含重复查询，不作为唯一状态数量；未穷举591节点整图的所有可能部分状态。

| 全Native单卡参考路径 | 状态数 |
| --- | ---: |
| 唯一合法动作 | 478 |
| 放置分支 | 57 |
| 融合分支 | 56 |
| 合计 | 591 |

57个放置分支为embedding所属卡及每层两个MLP分支的选卡；56个融合分支为28处AddRMS和28处SiLU×Mul。两张卡的参考路径统计一致。[逐状态记录](../data/decision_mcts_qwen3_20261110_v1/native_gpu0_action_diagnosis.json:10951)

478个单选状态都受到“embedding/lm_head共享参数同卡＋主干不可返回”的设备锁定：lm_head尚未分配时，主干不能离开它必须返回的卡。候选方面可再分为：

| 单选状态组合原因 | 数量 |
| --- | ---: |
| 当前规则仅一个单算子候选＋设备锁定 | 422 |
| 之前选择Native，融合备选因覆盖重叠失效＋设备锁定 | 56 |

本次不移除这些约束，也不引入共享权重复制假设。

## P0-2：Forced-action Closure

新增 `legal_mcts_search(..., forced_action_closure=True)`，默认False保留原行为。

- 根状态及每条展开边连续应用唯一合法动作，直到分支、终止或无合法补全；Selection只经过归一化后的树节点，终止叶保留。
- 原始choices、已分配节点与设备信息完整保留，导出的Plan不丢算子，不把不同融合或放置方案合并。
- Expansion、随机补全、融合引导补全、Native引导补全共用closure；中间状态仍检查sound prune及既有动作合法性。
- 随机补全保留原先单选时的randrange(1)消耗。同一部分状态和随机状态下，压缩开关不单独改变补全结果或后续RNG状态。
- UCT、uniform prior、Q归一化、best-completion回传、融合概率及代价模型不变。

诊断数据区分树层数、真正决策深度、已确定原图节点数、单选扩展与动作查询次数。动作查询观察器不评分。

## P0-3：小图Exact正确性

复用3B已有serial_tail、attention_fork、mlp子图，分别枚举node与hybrid动作域。原树与压缩树独立冷缓存评价，共4340次小图完整方案评分；Exact标签不传入完整图搜索。

| 子图（hybrid域） | 两组完整方案数 | 树节点数：原→压缩 | free最优预测 / ms（两组相同） |
| --- | ---: | ---: | ---: |
| serial_tail | 38 | 129→71 | 0.113643500 |
| attention_fork | 24 | 67→47 | 0.102901910 |
| mlp | 60 | 189→119 | 0.512411400 |

node域方案数1536、128、384同样保持不变。六个动作域的完整Plan集合、导出字段、free/cap75可行集合及最优方案集合均一致；所有压缩树单选节点为0。serial_tail与attention_fork的cap75原本无可行解，两组都保留该结果；MLP的node/hybrid可行数分别288/42。[Exact回执](../data/decision_mcts_qwen3_20261110_v1/exact_validation.json:2)

76项相关CPU测试通过，覆盖根状态直接终止、无解、共享权重、非相邻融合、sound prune、部分可行性过滤、随机数序列及原调用路径回归。

## P0-4：Qwen3完整图消融

591节点、两张BW1000、BF16，种子0/7/23。每组64次不同完整方案实评，含两卡Native初始化；free与cap75共12运行、768次完整图评分。cap75仍为单卡Native前向峰值的75%，未加入witness或显存下界引导。

### 无容量限制

| 指标 | 单选未压缩 | 单选压缩 |
| --- | ---: | ---: |
| 最佳预测makespan中位数 / ms | 10.123955692 | 10.118076184 |
| 真正决策最大深度 | 3 | 7 |
| 评分时树最多确定原节点数 | 18 | 39 |
| 每种子单选树扩展数 | 26 | 0 |
| 每种子分支树扩展数 | 3 | 23 |
| 最终最佳首次评分（各3种子相同） | 3 | 48 |
| 含诊断的搜索墙钟中位数 / s | 32.406 | 26.991 |

[完整汇总](../data/decision_mcts_qwen3_20261110_v1/summary.json:2)

两组第3次评分均为10.123955692ms；原始组随后停滞，压缩组第48次降至10.118076184ms。预测差5.879508μs、约0.0581%。压缩后的最佳仍全部GPU0，保留28组AddRMS，将第0层SiLU×Mul改为Native，其余27层SiLU×Mul仍融合。

墙钟包含逐动作诊断，不能与上一轮不含完整动作诊断的14～15秒直接比较。逐评分收敛数据保留全部1～64预算，不只选取改进时点：[convergence.csv](../data/decision_mcts_qwen3_20261110_v1/convergence.csv:2)。

### 容量受限

两组均0/3找到可行方案、每种子0/64可行。最大决策深度2→5，评分时树确定节点16→37，单选树扩展27→0。[容量组汇总](../data/decision_mcts_qwen3_20261110_v1/summary.json:51)

此前已有可行witness；本轮只消融closure，没有更换先验或加入可行性机制。因此保留“预算内未找到”的结果，不解释为空间无解。

## 当前结果与复现

Closure消除了当前树中的单选扩展，完整方案空间和小图Exact结果不变；在固定64次评分下覆盖了更多真实决策，得到小幅预测成本改善。容量受限可行性发现仍未解决，完整图也仍没有最优性证明或GPU执行加速证据。

六条未压缩控制的前64个候选、评分、显存和可行性与上一轮逐项一致。最佳Plan覆盖全部原节点、融合合法、共享参数同卡，导出ID与评分候选一致；核对冻结源码与预算通过，复核额外耗时模型调用为0。[工件复核](../data/decision_mcts_qwen3_20261110_v1/artifact_review.json:2)

```bash
python -B -X utf8 tools/evaluate_decision_mcts.py --output /fresh/decision_mcts_qwen3
python -B -X utf8 tools/summarize_decision_mcts.py --directory /fresh/decision_mcts_qwen3
```

工件在 `data/decision_mcts_qwen3_20261110_v1/`：两条逐状态参考路径、Exact回执、冻结计划、12条逐次搜索轨迹与最佳Plan、summary、convergence及验证回执。默认搜索开关尚未整体切换。
