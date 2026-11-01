# 显存可行性修复与同任务重跑

2026-11-01。5 个冻结任务、20 次 Exact、180 次 MCTS 全部重跑。10 份图/硬件输入与 ee2c9bf 的修复前基线逐字节相同，未改计时 Profile、探索系数、融合概率或 rollout 规则。

## 当前入口

_CachedScorer 接收显式 capacity 字节元组；正式主入口从硬件快照 devices[].max_mem 传入，机制测试从任务清单传入。峰值来自 CostEstimate.peak_memory，不使用 Profile 的 overflow 标志替代容量比较，也不以 profile.max_memory 隐式决定准入。

可行方案搜索分数为 makespan；超容方案为 makespan + 1e6 秒。缓存保存原始 makespan 与可行性，Sequential 切阶段不会把罚分写回物理时间。1e6 只作有限搜索引导，不充当任意输入下的全局上界；最终选择严格从已评分的可行方案中取最小原始 makespan。测试覆盖可行时间超过 1e6 秒时仍正确返回可行方案。

Sequential 第一阶段是在 device0 上提出融合划分，capacity_bound=false；第二阶段冻结划分搜索放置，capacity_bound=true。第一阶段的单卡容量不代表最终多卡放置的准入。报告保留两个阶段的容量绑定状态及最终证明范围 frozen_partition。

| 搜索结果 | 报告 |
|---|---|
| 已评分到可行方案 | 最小 makespan 的可行 plan，reason=null |
| MCTS/Beam 没有评分到可行方案 | plan=null，reason=evaluated_no_feasible |
| Exact 穷举其搜索空间后没有可行方案 | plan=null，reason=no_feasible_in_space |

Exact 的 Sequential 证明仅适用于其冻结划分，不是所有融合划分的联合无解证明。MCTS 没找到时不借用外部 Exact 结果升级 reason。本批不存在返回超容方案的情况；无计划时不导出 PlacedOpGraph、不计算质量差距或相对不可行基线的 speedup。

## 原始容量回归

serial_tail 10 节点、attention_fork 9 节点、MLP 8 节点，实际快照容量每卡 65520 MiB。每图四模式、16/64/256 次迭代、seed0/7/23，共 108 次。

- 108/108 最佳方案签名、成本与修复前一致。
- 108/108 有序评分请求中的完整方案签名逐项一致。
- 108/108 返回可行方案，达到对应 Exact 参考。
- 原始容量档没有超容候选，容量约束未成为有效限制。

此前 layer 原始容量记录也未出现已评分候选超容，旧四模式表仍是对应冻结版本的有效解析模型记录；该批容量约束未成为有效限制。人工容量档与它分开报告。

## 机制容量结果

人工容量仍固定为 MLP 参数总字节的 75% / 90%，分别每卡 13.501465 / 16.201758 MiB，不代表 BW1000 物理容量。

| 任务 | MCTS 返回可行/运行数 | MCTS 未找到 | Exact 可行参考 |
|---|---:|---:|---|
| MLP 75% 容量，四模式 | 0/36 | 36 个 evaluated_no_feasible | 四模式均无可行解 |
| MLP 90% 容量，Fusion-only | 0/9 | 9 个 evaluated_no_feasible | 固定 device0 无可行解 |
| MLP 90% 容量，其余三模式 | 27/27 | 0 | 均为 271.964004 μs |

Exact 的 5 个空结果均标 no_feasible_in_space。MCTS 的 45 个空结果均标 evaluated_no_feasible。其余 135 次返回可行计划，未出现“评分到可行但返回非法”的情况。

90% 容量档，各模式实际返回方案相对自身参考的差距，表内为三个 seed 的范围：

| 模式 | 16 次迭代 | 64 次迭代 | 256 次迭代 |
|---|---:|---:|---:|
| Placement-only | 0.00756%–10.71894% | 0–0.00756% | 0–0.00756% |
| Sequential | 0–21.21075% | 0–0.00756% | 0–0.00756% |
| Joint | 0–3.72729% | 0–0.00756% | 0–0.00756% |

这些是有效输出的搜索质量差距，不再使用修复前的诊断候选代替输出。三个模式的 Exact 参考成本相同；预算内差异不能解释成融合与放置耦合的收益。数据是固定解析模型结果，没有 GPU 执行或统计显著性结论。

## 验证与产物

- verification.json：200 次运行、23,705 条运行内去重候选、46,794 次评分请求复核通过。
- 原始 108 次结果及有序评分请求与修复前一致，10 份输入 hash 相同。
- 所有 MCTS 已评分候选的原始成本与可行性匹配对应 Joint Exact 全表；罚分不进入评价器的物理成本。
- 显式容量逐设备对比、空计划原因、最优可行返回、14 个非部署工件复核通过。
- tests.xml：60 项相关 CPU 测试通过，含 7 项新增容量回归。旧 allow_unbound 诊断测试有 34 条警告，本实验不加载这些标签。
- 新增测试覆盖有限 reward/Q、罚分与原始缓存分离、显式容量独立于 Profile、超大物理时间下的硬过滤、无缓存路径、Sequential 中间划分及空工件重放。

搜索墙钟含观测开销，不含运行后复核/文件输出，180 次合计 17.646 s。没有新增 GPU 数据、标签升级、SROP 或 Beam 实验。

```bash
python tools/evaluate_mcts_medium.py prepare data/mcts_medium_capacity_replay
python tools/evaluate_mcts_medium.py run data/mcts_medium_capacity_replay
python tools/evaluate_mcts_medium.py verify data/mcts_medium_capacity_replay
```

当前批复核：verify data/mcts_medium_capacity_v2。源码 hash 固定在 plan.json；后续源码变化需用冻结版本复验，不能改旧 hash。正式主报告版本为 1.3，空结果 plan=null、candidates=[]、best_placed_op_graph=null。

## 修复前基线

 ee2c9bf / data/mcts_medium_v1 只保留为缺陷复现和本次回归比较输入，不再作为当前搜索质量结果引用。旧文件和旧 hash 原样保留。SROP 继续冻结。
