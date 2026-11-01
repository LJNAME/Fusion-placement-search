# 中等子图 Exact 与 MCTS 对照

2026-11-01，基线 3894f2b。20 次 Exact、180 次 MCTS。搜索算法、评分和 Profile 计时常数保持不变；两档容量限制只用于机制测试。无 Beam、SROP、GPU 或完整方案执行。

## 预先确定的任务

| 任务 | 节点 | 计算段 | Joint 完整空间 | 每卡容量 MiB |
|---|---:|---|---:|---:|
| serial_tail_actual_capacity | 10 | QK→Softmax→AV→OutProj→残差→Norm，含 4 个边界输入 | 1024 | 65520 |
| attention_fork_actual_capacity | 9 | Q/K 投影、Norm、RoPE 两支汇入 QK/Softmax，含 1 个边界输入 | 512 | 65520 |
| mlp_actual_capacity | 8 | Norm、gate/up、SiLU、Mul、DownProj、残差，含 1 个边界输入 | 384 | 65520 |
| mlp_parameter_cap_75 | 8 | 同一 MLP | 384 | 13.5015 |
| mlp_parameter_cap_90 | 8 | 同一 MLP | 384 | 16.2018 |

原始容量来自 BW 双卡快照。两档人工容量固定为本子图参数总字节的 75% / 90%，不代表 BW1000 物理显存。全部任务保留，未按 Exact 结果筛选或再调整容量。首两类只有 singleton 划分；MLP 有 2 种融合划分。

内部算子、内部边、形状与参数复制自当前 Qwen3 layer Builder。外部生产者替换为零计算、零参数的逻辑输入，保留输出尺寸与原 tensor 边；输入位置也在放置空间中搜索。外部调度和输出消费者未执行，终端输出保留至子图结束。该边界契约是开发夹具，不能外推为完整层或真实运行时间。

运行前 plan.json 保存图/硬件快照、源文件 hash、完整空间计数与阶段顺序。先做全部 20 次 Exact、保存 references.json，再运行四模式 × 16/64/256 × seed0/7/23。Sequential 迭代对半；rollouts=3、fusion_probability=.7、guided_rollout=true、value_mix=0、top_k=4。每次 scorer/cache 独立。

## Exact 可行参考

单位为固定解析模型预测 μs，括号内为完整可行候选数。Sequential 只统计其冻结划分后的放置阶段，不是全局最优参考。

| 任务 | Placement-only | Fusion-only | Sequential | Joint |
|---|---:|---:|---:|---:|
| serial_tail 原始容量 | 102.9562 (1024) | 102.9562 (1) | 102.9562 (1024) | 102.9562 (1024) |
| attention_fork 原始容量 | 109.9744 (512) | 109.9744 (1) | 109.9744 (512) | 109.9744 (512) |
| MLP 原始容量 | 126.3681 (256) | 116.2528 (2) | 116.2528 (128) | 116.2528 (384) |
| MLP 75% 容量 | 无可行解 (0) | 无可行解 (0) | 无可行解 (0) | 无可行解 (0) |
| MLP 90% 容量 | 271.9640 (96) | 无可行解 (0) | 271.9640 (48) | 271.9640 (144) |

复用主入口原始 makespan 评分和 Sequential 流程。从完整评分记录中独立筛选显存可行最小值，未把原始排名前 4 项直接当成约束参考。Sequential 第一阶段沿用忽略容量的评分选择划分，再穷举其放置；不另选一个对最终结果有利的划分。

75% 容量的 Joint 全部 384 项均不可行。90% 容量的 Fusion-only 固定 device0，不可行；其余模式存在可行方案。这是当前显存模型下的判定，不是设备实测 OOM 结论。

## MCTS 输出与初始化

每种模式每任务 9 次 MCTS。差距只在输出可行、且存在参考时计算。

| 任务 | 返回可行方案/运行数 | 实际输出参考差距 | 相对首个方案的改善 |
|---|---:|---|---|
| serial_tail 原始容量 | 36/36 | 四模式均 0 | 0 |
| attention_fork 原始容量 | 36/36 | 四模式均 0 | 0 |
| MLP 原始容量 | 36/36 | 四模式均 0 | 0 |
| MLP 75% 容量 | 0/36 | 留空；Exact 无可行解 | 不适用 |
| MLP 90% 容量 | 0/36 | 留空；返回方案超容量 | 不适用 |

原始容量的 108/108 次运行，首个最终阶段评分已经达到对应 Exact 参考。Sequential 的初始化单列为放置阶段首个评分，不拿其融合阶段的第一个方案混算。

90% 容量档：Placement-only、Sequential、Joint 的 27/27 次运行都评分过可行方案，实际返回却全部不可行。返回方案 peak=23.001953 MiB，超过每卡 16.201758 MiB 上限。Fusion-only 的 9 次在该模式内没有可行候选。原始 makespan 约 116–126 μs 的超容量方案不能与 271.964 μs 的可行参考比较质量。

## 已评分可行候选的诊断差距

只列 MLP 90% 容量；范围为三个 seed 的最小/最大相对差距。以下不是实际返回质量，也未自动替换 MCTS 输出。

| 模式 | 16 次迭代 | 64 次迭代 | 256 次迭代 |
|---|---:|---:|---:|
| Placement-only | 0.00756%–10.7313% | 0.00756%–10.7189% | 0 |
| Sequential | 0.00756%–21.2090% | 0.00756% | 0 |
| Joint | 0–0.00756% | 0 | 0 |

不能用这张表声称 Joint 搜索更好：当前引擎仍按包含非法方案的 makespan 回传和选优。它只能说明可行候选已出现，输出失败不能归因于完全没有探索到可行分布。

## 显存入口缺口

StructuralCostModel 计算 peak_memory 和 overflow_devices（cost_model/structural.py:280、321）。主评分器 _CachedScorer 仅缓存/返回 makespan（cost_model/search_evaluation.py:219–232）。主入口给 Exact 的同样是 float；exact_search._score_value 只有收到 CostEstimate 时才自动识别 overflow（cost_model/exact_search.py:98–105）。

本轮保留原行为并固化其失败。下一项应统一搜索可行性处理与无可行候选的结果语义，再跑相同冻结任务。不能只在最终导出时隐藏非法方案，也不能把 Sequential 的中间融合划分与最终可执行放置混为同一约束阶段。探索系数、奖励尺度和 rollout 优化暂不推进。

## 数据与复验

- inputs/：5 份图、5 份硬件快照；原始图定义与计时常数未改。
- references.json：20 次 Exact 的可行参考与计数；冻结在 MCTS 之前。
- candidates.csv.gz / requests.csv.gz：完整候选评分及有序请求；不是树节点访问。
- runs.jsonl / results.csv：原始输出方案、可行性、阶段预算、初始化和诊断差距。
- artifacts.json / best_plans/：6 个可行参考或实际可行输出工件，全部 deployable=false。不导出超容量赢家。
- verification.json：200 次运行、25,414 条运行内去重候选、48,760 次评分请求复核通过；所有 MCTS 候选成本和可行性均匹配对应 Joint Exact 全表。
- tests.xml：53 项相关 CPU 测试通过。旧测量诊断测试有 34 条预期 allow_unbound 警告；本实验未加载这些标签。

观测搜索墙钟包含插桩、计划记录开销，不含运行后检查/文件输出；180 次 MCTS 合计 18.109 s，不作未插桩引擎基准。没有独立测试或统计显著性结论。

```bash
python tools/evaluate_mcts_medium.py prepare data/mcts_medium_replay
python tools/evaluate_mcts_medium.py run data/mcts_medium_replay
python tools/evaluate_mcts_medium.py verify data/mcts_medium_replay
```

prepare 只生成任务与协议，不评分；已存在目录拒绝覆盖。当前批复验用 verify data/mcts_medium_v1。plan.json 保存本机源码与硬件绝对路径信息，跨机器应使用新目录 prepare/run。
