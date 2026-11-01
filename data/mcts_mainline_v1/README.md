# MCTS 主线：固定解析 Profile

2026-11-01；基线源码 cf6ce5c；4 次 small Exact、60 次 MCTS，未修改搜索算法或 Profile。未调用 Beam、SROP 或 GPU。源文件与硬件快照 SHA256 见 plan.json。

## 运行口径

- small：5 节点；四模式 Exact；MCTS 16 次迭代，seed=0/7/23，共 12 次。
- layer：21 节点；先运行四模式 × 1024 次迭代 × 3 seed，冻结已发现参考；再运行四模式 × 16/64/256 × 3 seed，共 48 次。
- rollouts=3、fusion_probability=0.7、guided_rollout=true、value_mix=0、top_k=4。Sequential 总迭代对半分，两个阶段使用同一 seed。
- 每次运行独立 scorer/cache。仅运行内部复用评分；观测逻辑不改变代价或搜索策略。
- observed_search_wall_s 包含观测记录开销，不含运行后校验和文件写入。60 次 MCTS 合计 37.267 s，不能当作未插桩性能基准。
- 固定解析开发模型，不加载 raw 标签、不使用 allow-unbound。planned 融合仍非正式能力，全部输出 deployable=false。

## small 参考差距

表中 MCTS 为三个 seed 的共同结果。相对差距分母是对应参考成本。

| 模式 | Exact 参考 ms | MCTS ms | 模式内最优差距 | Joint Exact 差距 |
|---|---:|---:|---:|---:|
| Placement-only | 0.108872 | 0.108872 | 0 | +10.2426% |
| Fusion-only | 0.098756 | 0.098756 | 0 | 0 |
| Sequential | 0.098756 | 0.098756 | 不适用；两阶段参考差距 0 | 0 |
| Joint | 0.098756 | 0.098756 | 0 | 0 |

Placement-only/ Fusion-only/ Joint 空间分别为 32/2/48。Sequential 只穷举先融合、后冻结划分搜放置的两个受限阶段，不是全局最优参考。全部 small MCTS 已评分候选的签名、成本逐一匹配 Joint Exact 的 48 项表。small 不追加预算。

## layer 预算与 seed

每格为 seed=0/7/23 的共同最佳成本；同一模式的三个 seed 也返回相同最佳计划。

| 模式 | 16 次 ms | 64 次 ms | 256 次 ms | 1024 次参考 ms |
|---|---:|---:|---:|---:|
| Placement-only | 0.316920 | 0.316920 | 0.316920 | 0.316920 |
| Fusion-only | 0.306805 | 0.306805 | 0.306805 | 0.306805 |
| Sequential | 0.306805 | 0.306805 | 0.306805 | 0.306805 |
| Joint | 0.306805 | 0.306805 | 0.306805 | 0.306805 |

融合三模式全部返回同一 device0 计划。冻结参考与事后已发现最佳均为 0.30680488599502565 ms；Placement-only 相对该参考差距约 3.30%。这两种 layer 参考都不是最优性证明。若网格发现更低成本，冻结参考差距允许为负。

60/60 次 MCTS 的最终最佳成本在第一次评分请求即已出现。当前首个补全沿用现有 guided rollout；随后迭代未改进最佳值。已有记录不能把改善归因于后续树搜索，也未显示 Joint 超过 Fusion-only 或 Sequential。

## 实际评分范围

下表只计 layer Joint，范围为三个 seed 的最小/最大值；融合划分数均为 2。

| 总迭代 | 不重复完整候选/次 | 跨设备候选/次 | 观测搜索墙钟 s |
|---|---:|---:|---:|
| 16 | 41 | 40 | 0.053–0.058 |
| 64 | 168 | 167 | 0.222–0.228 |
| 256 | 651 | 650 | 0.954–1.028 |
| 1024 | 2623–2626 | 2622–2625 | 2.995–3.733 |

Sequential 放置阶段每次只评分一个冻结融合划分。Fusion-only 始终只有 2 个不重复候选。所有已评分方案显存可行。

跨 64 次运行合计 26,442 条运行内去重候选记录、38,230 次评分请求，包含 4 次 Exact。运行之间未去重。layer Joint 合并去重后为 6,710 项；合法完整空间计数为 3,145,728，已评分比例约 0.2133%。空间大小由计数算法获得，未进行 layer 全空间代价穷举。

这些是完整候选评分记录，包含确定性/随机补全和终态，不是部分树节点访问记录。不能据此声称树覆盖充分、全局最优、同构图普遍没有放置价值，或 MCTS 一般无效。当前图与固定模型下未观察到增加预算的最佳值收益。

## 文件

| 文件 | 内容 |
|---|---|
| plan.json | 运行前冻结的阶段、配置和源码/硬件 hash |
| runs.jsonl | 64 次运行、阶段预算及观测计数 |
| results.csv | 60 次 MCTS 的成本、差距与开销 |
| evaluated_candidates.csv.gz | 每次运行去重后的完整候选签名、划分、设备、成本 |
| score_requests.csv.gz | 有序评分请求到候选行号的映射 |
| small_exact_references.json | small 模式内 Exact / 两阶段参考 |
| layer_frozen_reference.json | 网格之前冻结的已发现参考 |
| layer_discovered_reference.json | 全部 layer 输出的事后已发现最佳 |
| best_plans.json、best_plans/ | 4 个不同最佳计划及 PlacedOpGraph |
| verification.json | 离线重新计算计数、参考差距、工件和 hash |
| tests.xml | 49 项相关 CPU 测试，无失败或跳过 |

## 重放

仓库根目录，使用新的输出目录（已存在则拒绝覆盖）：

```bash
python tools/evaluate_mcts_mainline.py --out-dir data/mcts_mainline_replay
python tools/verify_mcts_mainline.py data/mcts_mainline_replay
```

校验本批：

```bash
python tools/verify_mcts_mainline.py data/mcts_mainline_v1
```

plan.json 中保存本机硬件绝对路径；跨机器校验需保留该路径或在新目录重跑。代码/hash 校验与逻辑计数均通过不等于 GPU 性能验证；没有标签升级或下游契约确认。

后续不继续加大本图预算。先核对可穷举中等图的搜索差距与真实约束结构，再决定是否需要树级访问诊断或算法修改；不能通过调整模型参数制造 Joint 优势。
