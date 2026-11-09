# 连续阶段放置与中等子图 Exact 对照

2026-11-09。沿用Qwen2.5-3B第0层的serial_tail、attention_fork、mlp三个既有提取模板，不构造新图、不新增GPU测量。目标仍是前向makespan。

## 动作空间

LegalFusionSearchSpace新增可选`placement_policy="contiguous_stages"`。两卡允许全GPU0、全GPU1、GPU0→GPU1、GPU1→GPU0；每张卡至多一个连续阶段。切分依据输入图固定拓扑顺序；动作同时选择融合候选及“留在当前阶段／开启下一张未用卡”，融合不是预先固定的。

非相邻融合候选跨越的拓扑区间必须留在同一阶段，不能把中间未处理节点放到另一卡后再返回。共享参数约束开启时也不能在切分后留下绑定旧卡的未来使用者。输出Plan.extra保存stage_devices和cut_indices。

旧逐节点空间仍是默认值，作为Current Joint MCTS对照。legal_mcts_search、legal_beam_search、exact_legal_search与_run_search均已传递该参数；未更改UCT、回传、代价公式或训练语义。完整3B搜索本轮没有重跑。

## 协议

- 三图分别10／7／8节点，Shape来自当前3B静态图；边界驻留沿用extract_graph的自由放置声明。不能把这些子图结论直接当成固定入口出口的整模型执行结果。
- 固定blocking HTAE、同一解析Profile＋既有LightOP校准；不新增标定，不引入async场景。
- free使用设备配置容量；cap75为单卡未融合前向峰值的75%，是合成压力约束，不是物理容量的75%或训练预算。
- Random、Sequential（先融合后放置）、Current Joint MCTS、Stage Joint MCTS均以64次不同完整方案的实际模型调用为上限，种子0／7／23。两张卡的未融合初始化均计费；不可行方案与Sequential第一阶段提案也计费；同一方案缓存命中不重复计费。所有已评分且实际可行的incumbent保留。
- Sequential第一阶段最多用总预算的一半，冻结融合划分后继续放置；两阶段共用同一缓存和计数。第一阶段可提出暂不可行的单卡划分，但不能将它伪装为最终可行方案。
- 原Joint和Stage Joint使用同一normalize_q／skip_exhausted／3 rollouts设置；差别仅为设备动作空间。Random从原逐节点空间采样。
- Exact单独完整穷举，不受64次预算约束；其标签不进入启发式搜索。另做Stage Exact，区分动作空间限制损失与MCTS搜索损失。结构化空间少于64时允许枚举完即停止，不重复评分补足64。

## 空间核算

| 子图 | 原空间 | 连续阶段空间 | 缩减 | Stage Exact相对原Exact |
| --- | ---: | ---: | ---: | --- |
| serial_tail | 1536 | 38 | 97.53% | free相同；cap75均无可行解 |
| attention_fork | 128 | 14 | 89.06% | free相同；cap75均无可行解 |
| mlp | 384 | 28 | 92.71% | free差6.5288%；cap75差14.9857% |

## 结果

单位ms，为三个种子的中位数。Exact为原逐节点联合空间最优值。

| 子图／容量 | Exact | Random | Sequential | Current Joint | Stage Joint |
| --- | ---: | ---: | ---: | ---: | ---: |
| serial_tail free | 0.110149 | 0.119483 | 0.110149 | 0.110149 | 0.110149 |
| attention_fork free | 0.102902 | 0.102902 | 0.102902 | 0.102902 | 0.102902 |
| mlp free | 0.512411 | 0.541281 | 0.512411 | 0.512411 | 0.545866 |
| mlp cap75 | 0.512411 | 0.541281 | 0.528584 | 0.541364 | 0.589200 |

另外两项serial_tail cap75、attention_fork cap75的完整Exact确认不可行，保留所有运行记录，makespan及Gap记null。

| 子图／容量 | Random Gap | Sequential Gap | Current Joint Gap | Stage Joint Gap | Stage域内Gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| serial_tail free | 8.4736% | 0 | 0 | 0 | 0 |
| attention_fork free | 0 | 0 | 0 | 0 | 0 |
| mlp free | 5.6341% | 0 | 0 | 6.5288% | 0 |
| mlp cap75 | 5.6341% | 3.1562% | 5.6503% | 14.9857% | 0 |

上表Gap同样取种子中位数，公式为`(best / Exact - 1) × 100%`。原Joint在mlp cap75最好的种子Gap为0.0145%，并非三个种子都为5.6503%；Sequential该场景有一个种子达到Exact。逐种子最优值、Gap、实际唯一候选数与跨卡边见summary.csv。

| 子图／容量 | Current Joint唯一候选／跨卡边 | Stage Joint唯一候选／跨卡边 |
| --- | --- | --- |
| serial_tail free | 64／0 | 38／0 |
| attention_fork free | 64／0 | 14／0 |
| mlp free | 64／2 | 28／0 |
| mlp cap75 | 64／2 | 28／3 |

各方法都遵守64次上限；Random、Sequential、Current Joint本次均用满64，Stage Joint完整覆盖其14–38个方案后停止。Stage Joint达到Stage Exact，但该结果不能证明大空间下的MCTS搜索效率优势。

MLP两个场景的差距来自连续阶段空间本身排除了更好的分支并行放置。跨卡边更少不自动等于makespan更小；本轮结果不支持直接用连续阶段空间替换所有逐节点空间。未为得到更好结果而放宽切分规则或改动成本。

## 入口与验证

```bash
python -B -X utf8 tools/evaluate_stage_search.py --output data/stage_search_medium_v1 --budget 64 --seeds 0,7,23
```

调用联合搜索时使用`_run_search(..., placement_policy="contiguous_stages")`，或直接传给`legal_mcts_search`。不传参数时保持原行为。

工件data/stage_search_medium_v1：report.json包含两个Exact、全部72个启发式运行、最佳方案与每次实际评分轨迹；summary.csv包含84行Oracle／运行记录；verification.json包含哈希、68项CPU测试回执和冷缓存重放结果。测试覆盖非相邻融合的独立穷举过滤核对、双向设备顺序、禁止回卡、共享参数、预算和默认路径回归。序列化后的分数、方案、计数与轨迹重放一致，排除CPU墙钟字段；无GPU测量。
