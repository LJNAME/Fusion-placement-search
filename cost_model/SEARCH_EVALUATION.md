# 主系统四模式开发评价

search_evaluation报告v1.3包含placement_only、fusion_only、sequential、joint。SROP验证保持冻结；本入口不导入srop_e0，不调用主动校准。默认标签准入闸门保留；--allow-unbound只用于明确声明的旧数据开发诊断。

## 固定Profile模式

--profile analytic只读硬件快照，不读raw标签，不需要allow-unbound；逻辑input按零时间处理。--profile measured仍为默认，保持accepted+契约一致的严格准入。--allow-unbound只用于显式旧标签诊断，解析模式会拒绝该参数。计费表、候选覆盖及链路/embedding边界见FIXED_PROFILE_SEMANTICS.md。

```bash
python -m cost_model.search_evaluation --profile analytic --scales small,layer \
  --mcts-iterations 16 --seed 23 --output data/analytic_run/search.json
```
## Sequential

第一阶段在device0固定放置下搜索融合，只取排名第一的划分；第二阶段把该划分收缩成macro图，仅搜索每个组的设备。所有评分都展开回原图，使用同一个StructuralCostModel；不能用macro图的估计代替原图代价。

第二阶段复制第一阶段模板，保留各组node_ids、pattern、backend及workspace等属性，只改变device。运行时检查最终划分与模板完全一致。并列方案沿用底层确定性排序，不事后挑有利划分。

MCTS总次数保持不变：stage1=floor(total/2)，stage2=total−stage1；16为8+8，17为8+9，至少需要2次。两个阶段均使用报告中的同一seed。Beam两个阶段分别使用同一宽度，宽度不是可相加的计算预算；报告每阶段请求数、唯一方案数、实际模型评估数和墙钟，不能声称与Joint计算量相同。

small的Sequential exact按顺序完整枚举两个受限阶段，不是Joint全空间最优解。因此optimality_gap=null，reference_kind=two_stage_exact_not_global_optimum，reference_gap为相对该过程的有符号差值，允许负值。两个阶段的搜索空间/评估数分别保留，不把它们相加称联合空间大小。其他三模式的exact仍是各自受限空间最优解。

## 可重放输出

每个尺度已有seed/beam/rollouts/iterations/top-k配置，v1.2补充完整开关、融合概率、Sequential固定设备和次数分配。顶层保存硬件快照、硬件/原始Profile输入路径及SHA256、标签诊断策略和主入口源码hash。候选带完整plan、节点覆盖数及deployable=false。

save_evaluation为每项搜索的最佳方案保存一个PlacedOpGraph，路径写入best_placed_op_graph。工件保存配置、预测值、原图节点、分组和设备标注，全部明确deployable=false。small是5节点，layer是21节点，不要求small覆盖21节点。内置planned融合只是机制开发，不能称正式融合能力或真实GPU加速。

```bash
python -m cost_model.search_evaluation --scales small,layer \
  --mcts-iterations 16 --seed 23 --allow-unbound \
  --output data/run1/search.json
python -m cost_model.search_evaluation --scales small,layer \
  --mcts-iterations 16 --seed 23 --allow-unbound \
  --output data/run2/search.json
python tools/verify_search_replay.py \
  --first data/run1/search.json --second data/run2/search.json \
  --output data/replay_verification.json
```

两个报告使用相同文件名search.json、分别放在不同目录，保证相对工件路径相同。验证器仅排除elapsed_s，其他报告字段全部一致；加载两次的所有最佳工件，检查图hash、候选标识、节点集合、预测值、冻结划分与非部署标记。工具分别验证合法性和记录是否选融合：校对placeholder注释与计划一致，Placement-only仍不允许融合；不强制Joint等模式最佳结果必须融合。fusion_observations保存逐模式实际选择，不把选到未融合参考当成非法结果。

现有快照data/mainline_four_mode_v1保存两次真实CPU CLI报告、日志及40个最佳工件；replay_verification.json记录140个候选检查通过。64项相关CPU回归通过，其中主评价器6项（含新增冻结划分、奇数次数、重放/工件和有符号参考差测试）。未采GPU、未放宽正式标签准入、未修改冻结SROP文件。
## Beam参考保底

Beam将每个允许设备上的合法singleton完整方案预存completed，只保留有限参考评分；在固定确定性scorer下，结果不劣于其中最好者。Sequential第二阶段在macro空间构造参考，展开后保持冻结划分。未改变原rollout均值排名或添加逐状态贪心槽位，活动beam宽度不变。额外参考评分计入阶段请求/评估/墙钟。

基线data/mainline_beam_floor_v1说明实际边界：layer Sequential Beam回到.306805ms，Joint Beam仅回到.316920ms，后者仍未找到Fusion-only方案。当前MCTS融合三模式返回同一device0计划，尚无联合收益证据。报告新增搜索实现源码hash，旧数据不回写。
## MCTS 预算与参考审计

主线工具 tools/evaluate_mcts_mainline.py 只运行 MCTS/小图 Exact，固定解析 Profile，不运行 Beam 或 SROP。data/mcts_mainline_v1/README.md 保存协议、60 次 MCTS 和 4 次 Exact 的结果；tools/verify_mcts_mainline.py 从压缩评分记录重新核对计数、参考差距和输出工件。记录对象是已评分完整候选，包含 rollout 补全，不是树节点访问。

small 三个 seed 均达到对应参考；Sequential 仍只称两阶段参考。layer 的 16/64/256/1024 次结果相同，融合三模式返回同一 device0 计划；60/60 次运行首个评分已经达到本次最终最佳。高预算结果先于网格冻结，事后最佳另存，两者都不是 layer 最优性证明。当前不追加同图预算，不据此宣称联合搜索优势。
## 中等子图与显存约束诊断

入口 tools/evaluate_mcts_medium.py 分 prepare/run/verify 三阶段；data/mcts_medium_v1/README.md 给出 5 个预定义任务、20 次 Exact 和 180 次 MCTS。容量机制档单列，未改计时 Profile、MCTS 或评分；可行参考从完整 Exact 评分表中计算，不把超容量返回值计入质量差距。

当前主评分器只返回 makespan，StructuralCostModel 的 overflow_devices 未进入主搜索选优。MLP 90% 人工容量下，27 次 Placement-only/Sequential/Joint 已评分可行候选但最终仍返回超容量方案；Fusion-only 无可行候选。下一项是统一搜索可行性与无可行候选的结果语义，再重跑同一任务，暂不调探索/奖励/rollout。
## 显存可行性入口（v1.3）

_CachedScorer显式接收capacity元组；正式调用从硬件快照传入。逐设备比较CostEstimate.peak_memory；超容评分加有限1e6秒，缓存仍保存原始makespan。最终只在已评分可行方案中按物理成本选优，不依赖罚分足够大的假设，不返回inf。

Sequential第一阶段只提出融合划分，不绑定单卡容量；冻结后放置阶段绑定。报告stage.capacity_bound标明二者，feasibility_scope区分mode_space与frozen_partition。Exact空结果reason=no_feasible_in_space；MCTS/Beam空结果reason=evaluated_no_feasible。均输出plan=null、candidates=[]、best_placed_op_graph=null，无工件。若未融合单卡基线不可行，speedup_vs_unfused和improvement_percent留空。

当前重跑data/mcts_medium_capacity_v2：20Exact/180MCTS，原始容量108次的最佳方案/成本/有序评分请求逐项不变；90%人工容量的27次Placement/Seq/Joint全部返回可行方案；75%容量下MCTS的未找到与Exact的无解证明分开。60项CPU测试通过。旧ee2c9bf与data/mcts_medium_v1仅作修复前基线，不再作为当前搜索质量引用。此前layer原容量已评分候选没有超容，容量约束未成为有效限制，旧四模式表仍是该冻结版本的有效解析记录。
