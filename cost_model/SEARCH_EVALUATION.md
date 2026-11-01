# 主系统四模式开发评价

search_evaluation报告v1.2包含placement_only、fusion_only、sequential、joint。SROP验证保持冻结；本入口不导入srop_e0，不调用主动校准。默认标签准入闸门保留；--allow-unbound只用于明确声明的旧数据开发诊断。

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

两个报告使用相同文件名search.json、分别放在不同目录，保证相对工件路径相同。验证器仅排除elapsed_s，其他报告字段全部一致；加载两次的所有最佳工件，检查图hash、候选标识、节点集合、预测值、冻结划分与非部署标记。此工具针对当前small/layer开发夹具，要求融合模式最佳方案确实包含planned融合，不是任意图的通用性能验收器。

现有快照data/mainline_four_mode_v1保存两次真实CPU CLI报告、日志及40个最佳工件；replay_verification.json记录140个候选检查通过。64项相关CPU回归通过，其中主评价器6项（含新增冻结划分、奇数次数、重放/工件和有符号参考差测试）。未采GPU、未放宽正式标签准入、未修改冻结SROP文件。
