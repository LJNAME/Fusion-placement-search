# 主系统四模式基线（2026-11-01）

Sequential已接入主评价器。两次独立CPU CLI运行的报告除elapsed_s外逐字段一致；加载/检查140个候选与40个最佳工件通过。SROP仍冻结，未采GPU。运行使用--allow-unbound，旧Profile标签只作为低置信诊断数据，所有输出deployable=false；下列时间是代价模型预测，不是GPU实测性能。

## 输入与配置

scales=small,layer；seed23、beam8、rollouts3、mcts_iterations16、top_k4。输入为仓库bw1000_2card.calibrated.json/raw_measurements.json，报告inputs保存路径与SHA256。每尺度configuration完整保存；source_sha256记录主入口字节hash。报告v1.2，两个原运行目录run1/run2，各含search.json、stdout/stderr日志与search.artifacts。

small为5节点/4边，12项搜索，缓存后48次实际代价评估；layer为21节点/25边，8项搜索，2266次实际代价评估。第二次完全相同；计数不代表相同墙钟预算，缓存跨模式共享。每个MCTS运行合计16次，Sequential为8+8。

## small机制对照

单位ms，均为各模式最佳方案预测。

| 模式 | Exact | Beam | MCTS | 组数 | planned组数 |
|---|---:|---:|---:|---:|---:|
| Placement-only | .073772 | .073772 | .073772 | 5 | 0 |
| Fusion-only | .066720 | .066720 | .066720 | 4 | 1 |
| Sequential | .066720 | .066720 | .066720 | 4 | 1 |
| Joint | .066720 | .066720 | .066720 | 4 | 1 |

Sequential Exact指两阶段各自穷举的顺序过程，不是全局联合最优；optimality_gap=null，reference_gap=0。其余模式相对自身受限空间的exact差距为0。本夹具融合相关模式质量相同，不证明Joint优于Sequential，也不把融合预测差异当实测加速。

## layer规模记录

| 模式 | Beam预测ms | MCTS预测ms |
|---|---:|---:|
| Placement-only | .540803 | .527617 |
| Fusion-only | .520565 | .520565 |
| Sequential | .526244 | .520565 |
| Joint | .515456 | .520565 |

layer不跑Exact，没有已验证的最优参考；以上只记录单seed、固定参数下的算法输出，不宣称模式优劣或搜索质量显著差异。两个算法在该预算下不一致，后续质量评估仍需参考与种子分析，不立即扩GPU任务。

## 机械验收

replay_verification.json：pass=true，唯一排除字段elapsed_s，双run共140候选/40工件。各候选完整且不重复覆盖各自原图节点；各最佳工件保存/加载成功，graph_hash/plan/预测值一致。融合相关模式最佳方案均包含1个planned组，工件extra.deployable=false；Placement-only为0个planned组，仍明确开发非部署范围。

Sequential MCTS两阶段预算合计16，第一阶段仅device0，第二阶段设备0/1；所有候选划分与budget.frozen_partition一致。测试另覆盖17=8+9、模板属性不被placement修改，以及heuristic优于两阶段Exact参考时reference_gap可为负。

64项相关CPU测试通过（58核心+主评价器6），其中重放验收亦拒绝seed篡改；测试XML在工作区transfer/mainline-resume-20261101/four-mode-tests.xml。旧标签警告保留在stderr.log，不把allow-unbound解释为accepted。复验命令见cost_model/SEARCH_EVALUATION.md。

## 当前交付与缺口

1. 图/融合规则、合法联合动作、Exact/Beam/MCTS、结构代价和PlacedOpGraph已实现；主系统Sequential缺口已补。
2. small/layer固定配置的搜索—工件流程可重放，配置和输入血缘在报告内。
3. small四模式机制对照已交付；layer仅规模记录。

正式融合能力、目标执行契约、正式Profile覆盖以及上下游适配仍待后续；完整方案执行和并行配置归下游。冻结SROP的旧复验如依赖主评价器源hash，应恢复冻结90f623c版本；不更新旧hash适配新源码。
