# Beam合法参考保底（2026-11-01）

本轮仅补Beam参考方案保留，不改Profile、rollout均值排序、beam宽度或MCTS逻辑，也未新增逐状态贪心支路。SROP保持冻结，无GPU。结果是固定解析模型下的CPU工程诊断，全部工件deployable=false。

## 修复边界

legal_beam_search在completed中预存每个allowed_device的合法singleton单卡参考。候选从当前LegalFusionSearchSpace的registry/actions构造，不硬塞device0，不把planned组混进executable-only空间。有限参考代价才入池，不消耗随机数或占用活动Beam槽位；查询耗时和scorer调用包含在正常搜索记录中。

固定确定性scorer下，最终最佳值不劣于这些有限参考中的最小值。该保证不是全局最优保证，scorer仍负责表达不可行代价；实际本次额外验算所有参考的显存约束均通过。Sequential第二阶段的singleton是macro节点，展开后保持第一阶段融合划分，仅改变设备，不能退回原图未融合划分。

原Beam按随机补全的平均分剪枝，未保留一般rollout完整方案；本轮未改这部分。默认fusion_probability=.5，主评价器实际传.7；固定seed可重放，不意味着均值估计没有剪枝误差。确定性逐状态分支属于进一步算法改动，本轮未加入。

## layer重跑

两次真实CLI均--profile analytic、small/layer、seed23、beam8、rollouts3、MCTS16、top-k4，保留输入/源码hash。新增search_implementation_sha256记录Beam/Exact/结构代价/融合规则源码，避免只记录入口文件遗漏算法变化。

| 模式 | 修复前Beam ms | 修复后Beam ms | MCTS ms |
|---|---:|---:|---:|
| Placement-only | .317027 | .316920 | .316920 |
| Fusion-only | .306805 | .306805 | .306805 |
| Sequential | .425065 | .306805 | .306805 |
| Joint | .381588 | .316920 | .306805 |

修复前来自mainline_analytic_v1，原始文件保留。Joint Beam回到未融合参考，仍未找到Fusion-only的方案；保底不能保证其达到.306805ms。所有模式8项small/layer Beam对各自可行参考的代价界检查通过，baseline_verification.json保留逐设备参考值。

## 结果解释

layer的Fusion-only/Sequential/Joint MCTS预测时间严格相同：.00030680488599502565s；三个plan_signature也相同，均只用device0。Placement-only为.0003169201069558956s。融合在该解析模型中的预测收益3.1917258%，本次放置/Joint相对Fusion-only增益0，不能写成实测融合收益或联合搜索收益。

Beam的原Sequential/Joint差异不是已验证的耦合收益；修复后Joint Beam仍落后于Fusion-only，是当前启发式搜索未找到更好已知候选。不得把算法误差当优化模式的本质差异。

同型号设备加链路开销并不对任意DAG构成单卡支配定理；多分支并行、通信和显存仍可能改变最优放置。本批没有layer精确最优参考，不宣称放置维度在一般场景无价值。后续若验证耦合，应按真实资源/图约束登记任务，不能调Profile制造收益；本轮不扩大任务或采GPU。

## 验收

small仍12项/48实际模型评估，layer8项/2297评估；两次仅elapsed_s不同，140候选/40工件重放检查通过。8个MCTS最佳值和计划签名与修复前完全一致。86项相关CPU测试通过（新增3项低层参考/允许设备/非有限代价及1项四模式本空间保底），XML在transfer/mainline-resume-20261101/beam-floor-tests.xml。

重放检查将输出合法性与是否选择融合分开：仍验证完整节点、冻结划分、placeholder注释与实际计划一致，但不强迫Joint最佳方案必须融合；fusion_observations逐项记录实际选择。原基线曾全部选融合只是历史观测，不是合法方案定义。

冻结SROP的41证据/11未提交文件hash未变，见baseline_verification.json。复验仍用tools/verify_search_replay.py；无需服务器操作。先前mainline_analytic_v1和mainline_four_mode_v1不回写。
