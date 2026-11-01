# 主系统独立解析基线（2026-11-01）

--profile analytic仅加载固定硬件快照，实际CLI传入不存在的--raw路径仍成功，报告inputs只有hardware；无allow-unbound、无Kernel标签读取。逻辑input为零执行时间，修复了旧解析路径给它计launch/访存的行为；硬件launch/fusion常数不变。所有工件deployable=false，无GPU性能或正式能力声明。

两次CLI配置seed23、beam8、rollouts3、MCTS16、top-k4，small/layer；报告除elapsed_s逐字段相同，140候选/40工件检查通过。small5节点12搜索48模型评估，layer21节点8搜索2296评估。输入/源hash与配置保存，链路由硬件参数公式预测，不查raw消息表。

| 模式 | small Exact/Beam/MCTS ms | layer Beam ms | layer MCTS ms |
|---|---:|---:|---:|
| Placement-only | .108872 | .317027 | .316920 |
| Fusion-only | .098756 | .306805 | .306805 |
| Sequential | .098756 | .425065 | .306805 |
| Joint | .098756 | .381588 | .306805 |

small表只验证四模式与参考一致；Sequential Exact仍是两阶段参考。layer没有Exact参考，以上不能证明模式优劣。尤其本次Sequential/Joint的Beam结果比未融合单卡基线更差，尚无输出保底；这是后续搜索质量检查项，不用改Profile或隐藏结果来消除。

strict_measured保留默认严格入口的真实CLI失败：MeasurementContractError、exit1、无search.json。解析模式不会偷偷忽略Kernel闸门；它根本不读取标签。82相关CPU测试通过（含解析input零时间、禁用raw加载/准入参数测试），XML在工作区transfer/mainline-resume-20261101/fixed-profile-tests.xml。

当前用于无旧标签依赖的工程开发。硬件参数来自历史固定快照，不能把本次预测与旧表诊断预测的差值当准确率/真实性或GPU加速；正式实测模式仍待能力与目标契约确认。SROP继续冻结，旧基线与原始标签未覆盖。
