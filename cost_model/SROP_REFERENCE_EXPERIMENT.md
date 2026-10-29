# SROP：Reference 执行语义下的下一阶段实验

2026-10-29。状态：开发集机会诊断完成，尚未冻结新的 GPU 实验。

## 1. 已完成的机会诊断

`python -m cost_model.reference_profiling_opportunity`

产物：`data/reference_profiling_opportunity_v1.json`。两批已暴露数据的 8 个独特任务、88 个已测候选，逐任务保持原始候选子集。旧四任务来自 seq1536/2048，新四任务来自 seq768/1280；各自使用本宿主的执行数据，不拼接同一候选的跨宿主样本。

| 诊断 | 结果 |
|---|---:|
| 初始 Reference+host 方案满足 1% | 8/8 |
| 固定 GPU0 单卡满足 1% | 8/8 |
| 全部替换为实测算子耗时后满足 1% | 8/8 |
| 逐项真值修正 | 116 次 |
| 逐项修正改变所选物理方案 | 0 |
| 逐项修正得到正向执行收益 | 0 |

主机 profile 沿用 92193b... 中的未验收版本；GPU0/GPU1 的算子标签继续聚合，不变成物理卡特定模型。当前诊断只修改被指定的 Key，其他 Key 不变，不是 Refit 的假设后验更新。全部替换只是一个端点，不能作为所有部分更新或未来任务的收益上界。

旧四任务的执行原始记录已复核，算子真值读取既有汇总，不在本脚本中重验旧 kernel 原始血缘；新四任务原始算子血缘已经上一轮审计。两批全部属于开发数据。本轮没有重新采 GPU，也没有将 Reference 模型接入旧 SROP 主循环。

## 2. 下一阶段的实验边界

现有 8 个任务全部保留为无需补测的对照。不能只汇报初始 regret 非零、某方法胜出的任务；接近最优时的额外测量成本也是结果。

新任务先确定执行需求和候选机制，再固定任务网格与预算。候选决策可来自真实可执行融合、下游给出的放置限制、实际显存约束，或支持有效分支并行的执行器；这些能力目前不齐，不能用 planned pattern 的预测时延代替 GPU 标签，也不能人为降低链路或增加主机成本来制造主实验优势。

模型、seq、batch 和约束的全列表在目标真值采集前固定；特征与预测可用于说明任务覆盖，不依据隐藏 regret 或 SROP 输赢决定保留任务。初始 regret 非零组只作预先定义的事后分层，全任务主表不变。当前不要求重复收集相同单层 eager 任务，也不要求提前实现上下游中间层或正式融合清单。

## 3. 两种评测模式

| 模式 | 候选来源 | 能说明什么 |
|---|---|---|
| 固定候选重排序 | 仅公开初始预测与无标签结构探针，冻结后共同使用 | 补测是否改善候选选择 |
| 重新搜索闭环 | 每种方法根据各自已公开标签重新运行搜索 | 补测是否改善实际发现的方案 |

主闭环不读取 Oracle 搜索候选。Oracle 只在独立评估器中用于参考；如只执行了候选子集，regret 的参考就是该子集，不能写成全局最优。

## 4. 数据访问与更新

- Zero-shot 测试不开放目标标签；Active calibration 单独启动，只逐次揭示选中的完整 Kernel Key。
- 原训练数据、公开特征、已揭示标签可供策略访问；未揭示算子标签和候选真实执行时间只归评估器所有。
- 完整 Key 和标签血缘去重。当前已暴露 112 个训练对象、56 个旧开发对象和 52 个新对象，均不能再称盲测。
- Proxy 保留为快速决策价值代理；Refit 的每个假设观测必须调用与实际更新一致的预测器更新路径，并重新评分其他受影响 Key。
- 预测物理并列集合在测量前按最小 ID 固定选择；完整并列最好/最坏仅作诊断。不能根据真值选择更快物理卡。
- 当前 q90 覆盖不足、host 未验收、body/GQA/布局覆盖未核实继续记录。有限扰动范围不作为置信区间或停止证书。

## 5. 对照与成本

固定 GPU0 单卡、initial-only、Random、Uncertainty-only、Impact-only、SROP-Proxy、SROP-Refit 全部保留。Random seed0/1/7/23，每个任务内先合并 seed/repeat，再平均任务；不同 seed 不充当独立任务。

主相对 regret 容忍值 1%，敏感性 0.1%/0.5%。开发阶段沿用总预算 1/2/4/8/16 秒、主参照 4 秒；正式预算在新任务的训练侧预跑后冻结，不能看目标结果改预算。候选预处理、初始预测、选点、真实测量、模型更新与重新搜索全部计费；固定单卡无需执行的搜索费用不强加给它。

主曲线同时报告原始相对 regret 和超出 ε 的决策损失。给定时间 B，只发布 B 之前完成更新/搜索的方案；测量完成而更新未完成时保留旧方案。数据加载和一次性准备如何摊销在协议中固定，不能只算 GPU kernel 计时而忽略 profiler 启动/预热/同步。

同宿主 GPU 上记录完整 wall time 作为正式总成本。CPU 实测加历史 GPU profiling wall 的串行相加只作开发回放；跨宿主 CPU 计时不解释成真实端到端加速。记录各阶段费用，单核预测误差另报，不替代真实方案级 regret。

## 6. 验收与停止

主结论要求多个独立留出任务上，在相同总成本下改善真实方案选择，或更低成本达到相同 ε 质量；必须同时比较最强主动基线、initial-only 和固定单卡。接近零 regret 时额外付费却不改善的结果照常计入。

当前不将 SROP 分数为零或预测 Top2 接近作为安全停止。先比较固定预算曲线；原启发式停止只作标明的消融。N 次后续执行的净收益停止是单独扩展，尚未实现。

执行入口变更前，先补 Reference 排序器对旧 policy/Refit 路径的接口测试，验证单卡/跨卡、完整覆盖、同 Key 多处传播和标签隔离，再冻结新的任务/候选/源码。现有冻结 59a32c2e... 与 92193b... 均保持不变。

## 7. A1 连续 decoder 数值验证

2026-10-29：CPU 数值验收完成，GPU 跨设备复制待验证。新入口 `execution.decoder_reference.ContinuousDecoderExecutor`，子图由 `simulator.model_graphs.decoder_stack.build_decoder_stack` 构造。仅含连续 decoder body，不含 embedding、最终 norm 和 LM head。

`execution/reference.py` 与 `execution/gpu_holdout.py` 都被旧实验源码 hash 引用，保持原样。A1 必须使用新入口；旧 `execution.run_reference` / `CanonicalReferenceExecutor` 不会自动获得层间透传修复，不能用于新的连续多层计算。新入口继承既有逐组提交/消费者复制，不新增流水调度或 TP/PP 搜索。

根输入仍按原 seed 生成；后续 input 节点只接收一个前驱、透传已路由到目标设备的张量，不创建新的随机输入。检查 Shape、dtype、目标设备及根/非根输入依赖数量。未改变旧冻结候选、代价模型或 profile。

小尺寸合成数据 batch2、seq5、hidden16、FFN32、4个Q头/2个KV组、head_dim4；2/4 层 × FP32/BF16 × 有/无 QK Norm × 单卡0/单卡1/交替层/输入边界分离，共32个CPU数值用例；与独立连续层公式的最大绝对误差0。首层输入加0.1后的32个对照也全部一致，旧逐层重置行为32/32被拒绝，根输入缓存始终只有1项。CPU上的两个逻辑设备均映射为cpu，不作为实际跨GPU复制验证。

新增6项回归通过：32用例、首层输出扰动传到第2/4层最终输出、单层与旧入口逐位一致、Q投影宽度不等于隐藏宽度、输入别名/Shape/dtype/设备、层间输入断连/多前驱拒绝。既有10项相关回归、compileall及两个freeze完整性检查通过。独立公式不遍历图、不调用执行器算子实现；遵循现有逻辑图的无mask/无attention scale计算，不是预训练Qwen全模型正确性认证。

CPU复验（无卡可跑）：

```bash
git pull
python -m execution.verify_decoder_stack --output data/a1_decoder_validation/cpu_report.json
python -m unittest execution.tests.test_decoder_reference -v
```

以后有两卡时可做数值检查，拒绝CPU回退，不采性能结果：

```bash
python -m execution.verify_decoder_stack --devices 0,1 --output data/a1_decoder_validation/gpu_report.json
```

本轮报告 `data/a1_decoder_correctness.cpu.json`。下一步 A2 离线审计候选覆盖；A3 真实性能竞争、B1 Reference SROP 主循环、B2 同总成本收益尚未完成。不因A1通过就认定存在多卡性能竞争。
