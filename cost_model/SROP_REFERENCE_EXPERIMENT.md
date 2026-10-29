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

2026-10-29：CPU与双卡GPU数值验收完成；GPU报告已回传并核对，见§9。新入口 `execution.decoder_reference.ContinuousDecoderExecutor`，子图由 `simulator.model_graphs.decoder_stack.build_decoder_stack` 构造。仅含连续 decoder body，不含 embedding、最终 norm 和 LM head。

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

## 8. A2 候选结构覆盖

2026-10-29：离线结构审计完成。入口 `python -m cost_model.decoder_candidate_coverage`，产物 `data/a2_decoder_candidate_coverage.v1.json`。5项回归、真实图离线审计、compileall及两个freeze完整性通过。

对92193b...四个旧任务分别检查完整提案池和已测子集：完整提案252个物理候选，已测66个。完整提案覆盖Q/K/V所有物理位模式及分支链同卡位模式，但每任务都缺4个局部QKV组合、1个局部gate/up组合。已测子集中两个3B任务还各缺2种QKV物理位模式。两种物理方向齐全；此结果不能解释成搜索空间不支持这些结构，只能说明这些有限提案/测量集合没有覆盖。

局部分支见证要求：只改变当前层对应分支链，所有其他节点固定在同一背景设备。Q分支含q_proj/q_norm/q_rope（按图实际存在），K同理，V为v_proj；gate分支含gate_proj/silu，up为up_proj。全局交换后用相对背景的位模式审计，不能仅凭每层都出现过变化就认定局部组合齐全。

新结构候选仅用于2/4层decoder body，seq256、batch1、BF16，采用现有Qwen3-0.6B/Qwen2.5-3B维度。基池为graph-wide-v1探针和singleton参考，未新跑MCTS；增加每层全部QKV/gate-up位组合、全部确定性拓扑前缀切分和全部整层边界切分，再展开两种物理方向。没有读取耗时、预测分数或目标标签，也没有改主机/链路成本；未改冻结提案和已测子集。

| 图 | 物理候选：基池→补齐 | 局部分支缺项：基池→补齐 | 拓扑前缀缺项：基池→补齐 |
|---|---:|---:|---:|
| Qwen3 2层 | 56→174 | 19→0 | 40→0 |
| Qwen3 4层 | 64→298 | 37→0 | 80→0 |
| Qwen2.5 2层 | 52→162 | 19→0 | 36→0 |
| Qwen2.5 4层 | 60→278 | 37→0 | 72→0 |

合计232→912物理候选，局部分支缺项112→0、拓扑前缀缺项228→0，交换方向缺项0，全部基池候选保留。所有新增候选通过合法性、registry executable与ContinuousDecoderExecutor构造检查；是静态可执行资格，不是912个GPU数值验证，更不是完整2^N放置空间覆盖。

审计分别报告搜索合法性、registry可执行性和Reference静态资格；当前Reference只有singleton组。planned多节点融合在回归中可搜索合法，但不计入Reference证据。报告另列每候选的分设备FLOPs/参数字节、整张量复制数量/字节、主机算子/复制提交次数。两端stream屏障数只是Reference模型假设，没有观测GPU同步，也未估算执行时长、峰值显存可行性或吞吐。

复验（无卡）：

```bash
git pull
python -m cost_model.decoder_candidate_coverage --output data/a2_decoder_validation/coverage.json
python -m unittest cost_model.tests.test_decoder_candidate_coverage -v
```

A1双卡数值已回传通过（§9）；下一步A3需预先确定开发集诊断协议，再判断这些结构是否存在真实性能竞争。新增候选没有GPU标签；不能把旧66个已测结果外推到912个新候选。A2补齐不证明多卡更快或SROP有效，B1/B2尚未开始。

## 9. A1 双卡数值结果回传

用户宿主40106终端为 `A1 cases=32 all_pass=True gpu_copy_validated=True`。原报告 `data/received_a1_decoder_validation/gpu_report.json`，一致性核对 `audit_report.json`。报告cuda:0/1、torch2.9.0/HIP6.3.26093；7项源码hash与本地一致，32个唯一用例覆盖2/4层×FP32/BF16×有无QK Norm×四放置。

| dtype | 用例 | 原始输出最大绝对误差 | 输入扰动后最大绝对误差 |
|---|---:|---:|---:|
| FP32 | 16 | 1.210719347e-8 | 2.235174179e-8 |
| BF16 | 16 | 0 | 0 |

所有报告误差小于各自绝对容差；32/32数值/有限性/形状/扰动传播/旧重置拒绝通过，根输入缓存均1项。核对依据为收到的汇总和源码hash，不包含原始输出张量，也没有本地重复GPU执行。

A1通过范围仍为batch2、seq5、hidden16、FFN32的小尺寸synthetic decoder body及四种放置；验证实际跨卡传递路径，不代表912候选全体、实际模型维度或性能已验证。下一步A3先固定开发任务/候选、重复与计时规则，再进行真实性能竞争诊断；B1/B2未开始，旧host未验收和SROP未证实状态不变。

## 10. A3 真实方案竞争协议（v1，采集前冻结）

工件 `data/a3_decoder_competition_v1/freeze.json`；freeze_id=`sha256:376f1c3644bc2dd7162cbab469c6f68caf90a2e9d3184b4fd9661a903f995714`。43项源码、6项输入hash受保护。冻结时尚未采新方案耗时，回传结果见§11；此协议为开发诊断，不是独立SROP收益实验。

### 任务与候选

| task_id | 层数 | Shape/dtype | 物理候选 |
|---|---:|---|---:|
| qwen3-0.6b-decoder2-s256-b1 | 2 | seq256/batch1/BF16 | 28 |
| qwen3-0.6b-decoder4-s256-b1 | 4 | seq256/batch1/BF16 | 32 |
| qwen2.5-3b-decoder2-s256-b1 | 2 | seq256/batch1/BF16 | 28 |
| qwen2.5-3b-decoder4-s256-b1 | 4 | seq256/batch1/BF16 | 32 |

沿用A2的实际模型维度，仅decoder body，无embedding/head。120物理候选全都属于原912结构候选；按结构选择，不按预测或真实耗时筛选，不扩展执行器和候选空间。

每任务包括GPU0/GPU1单卡；首层与末层的QKV相对背景位模式100/110/111，gate-up位模式01/10/11；全部整层边界切分。每个跨卡结构保留两种物理方向。只在首末层抽局部分支，不声称完整保留A2所有局部组合和拓扑前缀覆盖。

所有候选ID、完整图/方案和selection_reasons、初始预测/并列集合/预选物理ID、逐块schedule均封存在freeze.json。逻辑0/1固定映射到进程可见cuda:0/1，不根据实测选择方向。预选为初始预测并列集合的最小ID，评估时完整并列集合最好/最坏另报。

初始评分：原112训练数据拟合的nearest、原硬件链路、未验收host的ReferenceEager模型。seq256测量Key已在原训练集中出现；按local_name在各层复用对应预测，不称未见Shape或零样本泛化。新候选的方案级执行时间仍未采。host/body/GQA/布局未验收限制不因冻结解除。

### 运行与计时

两张BW设备，torch2.9.0/HIP6.3.26093；不允许CPU回退。输入/权重seed1234，canonical CPU模板在每任务内共享，保证候选和物理方向不改变张量值。每块重建executor、5次预热、20次正式前向；不在块间empty_cache，不加入逐组Event或profiler。

每块先检查实际维度输出与GPU0单卡参考的数值一致性，rtol=.01/atol=.001并要求finite。该参考使用相同连续执行器；独立公式证据来自A1小尺寸，不能把它写成实际模型维度的独立实现对照。

每任务3个窗口。窗口开始、结束均测GPU0和GPU1单卡锚点；前后锚点顺序相反，不同窗口交换起始方向。中间测除单卡外的全部入选物理候选，按固定seed20261029+100×任务序号+窗口号打乱，完整顺序已封存。

合计384块、7680次正式前向，另含每块1次数值检查和5次预热、每任务1次参考生成。正式延迟为run_once(False)同步前向墙钟，含逐算子Python提交/复制及末端同步；构造、缓存首次生成、数值检查和预热不计入稳态前向延迟。另记录block_wall_s，不能据此声称SROP总成本收益。

### 漂移、噪声和失败

- 主结果不删除任何样本或窗口，不重跑直到通过。
- 各单卡锚点abs(after/before−1)>10%标记窗口漂移；任一锚点越线，该窗口不满足质量参照，原数据仍参加主汇总。
- 断点续跑跳过已落盘slot，不覆盖原块；一个窗口跨越不同采集进程也标记不满足质量参照。环境变化拒绝合并。
- 每块保留20个原始墙钟值、正确性和峰值显存；按record hash、环境hash、固定slot核对。失败块保存后停止，不将候选从池里删除；初始化失败另保存setup_failure文件。缺块或有失败时不生成regret结论。
- 参考差异采用预设描述性规则：每窗口中，候选与该窗口较快单卡的差值绝对值大于max(0.1%×较小延迟，3×两者稳健MAD噪声尺度之和)，且3窗口差值同号、3窗口质量均过线，才标记“重复一致且超过噪声参照”。这不是显著性检验或置信区间。

### 汇总与三个问题

每块取20次中位数；单卡在每窗口取before/after两个块中位数的中位数，其余候选为本窗口唯一块中位数；再对3窗口中位数取中位数，作为主延迟。原始样本分布及各窗口regret另报。

性能层：报告候选延迟、原始分布、锚点漂移、逐窗口相对单卡差距、重复一致差异及跨卡优势。存在慢方案也会产生性能差异，不能直接称补测机会。

决策层：用封存初始物理选择，报告距已测120候选中各自任务子集最优的真实regret，主ε=1%、敏感性0.1%/0.5%；固定GPU0、固定GPU1分别保留。初始已经达到ε的任务照常报告，不按事后分层删任务。

补测层：本轮不采新kernel标签、不进行Proxy/Refit/重新搜索，因此报告kernel_profiling_value=not assessed。即使发生初始错误或跨卡更快，也不直接算SROP收益；后续才验证部分测量/更新能否改善、且收益是否超过成本。

### 服务器运行

只git pull，不重新prepare、不打部署包。两卡模式直接运行：

```bash
git pull
python -m calibration.decoder_competition check
python -m execution.decoder_competition run --out-dir data/a3_decoder_competition_v1_results
```

check应显示freeze376f1c...、4 tasks / 120 physical_candidates / 384 blocks / 7680 formal_forwards。采集完成末尾应显示complete=True blocks=384/384；各任务regret/质量窗口/跨卡优势如实打印，无预设成功值。

回传competition_report.json、execution_samples.jsonl和environment.json；失败时连同setup_failure文件和终端输出保留。可无卡复算已有日志：

```bash
python -m execution.decoder_competition evaluate --out-dir data/a3_decoder_competition_v1_results
```

9项本地协议回归及compileall通过；包括固定慢物理并列不事后换卡、漂移保留、窗口反转、跨进程窗口、失败/重复/hash篡改、来源锁定、抽样和调度、GPU入口拒绝CPU回退。CPU数据只为schema测试，不作为GPU性能证据。原两份freeze保持不变。

## 11. A3 GPU回传与原始记录复算

原件在 `data/received_a3_decoder_competition_v1/`，宿主29890，两个BW设备、torch2.9/HIP6.3.26093、双向peer_access=True。freeze376f1c...的43源码/6输入一致；384个预期slot、7680次正式前向齐全，记录/环境hash、块中位数、输出Shape/容差、block_wall及显存字段核对通过。由原始记录复算的报告与原competition_report.json完全一致，无浮点差异。

384块数值全部通过、最大绝对误差0；单采集进程，12/12窗口满足预设质量规则，最大锚点变化3.1660%。4个>2倍各块中位数长尾保留，未删样本或窗口。

| 任务 | 已测单卡最优ms | 最佳跨卡ms | 最佳跨卡慢于最优 | 初始regret | 初始模型方案MAPE |
|---|---:|---:|---:|---:|---:|
| Qwen3 2层 | 2.704249 | 2.866253 | 5.9907% | 0% | 2.3509% |
| Qwen3 4层 | 5.484878 | 5.582049 | 1.7716% | 0% | 3.7410% |
| Qwen2.5 2层 | 2.635811 | 2.739136 | 3.9201% | 0% | 3.7408% |
| Qwen2.5 4层 | 5.305389 | 5.376427 | 1.3390% | 0.2215% | 5.1018% |

各任务已测最快均单卡，最佳跨卡均为中间整层边界（2层cut1、4层cut2）。112个跨卡候选中100个满足“重复一致且超过预设噪声参照”的更慢差异，12个未满足该描述性条件；没有重复一致的跨卡优势。此参照不作为置信区间或因果分解证据。

初始合并ε=1%/0.5%均4/4，0.1%为3/4；固定GPU0、固定GPU1也分别4/4满足主1%。初始逐窗口1%为11/12，Qwen2.5 2层第三窗口1.5394%，不能宣称逐次稳定保证。初始模型方案Spearman0.907～0.946；本轮算子Shape来自已有训练，方案级新数据为开发集，不称新Shape独立泛化。

本轮给出可测性能差异，但没有超过主ε的合并初始决策损失。当前条件下不立即投入B1/B2或继续增加SROP公式/重复同类采集；保留负向结果，转向用现有数据离线做代价模型基线与执行语义消融，尤其比较原DAG、串行加和、无host Reference和含host Reference。该下一步尚未完成。

kernel_profiling_value仍为not assessed：未测新kernel、未Refit/重新搜索，不能证明普遍无补测价值。结果仅限4个同构BW、seq256、2/4层synthetic前向任务及120个入选候选，不覆盖全部912/全空间、真实融合、反向或下游流水训练。host未验收和body/layout限制保留。不要修改本冻结的阈值/候选后再声称前瞻验证。

复算入口：

```bash
python -m execution.analyze_decoder_competition --results data/received_a3_decoder_competition_v1
```

新分析只读，audit_report.json包含原件hash、逐任务差距/初始选择/MAPE/窗口检查；真实数据复算和compileall通过，三份既有freeze保持不变，未新采GPU或重复无关套件。
