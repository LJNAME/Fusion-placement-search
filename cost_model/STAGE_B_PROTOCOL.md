# 阶段 B 执行规范

2026-11-02 修订，基线 `6005868`。顺序为 **B0 → B1 → B2 → B3 → B4 → B5 → B5.5 → B6**；完整计算图验证另列C阶段。B0～B3仅CPU，B4首次使用GPU；B5.5仅CPU，不重新测量。B1已完成入口/出口GPU0的声明动作空间枚举；B2声明及覆盖核验已完成，B3 CPU矩阵/质量配方已冻结，B4 Smoke及后续实验未执行。

## 三种成本语义

| 概念 | 能证明什么 | 当前状态 |
|---|---|---|
| 比较证据（Evidence Coverage） | 指定Case和测量契约下Fused/Native的性能差异 | 两卡选定KERV配对比较 **2/2** |
| 搜索成本项（Search Cost Coverage） | 搜索器可以按独立Key和契约查询、合法评分 | **0**；singleton/link/whole-plan搜索用途未开放 |
| 完整方案成本 | 算子、依赖、通信、调度和执行边界共同形成的整体成本 | 尚未验收 |

三者不能互相替代。搜索需求集合与计费单位尚未固定，不计算“2/18”，不据此生成“补测16项”。KERV现有accepted只授权 `selected_fused_group_graph_batch_proxy` 局部比较用途，Native仍为comparison_only。（`cost_model/kerv_admission.py:16`）

## 阶段与出口

| 阶段 | 核心任务 | GPU | 出口要求 |
|---|---|---|---|
| B0 | Reference Identity Closure | 否 | 实现/布局身份机制、稳定Key和严格正负测试通过；已有CPU验收 |
| B1 | 搜索动作空间与可达放置枚举 | 否 | 合法分组、分组允许设备、节点可达设备、可跨卡边集合及合法方案见证 |
| B2 | 推导通信成本集合 | 否 | 对B1全部合法方案的跨卡边取并集，按实际方向/大小/同步语义生成唯一查询清单 |
| B3 | 冻结Fused/Native配对矩阵 | 否 | 同Case/Contract/Recipe，明确成本用途、布局声明及GPU准入必需证据 |
| B4 | BW1000身份捕获、成对测量与准入 | 是 | B0～B3通过；先捕获实际版本/源码/设备和布局，身份完整且与B3一致后再开始计时 |
| B5 | 通信成本与链路残差验证 | 是 | 通信符合B2动作域，单项/Native链/Fused链边界一致，组合与依赖验证通过 |
| B5.5 | 解析成本与正式Profile的搜索决策对照 | 否 | 同图/空间/约束/算法/参数，仅目标成本来源变化；输出决策差异及交叉评分 |
| B6 | FFN子图Exact / Guided / Sequential / Joint对照 | 视设计 | 冻结可信成本与相同合法空间，比较参考最优、初始/最终差距及后续搜索收益 |
| C | 混合覆盖下的完整计算图验证 | 另定 | 明确FFN实测与Attention/Norm/其他层等解析来源，另立验收；不冒充整图真实成本覆盖 |

当前B4入口关闭。B0本机CPU通过不等于BW1000目标Case身份已经捕获；CPU Key不能写进GPU Profile。B4的身份捕获是计时前的检查，不允许先计时后补身份。B0～B3先证明身份和准入流程可用；目标实际身份或布局与声明不符时，B4不得产生可消费的时延标签。

## B0四项验收

| 检查 | 已落地的验证 |
|---|---|
| torch.matmul、F.silu、tensor.mul的implementation.id/version明确 | 注册键为unfused:matmul/silu/mul，版本与源码来自实际加载Torch；Mul具体入口为Tensor.__mul__ |
| 布局状态与来源可追溯 | runtime_tensor读取真实Tensor；graph_declaration必须有source_id；unknown、缺来源与stride冲突均拒绝 |
| 实现版本、Shape、布局变化不错误复用Key | 逐项变更后原Profile精确查询不命中；绑定版本与实际运行环境不符时直接拒绝 |
| Admission正向与负向测试通过 | 完整身份+accepted单元夹具+匹配契约可消费；缺身份、conditional、错Contract、错误设备来源等拒绝 |

本机证据见 `data/reference_identity_b0_v1/identity.json` 与 `validation.json`。5个CPU Key无缺项，59项相关测试通过；该验证属于 `6005868` 基线，历史测试记录及其协议哈希不因本次文本修订而重写。测试中的accepted时延仅为单元夹具。生产singleton Profile尚未接受，两个旧P2配对Case和36条legacy_unbound状态不变。

## B1：由合法动作空间推导驻留

Placement是搜索决策。不能选一个参考方案后，将它的通信集合当成整个搜索空间的需求。

1. 固定FFN子图及入口生产者、出口消费者的边界约束；边界可以固定或可选，但必须明确允许范围。
2. 从实际registry及合法性约束得到允许的融合分组，保留对应Native分解。
3. 明确每个分组允许的GPU集合；每个方案必须是合法分组覆盖与合法设备赋值的组合。
4. 枚举全部合法动作形成的完整方案，得到节点可达设备及各边两端的可联合到达设备对。
5. 对每条可跨卡边保留至少一个合法方案见证，并对所有合法方案的跨卡边取并集。

只把生产者可达设备集合与消费者可达设备集合做笛卡尔积会忽略联合约束，可能引入不合法的跨卡需求；必须以完整合法方案或等价的精确可达性证明为准。禁止用采样、beam截断或一个参考放置替代全集覆盖。此处仅枚举结构，不评分，不运行MCTS。

B1产出：合法融合分组集合、分组允许设备集合、节点可达设备、带方向的可跨设备边集合、边界约束及合法方案见证。仅存在于融合组内部的边不计通信；同一条原图边在Native划分中仍可能跨卡。

B1首个场景已产出：入口/出口均GPU0，32 Native + 16 Fused = 48个带标签方案，24个不含边界的计算结构置换类；不合并评分。T1/T2/T3及独立核验见 `data/ffn_placement_b1/`、`cost_model/FFN_PLACEMENT_B1.md`。

## B2：通信成本并集

每条边至少记录：

| 字段 | 定义 |
|---|---|
| source_device / destination_device | 合法方案中该边生产者与消费者的设备 |
| tensor_shape / tensor_bytes | 实际边Tensor的Shape与传输字节数 |
| transfer_required | 该合法设备对是否需要跨卡；未解析时保持未定 |
| synchronization | 生产完成到消费者可用的依赖及同步边界 |
| witness_plan | 使此设备对合法的方案标识 |

只要存在一种合法方案使一条边跨卡，就必须能评价其成本；不要求所有边同时跨卡。两端同卡的动作不产生通信费用，但不能因此删除该边在其他合法动作下的传输需求。入口/出口约束必须一并覆盖，不能将边界外的传输隐含为零。

B2按方向、大小、dtype/layout、实际传输实现、双端身份与同步语义归并唯一查询；相同字节数并不自动代表相同传输契约。每个归并项追溯到B1原始边与见证方案。

0.5/1.5 MiB × 双方向是当前假设下的4种潜在类别，最终是否全部需要由合法动作空间确认。只在完整并集生成后确定通信Profile清单；不是测量某个参考方案的跨卡边。PP层切分仍归下游。

B2已核验4个声明请求覆盖160个跨卡边实例，46个有通信方案与2个零通信方案均被覆盖；数据见 `data/ffn_communication_b2/`。选定预分配 `Tensor.copy_`、目的流完成依赖与目标Graph Replay口径，布局来自显式 `empty_strided` 构造声明，不是生产者观测。B3配方和B4双端身份/真实route/捕获及Event完成证明仍缺，4个请求均非measurement_ready，无通信准入。详见 `cost_model/FFN_COMMUNICATION_B2.md`。

## B3：冻结配对矩阵

每个融合候选同时定义Fused与Native两臂：

- **Case_F = Case_N**：Shape、dtype、布局、参数/输入生成、设备与边界驻留一致。
- **Contract_F = Contract_N**：计时单位、执行模式、分配/提交是否包含、同步与输出保留语义一致。
- **Recipe_F = Recipe_N**：预热、重复、每轮调用量、配对顺序、正确性阈值、计时边界与统计规则一致。

Fused记录真实KERV路径，Native记录对应Reference分解；两路径分别取证。整个FFN的Fused臂仍包含GEMM，只有被替代的SiLU×Mul对应单个KERV Kernel。

GPU实测不能与解析估计配对，Graph Replay不能与Eager配对，一张GPU的数据不能直接替代另一张。按配对Case数量管理，Profile行数不决定实验数量。参数驻留、gate别名及旧值失效、各臂输出保留都属于Case/契约边界。

B3结果见 `data/ffn_measurement_b3/` 与 `cost_model/FFN_MEASUREMENT_B3.md`：18核心+1链路诊断，7有效轮×50调用、10预热；旧Fused缺新锚点/Recipe证据且输入/轮组上下文变化，两条均须重测，当前新测任务19而非17。用户确认最多9次尝试、取前7个锚点合格轮；IQR失败不替换样本补救。50 gate独立视图、1个共享只读up、每次Replay前计时外恢复；输入/别名均待GPU核验。清单7项CPU完成、1项HIP/跨卡Smoke待B4，正式计时仍禁止。

当前B4执行策略见 `cost_model/FFN_B4_SMOKE.md`：Smoke最小四类含0.5MiB 0→1 Copy，纯布尔管道验收，耗时仅诊断；失败立即保留traceback/pool/显存并停止。正式固定7次尝试、不补轮、不替换，逐格quality_failed后继续19格。这是用户对旧B3最多9轮规则的更新，通过新policy/recipe哈希生效，不改签旧产物。旧2/2在新审计中标superseded_by=c4fc415，历史证据与原准入保留。GPU实际运行仍待执行。

run03实际回传：最小Smoke 4/4通过，正式8/19格完成，跨卡诊断链Capture边界发生native段错误；计时Event与独立Trace跨度差异待核，未新增准入。当前B4 v2按10个轮组顺序隔离子进程，每组只启动一次；同组各臂保持共同锚点/配方，硬崩溃整组失败后继续其他组。新进程边界写入policy/Recipe，不与run03拼接替换；采样仍固定7尝试，不补轮。v2仅CPU验证，实际GPU及计时边界均待核验。
run04验证轮组隔离：19状态齐全、15格各7次样本，10 quality_passed/9 failed；链路capture_end段错误，计算Event在Profiler前后系统变化。暂停正式补测/准入，先执行独立Profiler控制与预分配链路诊断；不改既有阈值、不用预热补正式样本、不替换run04记录。见FFN_B4_RUN04_REVIEW.md。

当前v3规范见FFN_B4_V3.md：独立控制复现Profiler Replay既有Graph后Event计时状态改变；每轮组的全部正式/锚点采样都必须先于任何Profiler，后置Trace只影响路径准入，不覆写原始样本。组合跨卡链经单次动态/预分配输出两路径仍capture_end崩溃，按已知后端证据阻断，不重试、不回退；18核心格可继续。B5链路残差与完整FFN成本仍缺失。旧Recipe/旧P2决策保留，v3仅CPU验证，GPU结果待回传。

run05已核验v3：18核心格各7轮、0进程崩溃，9 quality_passed/9锚点失败+1组合链阻断；计时恢复稳定量级。当前不追加GPU、不调整阈值/补样本，先进行9候选的CPU Canonical Key导出与准入范围审核。新accepted仍0，FFN方案成本未闭合，B5.5/B6不启动。见FFN_B4_RUN05_REVIEW.md。

run05锚点CPU归因阶段（61557f7），见FFN_B4_ANCHOR_AUDIT.md。准入/覆盖报告主指标采用priceable_plans=0/48；Down0/1是各方案缺失的设备备选，并非每方案同时需要两者。去重后6/70轮组窗口、8/98设备前后对越线，全部在轮0～2；9格失败由共享判据与固定7轮传播。预热/正式中位比不能替代逐轮稳定性，现有时间/遥测缺口不足以区分纯协议/环境原因。该轮未改采集器/阈值、未新增GPU、未追溯升格；新锚点协议需另行预注册，不把记录口径修正当成本数据恢复。

锚点状态CPU分析/条件化设计见calibration/FFN_ANCHOR_CONDITIONING_DESIGN.md：Down/Up R3 pre首条相对后两条中位数的逐轮比值中位为+12.9705%/+4.7415%，post中位接近零，但41.411μs最大值在短主机间隔post，不能唯一归因空闲/时钟。建议正式成本明确条件化Graph batch与首发恢复两类条件；新设计只以Down0做A→B、B→A，唯一干预为每端点前0/1条件化Replay，三条测量全部保留、原预热位置及七轮/10%规则不改。仅CPU设计稿，无GPU collector/run入口、无自动采用或正式矩阵；29项CPU测试及三个产物重复字节通过，旧v2源/plan不变。

pilot01真机回传已CPU核验，见calibration/FFN_ANCHOR_PILOT01_REVIEW.md及data/ffn_anchor_pilot01_review：13/13来源匹配06af153、四格无崩溃且完整，pilot_passed=true但四格质量全部失败，R3质量闸门false。拒绝原因均为anchor漂移/端点IQR，target IQR均不超10%；R3 pre第一条14/14严格大于后两条。当前物理GPU0 UUID与run05不同，168项遥测全null/amdsmi不可导入，不能唯一判因或采用R3。旧样本/阈值/运行时不改，七项矩阵关闭；Mul0元数据Smoke仍独立待执行，不准入pilot数据。

测前修订采用collector.v2，见calibration/FFN_ANCHOR_CONTROL.md及data/ffn_anchor_control_v2：pilot_passed只验四格/锚点/时间戳/端点三类读数完整性；R3质量另列，不自动采用或开矩阵。报告首行Down0，n=1仅描述差值，不估计越线率、不证明采样有偏或Down/共享窗口因果。超过噪声判据尚未冻结，保持人工协议复核。另加独立Mul0元数据Smoke（calibration.ffn_mul_metadata_smoke），完整当前pilot原件CPU复核后才允许启动一次worker；无KERV探针，计时不参与通过，仍无准入/七项自动执行。58项CPU测试通过，本轮GPU执行0。

此前collector.v1已实现（2bc8f94），见calibration/FFN_ANCHOR_CONTROL.md及data/ffn_anchor_control_v1：默认CPU plan，显式run仅四格GPU0 pilot，父进程无GPU驻留；固定七轮、末尾Trace、UUID/build一致性与原始证据CPU复核已接通。新activation hook记录实际Mul left，但未在GPU执行，旧记录缺口不变。50项相关CPU测试通过；gpu_validated=false，七项后续矩阵仍关闭，无新准入或搜索。旧v3及冻结协议不改签。

前次CPU准备快照（c2f58a5），见FFN_B4_CPU_PREPARATION.md：run05九条conditional导出、八条完整边界/运行时身份可精确检查，Mul left元数据缺口保留；严格成本查询九条全部拒绝，priceable仍0/48。另冻结ffn-anchor-control.v1的1/3端点策略对照，原预热位置和七轮/10%规则保留，R3汇总/离散度阈值显式预注册；当时采集器尚未实现、GPU入口关闭；当前入口状态见上一段。旧B4 v3不改、新旧数据不混用。

## B4：准入不筛选性能赢家

**Profile Admission仅验证测量身份、正确性、来源与契约有效性，不以性能收益作为准入条件。Fused慢于Native时仍可准入，搜索空间必须保留Native作为合法替代。**

implemented表示该实现可执行，不代表必须融合。快、慢或持平均保留测量结果；是否选用由完整合法方案的成本决定。现有 `check_pair` 已明确不依赖融合收益（`cost_model/kerv_admission.py:26`）；当前动作空间要求每个节点存在singleton候选，并同时生成合法候选的设备动作（`cost_model/legal_search.py:97`、`cost_model/legal_search.py:112`）。这不自动扩大现有P2准入scope；Native独立搜索成本仍须另行准入，不能将comparison_only当作搜索标签。

## B5：通信与链路验证

验证B2所需方向与Tensor条件下的传输、生产者依赖和消费者可用边界；不能只计异步API返回。再验证同口径下的Native/Fused代表性链路，保留真实FFN分叉依赖及设备/流调度。

统一采用 `Δ_add = ΣT_single − T_chain`。逐轮配对计算后汇总，不用各项中位值相减代替。

| 历史seq256 Case | ΣT_single − T_chain（逐轮中位） | T_chain − ΣT_single（逐轮中位） |
|---|---:|---:|
| GPU0 | +0.887990 μs | −0.887990 μs |
| GPU1 | +1.075200 μs | −1.075200 μs |

正向值表示单项加和高于Native链路实测。来源为 `data/fusion_cost_validation_v1/inputs/gpu0_seq256/result.json` 与 `gpu1_seq256/result.json`，计算见 `calibration/probe_kerv_additivity.py:59`。历史8轮、仅保留末输出的诊断不能与P2的7轮、保留全部输出契约混用，也不能自动回写任意邻接修正。

## B5.5：成本来源是否改变搜索决策

只使用CPU读取已准入数据；不重新测量、不修改MCTS。Before固定为 `Analytic_target_contract`，采用与After相同的目标Graph Replay计时和分摊语义；launch及理想融合公式必须按该契约定义，禁止沿用旧eager解析Profile。After为同一目标Graph Replay契约下本研究范围内的正式Profile模式。研究对象均为同一个FFN子图，不扩成完整Qwen3图。

两次固定相同graph hash、registry及动作空间/约束、搜索算法及实现版本、全部参数、预算、seed、候选顺序与平局规则；唯一允许变化的是预先声明的目标成本来源。其他既定成本、内存与调度规则保持一致，解析回退逐项标记。After缺少目标范围所需的正式标签时拒绝实验，不悄悄用解析回退填补并宣称完整真实成本。

B1已确认本场景只有48个带GPU标签的合法方案；B5.5对全部48个方案分别计算两套成本及排名，不只比较两个Top-1。表身份、图、约束与方案集合固定，零GPU重测；当前仅约定实验，尚未评分。

记录Before方案A、After方案B各自的融合节点集合、GPU放置、预测成本、合法性、成本来源及Profile Key/Contract/record/decision身份。除各模式自身分数外，交叉评价 `C_analytic(A)`、`C_analytic(B)`、`C_profile(A)`、`C_profile(B)`，避免把不同成本尺度下的两个分数直接相减当成加速。

| 结果 | 可支持的判断 |
|---|---|
| 融合选择改变 | 当前Case的融合决策对成本来源敏感 |
| 放置选择改变 | 当前Case的设备分配对成本来源敏感 |
| 两者均不改变 | 当前Case决策稳定；不能单凭这一点证明解析成本数值准确或普遍足够接近 |

有随机性的算法使用相同预设seed组报告全部结果，不挑seed、不改参数制造变化。需要排除搜索误差时，可在同一小空间中用同算法的Exact对照核查；B5.5回答成本来源的影响，不据此宣称MCTS优于其他策略。预测选择变化也不等于实际端到端加速。

## B6与C的结论边界

B6限定为**FFN子图真实成本驱动的Exact / Guided Completion / Sequential MCTS / Joint MCTS对照**。B6以冻结的正式成本体系为共同评价器，目标是搜索算法价值；B5.5以固定算法比较成本来源，两者不合并成一次实验。

B6所需FFN singleton、KERV、Native替代计费、可达通信及组合规则必须分别具备适用的准入依据；下表是目标要求，不是当前完成情况：

| 范围 | B6需要的成本来源 |
|---|---|
| FFN 5个singleton × 2 GPU的可达用途 | 新准入的逐设备Profile |
| KERV融合组 | 与B6执行语义匹配的accepted Profile，现有局部proxy不自动扩展 |
| Native替代与链路 | 成对测量、明确组合语义与独立使用权限，不能双重计费 |
| 合法跨卡动作 | B2并集内正式通信成本与B5依赖验证 |

报告Exact参考最优与可行性，Guided/Sequential/Joint的Initial Gap、Final Gap、Search Improvement、首次最佳出现位置及预算；允许后续搜索无改进、Joint无额外收益。Exact只代表当前冻结成本模型的空间最优，不是未经执行验证的物理真值。

完整计算图的Attention、Norm、其他27层等若仍使用解析成本，放到后续C阶段，明确混合覆盖与回退，不称为整图真实Profile验证，不将FFN结论推广为完整模型收益。

## 当前执行点

B0实现及本机CPU验收通过，BW1000身份留待B4计时前捕获；B1声明空间及三表已完成（48/24/48）；B2声明与覆盖核验通过，B3 CPU矩阵与配方冻结，B4正式计时入口关闭；下一项为B4 Smoke和实际身份/路由/捕获/同步核验。B5、B5.5、B6与C均未执行。无新增GPU测量、Profile准入或搜索结果，SROP保持冻结。
