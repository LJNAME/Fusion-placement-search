# 阶段 B 执行规范

2026-11-02 生效。顺序固定为 **B0 → B1 → B2 → B3 → B4 → B5 → B6**；B0～B3不使用GPU，B4开始才使用GPU。当前只落实B0实现及CPU验收，后续阶段未启动。

## 三种成本语义

| 概念 | 能证明什么 | 当前状态 |
|---|---|---|
| 比较证据（Evidence Coverage） | 指定Case和测量契约下Fused/Native的性能差异 | 两卡选定KERV配对比较 **2/2** |
| 搜索成本项（Search Cost Coverage） | 搜索器可以按独立Key和契约查询、合法评分 | **0**；singleton/link/whole-plan搜索用途未开放 |
| 完整方案成本 | 算子、依赖、通信、调度和执行边界共同形成的整体成本 | 尚未验收 |

三者不能互相替代。搜索需求集合与计费单位尚未固定，不计算“2/18”，不据此生成“补测16项”。KERV现有accepted只授权 `selected_fused_group_graph_batch_proxy` 局部比较用途，Native仍为comparison_only。（`cost_model/kerv_admission.py:16`）

## 阶段与出口

| 阶段 | 工作 | 出口要求 |
|---|---|---|
| B0 | Singleton Identity Closure | 三个Reference实现id/version绑定到加载环境；输入、输出、weight布局有来源；稳定Key；严格正负测试通过 |
| B1 | FFN边界驻留确定 | 输入生产者、各节点、输出消费者的设备位置有明确Case；不能将边界输入当免费搬运 |
| B2 | 真实跨卡边清单 | 由B1的放置Case枚举跨卡边，记录必要性、大小与同步；生成实际所需的通信Case |
| B3 | Fused/Native配对矩阵 | 按Case管理，明确两条执行路径；Case、Contract、Recipe一致；成本用途与准入条件明确 |
| B4 | 成对补测与Profile Admission | B0～B3全部通过；先检查准入所需身份和证据能被采集，再生成GPU时延数据；未审核记录不能用于正式搜索 |
| B5 | FFN链路残差验证 | 同边界下检验单项加和与Native链路，并验证Fused链路；不直接回写任意邻接修正 |
| B6 | 真实子图Cost Model验证 | Key/Contract精确命中、完整计费与溯源通过后，才验证Exact/MCTS参考结果 |

当前GPU补测入口关闭。B0本机CPU通过不等于BW1000目标Case身份已经闭合；CPU Key不能写进GPU Profile。目标身份或布局证据尚未取得时保留未完成状态，不能通过先计时后补身份来绕过入口要求。

## B0四项验收

| 检查 | 已落地的验证 |
|---|---|
| torch.matmul、F.silu、tensor.mul的implementation.id/version明确 | 注册键为unfused:matmul/silu/mul，版本与源码来自实际加载Torch；Mul具体入口为Tensor.__mul__ |
| 布局状态与来源可追溯 | runtime_tensor读取真实Tensor；graph_declaration必须有source_id；unknown、缺来源与stride冲突均拒绝 |
| 实现版本、Shape、布局变化不错误复用Key | 逐项变更后原Profile精确查询不命中；绑定版本与实际运行环境不符时直接拒绝 |
| Admission正向与负向测试通过 | 完整身份+accepted单元夹具+匹配契约可消费；缺身份、conditional、错Contract、错误设备来源等拒绝 |

本机证据见 `data/reference_identity_b0_v1/identity.json` 与 `validation.json`；Key重新构造和JSON往返稳定。测试中的accepted时延仅为单元夹具，不是新实测或新准入。生产singleton Profile尚未接受，两个旧P2配对Case和36条legacy_unbound状态不变。

## B1与B2的边记录

每条候选边保留以下字段，不能仅用“型号相同”忽略传输：

| 字段 | 定义 |
|---|---|
| source_device | 该Case中的生产者设备 |
| destination_device | 该Case中的消费者设备 |
| tensor_shape | 实际边Tensor的Shape |
| tensor_bytes | 实际传输字节数 |
| transfer_required | 两端是否不同设备；驻留未定时不能填写false |
| synchronization | 生产完成到消费者可用的依赖及同步边界 |

0.5/1.5 MiB × 双方向只是潜在类别。两端同卡不生成传输任务；只有固定Case后枚举出的真实跨卡边才进入测量集合。出口要考虑下游消费者位置，边界外的生产与消费不能隐含为零成本。PP层切分仍归下游，不纳入本轮层内放置任务。

## B3的配对条件

每个融合候选同时定义Fused与Native两臂：

- **Case_F = Case_N**：Shape、dtype、布局、参数/输入生成、设备与边界驻留一致。
- **Contract_F = Contract_N**：计时单位、执行模式、分配/提交是否包含、同步与输出保留语义一致。
- **Recipe_F = Recipe_N**：预热、重复、每轮调用量、配对顺序、正确性阈值、计时边界与统计规则一致。

Fused记录真实KERV路径，Native记录对应Reference分解；执行路径分别取证。不能把整个FFN的Fused臂称为“只有一个Kernel”：只有被替代的SiLU×Mul是单个KERV Kernel，相邻GEMM仍存在。

GPU实测不能与解析估计配对，Graph Replay不能与Eager配对，一张GPU的数据不能直接替代另一张。按配对Case数量管理，Profile行数不决定实验数量。参数是否常驻、gate别名及旧值失效、各臂输出保留等都属于Case/契约边界，不能遗漏。

## B5的统一差值

统一采用 `Δ_add = ΣT_single − T_chain`。逐轮配对计算后汇总，不用各项中位值相减代替。

| 历史seq256 Case | ΣT_single − T_chain（逐轮中位） | T_chain − ΣT_single（逐轮中位） |
|---|---:|---:|
| GPU0 | +0.887990 μs | −0.887990 μs |
| GPU1 | +1.075200 μs | −1.075200 μs |

正向值表示单项加和高于Native链路实测；来源为 `data/fusion_cost_validation_v1/inputs/gpu0_seq256/result.json` 与 `gpu1_seq256/result.json`，原计算见 `calibration/probe_kerv_additivity.py:59`。历史8轮、仅保留末输出的诊断不能与P2的7轮、保留全部输出契约混用，不能用于自动修正搜索分数。

## 当前执行点

B0实现与本机CPU身份验收已有证据；目标BW1000身份仍未观测。B1/B2/B3待开展，B4入口关闭；无新增GPU测量、时延Profile、完整方案准入或搜索结果。MCTS不修改，SROP保持冻结。
