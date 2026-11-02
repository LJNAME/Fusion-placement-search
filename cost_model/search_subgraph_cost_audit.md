# Search Subgraph Cost Coverage Audit

阶段 A；Qwen3-0.6B Builder 的第 0 层 FFN；2026-11-02。未采集新 GPU 数据，未运行 Exact/MCTS，未修改准入状态。

## 覆盖结果

下表的 accepted 均指当前 P2 决策允许严格消费的成本，不能与历史采集器的完整性计数混用。

| 成本用途 | 需求 | 当前 accepted | 已有证据 |
|---|---:|---:|---|
| 完整 FFN 单节点 × 两卡 | 5 × 2 = 10 | 0/10 | 旧单算子表、旧组级实验；实现/契约未绑定到本次查询 |
| 其中 SiLU、Mul 单节点 × 两卡 | 2 × 2 = 4 | 0/4 | 旧四臂 Graph 诊断中有实测 |
| 选定 KERV 融合组 × 两卡 | 2 | **2/2** | P2 严格查询重放通过；仅局部 Graph batch proxy |
| Native SiLU→Mul 组合 × 两卡 | 2 | 0/2 | 原始记录 conditional_dev；comparison_only |
| 完整 FFN 的 Native/Fused × 两卡 | 4 | 0/4 | 本次证据包未覆盖 FFN 链路 |
| 两种边大小 × 两个传输方向 | 4 | 0/4 | 历史双向曲线各 6 点；未被本次契约准入 |

机器清单：`data/search_subgraph_cost_audit_v1/report.json` 的 `nodes`、`edges`、`fused_groups`、`singleton_cost_queries`、`chain_cost_queries`、`required_link_queries`。覆盖数按节点用途计算，不是待测独立 Kernel 数；逻辑边界输入不计入计算节点。

P2 明确设置 `singleton_costs_admitted=false`、`link_costs_admitted=false`、`hardware_profile_sharing=false`、`whole_plan_costs_admitted=false`。严格 overlay 遇到普通计算 singleton 或正字节通信会拒绝，不会偷偷退回解析估计。因此完整 FFN 的正式成本覆盖仍未闭合。（`cost_model/kerv_admission.py:16`、`cost_model/kerv_profile.py:229`、`cost_model/kerv_profile.py:238`）

## 子图与边界

来源是仓库的模型形状 Builder，不是 FlagScale 运行时捕获的真实计算图；输入测量使用合成边界 Tensor。完整 Builder 图 **591 节点**，28 层 FFN 仅 **1 种 Shape 模板**；本次重建图哈希与两份 P2 PlacedOpGraph 一致。hidden=1024，FFN width=3072，batch=1，seq=256，BF16，无 Linear bias。（`cost_model/search_evaluation.py:49`、`simulator/model_graphs/dense_layer.py:302`）

```mermaid
flowchart LR
  X["边界输入 X [1,256,1024]"] -->|"0.5 MiB"| G["Gate GEMM"]
  X -->|"0.5 MiB"| U["Up GEMM"]
  G -->|"1.5 MiB"| S["SiLU"]
  S -->|"1.5 MiB；融合后消除"| M["Mul"]
  U -->|"1.5 MiB"| M
  M -->|"1.5 MiB"| D["Down GEMM"]
  D -->|"0.5 MiB"| Y["边界输出；外部 residual_add_2"]
  subgraph K["KERV silu_mul 替代组"]
    S
    M
  end
```

| 节点（均带 L000. 前缀） | 语义及输出 | 实现状态 / Key 检查 |
|---|---|---|
| pre_mlp_norm | 已完成 Norm 的边界值 [1,256,1024] | 子图不计 Norm 算时或其参数；输入驻留位置必须另行固定 |
| mlp.gate_proj | GEMM (m,n,k)=(256,3072,1024)，输出 [1,256,3072] | reference 用 torch.matmul；无本次执行来源绑定；draft Key 有 implementation/layout 缺项 |
| mlp.up_proj | 同 GEMM Shape、同输出 | 独立图节点；semantic_slice=up，与 gate 的 Key 不同，不能默认共享 |
| mlp.silu | SiLU，[1,256,3072] | reference 用 F.silu；独立 Native Kernel 尚无 accepted Key |
| mlp.gated_mul | SiLU(gate) × up，同 Shape | reference 用 Tensor 乘法；独立 Native Kernel 尚无 accepted Key |
| mlp.down_proj | GEMM (256,1024,3072)，输出 [1,256,1024] | reference 用 torch.matmul；无本次执行来源绑定；draft Key 有缺项 |

三个 GEMM 参数各 **6,291,456 bytes**，总 **18 MiB**；权重按驻留参数处理，不凭空加入每次调用的搬运。扩展到权重分片或迁移时另建边和实现契约。reference executor 目前只执行 singleton；已有 KERV Adapter 只绑定选定融合组，尚无包含这三个 GEMM 的统一 FFN 融合执行器。（`execution/reference.py:124`、`execution/reference.py:195`、`integration/flagscale/kerv_execution.py:181`）

`GroupKernelKeys` 已能为每个 singleton 生成 draft hash，但缺 implementation.id/version 和未观测布局，不能将 draft hash 当成可信 Profile 身份。审计 JSON 保留每个节点×设备的完整 draft Key 和 identity_gaps；没有编造 source hash、原生 Kernel 绑定或布局。（`cost_model/fusion_kernel_profile.py:59`）

### 边的覆盖

| 边 | bytes | 放置变化的影响 |
|---|---:|---|
| pre_mlp_norm → gate_proj | 524,288 | 输入与 Gate 不同卡时传输 |
| pre_mlp_norm → up_proj | 524,288 | 输入与 Up 不同卡时传输；相同源不能未经验证就合并复制 |
| gate_proj → silu | 1,572,864 | 融合后仍为 Gate → KERV 的外部输入边 |
| silu → gated_mul | 1,572,864 | KERV 内部表达式，不再产生跨组 Tensor 边 |
| up_proj → gated_mul | 1,572,864 | 融合后仍为 Up → KERV 的外部输入边 |
| gated_mul → down_proj | 1,572,864 | 输出与 Down 不同卡时传输 |
| down_proj → residual_add_2（子图外） | 524,288 | 只有固定外部消费者位置后才能确定是否需传输 |

内部 **6 条边**，出口另 **1 条**。跨卡查询归并为 `(0→1, 524288)`、`(1→0, 524288)`、`(0→1, 1572864)`、`(1→0, 1572864)` 四类；不是每个方案都发生四次通信。激活核心子图仅涉及 1.5 MiB 的两个方向。

历史 `raw_measurements.json` 在两个方向各有 6 个尺寸点（4 KiB～256 MiB），可以被旧代价模型插值查询；这不等于通过同一 Measurement Contract 的准入。旧链路估计接口尚未经过 Kernel 使用的 MeasurementLabelLookup，当前 P2 overlay 则直接拒绝通信成本。（`cost_model/measured_profile.py:480`、`cost_model/kerv_profile.py:229`）

阶段 B 必须选择实际 copy/通信实现，固定双端 UUID、P2P 路径、Tensor layout、流、生产者完成到消费者可用的计时边界与同步方式；不能只计异步 API 返回。单次传输及其依赖要与单算子 Graph proxy 的组合口径验证。PP stage 边界由下游负责，本轮不把层间 Pipeline 通信纳入层内放置审计。

## 已准入的融合组

两卡均为 `gfx936:sramecc+:xnack-`，保持逐卡记录，不共享 Profile。以下为同卡配对、同契约的中位值（μs/call）；GPU0 与 GPU1 的完整 Contract ID 不相同，不能跨卡直接按同一个 ID 查询。

| 设备 | KERV accepted | Native comparison_only | KERV Canonical Key |
|---|---:|---:|---|
| GPU0 | 7.2511798143 | 10.3167796135 | sha256:2cfa617ace0ee3af1e7c9e12d8f99f6af409c3e64a575134e9be3679d74bca80 |
| GPU1 | 7.2320002317 | 11.2543594837 | sha256:d53d35cfc3cc280317d00b41ae897d5300c6f42be70fd7af959b891dee751d36 |

- GPU0 Contract：`sha256:f4cbdf92046f460ce1c8c76c3b35229d634717fab8a9323b604b274d608c9d70`。
- GPU1 Contract：`sha256:ce6b7d7f1a74fbbd95d58252764487ce5f7d707ef520585de5bfa0d7988af507`。
- 决策：`sha256:d515655fab5ce53f21bbea060298210d10c07a37b0b5a5e285a5ae4a47bdb795`；use=`selected_fused_group_graph_batch_proxy`。

每卡 7 轮 × 50 次，10 次预热；所有 50 个输出保留并逐轮检查，Event 包围 Graph replay，恢复 gate 在计时外。绑定与证据可从报告中的 record_id → source_record_id / admission decision → 原始 result、trace、placed 追溯。Native 两 Kernel 的组合总时间不能分摊成 SiLU 或 Mul 标签。（`data/kerv_profile_admission_v1/strict_cost_queries.json:1`、`cost_model/kerv_admission.py:26`）

KERV 原地覆盖 gate，Mul 输出与 gate 别名；此层 gate 在 SiLU 外没有额外消费者，SiLU 内部值也不被组外使用。搜索必须沿用 Adapter 的别名/旧值失效检查。两输入改为预置逻辑边界后可以严格查询局部融合成本；把输入 GEMM 算时或跨卡输入成本设为零并不适用于完整 FFN。

## 历史测量与组合误差

`data/fusion_cost_validation_v1/inputs/gpu{0,1}_seq256/result.json` 已有同 Shape 的四臂 Graph 诊断。以下数值均来自该旧实验，不能与上节 P2 的单项数值拼表评分。

| 设备 | SiLU 中位 | Mul 中位 | Native chain 中位 | Fused 中位 | 逐轮 ε 中位 | 逐轮 ε/chain 中位 |
|---|---:|---:|---:|---:|---:|---:|
| GPU0 | 7.289570 | 4.734360 | 11.033600 | 8.028760 | **−0.887990 μs** | −7.8884% |
| GPU1 | 7.555200 | 4.675200 | 10.731200 | 8.172800 | **−1.075200 μs** | −10.0466% |

这里 `ε_round = T_chain,round − (T_silu,round + T_mul,round)`，先逐轮计算再取中位；不是用各列中位相减。旧采集器报告的 residual 定义正好相反，本表已转换符号。（`calibration/probe_kerv_additivity.py:59`）

旧协议 **8 轮 × 50 次**，只保留最后一个输出；P2 是 **7 轮 × 50 次**，保留全部重放输出。旧 Contract `sha256:b0b0cd8a76c3bbe3d4924d718cc4d84807043c2c7643fdc12a7324a501f49c48` 也没有本次逐卡绑定的完整身份。这些是已有诊断数据，尚不能直接升级成当前 accepted 单算子/链路标签。（`calibration/probe_kerv_additivity.py:32`）

历史原始 BW1000 Kernel 表还有 **36 条 legacy_unbound**；本轮没有将其升格。（`data/fixed_profile_audit_v1/report.json:29`）

另有本地归档 `E:/Chatgpt_work/transfer/scnet-group-key-20261031-185721/received/group_key_mlp_dcu310_20261031_185721/formal/verified_profile.json`：`accepted_keys=12` 表示该轮 `development_isolated_kernels` 采集器的覆盖完整性，记录无本次 P2 admission 字段，且含旧设备身份与不同计时口径。它不是本表的 12 条正式 accepted；这里只保留历史来源，不参与覆盖分子。（该归档 `:5`、`:8`；SHA256 `4dbf612f07895a649e4529996c1e566fe7ab27b1e595e6b056d389711d87da40`，归档不在仓库内）

完整 FFN 的预测也不是全图耗时直接相加：结构模型按设备队列、前驱就绪、通信与共享链路资源推进时间；Gate/Up 的分叉是否并发取决于具体设备与执行流。单流链残差不能直接当成双卡并行 DAG 的修正项。（`cost_model/structural.py:194`、`cost_model/structural.py:234`）

## 阶段 B 的最小测量集

1. **先固定实验对象。** 激活核心用真实 Builder 节点 ID 与 Shape；分别说明两输入及输出的驻留位置。选择 Native 输出保留策略、单流/多流与计时边界，明确目标是局部 graph_batch_proxy 还是串接后的子图成本。若改变 P2 契约，重新生成并准入相应标签，不沿用旧 accepted 状态。
2. **激活核心先补语义身份。** SiLU/Mul × 两卡共 4 个 singleton 用途；先复核历史四臂数据可否满足新契约。不能满足时，同批采集 SiLU、Mul、Native chain、Fused × 两卡，共 8 个臂，计算逐轮 ε；不是从旧 Native 总时间倒推单项。Native 组合另行审核。若暂时只允许整段 Native/Fused 两个替代候选，则先补 Native 组合准入与组合语义，不宣称支持两单节点独立放置。
3. **再扩 GEMM。** 三个节点 × 两卡共 6 个身份用途，数学 Shape 只有两种 × 两卡共 4 个模板；Gate/Up 的语义字段、布局、实现及绑定一致性审核通过后才考虑共享实测。随后采集两卡各自完整 FFN Native/Fused，共 4 个链路臂，验证实际调度边界下的组合预测；两个分支要保留真实依赖。
4. **最后开放跨卡放置。** 在输入/出口位置确定后，审查 0.5/1.5 MiB × 双方向这 4 类传输，并验证至少一个含实际生产者→传输→消费者的代表性链路。四类大小/方向不足以证明任意重叠、资源争用或 Pipeline 成本，后续仅开放已验证的执行域。

最小集按上述阶段递增，不要求现在批量重测所有历史 Kernel。没有完整共同契约、组合验证与通信来源之前，不生成“真实 Profile 最优值”，也不启动 MCTS 对照。

28 层当前 FFN 形状相同，不能先验认定逐层融合收益有差异；两卡局部融合收益不同也尚不能证明最优放置改变。后续 Exact Oracle 应使用届时已准入的成本域，并检查融合选择是否改变层内分组或设备放置；14/14、16/12 的 PP 层切分仍由下游处理。

## 重放

在仓库根目录执行：

```powershell
& 'C:/Users/LJ/miniconda3/envs/fusion-dev/python.exe' -m tools.audit_search_subgraph_cost
```

脚本只重建图、核对已保存证据及严格查询、导出清单。它会检查图快照一致、28 个可执行 KERV 位置、目标组一致、两次 accepted 精确查询、两次 Native 严格拒绝，以及 10 个 singleton 缺项；不调用 GPU，不变更原始测量与准入决策。JSON 中保存输入文件 SHA256。Pinned FlagScale 源码需位于既有相邻目录，CPU 重放不加载 Triton 执行模块。
