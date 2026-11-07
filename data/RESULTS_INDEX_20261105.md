# 实验结果索引（2026-11-05 一轮）

本轮所有实验产物都在 `data/` 下，一个目录对应一次采集或一次CPU分析。说明文档：[标定记录](../calibration/STAGE_SIZE_CALIBRATION.md)、[搜索日志](../cost_model/FUSION_PLACEMENT_SEARCH_LOG.md)。

目录约定：

- `raw/`：服务器上原始输出目录的逐字节副本，含 `plan.json`、`attempt_*.json`、`process.log`。
- `review/evaluation.json`：CPU复核结果，按原始数据重算质量与身份。
- 其余子目录（`fit/`、`validation/`、`diagnose/`、`ffn_search/`）：基于该批数据的后续CPU分析。

"采集提交"即 `plan.json` 中记录的 git revision。stage_size batch-01 采集于未提交的源码，以 `plan.json` 中逐文件 SHA256 溯源，与提交 `a147a8ac` 一致。

## GPU采集

| 目录 | 内容 | 采集提交 | 质量 | 用途 |
| --- | --- | --- | --- | --- |
| `native_stage_calibration_20261105_01` | GPU0 SiLU、Mul独立阶段，S=256 | 50340281 | 2/2 通过 | 实测Profile（跨层复用） |
| `stage_size_calibration_20261106_01` | batch-01：SiLU/Mul/整链五尺寸，四组Copy | cab527d8＋未提交源码 | 16/19，Copy 3组失败 | `T(N)=a+bN`拟合与留出验证 |
| `stage_size_followup_mul_20261106_01` | batch-02：Mul S=192/256/384/768 | 54383825 | 5/8 | 偏差诊断；噪声批次，不进拟合 |
| `stage_size_batch03_mul_20261106_01` | batch-03：S=256/384/448/512，负载门槛 | 77028772 | 8/8 | 确认Mul跳变位于448～512 |
| `mul_holdout_20261106_01` | Mul独立留出 S=320/416/640/896 | ccd61424 | 4/4 | 两段模型留出验证 |
| `copy_sdma_default_20261106_01` | 四组Copy，默认SDMA（主契约） | a4af9f84 | 0/4 | 标记unstable，不作Profile |
| `copy_sdma_disabled_20261106_01` | 四组Copy，HSA_ENABLE_SDMA=0（对照契约） | a4af9f84 | 4/4 | 仅对照，不替换默认通信成本 |
| `gemm_projection_20261106_01` | Gate/Up/Down × GPU0/GPU1 | c863d286 | 5/6，Down GPU1失败 | 实测Profile |
| `gpu1_cost_completion_20261106_01` | GPU1 SiLU/Mul/LightOP SiLU×Mul/Down；Down GPU0同批参照 | f9e7012b | 5/5 | 实测Profile（Down GPU0仅参照） |
| `add_rmsnorm_calibration_20261106_01` | GPU0 LightOP Add＋RMSNorm完整调用、Native Add、RMSNorm、Native整链 | 30e54e52 | 4/4 | 实测Profile（整链仅对照） |

## GPU诊断与检查（非标定数据）

| 目录 | 内容 |
| --- | --- |
| `copy_warmup_diagnostic_20261106_01` | Copy每轮6次Replay诊断，默认SDMA及 `sdma0/`（SDMA关闭）；含探针脚本 `probe_source.py` |
| `add_rmsnorm_numerics_20261106_01` | LightOP Add＋RMSNorm在GPU0/GPU1的正确性检查，不计时 |
| `plan_previews_20261106/` | 采集前的计划预览（plan阶段输出），无GPU执行 |

## CPU分析与搜索结果

| 目录 | 内容 | 产生提交 |
| --- | --- | --- |
| `ffn_comm_instability_v1` | 默认SDMA通信不稳定性对48方案的影响 | 28dd024b |
| `ffn_gemm_rescoring_v1` | 接入GEMM实测后的48方案Exact/MCTS | f71a264c |
| `ffn_gpu1_rescoring_v1` | 计算成本全部实测后的48方案Exact/MCTS | 2a9bfcfe |
| `mul_piecewise_v1` | Mul两段拟合与冻结的留出预测 | f71a264c |
| `search_scale_model_v1` | 整模型搜索规模（MCTS/Beam，解析代价） | d2545f46 |
| `joint_vs_staged_v1` | 联合 vs 两种顺序搜索；中等子图含Exact | d2545f46 |
| `fused_placed_export_v1` | 整模型方案导出示例（默认规则；含不可部署方案） | d2545f46 |
| `executable_search_model_v1` | 只用可执行融合，解析代价 | 3963f2e6 |
| `executable_search_model_measured_v1` | 跨层实测Profile，64次迭代，单种子 | bff25764 |
| `executable_search_model_measured_v2` | 加入Add＋RMSNorm实测，256×3，修复前 | 751946c5 |
| `executable_search_model_measured_v3` | 修复rollout偏差与顺序搜索后，256×3；含各模式导出图 | 45211004 |

`executable_search_model_measured_v3` 是当前结论所依据的结果：三种模式均返回已知最佳参考方案（不融合、全GPU0、8.7730 ms）。`v1`、`v2` 保留为修复前对照。

## 仓库外副本

服务器 `~/private_data/LJ/` 下保留着各采集的原始输出目录，以及 `/tmp` 下的运行日志。2026-11-05核对结果：

- 11个GPU采集目录与本仓库 `data/*/raw` 逐字节一致。
- `/tmp` 下9个GPU运行日志与对应 `process.log` 一致，4个搜索日志已作为 `process.log` 归档。
- 3个原先未归档的目录（Copy诊断、两个计划预览）已复制入库。

这些仓库外副本均可删除，但未删除，由用户决定。`/tmp/start.log` 是Jupyter启动日志，与实验无关；`/tmp/jvs_smoke*`、`/tmp/arp` 等是冒烟测试的临时目录，不含结果。

## 2026-11-07 CPU对照

`search_quality_fixed64_v1`：同一591节点图与冻结成本，三模式×三种子、每次64个实际评分；report.json、summary.csv、逐次convergence.csv。原v2/v3结果不改。说明见[固定预算对照](../cost_model/SEARCH_QUALITY_FIXED64.md)。

`search_quality_qwen25_3b_audit_v1`（2026-11-07）：Qwen2.5-3B的P0输入与双卡成本来源审计。687节点、当前可执行融合0、每卡651计算解析回退；没有运行Fixed-64或GPU。见[输入审计](../cost_model/SEARCH_QUALITY_3B_INPUT_AUDIT.md)。

`3b_fusion_backend_audit_v1`：KERV双输出AddRMS入口与官方LightOP最新可见包离线审计，目标Shape尚无新GPU证据。见[后端审计](../cost_model/THREE_B_FUSION_BACKEND_AUDIT.md)。

`kerv_add_rmsnorm_2048_probe_v1`：固定KERV adaptive源码与36处图上覆盖安全的CPU计划，未运行GPU、未产生Profile。执行说明见[2048单Case验证](../cost_model/KERV_ADD_RMSNORM_2048_PROBE.md)。

`kerv_add_rmsnorm_2048_02`：GPU0单Case正确性/原地语义/唯一Kernel trace回传，raw保留6原文件，review含18项CPU复核。三次手写参考误差0、aten误差0.03125，声明容差通过。无计时、Capture、Profile准入或图Adapter。见[KERV回传](../cost_model/KERV_ADD_RMSNORM_2048_PROBE.md)。

`kerv_add_rmsnorm_adapter_v1`：现有3B图的36处KERV AddRMS条件映射；CPU参考PlacedOpGraph、实现身份及逐组I/O报告，序列化往返与36组bind通过。新增31项/相关56项CPU测试通过，无新GPU、Profile或默认搜索变更。见[KERV图Adapter](../cost_model/KERV_ADD_RMSNORM_ADAPTER.md)。

`kerv_add_rmsnorm_adapter_smoke_v1`：原探针的adapter选项CPU计划，仅第0层一个组；相关59项CPU测试通过。GPU未运行，无新计时/Profile或搜索。

`kerv_add_rmsnorm_adapter_01`：GPU0第0层单组Adapter正确性回传，raw保留6文件，review有22项离线复核。36项布尔检查/双参考数值/唯一Kernel trace通过；无计时、Capture或Profile。见[KERV Adapter回传](../cost_model/KERV_ADD_RMSNORM_ADAPTER.md)。

`kerv_add_rmsnorm_timing_plan_v1`：GPU0 AddRMS2048单Case Graph Replay成本采集的CPU计划，尚无GPU数据。固定50调用×7轮/10预热，恢复在计时外，条件候选而非自动准入。见[单Case测时](../cost_model/KERV_ADD_RMSNORM_TIMING.md)。

`kerv_add_rmsnorm_timing_01`：GPU0第0层单组AddRMS Graph摊销成本回传，median5.353559851646423us，IQR/Median0.6873858%；13原文件和独立准入/查询证据。accepted新增1条，仅该Key/Contract/组/设备条件，成本查询返回measured_group_kernel；无默认搜索或整图覆盖变更。见[测时与准入](../cost_model/KERV_ADD_RMSNORM_TIMING.md)。

`qwen25_add_rmsnorm_graph_v1`：3B直接模式37处审计（36静态兼容但信息不足、1个final_norm不支持）；逐设备成本与显式条件规划MCTS接线，实际Profile复用0。三模式各64评分，独立小切片Exact/MCTS一致；无GPU或真实加速结论。见[3B图级接线](../cost_model/QWEN25_ADD_RMSNORM_GRAPH.md)。

- 2026-11-07：Native Add→RMSNorm 单条对照入口，GPU0第0层固定Graph7×50；KERV不重测。CPU plan：data/native_add_rmsnorm_timing_plan_v1/plan.json；仅审计：data/qwen25_add_rmsnorm_reuse_check_v1/capability_cost_audit.json；说明cost_model/NATIVE_ADD_RMSNORM_COMPARISON.md。Native GPU读数待回传，准入新增0，36处模型运行时证据仍不足。
- 2026-11-07 Native AddRMS回传：data/native_add_rmsnorm_timing_01，accepted完整链10.217599869us，KERV旧5.353559852us，减少47.604526%；review/summary.json、profile.json、cost_query.json及strict_reuse_audit。7×50/12源码SHA/Key/契约通过，无重测/新搜索。模型实际布局/别名/执行契约仍缺，图复用0；benchmark对照闭合，非36层或整图性能结论。
- 2026-11-07：data/qwen25_static_joint_v1，0.6B式静态执行声明的3B严格联合搜索。36处GPU0复用Native/KERV成本，GPU1缺成本对保持Native；一次joint/seed0，64请求32实评32拒绝，36融合全GPU0，预测25.239290006ms（Native参考25.414395447ms）。joint_placed.json及joint_fused_placed.json往返与重评分一致。未新增测量/完整模型执行，详见cost_model/QWEN25_STATIC_SEARCH.md。