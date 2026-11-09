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
- 2026-11-07 同构硬件复用：data/qwen25_hardware_class_joint_v1；GPU0/GPU1指纹及栈相同，72个可查成本对（36同卡、36迁移），原测量仍2条。joint64实评/0设备拒绝，GPU1候选查询378次，最终36融合全GPU0、预测25.239290006ms，导出一致；未测GPU1 AddRMS或新增测量。说明cost_model/PROFILE_HARDWARE_CLASS.md。
- 2026-11-07 3B放置/编号审计：data/qwen25_placement_audit_v1。A/B/C预测25.239290006/25.308636502/25.255440040ms，B无计算重叠并多60.646081us通信。评分器重标号不变，但一次交换搜索仍逻辑0，错过已知参考16.150034us，确认有限预算编号偏好。无GPU或算法修复；说明cost_model/QWEN25_PLACEMENT_AUDIT.md。
## 2026-11-07：显存审计

`qwen25_memory_audit_v1/`：687节点/866边字节校验、三个固定放置的现有前向峰值、逐张量逻辑生命周期及17项CPU测试回执。三方案低于配置总容量，训练及实际空闲预算可行性未知；18/18跨卡共享embedding/head权重遗漏。没有新GPU、MCTS或生产算法改动。说明：[QWEN25_MEMORY_AUDIT.md](../cost_model/QWEN25_MEMORY_AUDIT.md)。

## 2026-11-09：结构化放置动作空间

`stage_search_medium_v1/`：3B第0层的三个既有中等子图，原空间Exact与Stage Exact、Random/Sequential/Current Joint/Stage Joint，64次独立实际评分上限、3种子；72次启发式运行，6个任务含2个完整空间不可行项。包含逐次Gap、唯一候选、跨卡边和最佳图。连续阶段空间14–38项可全覆盖，但MLP限制损失为6.53%／14.99%，不宣称新空间必优。说明：[STAGE_PLACEMENT_SEARCH.md](../cost_model/STAGE_PLACEMENT_SEARCH.md)。

## 2026-11-09：分支与混合动作对照

`branch_search_medium_v1/`：相同三子图、6任务、4种Joint、64实评上限、3种子；含原域及受限域Exact、分支重叠和融合—放置反事实耦合。MLP cap75中位Gap由原Joint5.65%降至分支0.0145%／混合0；混合完整覆盖60项，尚非大空间效率证据；未观察到融合偏好翻转或共同域最优放置变化。74项CPU测试与冷缓存重放通过，36条旧控制结果不变。说明：[BRANCH_PLACEMENT_SEARCH.md](../cost_model/BRANCH_PLACEMENT_SEARCH.md)。

## 2026-11-09：预算曲线与真实耦合候选

`hybrid_budget_v1/`：8/16次实际评分主实验96运行，保留混合Joint未优于Random的结果；单独标注探索性64轨迹跟进，MLP free达1%目标的中位评分数为混合21／原Joint51／同域Random30。`coupling_evidence_audit_v1/`：真实KERV两卡原始矩阵点估计最优卡翻转，但0.019μs差异的样本范围重叠，尚未确认。说明：[HYBRID_BUDGET_AND_COUPLING.md](../cost_model/HYBRID_BUDGET_AND_COUPLING.md)。18项CPU测试通过，无新GPU或完整3B。

## 2026-11-09：完整 Qwen 双模型搜索

`full_joint_models_v1/`：Qwen3-0.6B（591节点）／Qwen2.5-3B（687节点），四法×两个容量场景×三种子，48运行、每次64实评；summary、逐次trace、最佳Plan及源码哈希。3B cap75混合Joint30.23505ms，原Joint47.04910ms；Qwen3 cap75混合Joint／同域Random均0/3找到可行解，事后单个合法见证保留。21项CPU测试及全部计费／导出离线检查通过。仅前向解析预测，非完整模型GPU执行，参考差距非Exact Gap。说明：[FULL_QWEN_JOINT_EXPERIMENT.md](../cost_model/FULL_QWEN_JOINT_EXPERIMENT.md)。
## 2026-11-09：Qwen3可行性专项审计

`qwen3_feasibility_audit_v1/`：完整构造witness、591合法动作路径、主干／分支显存下界、同先验固定预算对照及最佳图。24运行／1536评分；两个混合方法后续可行数均0→61/61，Joint／Random中位15.132347／15.132019ms，逐节点Joint仍10.240171ms。59项CPU测试，12旧控制前缀及24运行预算／源码／导出校验通过。新增GPU为0；静态前向可行性不等于训练或真实部署。说明：[QWEN3_FEASIBILITY_AUDIT.md](../cost_model/QWEN3_FEASIBILITY_AUDIT.md)。