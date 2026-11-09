# 融合—放置搜索实验日志（2026-11-05）

本轮从代价模型标定转向融合与设备放置搜索。所有结果目录的索引见 [data/RESULTS_INDEX_20261105.md](../data/RESULTS_INDEX_20261105.md)。标定部分的逐批记录见 [STAGE_SIZE_CALIBRATION.md](../calibration/STAGE_SIZE_CALIBRATION.md)，本文不重复，只记录与搜索相关的工作与结论。

全程约定：先冻结计划/预测再采集；原始数据全部保留，不剔除异常值、不补测；新批次使用新目录；GPU专属参数与通用解析模型分离；代价模型公式与MCTS选择/回传公式不改。服务器上的提交统一使用仓库作者身份。

## 结论摘要

1. **整模型可运行。** Qwen3整模型（591节点、2张BW1000）上，MCTS每次评估约41 ms，256次迭代约1 min；所有返回方案合法且满足显存约束。Beam在该规模不可用（宽度2/4需97/190 s，且返回不融合方案）。
2. **只用可执行融合。** 默认规则中 `silu_mul`、`elementwise_chain` 仍为planned，早期整模型最优方案（28个planned `silu_mul`）不可部署。改用 `executable_registry()`（单节点规则＋LightOP `silu_mul`＋LightOP `add_rmsnorm`）并开启 `require_executable=True`。
3. **实测改变了融合结论。** GPU0上两种LightOP融合都比Native慢：SiLU×Mul完整调用17.81 μs对Native 10.17 μs；Add＋RMSNorm完整调用（含两次输入复制）10.26 μs对Native整链7.12 μs。解析模型对这两项方向都判断错误。
4. **跨层复用后，最优方案为不融合、全部放GPU0。** 已知参考方案中最好的是8.7730 ms（不是全空间精确搜索证明的全局最优）。
5. **搜索偏差已定位并修复。** 原MCTS每个叶子的确定性补全总是"尽可能融合"，随机补全又因设备均匀随机而远差于它，导致联合搜索稳定落在最差参考方案上。新增可选的"不融合、保持设备"补全（默认关闭），并修复顺序搜索丢弃第一阶段方案的问题后，三种模式在3个种子下均达到已知最佳参考，评估次数由768增至1024–1026。
6. **联合搜索与顺序搜索在当前问题上结果相同。** 中等子图的精确最优五种模式一致；整模型修复后三种模式一致。联合搜索只在低预算时（MLP 90%容量、16次迭代）收敛更快。没有构造只利于联合搜索的测试图，也未启用 `elementwise_chain`。

## 1. 搜索规模（Phase 1）

数据：`data/search_scale_model_v1`；驱动 `tools/evaluate_search_scale.py`；提交 `5fcc1c6e`、`d2545f46`。解析代价，默认融合规则。

| 方法/模式 | 迭代 | 最优 / ms | 耗时 / s | 评估次数 |
| --- | ---: | ---: | ---: | ---: |
| MCTS 联合/先融合后放置/只融合 | 16 / 64 / 256 | 9.9046 | 2 / 8 / 31 | 48 / 192 / 768 |
| MCTS 只放置 | 16 / 64 / 256 | 10.1878 | 2 / 8 / 31 | 48 / 192 / 768 |
| MCTS 联合（参考预算） | 1024 | 9.9046 | 126 | 3072 |
| Beam 联合，宽度2 / 4 | — | 10.1878 | 97 / 190 | 2424 / 4808 |

16次迭代即达到1024次的结果：首个确定性补全直接生成该方案，问题对搜索而言过于简单。超过100节点的图评分缓存关闭。

## 2. 顺序搜索基线与联合搜索对比（Phase 2–3）

代码：`cost_model/search_evaluation.py` 新增 `placement_then_fusion` 模式（`ebb22e68`），与已有"先融合后放置"相同，预算对半分配；第二阶段固定每个节点的设备，拒绝跨设备融合。原四种模式不变。数据：`data/joint_vs_staged_v1`；驱动 `tools/evaluate_joint_vs_staged.py`（`ef491002`、`8962c6c0`）。

显存约束复核：参数按组只计一次并全程驻留，激活从产生到最后消费者，无重复计数。9个容量配置中5个在整个联合空间内无可行解（穷举放置复核），驱动先用精确搜索判定，直接标记infeasible，不再搜索：

| 配置 | 原因 |
| --- | --- |
| serial_tail 90% / 75% | 单个 out_proj 权重4 MiB超过单卡上限 |
| attention_fork 90% / 75% | q_proj所在卡：4 MiB权重＋0.5 MiB输入＋1 MiB输出 > 上限 |
| mlp 75% | 256种放置中最优仍超0.4985 MiB |

可行配置结果（与联合精确最优的最大差距）：

| 配置 | 只放置 | 只融合 | 先融合后放置 | 先放置后融合 | 联合 |
| --- | --- | --- | --- | --- | --- |
| serial_tail / attention_fork 实际容量 | 0 | 0 | 0 | 0 | 0 |
| mlp 实际容量 | +8.71% | 0 | 0 | 0 | 0 |
| mlp 90%，16次迭代 | 10.72% | 不可行 | 21.21% | 14.45% | 3.73% |
| mlp 90%，64次迭代 | 0.007% | 不可行 | 0.007% | 3.72% | 0.007% |
| mlp 90%，256次迭代 | 0.007% | 不可行 | 0.007% | 0.007% | 0.007% |

整模型上"先放置后融合"同样为9.9046 ms。

## 3. 融合—放置图输出

`simulator/fused_placed_export.py`（`c00dbb6d`）：从现有 `PlacedOpGraph` 导出框架无关的组级DAG，包含设备、融合组（原节点、设备、模式、后端、实现状态、是否可执行）、组间依赖与张量字节、每条跨卡依赖对应的点对点拷贝、每卡执行顺序、预测时间（标明为预测）。校验节点覆盖、拷贝与依赖一致、执行顺序满足依赖、可部署标记一致。`save_placed_op_graph` 增加可选 `registry` 参数以保存非默认规则的方案（`3963f2e6`）。

已知缺口：尚无自动并行模块读取该格式；`integration/flagscale/plan_to_conf.py` 只支持层级切分。

## 4. 可执行融合规则与Add＋RMSNorm适配

`integration/lightop/add_rmsnorm.py`（`290f0d0c`）封装海光 `lightop.fused_add_rms_norm`。厂商kernel原地写两个输入（residual←x＋residual，x←RMSNorm(residual)×weight），适配器先复制到私有缓冲，图输入保持只读；一个组输出Add结果（残差流，可有其他消费者）和RMSNorm结果。限定BF16、[1,256,1024]、同层、前向；eps按Qwen3声明为1e-6（图中未记录）。整模型匹配28处 `residual_add_1→pre_mlp_norm`；`residual_add_2→下一层input_norm` 跨层且经过输入节点，不融合。

GPU正确性（`data/add_rmsnorm_numerics_20261106_01`）：GPU0、GPU1均通过，输入不变、重复调用确定、两路输出在BF16容差内。

## 5. 实测Profile跨层复用

`cost_model/layer_invariant_profile.py`（`090bef4d`）：匹配条件为算子类型、Shape、dtype、布局、GPU序号与软件栈、实现Recipe与源码版本、执行契约类别（Graph Replay、每次50调用、中位数），不含节点名称；节点名称在图与结果中保留。同条件的多次测量（Gate与Up形状相同）仅在相差≤10%时取均值。未命中回退解析成本并记录原因，例如 `no_record:fused:add_rmsnorm:gpu1`。在FFN案例上复现原48方案结果（最优相同、MCTS与精确一致，最大差0.21 μs）。

当前收录的记录（均为审核通过的Graph Replay中位数）：

| 条件 | GPU0 / μs | GPU1 / μs |
| --- | ---: | ---: |
| SiLU [1,256,3072] | 6.5376 | 7.3632 |
| Mul [1,256,3072] | 3.6352 | 3.8528 |
| LightOP SiLU×Mul 完整调用 | 17.8112 | 18.9792 |
| GEMM m256 n3072 k1024（Gate/Up） | 20.4816 | 20.5536 |
| GEMM m256 n1024 k3072（Down） | 27.4368 | 27.1264 |
| Add [1,256,1024] | 2.3456 | — |
| RMSNorm [1,256,1024] | 5.0976 | — |
| LightOP Add＋RMSNorm 完整调用 | 10.2624 | — |

Add＋RMSNorm测量（`data/add_rmsnorm_calibration_20261106_01`，`30e54e52`、`4d74b288`）：4/4质量通过，IQR/Median 0.28%～1.09%；Native整链7.1232 μs仅作同批对照，不作Profile。解析预测：融合9.335、Native整链17.496 μs。

注意力相关算子、q_norm/k_norm等未测Shape仍为解析成本，每个方案按来源计数记录在报告中。

## 6. 整模型可执行搜索（实测优先）

驱动 `tools/evaluate_executable_search.py`。已知参考方案（全部GPU0，同一评分器）：

| 参考方案 | 预测 / ms |
| --- | ---: |
| 不融合 | **8.7730**（已知最佳参考） |
| 只融合Add＋RMSNorm | 8.8520 |
| 只融合SiLU×Mul | 8.9869 |
| 两种都融合（56组） | 9.0658 |

修复前（`data/executable_search_model_measured_v2`，`751946c5`）与修复后（`data/executable_search_model_measured_v3`，`6c3917f9`、`45211004`），256次迭代、种子0/7/23：

| 模式 | 修复前 / ms | 差距 | 修复后 / ms | 差距 | 评估次数 | 平均耗时 / s |
| --- | --- | ---: | --- | ---: | --- | --- |
| 联合 | 9.0658（融合全部56组） | +3.34% | 8.7730（不融合） | 0 | 768→1024 | 58→74 |
| 先融合后放置 | 8.909～8.923 | +1.55%～+1.71% | 8.7730 | 0 | 768→1026 | 58→75 |
| 先放置后融合 | 8.909～8.923 | +1.55%～+1.71% | 8.7730 | 0 | 768→1026 | 57→75 |

偏差分析：确定性补全每步取最大融合并保持设备，在融合不划算时正好落在"两种都融合"；20个随机补全（融合概率0.7、设备均匀随机）为26.9～30.3 ms，约一半组在GPU1，跨卡拷贝占主导，因而从不优于确定性补全；56个融合决策位于591个锚点中的第14～584位，256次迭代的树无法到达。

修复内容：

- `legal_mcts_search(unfused_guided_rollout=False)`：开启后每个非终止叶子额外评估一次"不融合、保持当前设备"补全，与原补全并存；动作均来自合法动作集；选择与回传公式不变。默认关闭，已有实验可复现（测试覆盖）。
- 顺序搜索最终结果取第二阶段结果与可行的第一阶段方案中更优者，预算分配不变，额外一次评分计入预算。

单独验证第一阶段修复（新补全关闭，种子0，`incumbent_only_seed0.json`）：先放置后融合达到8.7730 ms（770次评估，选中第一阶段方案）；先融合后放置仍为8.9098 ms，因为其第一阶段结果本身就是含融合的划分。修复后的完整运行中，第一阶段方案参与比较但未被选中。

三种模式的种子0结果均导出并通过校验：591组、无融合、无跨卡通信、可部署。

局限：修复依靠覆盖两个极端的确定性补全。若最优方案是部分层融合、部分不融合的混合形态，两种补全未必足够；本轮问题未覆盖这种情况。

## 7. 仍缺的部分

- 完整FFN或整模型的实测参考仍为0；方案成本由独立算子测量经调度模型组合，Native整链相加偏差0.06%～7.58%（batch-01）。
- 默认SDMA通信标记为unstable，通信仍为代理；GPU1的Add＋RMSNorm、Add、RMSNorm未测。
- Add＋RMSNorm只测了含两次输入复制的完整调用，没有单测kernel本身，无法区分复制在3.14 μs差值中的占比。
- 下游自动并行模块尚未对接。
- Mul两段模型已通过独立留出验证（MAPE 3.84%对16.07%），保留为可选实现，未接入成本模型。

## 复现

```bash
python -X utf8 tools/evaluate_search_scale.py --output /fresh/search_scale
python -X utf8 tools/evaluate_joint_vs_staged.py --scope all --model-modes placement_then_fusion --output /fresh/joint_vs_staged
# 以下需厂商环境（LightOP），FUSION_PYTHON 为已验证的启动器
$FUSION_PYTHON -X utf8 tools/evaluate_executable_search.py --output /fresh/exec_before
$FUSION_PYTHON -X utf8 tools/evaluate_executable_search.py --unfused-guided-rollout --output /fresh/exec_after
bash tools/run_add_rmsnorm_calibration.sh /fresh/add_rmsnorm run
```

相关CPU测试：`cost_model/tests/test_placement_then_fusion.py`、`test_fused_placed_export.py`、`test_search_bias_fixes.py`、`test_layer_invariant_profile.py`（厂商环境）、`integration/tests/test_lightop_add_rmsnorm.py`。

## 2026-11-07：固定实际评分预算

三模式各seed0/7/23、统一64次实际评分，均为不融合全GPU0、8.773036ms；联合/先融合后放置第4次首次达到最终最佳，先放置后融合第1次，前4次之后改进均0。四个已有参考只需4次评价即可得到相同结果。当前案例没有树搜索后续收益或联合优于顺序证据。数据data/search_quality_fixed64_v1，口径和复现见[固定预算对照](SEARCH_QUALITY_FIXED64.md)。无新GPU、模型参数或搜索算法修改。

## 2026-11-07：3B扩展入口

模型锁定已有Qwen2.5-3B定义（36层、hidden2048/FFN11008、687节点866边）。P0审计：现有LightOP多节点可执行候选0；36处SiLU→Mul、36处Add→RMSNorm因Adapter固定Shape被拒，KERV SiLU宽度11008超过8192上限。每卡651项计算均解析回退、36逻辑输入，精确命中/Nearest/不可用均0；沿用当前Graph查询，不混入旧eager标签。没有启动Fixed-64或GPU。明细见[3B输入审计](SEARCH_QUALITY_3B_INPUT_AUDIT.md)。

## 2026-11-07：KERV AddRMS与LightOP最新发布包

KERV公共AddRMS虽覆盖2048，但只输出normalized、auto走Native；另一个adaptive AddRMS入口有真实双输出Triton源码，需核对BF16舍入、contiguous转换/分配和原地输入安全。最新可见LightOP日期构建g09a4f3要求Torch2.10，当前Torch2.9固定g6bfaca未变；最新wheel无相关C++/HIP源码及测试，11008/2048仍未核实，不把Adapter限制当厂商上限。详见[后端审计](THREE_B_FUSION_BACKEND_AUDIT.md)，数据data/3b_fusion_backend_audit_v1。未修改规则或执行GPU。

## 2026-11-07：KERV AddRMS2048单Case入口

新增calibration/kerv_add_rmsnorm_probe.py与tools/run_kerv_add_rmsnorm_probe.sh。CPU核对36处out_proj可覆盖、Add双输出必须保留；布局仍unknown。固定adaptive源码，合成BF16 [1,256,2048]，三次恢复/别名/只读/双参考检查，另一次唯一Kernel设备trace；只验证正确性，不计时、不Capture、不准入、不注册搜索规则。8项CPU测试通过，data/kerv_add_rmsnorm_2048_probe_v1仅plan，GPU未运行。CPU两参考131839个元素不同、max abs 0.03125，仅语义诊断。见[KERV单Case验证](KERV_ADD_RMSNORM_2048_PROBE.md)。

## 2026-11-07：KERV探针启动器选择

首次服务器run使用/usr/local/lib/python3.11的Triton，在读取AMD设备属性时UnicodeDecodeError，未进入Kernel；不能判断2048数值或执行失败。tools/run_kerv_add_rmsnorm_probe.sh的run改为必须显式设置FUSION_PYTHON，plan保留系统Python选项。先使用既有dcu-kernel310-20261031/run_python.sh重试新输出目录，不改Triton驱动或Kernel源码。该归因的最终确认仍待正确启动器回传。

## 2026-11-07：KERV AddRMS2048回传通过

kerv_add_rmsnorm_2048_02归档6原文件，18项CPU复核通过，源码/collector/图身份匹配。GPU0连续BF16 [1,256,2048]，三次手写参考max abs均0、aten均0.03125（rtol/atol=0.02通过），21个语义布尔检查通过；独立trace唯一_add_rms_norm_inference_kernel。只有单Case正确性证据，无计时/Capture/Profile/图Adapter或规则注册。见[KERV单Case验证](KERV_ADD_RMSNORM_2048_PROBE.md)回传节及data/kerv_add_rmsnorm_2048_02/review/report.json。未新增GPU，下一项固定Shape的图Adapter映射。

## 2026-11-07：KERV AddRMS2048图Adapter

新增integration/flagscale/kerv_add_rmsnorm.py和31项CPU映射测试，相关56项通过。固定BF16 [1,256,2048]、显式implementation及eps=1e-6；运行时布局/设备/梯度/别名检查，消费attention输入，按原节点发布Add和Norm两个输出，残差旁路保留。调用失败不恢复可能已改写的x映射，无Native回退。新opt-in registry在687节点3B图匹配36组，现有LightOP/default registry未变；参考放置图36组均绑定、双输出各组2MiB、序列化往返一致。data/kerv_add_rmsnorm_adapter_v1、说明[KERV图Adapter](KERV_ADD_RMSNORM_ADAPTER.md)。仅CPU mock映射，未执行Adapter的GPU路径、未Capture/测时/准入或改默认搜索，Profile新增0。

## 2026-11-07：最小Adapter GPU检查入口

原KERV AddRMS探针增加可选adapter路径：只绑定已有PlacedOpGraph第0层一组，复用三次正确性和一次device trace，增加live map发布/输入失效检查。无新矩阵、计时/Capture/准入/搜索。data/kerv_add_rmsnorm_adapter_smoke_v1仅CPU plan；相关59项CPU测试通过，新增3项。服务器命令见[KERV图Adapter](KERV_ADD_RMSNORM_ADAPTER.md)最小检查节；GPU尚未执行。旧直接Kernel数据保留对应原collector SHA，不因新入口改动作废。

## 2026-11-07：KERV图Adapter GPU单Case通过

kerv_add_rmsnorm_adapter_01归档6原文件并完成22项离线复核，collector/Adapter对齐aaae3d2，placed/source身份一致。GPU0第0层BF16 [1,256,2048]的PlacedOpGraph→Adapter→KERV链路通过；三次36项布尔检查（含15项映射）全通过，手写误差0、aten误差0.03125（rtol/atol0.02），唯一目标Kernel trace。旧attention输入失效、双输出按原节点发布、残差旁路保留。详见[KERV图Adapter](KERV_ADD_RMSNORM_ADAPTER.md)回传节。正确性验证收尾，不增加Case；正式Profile0、无计时/Capture/GPU1/完整模型/搜索变更。本轮离线归档，无新增GPU执行。

## 2026-11-07：KERV AddRMS单Case Graph测时入口

calibration/kerv_add_rmsnorm_timing.py、tools/run_kerv_add_rmsnorm_timing.sh：仅GPU0第0层BF16 [1,256,2048]，50份独立x、50调用/Graph、10预热Replay、7固定正式轮；恢复/校验在计时外，不补轮/替换/Profiler/锚点或Native对比。复用stage_size_calibration.timed，IQR/Median≤10%；记录完整运行环境、布局、双输出别名与现有Canonical Key及独立Contract。通过仅conditional_dev候选，待回传复核，不自动accepted或改搜索。新增9项、相关68项CPU测试通过；data/kerv_add_rmsnorm_timing_plan_v1仅计划，无新GPU耗时。见[KERV单Case测时](KERV_ADD_RMSNORM_TIMING.md)。

## 2026-11-07：KERV AddRMS2048成本查询闭合

kerv_add_rmsnorm_timing_01归档13原文件，八个collector源码SHA匹配95175f3，七轮/契约/双输出Key完整复现。Graph摊销median5.353559851646423us，min/max5.331159830093384/5.404760241508484，IQR/median0.6873858%。独立CPU review准入1条accepted，仅第0层组/GPU0/固定输入恢复契约；raw candidate不改。cost_model/kerv_add_rmsnorm_profile.py分离review与只读load_profile，复用既有KeyedKernelProfile，实际查询返回measured_group_kernel同值且不追加成本，其他组/设备/实现/布局/契约拒绝。新增9项、相关77项CPU测试通过。data/kerv_add_rmsnorm_timing_01/review保留profile/cost_query/receipt；说明[KERV测时](KERV_ADD_RMSNORM_TIMING.md)。本轮无GPU、拟合、搜索或默认配置修改；不扩大到36层/GPU1/完整方案，不需要重复此测量。

## 2026-11-07：3B AddRMS候选/契约审计与MCTS接线

现有687节点图扫描37条直接Add→RMSNorm：36符合v1 Adapter静态结构但缺物理布局/别名/执行契约，final_norm一处因输入角色和层约束不支持。默认严格模式不开放信息不足融合；显式conditional-planning仅CPU接线。cost_model/qwen25_add_rmsnorm.py支持逐组model_runtime证据和完整Canonical Key/Contract等价复用，本次证据为空，5.35356us图上复用0、Native同契约实测链0。条件场景每卡651解析/36逻辑输入，融合72解析/2不可用；旧eager launch不进入Graph评分，用既有1.5us显式探索假设，不作校准值。三模式seed0各64真实评分，最终36融合全GPU0、预测25.213223ms，对无融合参考25.295361ms；仅模型接线，无真实收益或联合优势证据。真实第0层6节点切片Exact96组合/6满足边界、MCTS model gap0；新增17项/相关68项CPU测试通过，慢融合保留Native。data/qwen25_add_rmsnorm_graph_v1含审计、轨迹与未部署放置图；见[3B图级接线](QWEN25_ADD_RMSNORM_GRAPH.md)。无GPU、新准入或默认配置改变。

### 2026-11-07：收缩为一次 Native Add→RMSNorm 对照（GPU待回传）

- 新增 calibration/native_add_rmsnorm_timing.py 与 tools/run_native_add_rmsnorm_timing.sh：GPU0、第0层、BF16 [1,256,2048]、eps=1e-6，完整 out-of-place Add→F.rms_norm 双输出链；50独立x、50次eager准备、10次预热、固定7×50，无锚点、补轮、Profiler或KERV重测。
- 旧KERV accepted 5.353559851646423μs保留；执行前核对原运行环境/构建/设备类，UUID仅溯源；公共Graph边界相同，分配/别名等实现配方分别保留。
- cost_model/native_add_rmsnorm_profile.py提供显式CPU review及只读load_record，输出比较与完整链成本，不拆成单算子标签。严格搜索在提供Native记录时要求同契约实测对；缺少逐组等价性证据不开放融合。
- NativeChainCostModel只在两个原生节点同卡且通过身份检查时，将整链计入一个连续执行单元，保留双输出依赖；原搜索图和导出节点不合并，导出带连续执行约束。跨卡拆分不命中整链成本。
- --audit-only可只检查复用而不新增搜索实验。现有687节点图仍37条直接连接：36处静态匹配、运行时证据不足；其余35处未复制第0层观察值，final_norm仍不支持当前映射。
- 相关51项CPU测试通过（包含Native快/慢两种选择、审核与防篡改、整链不拆分和跨卡拒绝）；CPU plan与严格复用审计已保存，无新GPU数据、准入或完整搜索结果。
- 操作见 cost_model/NATIVE_ADD_RMSNORM_COMPARISON.md；仅等待一次 native_add_rmsnorm_timing_01 回传。
### 2026-11-07：Native AddRMS回传准入与成本查询完成

native_add_rmsnorm_timing_01.tar.gz：8441字节，SHA256 3b147a973ed0afdc463aa38f4dd83878f67b1cf7d0ccbe35086e3811638da658；13原文件逐字节保存在data/native_add_rmsnorm_timing_01，receipt.json记录归档。既有review复算7×50和双参考/只读/恢复回执、同stream边界、Canonical Key/Contract及12个采集源码SHA（ff343bd），通过并新增1条Native完整链accepted；没有单算子标签。median10.217599869us、min10.179159641/max10.575959682、IQR/median1.424699%。对保留KERV5.353559852us，少4.864040017us（47.604526%），preferred=KERV；不同批次但公共Graph契约/runtime/构建/设备类匹配，UUID改变不影响。review/cost_query.json经现有KeyedKernelProfile严格查询同值，memory/launch追加0，GPU1拒绝。只做CPU归档/复核/查询/审计，无GPU重测或新搜索；benchmark上成本对已闭合，原3B模型36处缺真实布局/别名/执行契约，严格图复用仍0、final_norm在v1外，未冒充model_runtime。详见NATIVE_ADD_RMSNORM_COMPARISON.md回传节。该Case测量结束，后续补执行契约证据；README/SROP/transfer未纳入。
### 2026-11-07：按0.6B静态执行声明闭合3B联合搜索

用户明确完整模型执行不属Fusion职责，取消model_runtime作为静态搜索的全局前置条件；不新增GPU测量。新增qwen25_static_contract.py，复用LayerInvariantMeasuredProfile的明确执行声明思路，原图Shape/dtype/同层/角色/消费者/权重/双输出检查，目标连续布局和独立分配由现有Adapter运行时guard约束，不写成已观测。完整条件只排除layout_provenance，其余Key/runtime/硬件/实现/契约均匹配。GPU0层内36处可用Native/KERV已有实测对；GPU1缺成本对保守Native，final_norm在v1外。tools/run_qwen25_static_search.py只跑一次joint seed0：64请求、32实际评分、32因设备成本对不符拒绝，50不同请求方案。最终36融合全GPU0，预测25.239290006ms；Native全GPU0参考25.414395447ms。初次评分即最终最佳，后续改进0，无联合优势或整图真机性能结论。最终成本579解析/36逻辑输入/36融合实测条件复用。data/qwen25_static_joint_v1含逐项审计、执行声明、32有效评分轨迹、原节点图及651组下游图；原fingerprint不变、双输出完整、每组实现/Key/Contract一致、导出往返重评分相同。46项相关CPU测试通过，Native fallback组的连续执行与导出依赖也验证。原微基准数据/准入和MCTS核心不改；model_runtime旧路径保留，但不再阻断静态搜索。说明QWEN25_STATIC_SEARCH.md；README/SROP/transfer未纳入。
### 2026-11-07：按同构硬件类别开放GPU1 Native/KERV成本迁移

新增profile_hardware_class.py，从已有GPU1回传runtime/fused_runtime及review SHA核对两卡BW1000/BW、gfx936:sramecc+:xnack-、68702699520字节和Torch/HIP/驱动/构建/Triton一致，硬件类别Key均80afda64fe08dae7d6cf0b5280a8d33bf62eb0262f1c5cf8826cd540ca7c4247。qwen25_static_contract.py查询排除硬件物理编号/UUID，保留硬件类别/架构特性/软件构建/完整Kernel条件和契约，原accepted测量Key不改；每次复用保存measured_on/reuse_on/reuse_kind。GPU0原值Native10.217599869us、KERV5.353559852us迁移给GPU1，source标transferred、target_device_independently_measured=false，原始GPU1独立测量新增0。通信方向/链路/其他解析参数不改。单次joint seed0重跑64实际评分/64请求，设备成本拒绝32→0，GPU1融合候选成本查询378次；最终36融合全部GPU0、预测25.239290006ms，与前次相同；第1次最佳、后续改善0，不据此声称两卡精度一致或联合收益。58项CPU测试通过，cost_pairs可查询72=36同卡+36迁移，原始测量仍2条。data/qwen25_hardware_class_joint_v1含硬件指纹审计、逐组成本/来源、执行声明、轨迹/report及两种图导出；往返、身份/双输出/重评分一致，无新GPU/准入/完整模型执行，未改变MCTS核心。说明PROFILE_HARDWARE_CLASS.md；旧GPU0专项档案不改，README/SROP/transfer未纳入。
### 2026-11-07：3B三放置方案与编号交换CPU审计

tools/audit_qwen25_placement.py固定36融合，对整图A全0/B前18层0后18层1/C全1，预测25.239290006/25.308636502/25.255440040ms。B算子加和25.247990421ms、通信0.060646081ms、计算重叠0；唯一1MiB边L017.residual_add_2→L018.input。两卡busy11.769466/13.478524ms但依赖串行，当前是单图单微批latency，不是流水线吞吐。GPU0其他解析参数算力/带宽分别高0.068089%/0.047778%，融合共享不等于全成本完全对称。完整置换算力/带宽/容量/链路/准入/成本与来源，3个同物理方案的重编号评分误差均0。只新增一次交换joint seed0/64实评，仍全逻辑0（原GPU1），25.255440040ms，错过已知全逻辑1（原GPU0）16.150034us，参考gap0.063988%，非全空间regret。确认评分器编号等变，但有限预算搜索有逻辑0偏好；legal_search.py:120/271/304/498的设备顺序/补全/等prior首child路径支持归因。18相关CPU测试通过，data/qwen25_placement_audit_v1含report/findings/trace/plan；无新GPU/测量/算法修复/新融合。下一项建议同预算保留两卡完整融合/原生基线后复查编号，不继续扩大测量。说明QWEN25_PLACEMENT_AUDIT.md；README/SROP/transfer未纳入。
## 2026-11-07：3B显存CPU审计，初始化暂缓

687节点/866边Shape-dtype字节校验完整且差异0，全部forward。A/B/C现有峰值分别5.821262/0、3.180435/2.657551、0/5.821262GiB，低于每卡配置63.984375GiB；实际可用预算和训练可行性未知。现有搜索已有capacity惩罚及最终可行性过滤，不能再说只评分时间；但默认capacity是整卡总量。前述“满足显存约束”仅限该估算和配置预算，不是完整训练或运行时可行证明。

静态参数6,171,693,056字节；saved_activation/bwd_flops未进入训练时间线，workspace未声明，别名未去重，输出/接收缓冲从完成时刻才计。18/18方案lm_head与embedding跨卡，共享622,329,856字节权重的驻留未处理；不擅自添加复制策略或训练状态。36组融合均无独占内部输出被现有算法删除；全Native和全KERV的GPU0估算峰值相同不证明物理显存相同。

入口tools/audit_qwen25_memory.py；结果data/qwen25_memory_audit_v1（report、tensor_lifetimes、verification）；说明QWEN25_MEMORY_AUDIT.md。三个方案重评分/峰值与旧审计一致，17项已有CPU测试通过，无MCTS/GPU/生产代码修改。四基线初始化暂停，先确定估算范围、显式可用预算及共享存储/缓冲策略。

## 2026-11-09：连续阶段动作空间与中等子图对照

新增可选contiguous_stages，保留逐节点默认；设备只可留在本阶段或启用未用卡，两卡至多一刀，两种设备顺序均保留。融合仍联合选择，跨越拓扑区间的非相邻融合不允许被切开。Exact分别核算原空间和结构化空间。

从3B第0层按既有三个模板提取10/7/8节点，blocking、free/cap75、64次不同完整方案模型调用上限、3种子，72次启发式运行；Exact另行全穷举。两卡初始化、不可行提案及Sequential两阶段均计入共享预算。原三方法用满64；结构化空间38/14/28全部覆盖后停止。serial_tail/attention_fork cap75完整空间不可行，记录保留。

serial_tail与attention_fork free的Stage Exact等于原Exact；MLP free与cap75的Stage Exact分别差6.5288%和14.9857%，域内MCTS Gap均0。MLP原Joint cap75中位Gap5.6503%、最好种子0.0145%；连续阶段限制损失不能算成MCTS未收敛，也不能宣称新空间必然更优。Stage Joint已全枚举小空间，尚无大空间效率优势证明。68项CPU测试通过，冷缓存重放分数/最优方案/计数/轨迹一致。入口tools/evaluate_stage_search.py，数据data/stage_search_medium_v1，说明STAGE_PLACEMENT_SEARCH.md；没有重跑完整3B或新增GPU。

## 2026-11-09：分支感知与混合Joint

沿用三个3B中等子图和free/cap75，新增DAG简单fork/join线性分支同卡动作branch_aware，以及分支独立＋串行主干连续阶段的hybrid；融合仍搜索。统一64次独立实评上限、3种子，初始化与不可行项计费；Exact独立且不泄漏标签。

空间：serial_tail 1536/38/1536/38，attention_fork 128/14/32/24，MLP 384/28/192/60（原／阶段／分支／混合）。MLP free四法中位Gap0/6.5288/0/0%；cap75为5.6503/14.9857/0.0145/0%。混合三种子均达到原域Exact，但60项全部覆盖，不称选择策略效率优势。MLP free分支／混合恢复164.514us双卡分支重叠和2条跨卡边；cap75阶段也存在164.407us重叠但有3条跨卡边，不能将其全部损失解释为无分支并行。

新增离线Exact反事实耦合审计：MLP共同域Fused−Native为+12.697～+28.878us，存在数值交互与部分放置排序反转，但始终Native更快，共同域最优放置集合仍有交集；serial_tail始终Fused更快，收益变动仅0.000747us。当前未观察到融合偏好翻转或共同域最优放置变化，不把同时融合和分卡视作强耦合成功。保留两个完整域不可行任务及attention无融合对照项。

74项CPU测试通过，冷缓存完整重放一致（排除墙钟）；36条旧控制运行方案/成本/计数不变。入口tools/evaluate_branch_search.py，结果data/branch_search_medium_v1，说明BRANCH_PLACEMENT_SEARCH.md；无完整3B运行或GPU测量，原成本/UCT公式/数据不改。
