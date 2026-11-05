# 融合—放置搜索实验日志（2026-11-05）

本轮从代价模型标定转向融合与设备放置搜索。标定部分的逐批记录见 [STAGE_SIZE_CALIBRATION.md](../calibration/STAGE_SIZE_CALIBRATION.md)，本文不重复，只记录与搜索相关的工作与结论。

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
