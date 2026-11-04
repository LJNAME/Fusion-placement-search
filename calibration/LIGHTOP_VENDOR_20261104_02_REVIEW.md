# LightOP 11.4 第二批完整回执

完整包 `lightop_vendor_suite_20261104_02_full.tar.gz`：3,820,468 字节，SHA256 `0b248b3fd1c9c6f6405d39212845a1e17b975d56794d8530ee328388a7e23bef`。gzip CRC 通过，2,126 个文件；已取代先前 524,288 字节截断包作为本批审核输入。

| 原版任务 | 结果 | 耗时记录 |
| --- | --- | --- |
| SiLU / GELU / GELU-Tanh × Mul | 各 10 个 Case 打印 Pass | 各 10 行，μs |
| Permute + Quant | 14 个 Case 打印 Pass | 14 行，ms |
| RMS + RoPE | 16 个 Case 检查通过 | 16 行，μs |
| Reshape + Cache | 17 个 Case 打印 PASS | 17 行，μs |
| BMM | 128/128；32 个不同 Shape 各出现 4 次 | Excel 128 行，μs |
| Fill + RMS × 2 + Concat | 10 个 Case，两项差值均为 0 | 无 |
| RMSNorm + RoPE + Contiguous | 1 个 Case，四项差值均为 0 | 无 |
| 剩余 7 项 | 未完成，暂不续测（缺依赖退出） | 未完成 |

共 16 项、各尝试 1 次：9 项退出 0，7 项退出 1。7 项有检查和计时证据，2 项只有零误差输出。终端 77 行计时、BMM 表格 128 行，共 205 行。报告、逐项状态、原值和来源 SHA 在 `data/lightop_vendor_20261104_02_review/`。

## 比较边界

- SiLU×Mul 的 10 个 Case 均为 BF16、d=4096，tokens=32…16384，打印加速比 1.57–1.96。基准是 `torch.ops._C.silu_and_mul`，即 vLLM 的另一种融合实现；不能写成对 Native SiLU＋Mul 的加速。原脚本 `test/test_act_and_mul.py:41-70`。
- `test_fuse_groupgemm_permute_quant.py` 实际测的是重排＋INT8 量化，未执行 GroupGEMM；原脚本 `:20-38`、`:49-115`。10 次预热、100 次调用、同步前后用 wall clock 计时，不能与其他任务的 GPU Event / Triton 计时合并。
- BMM 的 `fused_bmm_benchmark.xlsx` 表头写 FP8，入口实际设 `in_dtype=torch.int8`、`out_dtype=torch.bfloat16`（原脚本 `test/test_fused_trans_bmm_trans.py:910-912`）；`q_quant_time=0.0`（`:504-505`），基准为 BF16 Torch BMM（`:507-508`）。这组数据排除了量化成本且比较精度不同，不能据此单独归因融合收益。Shape 列和全部原始计时保留在 `timings.jsonl`，未把 128 条当成 128 个不同 Shape。
- 两个零误差任务只报告差值，保留为 correctness-only，未补造耗时或准入容差。

## 7 项未完成（暂不续测）

| 缺失模块 | 原版任务 |
| --- | --- |
| `matplotlib` | `test_fuse_silu_mul_quant`、`test_fuse_silu_mul_quant_ep`、`test_fused_moe_gate` |
| `numa` | `test_fuse_silu_mul_fp8_quant`、`test_gemma_rms_norm_fp8_quant`、`test_rms_norm_dynamic_per_token_quant`、`test_rms_norm_fp8_quant` |

前三项先打印精度检查通过，随后在 Triton 性能测试导入 `matplotlib.pyplot` 时退出；不能把部分通过当成整项完成。

后四项在 `RMSNorm` 构造时，经 `CustomOp.dispatch_forward → get_cached_compilation_config → VllmConfig → ParallelConfig → ExecutorBase → WorkerBase` 引入 `numa`。固定 vendor vLLM 的 `vllm/worker/worker_base.py:5` 无条件导入该模块，`:35-46` 使用 `get_max_node` / `bind`，`:286` 使用 `get_membind`。这些任务尚未完成数值验证；文件名含 `silu_mul_fp8` 的脚本实际调用 RMSNorm FP8，按真实 API 归类。

本批日志未出现数值失败。此前批次的数值失败记录不被这份结论覆盖。

## 环境与源码

- 完整预检退出 0、无超时，168.945 s；实际平台 `vllm.platforms.rocm.RocmPlatform`。此前双内置平台冲突在本批未复现。
- 安装前后 81 个包快照一致；Torch、Triton、LightOP、NumPy、vLLM、lmslim 未替换。
- 临时 SMI `.pth` 和平台注册目录均已清理。
- 原始测试树逐文件 SHA 核对 2,057 份（121 × 17）；原版测试未修改。
- 能力文件按批次追加 `vendor_20261104_02` 证据，保留 `vendor_suite01`。静态 48 接口目录不改写，未增加 PlacedOpGraph Adapter。

新增 accepted Profile 为 0，完整 FFN 可定价方案仍为 0/48。厂商脚本各自的耗时边界尚未转换成项目统一的 Canonical Key / Measurement Contract。

## 当前处理决定（2026-11-04）

7 项均标记为 **未完成，暂不续测**，保留缺失模块、原始失败记录与源码 SHA。`remaining_jobs.json` 逐项写入 `completion_status=incomplete`、`followup_decision=deferred_by_user`，待执行集合为空；原 `report.json` 的历史审核结果不改写。

本阶段不补装 `matplotlib` / `numa`，不重新执行这 7 项，不扩大厂商测试矩阵。9 项已完成任务及其证据保留。若以后恢复，再核验原依赖和实际 `RMSNorm` 构造路径；该条件不再是当前主线的前置任务。

## 下一步计划

LightOP 作为真实融合 Kernel 的执行与测量后端，用于校准代价模型并验证融合—设备放置搜索。Native、KERV、LightOP 的性能比较不单独立题；仅在动作空间或控制变量需要时区分实现 Profile。输入打包问题按预测与实测的边界一致性处理。

代价模型采用已有研究中的成熟方法，当前任务是复用、校准与验证；具体论文方法及复现范围尚待选定，不新增代价预测算法作为创新任务。研究重点为融合—设备放置联合优化，系统实现贡献与算法贡献分别验收。SROP 保持冻结，仅作为后续测量预算与方案质量研究的候选，不因本次范围确认自动恢复。

校准数据与独立验证数据分开，先验证成本误差，再验证方案排名与 regret。Profile 加和构成的参考成本不等于完整方案真机耗时；Exact 仍限定于其成本模型。

| 顺序 | 工作 | 产出 | GPU |
| --- | --- | --- | --- |
| 1 | 将已完成任务对应的实际 API 与 Qwen3-0.6B 图匹配，核对 Shape、dtype、布局、层内分组和输入输出语义 | 真实图适用表：可直接映射、需要适配、不适用、证据不足 | 否 |
| 2 | 已实现 SiLU×Mul 单组 Adapter，明确输入打包、输出、生存期和计时边界 | 33 项 Adapter CPU 测试通过；真实 GPU 待验证，KERV 保留 | 否 |
| 3 | 绑定 Canonical Key / Measurement Contract，审查已有计时的复用范围与最小缺项 | 可精确查询的 Profile 身份；开发证据与正式数据分列 | 否 |
| 4 | 仅补齐选定候选所缺的执行、正确性与测量证据，完成 Admission 和成本查询 | 可信融合 Profile；不以加速比作为准入条件 | 按缺项决定 |
| 5 | 用实测数据校准代价模型，在独立验证数据上检查校准前后误差 | 同一执行边界的预测误差、验证集身份和适用范围 | 按缺项决定 |
| 6 | 成本覆盖满足后，在固定 FFN 图和动作空间上做 Exact / 搜索结果对照 | 独立参考成本下的排名、方案选择和 regret | 否（搜索评价） |

首项已完成，见 [LightOP × Qwen3 适用性审计](LIGHTOP_QWEN3_APPLICABILITY.md)：9 项中直接映射 0、需要适配 1、证据不足 1、不适用 7；记录 28 组 SiLU→Mul 和 28 组 Q/K Norm→RoPE 结构。原图布局未知，v1 审计证据保持历史状态。此后已完成 [SiLU×Mul Adapter 的 CPU 实现](../integration/lightop/SILU_MUL_ADAPTER.md)，动态能力目录单独标注 CPU 验证/GPU 待验；未新增计时或准入。下一项为实际运行身份、Canonical Key / Measurement Contract 与最小验证入口。

`op.silu_and_mul_opt` 接收打包输入和独立输出，而当前图中的 gate/up 来自两个投影；若需要拼接或复制，Adapter 必须明确该成本由谁承担，不能只引用 Kernel 内部耗时。现有这一批 d=4096 的测量，也不能直接用于 Qwen3 FFN 的宽度 3072（`data/ffn_placement_b1/identity.json:56`）。

7 项有计时证据的任务与 2 项仅零误差任务继续分列。厂商接口可调用、当前图可映射、Profile 可准入、完整方案可评分分别记录。BMM 的 INT8/BF16 边界、Permute+Quant 的实际范围沿用上文，不能补写成统一融合收益。

Exact 只表示当前成本模型下的最优值；完整方案的通信和其他成本缺失时，不宣称真实最优。MCTS 和 SROP 本阶段保持不变。
