# LightOP SiLU×Mul：身份绑定与最小 Smoke

当前实现只覆盖 Qwen3-0.6B 参考图中一个 SiLU→Mul 融合组，BF16、`[1,256,3072]`。边界输入由固定种子生成，不执行 Gate/Up/Down Projection，不执行整个 FFN 或模型。

## 已实现

- `integration/lightop/profile_identity.py`：检查实际导入的 LightOP 包、原生扩展路径及 SHA，核对 Adapter callable 与源码；观测 Torch 构建、HIP/驱动、设备架构和执行标志。缺失字段保持缺失。
- `cost_model/lightop_profile.py`：将实际 Tensor 观测、实现、设备和运行环境写入选定组的 Canonical Key，复用 `KeyedKernelProfile` 的严格成本查询。完整调用成本包含预分配 `cat(out=...)` 和 LightOP Kernel，查询后不再追加打包或 launch 成本。
- `calibration/lightop_profile_smoke.py`：单设备 Capture/Replay 验证入口，不生成时延标签，不自动准入。

UUID 和安装绝对路径保存在运行溯源中，不参与 Key；SCNet 重启后不单凭 UUID 变化拒绝数据。GPU ordinal、型号、架构和实际软件构建仍参与匹配，GPU0 与 GPU1 不自动共享时延。

原图未知布局不被默认改成连续。只在 Key 使用的图副本中，向选定组的边界节点写入运行时 Tensor 布局；内部 SiLU 标记为未独立物化的融合表达式。

成本查询只接受已审核的、契约完全一致的 measured/accepted 标签。缺 Key、身份缺项、`conditional_dev`、`legacy_unbound`、synthetic、Smoke、eager 或仅 Kernel 的标签均拒绝，不用解析成本替代本组缺项。其他节点不在此接口的成本覆盖范围，通信仍由既有基础接口提供，不能据此声称完整方案具备正式成本。

当前没有新增 accepted Profile。严格查询的正向测试使用明确标注的 CPU 假数据，不能当作实际准入。

## 2026-11-04 首次 GPU0 回执

回执 `lightop_profile_smoke_20261104_02.tar.gz` 完整，35,391 字节，SHA256 `763ef934ab3a227118982cc245ad27720d4296368f4ed1329d1c38b89cab0da1`。审核记录：[report.json](../data/lightop_profile_smoke_20261104_02_review/report.json)。

| 项目 | 本次结果 |
| --- | --- |
| 场景 | GPU0，`gfx936:sramecc+:xnack-`，第 0 层 BF16 `[1,256,3072]` |
| 进程 / Capture / Replay | 进程退出 0；Capture 和 3 次诊断 Replay 完成 |
| 输出 | 3×50=150 个诊断输出全部通过，每个分别对照 BF16 Native 和 FP32 Reference |
| 输入与存储 | gate/up 字节不变；输出与输入独立；调用缓冲独立 |
| 身份 | 从 PlacedOpGraph、运行环境和 Tensor 观测重建 Key/Contract，与原记录精确一致；`identity_gaps=[]` |
| 计时边界 | `cat(out=packed)`＋LightOP 完整 Adapter 调用，Graph 批摊销；计时不作通过条件 |
| 正式标签 | 未写入，新增 accepted=0 |

三次诊断 Event 批耗时为 0.905918 / 0.879200 / 0.914878 ms，每批 50 次调用；不作为正式成本标签，不用于加速结论。CPU 审核共 25 项一致性检查通过，包含 UUID 改变不影响 Key、Smoke 契约被正式成本接口拒绝。

本次只覆盖一个固定输入场景和 GPU0。没有独立 dispatch trace，没有 GPU1、完整 FFN 或模型执行证据。执行检查来自回执及本地冻结代码核对，未保存原始 GPU Tensor，未独立重做数值计算；Adapter 源码 SHA 匹配，采集器本身的 revision/SHA 未单独记录。

`preparation.one_call.gpu_execution_verified=false` 是执行前的准备快照，冻结 Recipe 的 `capture_support=unverified_on_GPU` 也不在本次修改，避免改变已执行实现的身份。这两项不覆盖事后 `capture_numeric_passed=true` 的结果；完整执行路径认证仍等待独立 trace。

独立 profiling 入口现已准备，见 [dispatch 采集](LIGHTOP_DISPATCH.md)，CPU 验证完成、真实 trace 待回执。下一项先补同一完整 Adapter 调用的独立 profiling 证据，区分打包与 LightOP 目标 Kernel；再冻结正式采集与 Admission，记录采集器自身身份和校准/验证 Case。剖析与计时分进程，不把本次 3 轮诊断转换为正式标签，不启动 GPU1、全矩阵或搜索。
## SCNet 命令

使用已安装的固定 LightOP 与现有私有 Python，不安装依赖，不运行暂停的 7 项厂商任务。

```bash
export FLAGSCALE_ROOT="$HOME/private_data/LJ/FlagScale-main"
export FUSION_PYTHON="$HOME/private_data/LJ/dcu-kernel310-20261031/run_python.sh"

# 先只生成计划，输出目录必须全新。
bash tools/run_lightop_profile_smoke.sh \
  "$HOME/private_data/LJ/lightop_profile_smoke_plan_20261104_01" plan 0

# GPU0，单次尝试；失败后返回完整目录，勿覆盖或自动重跑。
bash tools/run_lightop_profile_smoke.sh \
  "$HOME/private_data/LJ/lightop_profile_smoke_20261104_01" run 0
```

本次先执行 GPU0。GPU1 可显式传 `1`，但不随 GPU0 自动启动。

## Smoke 执行边界

1. 从完整逻辑参考图选择第 0 层 SiLU→Mul，保存带实现标注的 PlacedOpGraph，只执行该组。
2. 准备共享只读 gate/up、50 份独立 packed/output 缓冲。每份显式可变缓冲 4.5 MiB，50 份合计 225 MiB；不含输入、参考值、快照、库工作区和 Graph pool，不称为显存峰值。
3. 在侧流上执行 50 次 eager 准备调用，然后捕获包含 50 次完整 Adapter 调用的一个 Graph；`pool=None`，保留输入和所有缓冲至同步完成。
4. 10 次 Graph 预热，再执行 3 次 Smoke Replay。每次完成后逐一校验 50 个输出，分别对照 BF16 Native 和 FP32 Reference，`rtol=0.01, atol=0.001`。Native 只用于正确性，不测性能。
5. 在预热检查前及每次 Smoke Replay 前，将 packed/output 填为 NaN；在同一流上、计时区间外执行，避免空 Replay 使用 eager 预热留下的正确结果。校验 gate/up 字节不变、输出独立、各调用缓冲不复用。LightOP 输入只读，无 KERV gate 原地恢复步骤。

计时仅打印 50 次调用的 Event 批耗时，不作通过条件。Event 诊断异常不跳过、不重复 Replay；Capture、Replay、同步或数值异常仍使 Smoke 立即停止。失败记录保留 traceback、pool 配置、阶段及尽力取得的显存快照。进程崩溃时以 `process.log`、`process.json` 和 `last_stage.json` 为依据，不能承诺 Python 异常处理能捕获原生段错误。

## 回传与判读

返回整个新输出目录，包含失败文件。正常产物：`plan.json`、`placed.json`、`runtime.json`、`preparation.json`、`identity.json`、`last_stage.json`、`replay_0..2.json`、`result.json`、`process.log`、`process.json`。

`capture_numeric_passed=true` 只表示本次捕获、重放及正确性检查通过。`identity_complete` 单独报告；未知架构或驱动等会留下 `identity_gaps`，不能由 Smoke 通过替代身份闭合。

本入口不采 dispatch trace，不认证实际 Kernel 数量或 fallback 路径。后续若需要执行路径证据，使用独立 profiling 进程，避免污染计时。整个 Adapter 包含打包操作，不能要求它总共只有一个 GPU Kernel。

`measurement_contract(..., purpose="calibration")` 当前用于成本接口的 CPU 契约测试，尚无对应正式采集与 Admission 实现；不将 Smoke 的 3 轮诊断值改名为 7 轮正式标签。下一步先审 Smoke 回执，再确定正式测量、独立执行证据及准入所缺的最小工作，不扩大矩阵或启动搜索。

## CPU 验证

```bash
python -m pytest cost_model/tests/test_lightop_profile.py \
  integration/tests/test_lightop_profile_identity.py \
  calibration/tests/test_lightop_profile_smoke.py -q
bash -n tools/run_lightop_profile_smoke.sh
```

测试使用 CPU Tensor、临时假二进制和显式假标签验证工程规则，不提供 GPU 执行或性能证据。
