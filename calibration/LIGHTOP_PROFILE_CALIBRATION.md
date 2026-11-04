# LightOP GPU0 正式时延采集与 CPU 准入

承接已通过的 Smoke02 和 dispatch02。只测 GPU0、第 0 层 SiLU→Mul、BF16 `[1,256,3072]` 的完整 Adapter 调用，边界输入由固定种子生成。当前只有 CPU 实现与测试，尚无本版正式 GPU 回执，新增 accepted=0。

## 固定执行契约

沿用 `cost_model/lightop_profile.py` 的 calibration 契约，Adapter、Recipe 及 dispatch 使用的 7 个源码文件保持不变。

| 项目 | 固定值 |
| --- | --- |
| 单次调用 | 预分配 `cat(out=packed)`＋`lightop.op.silu_and_mul_opt` |
| 缓冲 | 共享只读 gate/up；50 份独立 packed/output，保留至同步完成 |
| 准备与预热 | 50 次 eager 准备，Capture 一个 50-call Graph，10 次预热 Replay |
| 正式样本 | 按时间顺序固定 7 次尝试，每次 Replay 含 50 次调用；不补轮、不替换 |
| GPU 计时 | GPU0 同一流的 start Event→一次 Replay→end Event；区间外同步 |
| 输出准备 | 每轮在同一流、计时区间外填 NaN 并同步，防止空 Replay 复用旧输出 |
| 正确性 | 每轮全部 50 个输出，各对照 BF16 Native 与 FP32，rtol=0.01 / atol=0.001 |
| 存储 | gate/up 逐字节保持；输出不与输入/packed 重叠；50 对可变缓冲独立 |
| 主统计量 | 7 次批 Event 耗时分别除以 50，取中位数，单位 μs/call |

分配、数值参考、NaN 填充、检查、日志与 Host 提交开销不纳入 GPU Event 区间。Native 只作正确性参考，不采性能对照。这里的标签是 Graph 批摊销完整调用时延，不是单次 eager launch 或独立融合 Kernel 时延。

## 准入规则

质量规则单独保存为 `quality_policy`，纳入计划及采集器身份；不改已冻结的 Measurement Contract 模板。

- 7 个正且有限的计时值齐全，逐轮数值、存储、Event 设备与流检查通过。
- 四分位采用 `statistics.quantiles(..., method="inclusive")`；IQR/Median ≤10% 通过，超过则 `quality_failed`。
- Event 读取异常记录原错误，继续余下尝试；不重复该 Replay、不丢弃样本。GPU 执行、同步或数值错误停止本 Case，保留 traceback、阶段、pool 和尽力取得的显存快照。原生崩溃以 faulthandler 和进程码为准。
- 不使用旧 FFN 实验的共享窗口锚点，不声称通过跨窗口漂移验证。7 轮离散度门槛只覆盖本 Case 的短测量序列。
- 准入不要求比 Native 或 KERV 更快。

运行入口不启动 profiler；记录并拒绝已知 profiler 环境注入或 Python profile/trace。该 guard 不保证发现所有外部 profiler，SCNet 正式命令应直接启动，勿套在 hipprof 下。

采集结果先为 `conditional_dev`。CPU 审核重建 Key/Contract，核对实际源码、软件构建、布局、设备以及 dispatch02 证据，重算全部统计量，并通过 `selected_group_profile` 试查一次成本；只有这些检查均通过才写 `accepted` 决定。UUID 仅保留为 SCNet 运行溯源，GPU ordinal、架构和软件构建仍精确匹配。

## SCNet 运行

使用包含本入口的完整 Git bundle 更新服务器后：

```bash
export FLAGSCALE_ROOT="$HOME/private_data/LJ/FlagScale-main"
export FUSION_PYTHON="$HOME/private_data/LJ/dcu-kernel310-20261031/run_python.sh"

# 可选 CPU 计划，不加载 GPU 或厂商算子。
bash tools/run_lightop_profile_calibration.sh \
  "$HOME/private_data/LJ/lightop_profile_calibration_plan_20261104_01" plan

# 单次正式采集，只运行一个 GPU0 Case；不自动准入。
bash tools/run_lightop_profile_calibration.sh \
  "$HOME/private_data/LJ/lightop_profile_calibration_20261104_01" run
```

输出目录必须全新；成功或失败均返回完整目录，不重复运行覆盖。七轮质量失败时，保留全部轮次及 `quality_failed` 结果，进程退出 1；它不等于 Capture 或数值崩溃，不写异常 `failure.json`。其他异常另写 `failure.json`。包装脚本单独保存 Python/日志进程码。

## 回传后的 CPU 审核

采集目录保持原样，另开审核目录：

```bash
python -m calibration.lightop_profile_calibration review \
  --input /path/to/lightop_profile_calibration_20261104_01 \
  --output /path/to/lightop_profile_calibration_20261104_01_review
```

`admission.json` 保存原始 JSON SHA、dispatch 证据 SHA、Kernel Key、Contract、质量判断及准入 ID。通过时同时记录 `latency` 和严格成本查询结果；质量失败时保留拒绝原因，不生成可用成本。原始 GPU Tensor 不归档，CPU 只能核对回执、冻结执行代码及其一致性，不独立重做 GPU 数值计算。

代价模型可直接重放审核并读取该单组 Profile：

```python
from calibration.lightop_profile_calibration import load_reviewed_profile

profile = load_reviewed_profile("/path/to/lightop_profile_calibration_20261104_01")
# 对同一观测组调用 estimate_group；缺项或不匹配时拒绝，不回退。
```

标签已含打包与融合执行，不能再追加 pack 或 launch 开销。本接口不覆盖 Singleton、链路、通信、GPU1 或其他 Shape，不扩展完整方案覆盖率；即便单组准入成功，也不能声称完成 FFN/模型时延验证。后续校准与独立验证 Case 尚未建立，本轮只取得第一个可查询成本项。

## CPU 验证

```bash
python -m pytest calibration/tests/test_lightop_profile_calibration.py \
  calibration/tests/test_lightop_profile_smoke.py \
  calibration/tests/test_lightop_dispatch.py \
  calibration/tests/test_lightop_dispatch_review.py \
  cost_model/tests/test_lightop_profile.py \
  integration/tests/test_lightop_profile_identity.py \
  integration/tests/test_lightop_silu_mul.py -q
bash -n tools/run_lightop_profile_calibration.sh
```

测试中的时延与 accepted 记录均为明确标注的 CPU 夹具，不能作为 GPU 成本。
