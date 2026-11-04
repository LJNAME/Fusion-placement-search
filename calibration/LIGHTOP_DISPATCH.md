# LightOP GPU0 独立 dispatch 采集

承接 `lightop_profile_smoke_20261104_02`：同一 BF16 `[1,256,3072]`、第 0 层 SiLU→Mul、预分配拼接＋LightOP 调用。只采执行路径证据，不采正式时延，不写 Profile，不运行 Native 性能对照。

## 入口

```bash
export FLAGSCALE_ROOT="$HOME/private_data/LJ/FlagScale-main"
export FUSION_PYTHON="$HOME/private_data/LJ/dcu-kernel310-20261031/run_python.sh"

bash tools/run_lightop_dispatch.sh \
  "$HOME/private_data/LJ/lightop_dispatch_20261104_02" run
```

默认 `plan`；显式 `run` 才启动既有 `hipprof --hip-trace`。不安装依赖、不切换实现、不重试。父进程只编排和 CPU 审核，唯一子任务为独立 profiling worker，当前只覆盖 GPU0。profiler 在 `trace/` 工作，worker 使用独立的 `trace/worker/` 产物目录。

## 2026-11-04 首次回执与入口修复

首批 `lightop_dispatch_20261104_01.tar.gz` 完整，3,608 字节，SHA256 `d4166e7591f18a107d378ed9750e262fb710602ad6d44ffcfc2000946b5dd742`。父计划中的 7 个源码 SHA 与提交 `110294b` 一致。[审核记录](../data/lightop_dispatch_20261104_01_review/report.json)。

worker 在进入运行时检查前报 `fresh output required` 并退出 2：hipprof 已先在同一 `trace/` 创建数据库，与 worker 的空目录要求冲突。hipprof 自身退出 0，父进程随后因缺少 worker 产物退出 1。数据库仅有 CONFIG、STR_TABLE、TRACE_COUNTER 三张空表，没有 Capture、数值或目标 Kernel 证据；此前 Smoke 通过的结论不变。

修复后 profiler 工作目录仍为 `trace/`，worker 独立写入空的 `trace/worker/`；父进程在 hipprof 退出后先检查 worker 完成记录，不能将 profiler 退出 0 当成 worker 成功。CPU 回归覆盖“profiler 先创建数据库再启动真实 worker 入口”的顺序，以及 profiler 退出 0 但 worker 没完成的情况。空目录保护保留，不放宽为任意旧目录复用。

首批修复后以全新 `lightop_dispatch_20261104_02` 目录重跑，结果见下节。首批原始记录保留；Adapter、Recipe、Kernel Key、标记方式和数值阈值未改。

## 2026-11-04 第二批 dispatch 回执

`lightop_dispatch_20261104_02.tar.gz` 完整，1,364,600 字节，SHA256 `a014e82019b9a3d9865e087e7cab9785006739afd88e3e0e914f8e96ca0be5ec`。本地从原始 SQLite/CSV 重新审核，与回传的审核结果和目标时间线精确一致；原始文件未改。[审核记录](../data/lightop_dispatch_20261104_02_review/report.json)、[目标时间线](../data/lightop_dispatch_20261104_02_review/target_kernel_timeline.json)。

| 项目 | 结果 |
| --- | --- |
| 进程与源码 | 父进程、profiler、worker 完成；7 个采集器源码 SHA 匹配 `ab067e8` |
| 目标区域 | worker PID 575，GPU0；257/263 字节标记唯一，隔离 1 次目标 Replay |
| 打包 Kernel | `CatArrayBatchedCopy_contig<...OpaqueType<2u>...>`，50 次 |
| LightOP Kernel | `act_and_mul_kernel_opt1<BFloat16, ...silu_kernel...>`，50 次 |
| 顺序与队列 | 50 对打包→融合，均在 GPU0 的队列 1；窗口内无其他 Kernel/Copy，无独立 SiLU/Mul 替代 |
| Graph API | Capture/EndCapture/Instantiate 各 1 次；`hipGraphLaunch=11`（10 次预热＋1 次目标 Replay） |
| 数值与存储 | 50 个输出逐个对照 BF16 Native / FP32；gate/up 字节保持，输出独立 |
| 身份 | `identity_gaps=[]`；Kernel Key / Smoke Contract 与首次 Smoke 精确一致 |
| 正式标签 | 未产生；新增 accepted=0 |

全进程 2,864 个 Kernel 包含准备、预热及数值参考；认证只针对标记内的 100 个事件。`at::native` 是本次实际目标符号的一部分，不能仅凭命名空间判定发生回退。原始 GPU Tensor 未归档，数值结论来自 worker 执行检查，CPU 复核其记录与冻结实现；pftrace 保留但本次不解码。

本次只验证 GPU0、第 0 层 BF16 `[1,256,3072]` 的完整 Adapter 路径，边界输入来自逻辑参考图的固定种子生成。未执行 GPU1、完整 FFN 或模型。

下一项冻结完整 Adapter 调用的正式计时契约和 Admission，在无 profiler 的独立进程采集。profiling DurationNs、标记间隔和 Smoke 诊断耗时均不转为正式标签。Adapter 与 Recipe 源码保持冻结；历史执行前的未验证声明保留，事后证据以本节和审核记录为准。

## 执行与定位

1. 核对实际 LightOP/Adapter/Torch/HIP/设备身份，重建 Key 与 Smoke 契约哈希，要求与已通过场景精确一致。UUID 仅作溯源，不参与匹配；其他执行条件变化则停止并保留观测。
2. 使用同一单组 PlacedOpGraph 和 50 份独立 packed/output 缓冲，共享只读 gate/up；50 次 eager 准备调用，捕获 50-call Graph，预热 Replay 10 次。
3. 在目标区域外将 packed/output 填 NaN 并同步，避免使用预热残留值。
4. 在同一侧流上执行 257 字节 pinned-host→GPU0 拷贝并同步，Replay 一次并同步，再执行 263 字节拷贝并同步。标记缓冲不与计算缓冲共享。
5. 结束标记后再计算 BF16 Native / FP32 参考，检查全部 50 个输出、输入字节保持及独立缓冲，`rtol=0.01, atol=0.001`。

两个拷贝只用于定位，不属于 Adapter 或成本标签。小型标记拷贝已在第二批 GPU0 回执中验证；标记缺失或有歧义时不自动换成 eager 或全进程计数。

## CPU trace 审核

复用已有回执核实过的 DTK hipprof SQLite/CSV 格式：从 `HIPCOPY_*` 找到 worker PID、GPU0 的两个唯一字节数标记；从 `STR_TABLE` 按 `(PID, STR_ID)` 解码 `HIPOPS_*` 名称，不从文件名推断 worker PID，不把 `at::native` 命名空间视为 fallback。

目标区域为起始拷贝 EndNs 到结束拷贝 BeginNs，必须含 50 组按序出现的 BF16 打包与目标融合事件：

- 打包：已观察到的 `CatArrayBatchedCopy_contig`、`OpaqueType<2...>`；
- LightOP：已观察到的 `act_and_mul_kernel_opt1`、`BFloat16`、`silu_kernel` 模板。

原始 Kernel 次数和 DurationNs 与汇总 CSV 交叉核对。窗口内额外 Kernel、独立 SiLU/Mul、缺失调用、错误设备、其他 Copy、跨边界事件或队列不符均不认证。未知符号只报告审核失败并保留原始 trace，不能据此直接判定 Kernel 故障。窗口外的预热、随机输入和数值校验不混入目标计数。

采集完成且自动 CPU 审核通过时，`summary.pipeline_passed=true`，`dispatch_review.dispatch_observed=true`。退出 0 或 `profile_worker.complete=true` 单独不代表 dispatch 通过。原始数据库、CSV、pftrace、日志全部回传；pftrace 当前不解码。

## 产物与身份

输出根目录保留 `plan.json`、`profiler.json`、`summary.json`、`dispatch_review.json`、`target_kernel_timeline.json`、`process.log/json`，以及 `trace/worker/` 下的图、运行身份、Key、准备记录、阶段和 worker 结果；hipprof 的数据库、CSV、pftrace 及 worker.log 留在 `trace/`。

计划、worker 和 Key 记录中均附采集器 Git revision 与实际源文件 SHA；审核器另记录自身 SHA。后续仅文档或报告提交不使相同源码的回执失效。冻结 Adapter 及 Recipe 不改，避免改变已验证实现的 SHA。

失败立即停止；Python 异常保留 traceback/pool/阶段及 worker 显存快照，原生崩溃依靠 faulthandler 日志与进程退出码。父进程不查询 GPU 显存。

本入口不产生任何正式延迟标签，`automatic_admission=false`。通过后再冻结独立时延采集与 Admission；不把 profiling DurationNs 用于校准，不启动 GPU1、其他 7 项厂商任务、MCTS 或 SROP。
