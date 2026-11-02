# 条件化 A/B 对照采集器

入口 calibration.ffn_anchor_conditioning，默认 CPU plan。collector_implemented=true、gpu_validated=false；本轮未执行 GPU。旧 pilot 仍四格质量失败，七项正式矩阵关闭。

冻结设计沿用 data/ffn_anchor_conditioning_design_v1/design.json。设计稿内的“未实现”字段保留历史状态；当前执行实现、源码 SHA 和 plan_id 读取 data/ffn_anchor_conditioning_v1/plan.json。新采集器使用独立 executor/Recipe/Measurement Contract，不改旧 collector.v2 或其原件。

## 固定执行范围

| 顺序 | Worker | 负载 | 每个 pre/post 端点前条件化 Replay | 端点主测量 |
|---|---|---|---:|---:|
| 1 | block1_A | Down0 | 0 | 3 |
| 2 | block1_B | Down0 | 1 | 3 |
| 3 | block2_B | Down0 | 1 | 3 |
| 4 | block2_A | Down0 | 0 | 3 |

CPU 父进程按此顺序各启动一次 fresh worker，无重试。每格只访问 GPU0，现场身份/build 必须在四格间一致；不按型号共享或替代身份。Case、条件化次数和所属块均绑定到新契约。

anchor capture/10 次预热仍在 target 准备之前；Down0 capture 后 10 次预热。固定 7 轮，每 Replay 50 次函数体调用，原 fixture/buffer、恢复/同步/校验、Event 边界和 10% 质量规则保留，不补轮、不替换。

每轮顺序：遥测 → pre 条件化 0/1 次 → pre 主测量 3 次 → Down0 主测量 1 次 → post 条件化 0/1 次 → post 主测量 3 次 → 遥测 → 落盘。条件化到首条测量之间，采集器不插入 checkpoint 文件、遥测或 Profiler；两阶段都记录原始逐阶段时间戳，并计算实际间隔。

A 每格记录 49 个主测量 Replay；B 也是 49 个，另记录 14 个条件化 Replay。全批共 196 个主测量 Replay（9800 次函数体），另有 28 个条件化 Replay（1400 次函数体）；初始化、Capture、预热、Trace 另计。

## 原件与统计

samples.json 在每轮 post 结束后保存，包含 pre_conditioning、pre、target、post_conditioning、post。条件化样本标 sample_role=conditioning、primary_measurement=false；其耗时只作诊断，不进入端点 IQR、漂移、target 成本或主读数。三条主测量全部参与冻结质量计算，没有丢首条。

每条均保留 reset_begin、timing_begin、event_wait_end、all_device_sync_end、validation_end，及数值/只读检查、Event 设备、输入调用数。条件化虽然不参加成本统计，其数值、Event 或计数管道异常仍须停止；“变慢”本身不是管道失败。

全部七轮及所有条件化 Replay 完成后才 Trace target 和 anchor，此后不再计时。result.json 单独附路径与原文件 SHA，samples.json 的 pending_post_timing_trace 不覆盖。父进程重新核对原始时序、0/1 次条件化、样本角色、全三条测量、预热数量、身份、契约和原始 Trace，再计算质量及主读数。

每个 worker 输出 identity.json、preparation.json、samples.json、两个 Trace、result.json、runtime_identity.json、worker.log、parent_plan.json 和 last_stage.json。全局 summary.json、report.json/report.md 在每格后刷新，包含失败和阻断。

## 通过与失败

pipeline_passed 只表示四格完整且原件核验通过；quality_passed/quality_failed 单独显示。质量失败保留七轮并继续固定顺序，不能通过补轮或替换修复。

Capture、数值、Event、角色/计数/时序、路径或身份失败立即阻断后续格。普通异常留完整 traceback、pool、GPU0 显存及 context；父进程另留 parent_failure.json，含 worker 故障日志和最后持久化的 pool/显存。原生崩溃不重启，不由父进程补采 GPU。紧邻测量阶段不为日志额外落盘，故 last_stage 可能只定位到轮次；具体故障阶段以可用 fault stack 为准。

报告首项为 Down0 pre 的 first−median(next two)。先逐端点计算，再按每次独立运行的七轮取中位数；两个块分别报告 B−A，禁止用合并均值隐藏相反方向。计划每条件 2 次独立运行，报告同时列实际完成次数。

报告表头/行带 conditioning_diagnostic_only_not_profile_admission。即使两个块都下降或 B 质量通过，也不自动证明无偏、失败率改善、热/时钟机制或稳态已建立；protocol_adopted=false、followup_matrix_open=false。没有 Profile 写入或准入接口，不启动 Mul0 Smoke、七项矩阵或搜索。

## CPU 验收

76 项相关测试通过，含 A/B 次数和角色分离、条件化不计入主读数、三条主测量完整性、紧邻阶段无采集器 I/O/遥测、原件篡改负向检查、相反块方向、崩溃/UUID 变化阻断及旧 B4/Mul/导出回归。Bash 语法通过，三个 plan/模板产物重复生成逐字节一致，旧 collector.v2 plan/source 身份未变。

CPU 重放（全新目录）：

```bash
python -m calibration.ffn_anchor_conditioning --output /fresh/cpu_plan
```

显式 run 才启动四个 GPU worker，需 --flagscale-root。Bash 入口 tools/run_ffn_anchor_conditioning.sh 默认 plan，解释器读取 FUSION_PYTHON，run 才要求 FLAGSCALE_ROOT；沿用私有 Python 3.10 启动器和固定 FlagScale checkout。本轮没有执行 run，不能把 CPU 替身检查当作实际 HIP 验收。
