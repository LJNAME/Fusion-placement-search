# 锚点状态与条件化对照设计

本轮仅 CPU 分析和设计稿。旧 pilot 四格仍质量失败，R3 未采用，七项矩阵关闭；没有新增 GPU 调用或准入。

## 原件中的位置差异

数据来自已核验的 pilot01，scope=posthoc_endpoint_position_diagnostic_not_profile_admission。定义逐端点差值：首条 − median(第二条、第三条)，然后按七轮取中位数；相对差也先逐端点计算，再取中位数。

| Case / 端点 | 首条严格最大 | 差值中位数 μs/call | 相对差中位数 |
|---|---:|---:|---:|
| Down R3 / pre | 7/7 | +0.847960 | +12.9705% |
| Up R3 / pre | 7/7 | +0.312030 | +4.7415% |
| Down R3 / post | 5/7 | +0.011220 | +0.1718% |
| Up R3 / post | 3/7 | 0 | 0 |

每个 Case 仍只有一次独立运行。表中 7 个端点属于同一运行，不能当成七次独立实验，也不估计失败率。

普通 pre 轮间主机间隔中位数：Down 453.030 ms、Up 49.074 ms；端点内第二/三条之间分别约 0.091、0.100 ms。第一轮 pre 的时间戳参考是 anchor_warmup_end，中间含 target 准备和 GPU 工作；单独标注，不与普通间隔混算。只有间隔统计作此区分，所有读数和逐端点差值仍保留七轮。

最大首条读数却在 Down R3 第 0 轮 post：41.411161 μs/call，前面的 target validation_end 到本条 timing_begin 只有 0.102431 ms。因此，现有证据不足以把全部高值归结为长间隔。

这些是主机时间戳差，不是 GPU 空闲时长。校验本身包含 GPU 工作；频率/温度遥测又全部缺失，首发、缓存、时钟、前序工作或计时状态的机制尚未区分。没有按位置删除读数、重算通过或回填 Profile。

## 建议的代价对象

现有边界是 Graph Replay、每图 50 次调用、10 次预热，计时外恢复/校验。建议后续正式成本明确描述为“固定条件化配方下的 Graph batch 成本”，并让锚点比较同一配方下的 pre/post 条件。这个建议尚未进入正式契约。

暂停后的首发/恢复成本若需要，应单列条件与契约，不与条件化成本共用标签。一次条件化 Replay 是否足够，当前未知；不能直接把其后测量称为已经证明的稳态。

## 最小对照：只改变端点前是否运行一次同图

设计名 ffn-anchor-conditioning-design.v1，状态 cpu_design_for_review_not_active_measurement_contract。没有 GPU collector 或 run 入口。

| 固定顺序 | 对照块 | Down0 条件 | 每端点条件化 Replay | 正式记录的端点 Replay |
|---|---|---|---:|---:|
| 1 | 1 | A | 0 | 3 |
| 2 | 1 | B | 1 | 3 |
| 3 | 2 | B | 1 | 3 |
| 4 | 2 | A | 0 | 3 |

每格 fresh 进程，只用 GPU0，CPU 父进程不保留 Smoke 状态。四格重新观察 UUID/build，要求一致；不按型号替换身份。AB/BA 只提供两个顺序相反的工程对照块，不消除时间/环境混杂，也不支持失败率或无偏估计。

A/B 均固定七轮、每 Replay 50 次调用、端点三次测量、原 10 次预热及其位置、原 fixture/buffer、恢复/同步/数值校验顺序；不补轮、不替换、不放宽 10% 诊断规则。B 只在 pre 和 post 的三条测量之前，各增加一次同一 anchor Graph 的条件化 Replay。

条件化是新配方中的显式阶段，单独记录边界、校验及诊断耗时，不能把测量首条改名丢弃。完成条件化到第一条测量之间不插入文件写入、遥测或 Profiler，保留真实间隔。三条测量值全部参与原端点 IQR/Median；Trace 仍在全部计时之后。

计划四个 worker，共 28 次 target 测量、168 次 anchor 测量，合计 9800 次计时函数体；另有 28 次条件化 Replay、1400 次函数体，单独统计。初始化、Capture、预热和 Trace 不含在这些计数中。

## 读数与后续边界

主读数是每次独立运行的七个 pre 端点差值中位数。两个块分别报告 B−A；原三值、post、端点 IQR、pre/post 漂移、target 七轮成本和 IQR 均保留。若两个块方向不一致，结论保留 inconclusive，不用跨块平均隐藏差异。

条件化会同时改变前序 GPU 工作、缓存/设备状态及主机历史。即使差值缩小，也只能说明该配方下的位置差异变化，不能证明具体热/时钟机制。没有预设自动采用规则；管道完整性与工程质量诊断继续分开。

旧数据不重判。若未来采用新配方，须有新的 Recipe/Measurement Contract，并对 Native/Fused 及所需 singleton/通信成本保持契约一致，不能把新锚点配方与旧成本混用。七项矩阵、准入、搜索均不由此自动开放。

Mul0 元数据 Smoke 与这份设计相互独立，仍可按既有入口执行；它不提供锚点计时有效性的证据。本轮没有启动该 Smoke。

CPU 产物：data/ffn_anchor_position_v1/report.json、endpoint_positions.csv、verification.json，以及 data/ffn_anchor_conditioning_design_v1/design.json。29 项 CPU 测试通过，三个产物重复生成逐字节一致，既有 collector.v2 的 plan/source 身份未变。

```bash
python -m tools.analyze_ffn_anchor_position /path/to/ffn_anchor_pilot_20261102_01 --output /fresh/position_analysis
python -m calibration.ffn_anchor_conditioning_plan --output /fresh/design
```
