# LightOP 第二批试验复核

LightOP 优化接口、四臂数值及只读检查、独立 hipprof 采集均完成。已合并输入下观察到局部收益；从独立 gate/up 开始计入拼接后，本轮路径耗时高于 Native。仅为单批探索结果，不进入正式准入。

## 四组计时

单位 μs/调用，普通 Eager 循环的 GPU Event 批摊销；各 10 次预热、100 次计时调用、每臂仅一批，按表顺序执行。四臂均预分配输出；输入准备和数值/只读校验不在计时区间。Native 的 SiLU 中间量分配与 split LightOP 的 cat 均在计时内。

| 输入边界 | 实现 | GPU Event | 同步 host wall |
|---|---|---:|---:|
| packed | Native SiLU→Mul | 31.147161 | 32.942520 |
| packed | LightOP | 7.806390 | 8.670790 |
| split | Native SiLU→Mul | 17.111990 | 17.693580 |
| split | 拼接＋LightOP | 21.812770 | 22.700220 |

同边界比较：packed 的 LightOP 相对 Native 耗时下降 74.937074%（比值 3.989957）；split 的 LightOP 路径相对 Native 增加 27.470679%（+4.700780 μs）。

packed 为连续 [256,6144] 合并输入，Native 从中切片；split 为两个独立连续 [256,3072] 输入。两组 Native 输入布局不同，基线不能互换，也不能把 packed 的差值作为当前 FFN 独立双输入的融合收益。

单批、固定顺序尚不能估计稳定性、置信区间或顺序效应；不能确认 Native 两组差异的唯一原因。两个 LightOP 路径之差包含拼接、分配与提交等差异，未单独测量 cat，不将其解释为纯拷贝时延。此次数值不能与旧 KERV Graph Replay 时间直接横比。

## 真实执行证据

| 独立剖析进程 | 源码算子调用数 | 目标 Kernel 事件 | 全进程 Kernel 事件 |
|---|---:|---|---:|
| Native packed | 11 | SiLU 11＋Mul 11，逐对相邻 | 100 |
| LightOP packed | 11 | act_and_mul_kernel_opt1 11 | 89 |

LightOP 设备符号为 `void at::native::act_and_mul_kernel_opt1<c10::BFloat16, &(c10::BFloat16 at::native::silu_kernel<c10::BFloat16>(c10::BFloat16 const&)), 8, true>(...)`。这是独立进程中观察到的优化融合路径，不能因符号含 at::native 就误判为两个算子的 Native fallback。

hipprof SQLite 的 Name 是字符串表 ID；按 PID/STR_ID 解码后核对。所有 Kernel 事件均在 GPU0，每条 EndNs−BeginNs 与 DurationNs 相等，逐符号调用次数和时长与 CSV 汇总一致。全进程事件还包含 fixture 和正确性检查，没有将 89 或 100 当作一次调用的 Kernel 数。pftrace 保留原件，未解码。

剖析在所有无 Profiler 计时结束后才启动，剖析数据不作为性能标签。没有对那 100 次计时调用逐次 Trace；目前证据支持同一接口和输入条件下的独立剖析观察。

## 来源与检查

- 原包 SHA256：41a98bd37414ad2563bb10118c975d157f2fa106418ab8b0c39d1ca6cc073f92；24 个文件，归档 transfer/hygon-silu-trial-received-20261103-02/hygon_silu_trial_20261103_02。
- plan 与冻结 v2 一致，2/2 采集来源哈希匹配 228ee20。四个子进程各一次，退出码均 0，无重试；四臂完整汇总与 worker.log 逐项一致，ms→μs/调用重算一致。
- LightOP 0.6.0+das.dtk2604.torch290.2605220936.g6bfaca；回传记录的 49 项包文件哈希均匹配已验证 wheel（含重复观测的 __init__.py，不表示 49 个独立文件）。实际 op.so SHA 为 594deb22551cacf50edfcc1104bad25abeb7f00e087612c9fdc024a0aa4a142f。
- 三个执行进程的设备、架构、软件和实现身份一致，实际为 BW / gfx936:sramecc+:xnack-。跨批次 ID 变化按用户 SCNet 重启说明处理。
- Torch 分发版 2.9.0+das.opt1.dtk2604.2605140906.gcbed15、Triton 3.5.1+das.opt1.dtk2604.torch290.2605141628.ge5e8df、HIP6.3.26113，与首批检查记录相同。包内未含安装报告，未重建完整安装过程。
- 原始输出 Tensor 未回传，数值结论依据源码匹配的采集器首/末输出与只读断言，不声称 CPU 独立复算 GPU Tensor 或校验全部 100 个中间输出。

CPU 入口 tools.review_hygon_silu_trial；产物 data/hygon_silu_trial02_review/report.json、profile_kernel_events.csv、verification.json。核验前后原件 SHA 保持不变。新审计测试覆盖字符串表解码、调用数、设备、时长、CSV 不一致和单位错误；与采集器相关测试合计 19 项通过。

## 当前结论与下一项

海光 LightOP 与 hipprof 路径已完成此次功能试验。候选成本需要包含输入布局与拼接，当前 FFN 的独立双输入条件下不能直接使用 packed 单 Kernel 成本评分。

下一项先固定 LightOP 候选的输入边界与转换计费，再以少量交替顺序重复核对 Native/split＋LightOP；无需先扩大 Shape、设备或搜索规模。尚未实现 LightOP 的 PlacedOpGraph Adapter、Canonical Profile 准入或搜索评分，不改变旧 KERV 准入；新增 accepted=0，priceable_plans=0/48，MCTS/SROP 保持冻结。本次仅 CPU 审核，不自动启动下一批 GPU。
