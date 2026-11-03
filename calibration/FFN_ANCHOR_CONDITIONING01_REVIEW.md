# Conditioning01 回传复核

四格均完成并通过冻结质量检查；两个区组的 pre 位置差值 B−A 均为负。仅作 conditioning_diagnostic_only_not_profile_admission 诊断，不采用新协议、不新增准入。

## 主读数

位置差值为每轮 first−median(next two)，再取七轮中位数。相对差值逐端点计算后取中位数，与绝对差值分别汇总。

| 区组 / 执行顺序 | A：无条件化（μs） | B：端点前一次条件化（μs） | B−A（μs） | 相对差值 B−A（百分点） |
|---|---:|---:|---:|---:|
| 1：A→B | 0.646380 | 0.011220 | −0.635160 | −9.643670 |
| 2：B→A | 0.025620 | 0.001600 | −0.024020 | −0.367926 |

每条件只有两次独立运行；七轮是运行内重复，不当作七次独立实验。第二块 A 本身已接近零，两块不合并成单一收益率。A 也全部通过，不能据此声称 B 降低越线率、为必要条件或建立稳态；具体热、时钟及缓存机制未确定。

## Down0 与锚点

| Worker | Down0 中位（μs/call） | Down0 IQR/Median | pre 端点最大 IQR/Median | 最大前后锚点漂移 | 质量 / 无效轮 |
|---|---:|---:|---:|---:|---|
| block1_A | 20.502379 | 0.202900% | 9.327167% | 3.780712% | passed / 0 |
| block1_B | 20.480001 | 0.304641% | 0.463334% | 0.391046% | passed / 0 |
| block2_B | 20.448000 | 0.500734% | 1.789018% | 1.547383% | passed / 0 |
| block2_A | 20.486400 | 0.296830% | 1.739889% | 0.585076% | passed / 0 |

Down0 的 B−A：区组 1 为 −0.022378 μs（−0.109148%），区组 2 为 −0.038400 μs（−0.187440%）。没有预注册等价区间或显著性判据，不称无偏、等价或真实加速。旧 pilot 的失败不重判，也不能据此次通过断定 run05 仅由共享窗口引起。

## 身份与原件

- 压缩包 SHA256：dbd03de011b97ba1561eca3c4eaf83624b4fc428b53ab8beb603378d70a57123。
- 54 个文件，展开 5,000,612 字节；工作区原件 transfer/ffn-anchor-conditioning-received-20261103-01/ffn_anchor_conditioning_20261102_01。
- 12/12 来源 SHA 与 3adcddc 一致；父/四 worker 的 plan、身份、契约、七轮原始样本、质量、Trace、报告与冻结模板均 CPU 重算一致，原件 SHA 不变。
- 计划顺序 A→B→B→A，四进程各启动一次、退出码均 0，无重试、无替换。四格 GPU0/build 一致，peer_capabilities 为空。
- 本批 UUID：30303134-6330-3532-3338-383230386131；架构 gfx936:sramecc+:xnack-。上批 pilot01 UUID 为 30303134-6331-6162-3532-646331313231，两批物理卡不同，不直接比较失败率或归因协议变化。
- Torch 分发版 2.9.0+das.opt1.dtk2604.2605140906.gcbed15，HIP 6.3.26113，Triton 分发版 3.5.1+das.opt1.dtk2604.torch290.2605141628.ge5e8df，驱动 60326113；实际加载 FlagScale revision f8e9a62c499d526a6ffa47f533bcf342330b7326，源码哈希保留在身份原件。
- 168 个遥测值全 null，原因 amdsmi 不可导入；没有温度/时钟证据。

## 样本与时序

196 次主测量 Replay（28 target、168 anchor，合计 9800 次函数体）；另有 28 次 conditioning Replay（1400 次函数体）。初始化、Capture、预热、Trace 不含在此计数中。

A 每格 49 次主测量；B 每格同样 49 次，另 14 次 conditioning。所有端点三条主样本完整，conditioning 标记为 primary_measurement=false，不参与主差值、IQR、漂移或 target 成本。所有恢复/同步/数值不变量和 Event 设备检查通过；每格 target/anchor 后置 Trace 各 50 个 Kernel。这里没有运行 KERV 融合候选或跨卡链路。

B1/B2 条件化完成到首条主测量 timing_begin 的主机间隔中位数分别 0.091061 / 0.089831 ms；范围分别 0.086601～0.122450 / 0.086341～0.124341 ms。它包含下一次 reset/同步等主机过程，不解释为 GPU 空闲时间。时间线保留全部 224 次 Replay，端点表保留 56 个端点。

## 可重放产物与后续

CPU 入口 tools.review_ffn_anchor_conditioning；data/ffn_anchor_conditioning01_review 含 report.json、replay_timeline.csv、endpoint_positions.csv、verification.json。46 项相关 CPU 测试通过（5 项新审计测试），三个审计产物重复生成逐字节一致。来源、运行时身份、summary 和报告四种负向修改均被拒绝；旧采集器/设计/阈值不改。

新 accepted=0，priceable_plans=0/48，protocol_adopted=false，七项矩阵关闭，MCTS/SROP 保持冻结。下一项可执行独立 Mul0 元数据 Smoke；它不提供锚点质量证据。若采用条件化，先单独冻结新 Recipe/Measurement Contract 与目标计量对象，不能用本批诊断数字作正式成本，也不能拼接旧 Profile。

```bash
python -m tools.review_ffn_anchor_conditioning /path/to/ffn_anchor_conditioning_20261102_01 --output /fresh/conditioning01_review
```
