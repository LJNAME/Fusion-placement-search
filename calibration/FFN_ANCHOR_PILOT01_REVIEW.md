# B4 锚点 pilot 01 复核

Down0 的 R1、R3 均 quality_failed。四格管道完整、零崩溃；pilot_passed=true，r3_quality_gate_passed=false。

数据来自 ffn_anchor_pilot_20261102_01.tar.gz，SHA256=01ce0469e98659d1f3e7540cf4d59450808a4fd6d486229e73200ead17ce0196。原包及 54 文件归档在 transfer/ffn-anchor-pilot-received-20261102-01，13/13 来源哈希匹配 06af153。全局/worker plan、身份、契约、原始 Trace、样本、质量、报告 JSON/Markdown 均 CPU 重算一致。

## 质量结果

scope=pilot_only_not_profile_admission；下表数字不得进入 Profile Admission 或搜索成本。

| Case | target 中位数 μs/call | target IQR/Median | 无效轮（0 起） | 拒绝原因 |
|---|---:|---:|---|---|
| Down0 R1 | 20.499179 | 1.0850% | 0 | anchor 漂移 20.202% |
| Down0 R3 | 20.483201 | 0.1249% | 0、1 | pre/post 端点内 IQR 超限 |
| Up0 R1 | 16.579200 | 0.8301% | 1、2、3、4 | anchor 漂移 12.302%～16.592% |
| Up0 R3 | 16.566401 | 0.2800% | 4、6 | pre 端点内 IQR 超限 |

四个 target 的离散度均未超 10%，拒绝的直接原因全部在锚点判据。数值、输入只读、50 输出检查、Event 设备与逐阶段时间戳通过；每格 target/anchor 的后置 Trace 各 50 个 Kernel，身份与路径检查通过。

R3 的 14 轮 pre/post 中位数漂移全部不超过 10%，但端点内离散度仍触发质量失败。Up R3 第 4、6 轮 pre IQR/Median 为 12.2301%、10.0953%；Down R3 第 0、1 轮 pre 为 10.7123%、10.5209%，第 0 轮 post 为 255.6955%。第 0 轮 post 原始三值为 [41.411161、6.799960、6.636780] μs/call，原样保留。

## 端点位置与间隔

两个 R3 Case 的 14 个 pre 端点，第一次 Replay 全部严格大于后两次。Down R3 第 0 轮 pre 为 [7.999960、6.585600、6.601600]；Up R3 第 4 轮为 [8.204760、6.672000、6.572780] μs/call。

锚点预热结束后，target 准备耗时 4.973～6.476 s。端点第一条与同端点后续条的主机间隔也不同：例如 Down R3 第 1 轮，前一次 validation_end 至 pre 第一条 timing_begin 为 1046.805 ms，同端点第二、三条为 0.107、0.089 ms。CSV 明确使用这两个时间戳，不把它解释为已证明的 GPU 空闲时间或时钟状态。

这是本包的描述性位置现象。恢复/首发瞬态、主机准备/落盘间隔或执行环境均待解释；不能删除第一条、换样本、移动预热后重新判本包通过。

## R1/R3 对照

| 同一 workload 的独立两次运行 | R1 | R3 | R3−R1 |
|---|---:|---:|---:|
| Down target 中位数 μs/call | 20.499179 | 20.483201 | −0.07795% |
| Up target 中位数 μs/call | 16.579200 | 16.566401 | −0.07720% |
| Up pre-anchor 汇总 μs/call | 7.862360 | 6.672000 | −15.13998% |

pre-anchor 的 R1 列为七个单值的中位数，R3 列为七个端点中位数的中位数。不是同一次运行内配对的性能收益，也没有冻结“超出噪声”的判据。每个 workload×策略只有一次独立运行，不能估计越线率、宣布 R3 更稳或证明 R1 有偏。target 相近也不足以证明锚点两种策略等价。

## 设备与遥测

四格 UUID/build 一致，GPU0=gfx936:sramecc+:xnack-，UUID=30303134-6331-6162-3532-646331313231。run05 GPU0 的 UUID=30303134-6331-6162-3532-646133313431，两轮不是同一物理卡；Torch/HIP/Triton 版本相同不能替代物理身份。

频率、利用率、温度共 168 项观测全部为 null，报告 amdsmi 未安装或不可导入（ModuleNotFoundError）。本包没有可用遥测，不能确认热行为、频率或争用根因。

Down 的质量失败在单卡独立窗口中仍出现，但不能由此判成 Down 专属故障，也不能唯一解释 run05 的共享窗口效应。Up 同样未通过，当前直接阻断项是锚点采样/判据链路。

## 后续范围

保持 R3 未采用、七项矩阵关闭、旧数据不重判、新 accepted=0、priceable_plans=0/48。先明确锚点要检验首发/恢复状态还是连续执行状态，再冻结下一版采样方案；本轮未修改运行时、预热位置、阈值或环境。

Mul0 仍未 GPU 冒烟。四格完整性前置条件已 CPU 复核，可以单独执行纯元数据 Smoke；其通过不能挽救本包的质量失败，也不能自动开放正式矩阵。

复核产物 data/ffn_anchor_pilot01_review/report.json、replay_timeline.csv、verification.json；记录 28 个 target Replay、112 个 anchor Replay，共 7000 个计时函数体调用，不含预热/Capture/Trace。四项源码/样本/契约/报告篡改负向检查均拒绝，报告重复生成逐字节一致，原始文件 SHA 不变。

```bash
python -m tools.review_ffn_anchor_pilot /path/to/ffn_anchor_pilot_20261102_01 --output /fresh/review
```
