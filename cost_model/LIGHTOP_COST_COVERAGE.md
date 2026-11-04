# LightOP 正式成本接入与 FFN 覆盖审计

2026-11-04；重放首条 accepted Profile，仅 CPU 审计。机器记录：[report.json](../data/lightop_cost_coverage_v1/report.json)；重放入口：`python -m tools.audit_lightop_cost_coverage --output /fresh/audit`。

## 当前覆盖

| 项目 | 结果 |
| --- | ---: |
| LightOP 单组正式 Profile | 1 条，GPU0，第 0 层 BF16 `[1,256,3072]` |
| 严格查询值 | 17.8111803532 μs/call（完整打包＋LightOP，Graph 批摊销） |
| 合法带设备标签方案 | 48：Native 32，LightOP Fused 16 |
| 计算结构置换类 | 24；不合并评分，入口/出口固定 GPU0 |
| 含已准入融合成本的方案 | 8；仍有其他缺项 |
| 完整可定价方案 | **0/48** |
| 零通信方案 | 2：Native/Fused 各 1，全部计算放 GPU0 |
| 图上计算成本用途 | 12 个 `(node/group, device)` 用途；11 个未覆盖 |
| 逻辑通信类别 | 4，正式通信成本 0 条 |

这 12 个用途不是 12 个已闭合的 Canonical Key，也不等于 12 个独立测量任务。枚举复用 B1 的动作算法，但使用 LightOP registry 重新验证结构；原 KERV B1 工件未改，KERV 实现与原地语义不转成 LightOP 身份。合法性只覆盖结构与声明的放置约束，未认证 HBM、生产者布局或完整方案执行。

每方案的计算缺项、跨卡边缺项、已有本地融合成本均保存在 `plan_coverage`。48 个方案没有计算总成本或排名；未知项不计为 0。

Gate、Up、Down Projection 三个节点在每个方案都出现，且两卡都缺本路径可用的正式成本。每个方案只需要该节点所选设备上的一项，不能写成“每个方案同时需要两卡的 Down”。Native SiLU/Mul 另有缺项，Fused/GPU1 也未覆盖。

7 条原始图边×48方案得到 336 行，160 个跨卡边实例；从逐方案边自动取并集，保留 **0.5/1.5 MiB×双方向** 四类。未融合 SiLU→Mul 的 1.5 MiB 需求仍在并集中。传输机制、布局与同步契约未在本 LightOP 路径绑定，四类仍是逻辑用途。

## 解析值不能直接作为 Before

| 成本边界 | 数值 / 状态 |
| --- | --- |
| 现有理想融合组解析值 | 11.2876288076 μs，仅诊断 |
| 其中访存项 / launch 项 | 3.5172235403 / 7.7704052674 μs |
| 正式完整 Adapter 调用 | 17.8111803532 μs/call |
| 同契约解析预测、预测误差、校准系数 | **未形成，不计算** |

现有 `AnalyticHardwareProfile` 按组边界读写与一次 launch 估算（`hardware_profile.py:109`），没有 LightOP 的 `cat(out=...)`。旧 launch 来自同步 eager `scalar.add_` 的 Host wall loop（`calibration/calibrate_bw1000.py:92`、`:512`），不能改称 Graph Replay launch。旧硬件率参数来自 2026-10-19 / HIP 6.3.26093，也未在当前软件条件下重新绑定。

按 BF16 元素读写计算的逻辑搬运量：理想融合边界 4.5 MiB；pack 阶段 6 MiB；LightOP 阶段 4.5 MiB；完整调用合计 **10.5 MiB**。这些是接口语义下的逻辑字节，不是实测 HBM 流量，缓存复用等未由此证明。

因此不拿 17.8112/11.2876 拟合系数，也不据此报告模型误差。后续 Before 需要显式覆盖完整调用，并绑定目标 Graph 契约的开销和率参数；独立验证 Case 仍待建立。

## 通信接口的范围

`KeyedKernelProfile.strict=True` 约束组成本；`estimate_transfer` 仍委托给基础 Profile（`fusion_kernel_profile.py:190`）。当前真实查询返回 `analytic_link`，不是已准入通信成本。四个诊断值保留在机器记录中，`accepted_query=false`，不参与正式方案覆盖。

本轮只显式暴露该范围，未改被冻结的 Profile/Adapter/Recipe。首条组成本仍有效，不自动授予整个 HardwareCostProfile 正式成本覆盖。

## 最小补测范围

先闭合两个 GPU0 零通信方案的计算成本，暂不扩大放置范围：

| 图上成本用途 | 待完成 |
| --- | --- |
| Gate Projection / GPU0 | 实际 Reference 身份、输入与权重载体、布局、Graph 契约及正式成本 |
| Up Projection / GPU0 | 同上；与 Gate 完整 Key 一致后才可复用测量 |
| SiLU / GPU0 | 单算子成本，不用 Native 链组合时延替代 |
| Mul / GPU0 | 实际左输入元数据及单算子成本 |
| Down Projection / GPU0 | 实际 Reference 身份、布局与正式成本 |

这是 **5 个成本用途**，实际 GPU 测量任务数尚未冻结。Gate/Up 的同形状只列复用候选，不能在身份闭合前删一个任务。全 48 方案范围仍有 11 个计算用途及 4 类通信用途未覆盖；也不能把它们直接相加当作补测任务数。

随后用同语义 Native SiLU→Mul 链检查单算子加和误差，再验证代表性 FFN 链的组合成本。它们是组合模型诊断，不自动替代单算子 Profile，不以实现间性能优势作为研究目标。原 KERV pair、旧 run05 数据不追溯升格。

本轮没有新 GPU 测量、完整方案评分、Exact 或 MCTS。7 项 CPU 测试覆盖重放、48 方案缺项、8 个已覆盖融合用途、通信解析委托、设备备选语义与错误边界禁止计算误差。
