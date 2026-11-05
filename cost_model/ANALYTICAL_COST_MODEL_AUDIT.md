# 解析代价方法适配审计

2026-11-04；CPU源码核对与官方方法说明审阅，未运行第三方预测器。当前优先级是不需要大规模训练集的成本评价；代价模型作为搜索基础组件，不开展预测器创新。

## 选择与适用范围

| 方法 | 本轮核实范围 | 当前用途 |
| --- | --- | --- |
| OpenXLA GPU | 官方2024设计说明：HLO算子/融合解析成本、实现依赖与已知限制 | 算子与融合建模的主要方法参考；不是现成DCU预测器 |
| Calculon | 官方README：LLM模型、系统配置与并行执行优化入口 | 系统级参照，不替换任意算子图搜索接口 |
| LLMCompass | 官方README：ISCA 2024、硬件设计工具及软硬件/阵列模拟模块 | 后续矩阵分块建模候选；本轮未审计具体公式或建立DCU配置 |
| TileSight | 2026预印本摘要：Tile资源、缓存、依赖与通信分层 | 后续参考；公开评估设备未列BW1000，未核实DCU适配 |
| NeuSight / nn-Meter | 沿用v1固定源码审计 | 有目标验证数据后再评估，不自动进入训练或重训 |

免训练不等于不需要参数。硬件吞吐、带宽与执行契约仍须有来源；厂商Kernel的实际实现不能仅由逻辑算子名推出。上述方法均未在本项目完成DCU准确性验证。

官方来源：

- OpenXLA：[Cost Models in XLA GPU – Present and Future](https://github.com/openxla/xla/discussions/10065)，2024-02-28说明；不是对2026最新源码的固定版本复现。
- Calculon：[官方仓库](https://github.com/calculon-ai/calculon)，README的LLM Example与Testing and validation。
- LLMCompass：[官方仓库](https://github.com/PrincetonUniversity/LLMCompass)，README及公开目录。
- TileSight：[论文v1](https://arxiv.org/abs/2607.22432v1)，摘要与评估范围；不将其NVIDIA误差外推为DCU精度。
- 学习式方法与FP32/BF16差异：[首轮选型记录](COST_MODEL_SELECTION.md)，固定源码清单 `data/cost_model_selection_v1/sources.json`。

## 与当前实现的差异

OpenXLA说明中的顶层结构也是max(计算,访存)＋启动项。仅替换顶层公式不构成新预测能力。可借鉴的细节与本项目条件如下：

| 成本要素 | 当前实现 | 适配条件与处理 |
| --- | --- | --- |
| 顶层Roofline | `hardware_profile.py:103`计算、`:109`访存、`:113`聚合 | 保留基线，明确适用范围 |
| 运算成本 | `shape_model.py:119`按FLOPs累加，matmul默认K曲线在`:40` | SiLU等不能只凭普通算术峰值代表；实现细节未知时列缺项 |
| 访存 | 组入口＋出口字节/显存带宽 | 区分逻辑字节与HBM流量；布局、缓存复用没有证据时不填写已确认收益 |
| 并行度 | K饱和曲线默认eta_max=0.9、k_half=512 | 无实际分块/占用率来源，不复制XLA代码生成器参数作为DCU参数 |
| 调用边界 | 组级一次launch＋融合附加项 | 实际Recipe可含多个阶段；pack不能由理想融合消除 |
| Graph开销 | local_pair使用历史eager代理 | 独立列参数和来源；不把Host launch墙钟直接累加为Graph设备时间 |
| 图调度 | `structural.py:212`取组成本、`:234`取通信 | 保留现有接口；单Kernel参数改进不证明完整方案聚合准确 |

OpenXLA的访存与并行度修正依赖自身编译信息及硬件假设；其启发式不直接当作LightOP或海光硬件的实测属性。

## 现有诊断证据

来源：`data/lightop_local_pair_20261104_01/raw/{plan,prediction}.json`与`review/evaluation.json`，采集版本04c6d8c。计算为固定GPU0 BF16 `[1,256,3072]`、Graph Replay、每轮50调用。

| 项目 | μs |
| --- | ---: |
| Native冻结解析估计 | 21.4028497685 |
| Native链参考中位数 | 10.0927996635 |
| Native中两次旧launch代理合计 | 15.5408105347 |
| pack＋LightOP参考中位数 | 18.8479602337 |

旧launch代理合计超过Native整链参考。这足以拒绝“直接沿用该代理即完成Graph适配”的判断，不能证明去掉launch后剩余模型即准确，也不能从整链实测反推出纯Graph启动成本。两臂没有阶段独立计时，不能把完整Fused值拆成已测pack与已测LightOP。

本批是新方法开发/诊断案例；原冻结评价保留。350次调用不是350个独立Shape训练样本，也不按重复轮次拆训练/测试。

## CPU阶段成本分解规格（已实现）

1. 新模块接入`HardwareCostProfile.estimate_group`，保留当前运行时默认值，提供可选路径。Native显式SiLU、Mul；Fused显式pack、LightOP，分别记录逻辑字节、参数来源、实现与执行契约。布局未知不补成连续。
2. 精确匹配的正式完整调用Profile优先返回完整耗时，不再叠加pack或launch；local_pair仅作诊断，不自动准入。缺少对应成本时显式返回估计来源或缺项。
3. Graph开销、有效带宽与运算率分别给来源。缺少Graph参数时保留未知；需要展示数值时采用明确标注的假设区间做敏感性分析，不默认0、不由本批实测反解后宣称独立预测。
4. 验证阶段/单位/字节数、两臂执行契约、完整Profile不重复计费及参数变化下的排序。CPU构造结果只验证实现，不作为DCU精度证据。
5. 根据参数敏感性决定是否需要少量标定。未确认硬件常数前不冻结GPU测量数量；另留Shape/实现/批次验证排名与局部regret，不要求先补完整模型数据库。

第一版先覆盖连续逐元素与pack边界。GEMM分块、跨卡重叠、更多融合实现按实际搜索需要后置；不为这一版迁入完整XLA、Calculon或LLMCompass工具链。

训练集用于拟合泛化预测器；标定数据用于有限参数；独立验证用于检查误差与决策。三者分别记录。只有按同执行边界对齐后，预测误差与候选排序才进入正式校准结论。

实现与当前限制见 [Graph解析适配v1](GRAPH_ANALYTIC.md)。
