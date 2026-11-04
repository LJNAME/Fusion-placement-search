# 代价模型选型：NeuSight、nn-Meter、Proteus

2026-11-04。先评估已有预测器，暂停继续补测Native/Fused和修补当前效率曲线；MCTS不改。现有Adapter、身份查询、Profile准入及冻结实验保留。

选择 **NeuSight作为算子预测的主技术候选**；实现专属融合成本继续走现有严格Profile；图级先沿用StructuralCostModel，后续按需要参考Proteus。nn-Meter作为Kernel粒度回归备选。不同时迁入三套框架。

这是论文与源码选型，尚未运行第三方预测器、加载权重或验证DCU精度，不能称已完成替换。

## 1. 比较结果

| 项目 | NeuSight | nn-Meter | Proteus |
| --- | --- | --- | --- |
| 主要用途 | Tile/Wave级学习预测，聚合为算子延迟 | 实际Kernel粒度、按设备训练回归器 | 分布式执行图调度、计算通信重叠与带宽共享 |
| 已有实现 | 5类MLP、AMD配置和权重、Tile数据 | Kernel识别、RandomForestRegressor、采样和后端接口 | Strategy Tree、HTAE、Profile和通信模型 |
| 原有GPU/环境 | NVIDIA及AMD MI100/MI210/MI250；论文测量为FP32 | 移动CPU/Adreno GPU/Myriad VPU；现成后端为TFLite/OpenVINO | NVIDIA/NCCL导向，运行成本可来自Profile |
| BW1000可直接使用 | 未核实；BF16、库实现、Tile和Graph边界需适配 | 无已核实的DCU模型；需新后端和目标数据 | 无已核实的DCU运行路径；不能替代单Kernel预测器 |
| 新数据需求 | 原FP32复现可用公开权重/表；BF16/DCU准确性仍需目标验证，数量待CPU评估后确定 | 每类实现/设备需训练样本；现有同Shape两臂不足以训练通用预测器 | 仍需算子与通信成本，不会消除数据来源问题 |
| 项目适配复杂度 | 中到高，优先隔离低层预测器 | 回归器本身较轻，完整Builder迁移较重 | 整体迁移较重；借鉴调度方法较轻 |
| 本次定位 | 主候选 | 方法备选 | 图级参考 |

复杂度为基于源码依赖的工程判断，没有估算确定工时，也没有以论文其他硬件的精度作为DCU精度承诺。

论文与官方实现：[NeuSight论文](https://arxiv.org/html/2407.13853v2)、[NeuSight代码](https://github.com/scai-tech/NeuSight)、[nn-Meter论文介绍](https://www.microsoft.com/en-us/research/publication/nn-meter-towards-accurate-latency-prediction-of-deep-learning-model-inference-on-diverse-edge-devices/)、[nn-Meter代码](https://github.com/microsoft/nn-Meter)、[Proteus论文](https://ieeexplore.ieee.org/document/10636756/)、[Proteus代码](https://github.com/JF-D/Proteus)。

## 2. NeuSight可以复用什么

官方源码固定到 `6945927d9afcca2b9daf021f8395e53edc5b4eef`。可复用MLP_WAVE_MM、MLP_WAVE_VEC及Tile/Wave聚合方法；设备配置包含SM/CU数量、缓存、内存、带宽和计算吞吐特征。

官方树中有BMM、LINEAR、VEC、SOFTMAX、LN五组AMD权重，共31,633,540字节。本轮只核实文件清单，未下载或加载权重。Tile表与权重是不同依赖：缺少Tile数据时不能把随机Tile当作真实实现的Tile。

需要处理的实际差异：

- **FP32→BF16**：论文§6.1明确全部测量为FP32；预测入口与矩阵模型有4字节假设。修改为2字节只修正访存量，不代表BF16计算吞吐、分块算法与学习权重已经适配。
- **算子覆盖**：dispatcher有Mul、Add、GELU、ReLU等，没有SiLU和RMSNorm的现成映射。RMSNorm不能直接改名为LayerNorm；RoPE也不是现成专用预测类别。
- **实现差异**：论文融合方法合并计算量、消除中间访存，不能直接当作“pack＋LightOP”的完整调用成本。LightOP与打包保持实际实现身份；现有完整调用Profile优先精确查询。
- **执行边界**：官方collector采用Event测量与重复调用，没有本项目冻结的Graph 7×50契约。原公开标签不能直接升格为当前Graph标签；新模型也不自动解决旧launch代理问题。
- **CPU入口**：顶层`neusight/__init__.py`导入Dataset.collect，后者在模块级创建CUDA Event。源码虽允许低层模型选CPU，但CPU部署应隔离采集入口；尚未实际运行验证。
- **权重加载**：MLPredictor只在`model.pth`存在时加载。接入时必须显式确认权重与Tile表已加载，不能把缺权重时的输出称为论文预测结果。

源码证据（固定提交）：[预测入口](https://github.com/scai-tech/NeuSight/blob/6945927d9afcca2b9daf021f8395e53edc5b4eef/neusight/Prediction/predictor.py#L44)、[矩阵模型4字节假设](https://github.com/scai-tech/NeuSight/blob/6945927d9afcca2b9daf021f8395e53edc5b4eef/neusight/Model/mlp_wave_mm.py#L20)、[Tile查询](https://github.com/scai-tech/NeuSight/blob/6945927d9afcca2b9daf021f8395e53edc5b4eef/neusight/Model/meta.py#L34)、[包初始化](https://github.com/scai-tech/NeuSight/blob/6945927d9afcca2b9daf021f8395e53edc5b4eef/neusight/__init__.py)、[采集入口](https://github.com/scai-tech/NeuSight/blob/6945927d9afcca2b9daf021f8395e53edc5b4eef/neusight/Dataset/collect.py#L17)。

## 3. nn-Meter和Proteus的取舍

nn-Meter固定到 `cd8dab49b735d58d03746141f73ef5934559ae68`。可以借鉴“实际Kernel/完整调用边界＋Shape特征→按设备回归”，但现成设备模型不能用于DCU。仓库于2025-09-12归档。检查的配置中，fc初始采样为3000、conv-bn-relu为10000；这些是原工具参数，不是本项目需要照搬的测量量，也说明完整Builder不是拿两条Profile就能训练的模型。

只借鉴其Kernel粒度回归时，应称“采用nn-Meter方法”，不能称“已接通原nn-Meter框架”。现有注册表已明确融合候选，不需要为这一实验重新运行整套融合规则发现。SROP与其自适应采样也不在当前阶段引入。

证据：[RandomForestRegressor实现](https://github.com/microsoft/nn-Meter/blob/cd8dab49b735d58d03746141f73ef5934559ae68/nn_meter/builder/kernel_predictor_builder/predictor_builder/predictor_lib.py#L3)、[采样配置](https://github.com/microsoft/nn-Meter/blob/cd8dab49b735d58d03746141f73ef5934559ae68/nn_meter/configs/builder/predictor_builder/predictorbuild_config.yaml#L25)、[自定义预测器](https://github.com/microsoft/nn-Meter/blob/cd8dab49b735d58d03746141f73ef5934559ae68/docs/builder/customize_predictor.md)。

Proteus固定到 `ef8730d3f45025ff4f7119ad12b4176bb0ace212`。它解决图级执行时间聚合，不是已训练好的通用GPU算子预测器；其OpCostModel仍查询缓存或调用已注册的成本函数。整体迁入会带入Strategy Tree、NCCL绑定与拓扑依赖，超出当前替换算子成本的需求。

项目已有`StructuralCostModel`：`cost_model/structural.py:212`取组成本，`:234`取通信成本，`:241`以后按链路可用时间调度，`:267`处理发送是否阻塞设备。先保留这一层，后续针对重叠和带宽共享与Proteus逐项核对，不把现有实现称为HTAE复现。

证据：[Proteus成本接口](https://github.com/JF-D/Proteus/blob/ef8730d3f45025ff4f7119ad12b4176bb0ace212/proteus/simulator/cost_model.py#L148)、[安装与拓扑](https://github.com/JF-D/Proteus/blob/ef8730d3f45025ff4f7119ad12b4176bb0ace212/README.md)。

## 4. 对应当前Qwen3参考图

CPU重建参考Builder：591个节点。下面是算子家族对应，不是已可执行、已适配或accepted成本覆盖。

| 逻辑类型 | 节点数 | NeuSight家族及缺口 |
| --- | ---: | --- |
| matmul | 253 | LINEAR/BMM候选，需映射矩阵维度、权重转置、dtype与实际库实现 |
| add / mul | 56 / 28 | VEC候选；Mul链成本不能拆成任意放置的Singleton成本 |
| softmax | 28 | SOFTMAX候选，需核对轴、布局与dtype |
| rmsnorm | 113 | 不能直接等同LN，保留独立缺口 |
| silu | 28 | 无现成dispatcher映射；与Mul融合时保留LightOP实现专属成本 |
| rope | 56 | 无现成专用预测类别 |
| embedding | 1 | 原方法可用访存估计，但仍是估计且需标来源 |
| input | 28 | 逻辑边界，不生成GPU耗时标签 |

这是项目参考图，不是上游实际捕获图的覆盖声明。当前GPU回执只含1个Shape的两种激活实现；350次重复调用不能算350个独立Shape训练样本。

## 5. 接入边界与开发顺序

先替换`HardwareCostProfile.estimate_group`背后的预测来源，保留`estimate_transfer`和`StructuralCostModel → CostEstimate → SearchCandidate`链路；不改MCTS。精确实现/输入/设备/契约匹配的Profile继续优先查询；论文预测必须标为预测，未支持项明确列出，不冒充实测或默认为已覆盖。

1. CPU可复用性验证：隔离NeuSight低层推理，加载固定权重和Tile表，在论文原有FP32支持样例上核对输出与单位。先证明原方法能工作，不用改过的BF16案例代替原方法复现。
2. BF16/DCU映射：核对已有硬件资料中的CU、L2、BF16矩阵/向量吞吐及带宽单位；当前local_pair runtime没有完整NeuSight设备特征，未核实项保留缺失。按真实库实现核对Tile来源，再决定是否值得做有限适配；不承诺预训练权重直接可用。
3. 范围内的校准与验证：只有CPU评估确定缺项后才设计必要的小测量集。LightOP完整调用继续查专属Profile；Native与Fused两臂对齐来源与执行边界。若NeuSight适配需要超出课题的训练工作，转用nn-Meter式实现专属回归，而不是完整重训全部论文模型。
4. 新模型可靠后，再恢复小型搜索域的Exact/MCTS对照。完整多GPU聚合按需求扩展，SROP后置。

旧Native两次launch代理本身为15.54081 μs，已高于该链10.09280 μs的本批Graph实测。这至少说明旧执行边界代理存在问题；引入成熟方法仍须处理Graph语义，不能把所有差异简单归因为模型容量小。

现有两臂数据保留原冻结评价；在新方法设计中使用它们后，作为开发/诊断案例，不再称新方法的独立测试集。后续独立集按Shape/实现/测量批次隔离，不按同一Case的重复轮次随机拆分。

## 6. 本轮核验范围

读取3个官方仓库的固定树，下载27个源码/配置文件共202,656字节，逐文件验证SHA256与Git blob SHA1。清单为`data/cost_model_selection_v1/sources.json`，机器结论为同目录`report.json`；本地只读快照在`E:/Chatgpt_work/transfer/cost-model-selection-20261104/`。

没有下载模型权重、安装依赖、执行第三方代码、新GPU测量或修改运行时预测器。本轮交付是选型结论与下一步CPU适配范围。