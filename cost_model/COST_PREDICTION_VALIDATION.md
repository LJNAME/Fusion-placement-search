# 成本预测验证：P0、P1、P2

2026-11-05。复用已有GPU记录，只做CPU审计、隐藏标签预测和48方案敏感性分析；没有新测量、解析参数重拟合、MCTS修改或SROP实验。

## P0：可复用数据

| 数据 | 数量 | 本轮用途 |
| --- | ---: | --- |
| 原E2 Kernel Key | 112 | 原eager契约下的回顾性留出预测 |
| 原新序列验证报告 | 56 Key、112条双卡记录 | 保留其原冻结eager结论，本节未重验原始回执 |
| GPU0 SiLU、Mul阶段 | 2 | 当前Graph契约的开发诊断与已有成本查询 |
| GPU0 Pack＋LightOP完整调用 | 1 | 严格命中时直接成本；隐藏精确值后作解析诊断 |
| 原GPU0 Native/Fused整链 | 2 | 局部决策的历史开发参照，不是FFN参考 |

原112 Key从raw_measurements、environments、manifest、split_contract和硬件文件重建，得到的记录及metadata与归档完全一致。算子数为matmul 56、rmsnorm 16、rope 12、add 8、mul 8、silu 8、softmax 4。

112条均采用eager循环内GPU Event，输出由Reference调用分配，HIP 6.3.26093。当前目标是预分配输出、Graph Replay，HIP 6.3.26113，具体实现与Recipe也不同。112条中可直接作为当前Graph兼容测试标签的为**0**，不能靠修改mode字段完成转换。

当前3条Graph成本记录已用于标定/开发或直接查询，本轮不重新命名为独立测试集。原112条也已用于此前预测器及实验；现在隐藏标签可检查预测数据流，不能恢复“首次盲测”状态。UUID变化没有作为排除原因。这里的0是本轮数据范围的结论，不声称审计了仓库外所有测量。

## P1：原eager契约的留出复核

沿用原两个完整Key划分：跨模型68 train/44 hidden，跨序列84 train/28 hidden；不按重复轮次切分，也不汇总成72个独立样本。预测器只拿到make_dataset_view遮蔽后的query，先生成预测ID，再由评价器读取标签。

- Analytic：原硬件均值、shape-aware Roofline与eager启动代理，不是当前Graph模型。
- Hybrid：隐藏Key不能精确命中Profile，采用同一解析回退；因此留出集上与Analytic相同。
- Nearest Shape：沿用现有特征距离和log residual方法，仅在实现/版本、执行后端、dtype、布局、硬件类型及其他非Shape身份均一致的训练记录中选邻居。归一化只用可见训练集；没有兼容邻居就记缺失，不跨实现兜底。

| 留出划分 | 方法 | 覆盖 | MAPE | Median APE | P90 APE | Signed Bias | Spearman |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 跨模型 | 原Analytic / Hybrid | 44/44 | 96.36% | 46.01% | 248.17% | +65.74% | 0.580 |
| 跨模型 | 严格Nearest | 40/44 | 19.98% | 8.91% | 34.01% | +10.12% | 0.964 |
| 跨序列 | 原Analytic / Hybrid | 28/28 | 127.01% | 89.99% | 306.55% | +85.45% | 0.510 |
| 跨序列 | 严格Nearest | 28/28 | 14.86% | 9.31% | 36.27% | +7.24% | 0.982 |

Signed Bias为mean((预测−实测)/实测)。Nearest的误差只统计可预测Key，同时保留原测试集覆盖分母。跨模型共同可预测的40条上，Analytic MAPE为88.15%，Nearest为19.98%；缺少兼容邻居的4条matmul没有删除，仍记作未覆盖。

| 算子 | 跨模型测试Key | 原Analytic MAPE | 跨序列测试Key | 原Analytic MAPE |
| --- | ---: | ---: | ---: | ---: |
| matmul | 24 | 145.18% | 14 | 204.82% |
| rope | 4 | 93.39% | 3 | 91.02% |
| rmsnorm | 4 | 28.95% | 4 | 45.01% |
| silu | 4 | 23.71% | 2 | 29.71% |
| mul | 4 | 21.82% | 2 | 17.09% |
| add | 4 | 20.99% | 2 | 36.35% |
| softmax | 0 | — | 1 | 69.33% |

原模型matmul总体高估，RoPE总体低估。逐Key的标签、预测、邻居ID和逐算子统计保留在JSON。原56 Key新序列报告中Analytic/Nearest MAPE为144.96%/11.23%，仍只属于原冻结eager实验，不升级为当前模型的Graph精度。

## P1：当前Graph模型的诊断

采用原configs/graph_analytic_scenarios.json中固定的illustrative assumptions，不遍历729场景选最低误差，不重新标定。独立阶段按原假设core＋h＋c，完整Fused按Pack＋LightOP两个阶段及调用边界评价。

| 成本对象 | 假设预测 / μs | 已有实测 / μs | Signed Error |
| --- | ---: | ---: | ---: |
| SiLU独立阶段 | 18.80150 | 6.53760 | +187.59% |
| Mul独立阶段 | 6.52461 | 3.63520 | +79.48% |
| Pack＋LightOP完整调用 | 28.07384 | 17.81118 | +57.62% |

隐藏精确Profile后，Hybrid也必须回退到上述预测；不允许把命中17.81118等实测值的0误差计入预测精度。三个实现各仅有该固定Shape的有效当前记录，没有兼容的其他Shape邻居，Nearest覆盖0/3，误差记null。未提供假设参数时，解析模型仍保留未知，数值覆盖0/3。

三个已知开发Case不能证明未见Shape泛化，也不用于拟合后再自证准确。当前FFN在GPU0的这三项仍使用已有实测成本；上表不是对FFN Native评分的直接替换。GPU1回退和Projection预测精度仍无独立当前验证。

本轮没有获得当前Graph模型的新独立测试Key，因此未通过“未测Kernel预测可靠”的验收；完成的是数据复用判定、有限误差诊断与可重复的隐藏标签评价入口。

## P2：48方案的通信翻转阈值

沿用dependency_level_node_id_v1及全部48方案，所有Profile和基准成本不变。每次重新计算资源约束makespan，记录逐组/传输轨迹、计算耗时总和与通信总和；后二者仅为诊断量，不能相加冒充方案时间。

设通信代理Tcopy＝L＋B/BW。每一轴均单独变化，其余成本固定；[0,1]是探索区间，不是可信范围。

| 单轴变化 | 全GPU0 Native与跨卡方案的切换 |
| --- | --- |
| 全部通信成本×α | α≈0.2425010616 |
| 固定延迟L×α，保留字节项 | α≈0.1610317239 |
| 字节服务时间×α，保留L | [0,1]内不切换 |
| GPU1 Projection成本×α，保留原通信 | [0,1]内不切换 |
| GPU1激活/融合成本×α，保留原通信 | [0,1]内不切换 |

两处阈值由全48方案的最优跨卡成本减最优零通信成本求根，二分42次，完整区间及正负边界保留在报告。固定优先级、零计算/发送干扰及非负成本使每个方案的成本随相应α单调不减；零通信方案在这些轴上保持不变。这不枚举多参数同时变化，也未扫描GPU0 Projection成本增大或链路组合残差。

基准最优为Native全GPU0，78.83045049 μs。整体通信缩放低于阈值后，最优变为Native的Up在GPU1，其余GPU0；高于阈值则保持全GPU0。在阈值附近，该跨卡替代需要：

| 传输 | 临界整体缩放下的成本 / μs |
| --- | ---: |
| 0.5 MiB，GPU0→GPU1 | 13.99139 |
| 1.5 MiB，GPU1→GPU0 | 15.41639 |

这是“通信足够便宜会改变放置”的条件，不是实测跨卡加速。仅增加带宽而不改变现有固定延迟，在上述单轴区间中不足以改变选择。

## 现有参考能否判断排序

旧Host wall通信点涵盖双方向4 KiB、64 KiB、1 MiB、16 MiB、64 MiB、256 MiB，且同批数据生成当前通信代理；它既不是独立约束，也不是当前Graph计时边界。

run05的Graph Copy审核保留如下，不重新改状态：

| 大小 / 方向 | 历史中位数 / μs | 原质量状态 | 当前搜索准入 |
| --- | ---: | --- | --- |
| 0.5 MiB，0→1 | 37.46560 | failed | 否 |
| 0.5 MiB，1→0 | 33.78238 | passed | 否 |
| 1.5 MiB，0→1 | 52.59202 | failed | 否 |
| 1.5 MiB，1→0 | 44.58558 | passed | 否 |

这些记录属于旧Recipe/窗口契约；单个历史值不能成为当前执行成本的可靠上下界。本次核查的通信证据尚不能给出当前Graph的可信α区间，因此无法断言该区间位于阈值哪一侧。Projection与真实调度的误差仍可能改变结论；FFN实测参考覆盖仍为0/48，方案级regret保持null。

原GPU0整链Native 10.09280 μs、Fused 18.84796 μs支持该已观测局部Case选择Native，但不是完整FFN成本或新的独立验证。这里不重复测SiLU/Mul，也不把原局部regret移用为48方案regret。

若后续进入P3，先围绕Up放GPU1替代方案核对0.5 MiB 0→1、1.5 MiB 1→0及Up Projection的可复用参考。两类Copy并不自动构成充分的FFN参考；不据此启动测量任务或强行准入旧记录。

## 复现与检查

```bash
python -X utf8 -m cost_model.profile_prediction_validation --output /fresh/profile_prediction_validation
python -X utf8 -m cost_model.ffn_decision_sensitivity --output /fresh/ffn_decision_sensitivity
```

结果分别在data/profile_prediction_validation_v1/report.json与data/ffn_decision_sensitivity_v1/report.json。21项相关CPU测试通过，新增9项，覆盖标签泄漏、完整Key冲突、严格邻居缺失、覆盖分母、旧契约排除、Graph未知传播、阈值正负见证与参考范围限制。

P0完成复用范围审计；P1完成可运行的回顾性评价，当前Graph独立泛化结论仍缺证据；P2完成单轴翻转阈值，可信范围下的排序稳健性仍未确定。模型、测量记录与调度器均未因本轮诊断改变。
