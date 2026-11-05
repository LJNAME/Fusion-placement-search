# Graph执行阶段解析模型：CPU适配v1

固定GPU0、连续BF16 `[1,256,3072]`。实现为 `cost_model/graph_analytic.py`；入口 `tools/analyze_graph_cost.py`；场景参数 `configs/graph_analytic_scenarios.json`；结果 `data/graph_analytic_v1/report.json`。这是可运行CPU分析模块，尚未接入MCTS默认成本接口，也未验证DCU预测精度。

## 成本对象

| 对象 | 读取 / 写入 | 计算项 |
| --- | --- | --- |
| Native SiLU | 1.5 / 1.5 MiB | 逐元素乘法＋单独的非线性服务项 |
| Native Mul | 3 / 1.5 MiB | 逐元素乘法 |
| Pack | 3 / 3 MiB | 逻辑搬运，不虚构算术量 |
| LightOP | 3 / 1.5 MiB | 融合计算，允许与Native不同的有效服务率 |

Native两阶段逻辑读写总量7.5 MiB，pack＋LightOP为10.5 MiB。字节数来自实际Recipe，不等于HBM流量；两个阶段不宣称已通过新trace观测到两个Kernel。

每阶段先计算算术与非线性服务时间之和，再与访存时间取max。串行阶段加和，分别加入Graph阶段、调用级及Fused额外开销。计算/访存重叠采用OpenXLA式近似；算术与非线性串行加和是本版假设，不是实际指令管线测量。

访存分成DRAM读、缓存读和写，支持显式read utilization、coalescing efficiency及cache hit fraction。缓存命中率大于0且缺少缓存带宽时，总成本保持未知；不按Tensor小于L2容量自动宣称缓存命中。

与旧模型的差别：实际pack执行阶段、独立非线性有效吞吐、实现差异、访存访问修正、Graph参数来源与未知传播。没有移植XLA编译器索引分析、占用率推断或GEMM分块搜索。方法依据：[OpenXLA官方设计说明](https://github.com/openxla/xla/discussions/10065)。

## 参数来源

每项保存value、unit、category、source、scope、status、needs_calibration。阶段字节/FLOP数量另有Recipe来源和单位。

- `memory_bandwidth_Bps`：旧设备带宽测量，仅历史代理。未据此称当前软件栈率参数已校准。
- `vector_flops_per_s`：未知；不把旧矩阵实测吞吐直接当作向量吞吐。
- `silu_elements_per_s`：未知非线性服务率；不以普通FLOPs峰值代替exp/除法等成本。
- `compute_utilization`、`lightop_compute_multiplier`：未知；有效率与实现修正可以互相混淆，不能从完整调用总和唯一分离。
- 有效带宽、访问合并、缓存比例及缓存带宽：显式假设或未知；Tensor连续声明不自动证明所有Kernel访问合并。
- Graph阶段、调用与Fused额外开销：全部未知，不读取旧eager launch，也不由本批完整耗时反解。

`unresolved`分支的总成本为null、choice为unknown。CPU场景只在显式填写假设后计算数值；假设不升格为实测或准入成本。

正式Fused路径调用原CPU reviewer重放准入及严格查询，返回17.811180353164673 μs完整调用值与Key/Contract/Admission身份；pack和launch追加均为0。该值只适用于其已观测实现、Case和环境，不认证当前新环境。Native的10.09280 μs局部参考没有进入参数拟合或本报告的预测输入。

## 敏感性结果

配置为6个轴、每轴3值，共729个组合。另给逐轴单因素结果；翻转只统计同一其他参数上下文中，单独改变该轴出现Native与Fused两种选择，保存两行见证。范围未经统计估计，不是置信区间。

| 参数 | 两臂解析：出现翻转的上下文数 | Fused完整Profile：出现翻转的上下文数 |
| --- | ---: | ---: |
| bandwidth_utilization | 21 | 90 |
| compute_utilization | 30 | 54 |
| fused_extra_overhead_us | 18 | 0 |
| graph_stage_overhead_us | 0 | 45 |
| lightop_compute_multiplier | 153 | 0 |
| silu_elements_per_s | 66 | 135 |

每轴对应243个其余参数上下文；这些是相关假设组合，不是独立实验、翻转概率或越线率。两臂各两个阶段时，共同Graph阶段开销抵消；实测完整Fused替代整臂后，该项只作用于Native。Fused内部参数在完整Profile模式下不改变成本。

向量率、缓存、访问合并、read utilization与调用级Graph项未进入本联合网格；不能称它们不影响排序。例示baseline只用于走通计算，不发布为已校准成本或推荐方案。

## 条件标定清单

| 使用模式 | 先核实的成本组合 | 当前不需重复拆测 |
| --- | --- | --- |
| 当前固定Case，Fused继续用严格完整Profile | Native有效访存、非线性/计算服务、Graph执行开销；先按已有资料缩小范围 | Fused的pack/Kernel内部拆分不是这个模式的前置 |
| 预测没有完整Profile的新融合条件 | 上述项，加LightOP相对计算效率与Fused额外开销 | 不为每个场景单独建立Profile |

这是范围依赖的候选清单，不是数学上唯一最小参数集，也不是已冻结的GPU任务。利用率、基础率与Graph开销存在可辨识性问题；以能约束决策的组合参数为目标，不强求从整调用总和拆出每项。

下一步先冻结本版适用边界和场景范围，查询已有硬件/软件资料收窄参数，再生成最小标定方案。未确定前不启动GPU、不承诺固定测量条数。后续独立验证使用未参与本轮设计的Shape/实现/批次；已有local_pair只作开发诊断，原冻结实验保留。

## 重放

在仓库根目录执行：

```bash
python -X utf8 -m tools.analyze_graph_cost --output /fresh/graph_analytic
python -X utf8 -m pytest cost_model/tests/test_graph_analytic.py -q
```

输出report.json必须不存在；没有GPU调用、训练或自动Profile准入。相同文件与路径重复构建应产生相同报告。修改场景文件使用`--scenarios`，所有数值仍是显式假设。

## 冻结状态

公式、固定Case与模式A/B已按 [冻结规范](GRAPH_ANALYTIC_FROZEN_SPEC.md) 固定；机器记录为 data/graph_analytic_freeze_v1/。本轮未改变解析实现与729探索范围，参数可信区间仍未闭合；2条Native路径及条件性的2条Fused路径仅为起步标定设计，不是已证明充分的参数集。

独立Native阶段采集入口与CPU复核已实现，见 [阶段校准入口](../calibration/NATIVE_STAGE_CALIBRATION.md)。GPU路径尚未验证，已有正式完整Profile和历史整链数据继续保留使用。

## 当前使用顺序

匹配的已复核Profile直接查值，未测项由解析模型补缺；见 [Native实测优先接口](NATIVE_MEASURED_FIRST.md)。729场景保留作历史开发诊断，不作为单算子测时或Profile使用的准入前置。暂停围绕Mul单点扩大参数分析，转向有限方案域的成本来源与排序验证。
