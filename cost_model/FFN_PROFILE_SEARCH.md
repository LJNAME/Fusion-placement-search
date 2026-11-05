# 小型FFN的混合成本与搜索检查

后续CPU调度审计已完成，见 [FFN调度评分审计](FFN_SCHEDULING_AUDIT.md)。本文数值及data/ffn_profile_search_v1保留为旧顺序策略记录；当前入口采用明确的依赖层级/节点ID策略，新结果在data/ffn_schedule_audit_v1。

2026-11-05。固定第0层Gate/Up Projection、SiLU→Mul或LightOP、Down Projection。入口和出口驻留GPU0，内部5个算子可独立放到GPU0/1，融合组不可拆分。只做CPU预测与开发检查，原GPU数据和搜索算法不改。

入口：`python -X utf8 -m cost_model.ffn_profile_search --output /fresh/ffn_profile_search`。配置 `configs/ffn_profile_search.json`，结果 `data/ffn_profile_search_v1/report.json`。

## 空间与成本来源

32个Native方案＋16个Fused方案，共48个内部放置方案。评分时补入两个固定GPU0逻辑驻留端点，保留7条真实依赖边，包含0.5 MiB入口/出口与1.5 MiB内部边。Down放GPU1时，返回GPU0的边必须计费。

| 图上成本用途 | 来源 |
| --- | --- |
| GPU0 SiLU、Mul | 原独立阶段实测，按明确的实现/布局/契约绑定 |
| GPU0 LightOP组 | 原严格完整Profile，含Pack，不追加开销 |
| Gate/Up/Down × 2卡 | shape-aware Roofline＋历史矩阵/访存率，仍是解析代理 |
| GPU1 SiLU、Mul、LightOP | 显式向量/非线性/访存/Graph假设，不复用GPU0为GPU1实测 |
| 0.5/1.5 MiB × 双方向通信 | 历史链路率的解析代理，无当前Graph Copy契约实测 |

12个图上计算查询用途，3个在声明的执行条件下命中实测，9个采用解析代理；4类通信均为代理。48/48能评分，**0/48具备全实测方案成本**。用途数不是完整Canonical Key覆盖率，正式Profile与reviewed阶段记录也不合并成一个accepted计数。

每方案保存组→节点→设备、每组评分来源、Profile身份或假设ID、跨设备传输与边界成本。Projection的连续预分配输出是规划契约，不能称已捕获真实GEMM输出布局；权重假定已驻留。实测复用具有这个执行条件前提，独立阶段缓存上下文仍另标。

Projection的Graph开销显式假设1.5 μs；旧eager launch 7.7704 μs没有用于本轮评分。GEMM仍为现有历史率参数模型，不称已完成OpenXLA GEMM预测精度验证。通信采用现有逐边传输、共享链路和发送阻塞假设，不擅自宣称Fan-out只传一次或已实现真实通信重叠。

## 五组探索假设

| 场景 | 当前最优评分 / μs | 当前最优放置 |
| --- | ---: | --- |
| 基准 | 78.83045 | Native，全GPU0 |
| 共同Projection成本×4 | 284.80340 | Native，全GPU0 |
| 通信代理成本×0.1 | 61.71533 | Native，Up在GPU1，其余GPU0 |
| GPU1 Projection成本×0.8 | 78.83045 | Native，全GPU0 |
| 通信×0.1且GPU1 Projection×0.8 | 61.71533 | Native，Up在GPU1，其余GPU0 |

这些倍数是探索性压力探针，不是可信区间、设备规格或实测改进；不同场景最优评分不能当作真实加速。测试场景的最优融合选择均未变化，但通信假设可改变设备分配。

全GPU0的Native/Fused评分为78.8304504900/86.4688307195 μs。将共同Projection成本×4后，二者差仍为7.6383802295 μs；固定同卡串行比较中，共同成本抵消。跨设备场景有依赖、通信和调度竞争，不能套用同一个C相消。

## Exact与MCTS

每场景Exact评估48个合法内部方案。原MCTS按128迭代、2次rollout、seed0/1/7运行；每次独立清空评分缓存，不提供Exact方案评分表。

5×3次搜索均达到当前模型的Exact最优评分，gap=0。但每次MCTS也评估了全部48个唯一候选，**没有搜索预算或效率优势证据**。多个seed/假设场景不是独立GPU工作负载，不报告独立参考regret、预测准确率或联合优化真实收益。

## 调度优先顺序诊断

基准中，Gate在GPU1，其余算子在GPU0的同一放置：

| 同一来源/设备分配下的计划 | 评分 / μs |
| --- | ---: |
| Native，现有规范顺序Gate→SiLU→Up→Mul→Down | 200.11369 |
| Fused，现有规范顺序Gate→Up→Fused→Down | 184.86733 |
| Native，另一合法顺序Gate→Up→SiLU→Mul→Down | 177.22895 |

旧调度器按组优先序逐项占用资源，不在等待间隙自动填入独立算子。规范Native顺序使Up等待SiLU，改变合法顺序会改变评分。因此某些局部融合分差包含调度顺序效应，不直接归因Kernel融合收益。

这条诊断没有加入48方案域，也没有修改调度器；本轮Exact仅对现有规范顺序下的融合—放置空间穷举，不覆盖所有合法执行顺序。原始评分和诊断都保存在报告。

## 当前结论与下一项

已具备固定FFN的成本来源清单、48方案预测、缺项敏感性、Exact/MCTS对照及逐方案追溯。真实完整FFN成本准确性仍未验证，设备分配受未校准通信代理影响，部分排序也受固定组顺序影响。

后续调度审计已消除本实验输入表示的评分差异，48方案重新评分与15次Exact/MCTS对照见上述审计。下一项核查成本预测有效性的现有参照与缺项；SROP仍冻结，不因本次CPU结果增加GPU测量或返回Mul单点拟合。
