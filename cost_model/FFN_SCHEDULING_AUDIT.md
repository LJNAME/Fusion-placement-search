# FFN调度评分审计

2026-11-05。固定48个融合—放置方案与全部成本输入，只调整CPU评分中的调度优先级。没有新GPU测量，也未修改Exact/MCTS算法。

## 评分语义

评分为资源约束下的模拟完工时间（makespan），不是算子时延总和，也不是无限资源DAG的关键路径下界。

旧策略`plan_order_v1`按Plan.groups预留设备资源。节点列表同时充当调度优先级，因而两个合法拓扑表示可能给同一融合—放置方案不同分数。该策略仍为StructuralCostModel默认值，供历史调用重放。

FFN入口显式采用`dependency_level_node_id_v1`：

- 融合后的组DAG中，源组层级为0，其余层级为最长前驱路径长度；按层级、再按排序后的节点ID元组安排。
- 每个GPU保留一个串行计算资源，通信沿用Profile定义的链路资源、发送阻塞与干扰假设。
- 同组发出的传输按目标组节点ID、源/目标节点ID、字节数排序，不依赖边列表排列。
- 保留现有资源预留算法，不填补空闲间隙，不优化Stream或执行顺序。组ID仍对应输入表示；比较轨迹时按节点身份对齐。

组开始时间为设备可用时间、同卡前驱完成时间、跨卡传输完成时间的最大值。轨迹记录start/end、device_ready、local_dependency_ready、communication_ready和communication_wait。communication_wait仅表示通信到达时间超出设备与同卡依赖就绪时间的部分，不是独立测量的通信开销。

该策略是预测执行契约，尚未证明真实FFN执行器采用同一调度。节点ID在本实验中固定；调度策略不属于搜索变量。

## 全枚举检查

| 项目 | 数量 / 结果 |
| --- | --- |
| 每场景候选 | 32 Native＋16 Fused＝48 |
| Native合法组拓扑表示 | 每方案3种 |
| Fused合法组拓扑表示 | 每方案2种 |
| 每场景表示检查 | (32×3＋16×2)×2＝256 |
| 5场景总检查 | 1,280 |
| 节点、边列表 | 原表示和同时反转表示 |
| 按节点身份对齐的评分、轨迹、显存、关键路径 | 全部一致 |
| 依赖、设备占用、链路互斥检查 | 全部通过 |
| Profile身份、查询成本、假设与配置 | 和旧报告一致 |

## 旧例复核

基准场景：Gate Projection在GPU1，Up、SiLU、Mul、Down在GPU0。

| 评分方式 | Native / μs | Fused / μs |
| --- | ---: | ---: |
| 旧规范组顺序 | 200.11369 | 184.86733 |
| 旧策略＋另一个合法Native顺序 | 177.22895 | 184.86733 |
| 固定新策略，全部合法拓扑表示 | 177.22895 | 184.86733 |

旧Native顺序使GPU0先等待SiLU，独立Up不能回填等待间隙。固定策略将同层Gate/Up先排，表示顺序不再改变资源预留。

## 48方案重新评分

| 场景 | 评分改变的方案 | 最优评分 / μs | 最优方案变化 |
| --- | ---: | ---: | --- |
| 基准 | 12/48 | 78.83045 | 无，Native全GPU0 |
| 共同Projection×4 | 8/48 | 284.80340 | 无，Native全GPU0 |
| 通信×0.1 | 8/48 | 61.71533 | 无，Native的Up在GPU1 |
| GPU1 Projection×0.8 | 12/48 | 78.83045 | 无，Native全GPU0 |
| 通信×0.1＋GPU1 Projection×0.8 | 8/48 | 61.71533 | 无，Native的Up在GPU1 |

五组完整排名均有变化。每场景16组同放置的Native/Fused比较中，旧策略有4组Fused评分更低，新策略为0组；这描述当前混合成本和固定策略，不是Native对所有融合实现的性能结论。

Exact与原MCTS再次运行。5场景×3 seed全部模型gap=0，每次MCTS仍评分48个唯一候选；没有搜索效率优势证据。48/48可评分，0/48全实测，预测精度与完整方案真机成本仍未验证。

## 文件与复现

入口：`python -X utf8 -m cost_model.ffn_schedule_audit --output /fresh/ffn_schedule_audit`。

代码：`cost_model/structural.py`、`cost_model/ffn_profile_search.py`、`cost_model/ffn_schedule_audit.py`。结果：`data/ffn_schedule_audit_v1/report.json`，含48方案逐节点/逐传输轨迹、五场景新旧对照、1,280次表示检查和Exact/MCTS结果。旧`data/ffn_profile_search_v1/report.json`保持原字节。

67项相关CPU测试通过，其中4项为本次审计测试；包含已有prepared scheduler回归。未修改冻结采集器、Profile原始记录、解析模型参数或MCTS。

调度表示一致性审计到此结束。下一项为已有成本与参考证据的有效性检查：区分可验证的局部排序和仍无独立参照的FFN预测，不把模型Exact当作真机最优，也不继续扩大调度器或单点测时。
