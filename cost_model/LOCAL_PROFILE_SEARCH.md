# GPU0两候选的Profile驱动搜索检查

2026-11-05。固定第0层激活边界，Native与LightOP Fused两个合法选择；只在GPU0评分。入口 `python -X utf8 -m cost_model.local_profile_search --output /fresh/local_profile_search`，结果 `data/local_profile_search_v1/report.json`。

## 数据与候选

从已准入LightOP图快照取gate/up→SiLU→Mul边界子图，gate/up作为已驻留输入，不包含Projection。Native和Fused保留各自已观测的执行视图；Fused在切片视图上的Canonical Key必须与原正式Key完全相等，再调用原严格查询。Native显式绑定已测aten.out路径和实际布局。

| 候选 | 当前评分 / μs | 成本来源 |
| --- | ---: | --- |
| Native | 10.1728001237 | SiLU6.5376001596＋Mul3.6351999640，经现有调度器聚合 |
| Fused | 17.8111803532 | 正式完整Pack＋LightOP Profile，不追加开销 |

图上入口为逻辑边界，成本0。其他所有计算成本都有明确实测来源；该检查遇到缺项直接停止，不静默使用旧eager解析代理。通信为0，GPU1没有参与评分。

Native两个stage identity、Fused Canonical Kernel Key与契约身份均写入报告；候选逐项列出所用Profile身份。Torch构建、HIP/驱动、架构与flags的可比字段先核对，不因两者都叫GPU0就混用。

## 搜索输出

使用原`StructuralCostModel`、`exact_legal_search`与`legal_mcts_search`，没有修改MCTS算法。穷举得到2个方案，最优为Native。MCTS固定24次迭代、1次rollout、seed 0/1/7，三次均选择同一Native方案，相对于当前代价模型的Exact gap均为0。

本例中的implemented融合没有被强制选择，Native仍为合法替代。三次seed是小空间工程复核，不是三个独立工作负载；singleton参考方案已经是本模型最优，不据此评价后续树搜索收益。

## 结论边界

这是已有数据上的开发检查，说明Profile身份→组评分→调度→Exact/MCTS输出可接通。Native为独立阶段聚合估计，仍携带`measurement_context:isolated_stage`；Fused为完整调用实测，两者不是同批完整方案实测对照。

没有新增GPU执行，没有独立参考regret、预测泛化精度或完整FFN覆盖结论。旧整链数据参与过开发，不重新作为新方法独立验证集。这个两候选同卡空间不能证明设备放置收益、联合搜索优势或MCTS效率。

下一步围绕小型FFN方案列出可比较成本来源：有匹配Profile直接使用，缺项保留解析来源；先检查排序是否依赖未知成本，再决定是否值得补充测量。当前不启动GPU1、通信或Pack测量，也不回到单个Mul的参数追拟合。
