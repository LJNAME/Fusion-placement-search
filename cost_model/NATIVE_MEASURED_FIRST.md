# Native实测优先成本接口

2026-11-05。SiLU 6.537600159645081 μs、Mul 3.6351999640464783 μs已可通过Profile直接查询。解析误差分析与测时质量分开；原729假设网格不是实测数据使用的前置条件。

## 使用

```python
from cost_model import ReviewedStageStore, NativeMeasuredFirstProfile, StructuralCostModel

store = ReviewedStageStore("data/native_stage_calibration_20261105_01/raw")
profile = NativeMeasuredFirstProfile(graph, analytic_profile, store, observed_bindings)
cost = StructuralCostModel(registry=registry).predict(graph, plan, profile)
```

`analytic_profile`由调用者传入，保留其预测来源和不确定性；也可将现有完整Fused成本查询作为base的一部分。没有可用解析值时保留原错误/未知，不默认0。本轮提供可选接口，没有切换MCTS默认配置，也没有完成完整FFN实测覆盖。

每个`observed_bindings[node_id]`包含当前执行条件的完整stage identity和`operands`角色→生产者节点映射。SiLU为gate；Mul为left/up，left生产者必须为SiLU。图节点显式声明实际implementation id/version、输出Shape、dtype、layout和stride；与观测输入/输出及设备相符才命中。不能仅按算子名或Shape查询，也不能把F.silu、tensor.mul自动改名为已测的aten.out实现。

完整identity来自调用者的实际执行绑定，包含Torch构建、HIP/驱动、架构、Recipe与Graph契约。原数据的Key保留原身份，换Shape、设备、实现或eager契约不会复用该值。未知布局不补为连续。

## 查询行为

- 匹配：`source=measured_native_isolated_stage`，返回整段测量值，不另加memory或launch；GroupCost的compute_time字段承载不透明完整时长，不表示测出了纯计算部分。
- 未匹配：交给base，保留其source，并附带stage_profile miss原因。
- 多节点组：整体交给base，不拆成SiLU/Mul加和；完整Fused Profile的原查询规则保留。
- 通信、显存和调度仍由已有base/StructuralCostModel负责。

store读取已完成的CPU review及其绑定的原始记录，验证原始文件一致性并重算中位数。不再次调用模型plan或参数网格，也不要求新的解析模型仍与旧模型源码相同。测量原始记录和review标签不改；这些条目是限定契约的reviewed实测成本，不改写旧统一admission清单。

## 方案级解释

StructuralCostModel已能消费上述来源并调度。两个独立阶段在加性调度模型下会得到10.1728001237 μs，这是Profile驱动的预测，不是新增整链GPU测量。输出保留`measurement_context:isolated_stage`，缓存与链路可组合性仍是验证问题。

原Native整链10.09280 μs、完整Fused18.84796 μs继续用于已冻结历史对照和新方法开发诊断。正式Fused17.81118 μs仍按其完整Key/Contract使用。后续只有为新方法报告独立结果时才另设隔离批次，不为修正单个Mul的网格偏差重新测量。

## Mul的有限口径检查

原网格最低点为：4.5 MiB逻辑IO / 历史有效带宽 = 3.5172235403 μs，加假设h=0.1 μs、c=0.5 μs，得到4.1172235403 μs。与实测3.6351999640 μs的差为0.4820235762 μs。

0.1/0.5是探索假设，旧Copy率也不是当前Graph带宽或缓存行为。13.26%只描述该条件网格的最低预测误差，不代表整个解析方法的精度。这里停止继续追单点：不减实测反解纯Graph开销、不挑参数凑观测，也不把这个偏差当作测量失败。

后续转到小型方案的来源核查、预测排序与搜索评价。GPU测量仅补影响候选比较的真实缺项；本次没有新GPU任务或测量数据。
