# 输出图与逻辑切分契约验收

2026-11-01，使用现有18个主线PlacedOpGraph工件，8个含融合组。不生成TP/PP配置、不接管下游执行，不新增搜索、GPU或Profile标签。

## 工件验收

| 项目 | 结果 |
|---|---:|
| 原节点ID、边、张量属性与分组/设备标注往返一致 | 18/18 |
| 原工件未被消费者修改 | 18/18 |
| 节点ID/依赖/唯一归属/设备/backend篡改被拒绝 | 90/90 |
| 当前registry可执行性检查拒绝planned融合 | 8/8 |
| 所有工件仍标deployable=false | 18/18 |
| 正式融合能力、下游schema已确认 | 否 |

严格能力检查只针对当前registry的reference/implemented/planned状态。10个singleton工件通过该局部检查，不代表正式部署准入；正式能力文件和执行契约尚未提供。当前36条Kernel标签仍为legacy_unbound，不在本轮升格。

## A/B与负例C

A：在现有plan的组拓扑序上，用解析组duration之和选两个连续逻辑stage的平衡切点，保留组原子性。它是本地诊断提案，不是下游自动并行算法。

B：真正按layer元数据均分；不会把“按层均分”偷偷换成“按算子数量均分”。

C：按原图确定性拓扑序等数量切算子，仅作接口负例，不能称下游实际行为。另枚举所有单切点以覆盖组原子性拒绝分支。

| 提案 | 适用数 | 合法/拒绝 | 负载与字节口径 |
|---|---:|---|---|
| A1、B1 | 各18 | 各18合法，映射相同 | 单stage不均衡0、跨stage字节0 |
| A2 | 18 | 18个开发提案合法 | 解析组工作量之和，非运行时stage延迟 |
| B2 | 0 | 18个not_applicable | 全部工件仅一层，不能切成两个非空层stage |
| C2算子均分 | 18 | 11合法、7拒绝 | 非法提案不计算负载/收益 |
| 全部算子单切点 | 149 | 16个切组负例全部拒绝 | 只证明本地检查器能发现违约 |

当前融合规则要求组内同层（fusion_rules.py:46、106）。所以按完整层边界切分时，这个约束已经由粒度自动满足。现有真实按层路径没有观察到切组；不能把C的阳性构造偷换成B的必要性或性能收益。

## 约束范围

integration/partition_contract.py接受一个独立的node_id→logical_stage提案，检查原节点全集、组唯一stage、依赖不能逆stage、原设备标签不能静默改变。默认假设消费者保留所选融合实现，不允许静默解融合；正式下游是否遵守/重写该契约尚未确认。

stage_metrics只接受合法提案。stage_group_work_s是固定解析Profile的组duration求和；不均衡为max/mean−1；interstage_dependency_bytes按组DAG外部边逐边计逻辑字节，不代表实测链路流量；original_cross_device_bytes则沿用原放置，独立于逻辑stage标签。

没有把stage映射到rank/GPU mesh；两个逻辑stage可能保留同一物理设备，不能把A2当可部署PP配置。原容量/显存标注仍由工件加载校验，不重算下游并行执行峰值。非法C只报告would_cut_internal_bytes，不把它计作已实现通信节省。

## 当前结论

已验收本地输出保存/加载与违约拒绝能力。尚未证明真实按层下游需要这条额外约束，也没有A/B运行性能优势。约束是接口语义的一部分，不因此自动成为科研创新。

在当前按层粒度下，组不跨层已经保证边界安全；这不等于任意算子级消费者都无需组信息。后续价值验证需要真实下游粒度/能力契约，不能仅改用一个会犯错的消费者来证明必要性。

## 产物与复验

- plan.json：18份原工件与源码hash、A/B/C定义、未确认契约状态。
- comparison.csv / artifacts.jsonl：逐工件提案、指标、拒绝原因、149切点覆盖和90项篡改。
- roundtrip/：18个可重放的非部署工件；原文件未修改。
- report.json：汇总；verification.json：复算汇总和双次输出一致性。
- tests.xml：37项相关CPU测试通过，含6项新增分区契约测试。

```bash
python tools/audit_output_partition_contract.py data/output_partition_contract_replay
```

输出目录须不存在。没有新测量、标签升格、MCTS预算或SROP验证；本轮关于14/15MiB的补充只修正证书适用范围，旧CSV和证书数值未动。
