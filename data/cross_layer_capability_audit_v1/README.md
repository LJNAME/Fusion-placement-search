# 默认候选跨层范围核对

2026-11-01，用户确认正式融合范围仅允许层内。具体能力文件仍未提供，执行/测量契约仍未确认；本轮不注册新pattern、不升级部署或标签资格。

## 591节点完整参考图

| 项目 | 数量 |
|---|---:|
| Decoder层 | 28 |
| 节点 / 边 | 591 / 730 |
| 默认开发候选 | 619 |
| 多节点融合候选 | 28 |
| 跨层候选 | 0 |
| registry可执行候选 | 591（全部单节点） |

默认注册4类：silu_mul、elementwise_chain、rmsnorm_reference、unfused。最后一个是单节点fallback。实际候选为unfused478、rmsnorm_reference113、silu_mul28；elementwise_chain在本图没有另列出的候选。所有多节点pattern继承require_same_layer=true（fusion_rules.py:46、106、198–225）。

这不是单层工件不足造成的结果。当前完整图已包含28层；在正确完整layer元数据下，默认规则禁止跨层候选，按完整层边界切分不会切破融合组。用户确认的正式范围与此一致，跨层A/B路线关闭，不再为此扩图。

没有跨层融合只保证整层切分不违反组原子性，不能推出不同切分算法的边界、负载或通信指标全部相同；那些差异可能来自负载策略或不同合法空间，不是该原子性约束的收益。

## 源码存在与能力准入

Pinned FlagScale为f8e9a62c499d526a6ffa47f533bcf342330b7326；kerv_ops.py与adaptive_linear_fusion.py两份hash均匹配已有源锁定清单。kerv_add_rms_norm存在（kerv_ops.py:362），默认registry没有注册。

该入口原地覆盖hidden，返回归一化张量；包装器另保留combined_residual（adaptive_linear_fusion.py:506–520）。源码存在不证明跨Transformer层融合已允许，也不替代布局、别名与多输出能力契约。正式范围既已确认为仅层内，本轮不再追问跨层守卫或接入跨层实现。

## 当前主张

不把“融合组额外限制层粒度PP切分”作为已成立研究贡献。输出图仍携带融合实现/组归属/设备等可执行性信息；它是工程契约，不由CPU负例自动变成科研创新。

18工件的往返、篡改拒绝和本地消费者检查结果保留。算子级负例只证明该检查器能拒绝违约，不能用来声称真实按层下游需要额外约束。36条旧Kernel标签仍legacy_unbound，未因层内范围确认自动accepted。

## 复算

```bash
python data/cross_layer_capability_audit_v1/reproduce.py <new-output-dir>
```

report.json保存代码和pinned源码hash、用户确认范围及未确认项；candidates.csv列出619项node_ids/layers/能力状态。仅枚举候选，不运行MCTS、Exact完整方案搜索、GPU或新Profile。
