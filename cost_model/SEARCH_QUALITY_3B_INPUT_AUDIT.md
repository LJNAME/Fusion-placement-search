# Qwen2.5-3B Fixed-64 输入审计

2026-11-07。采用项目已有qwen25_3b_spec()：36层、hidden=2048、FFN=11008、16个Q头/2个KV头、seq256、batch1、BF16。完整逻辑图687节点、866边，非运行时捕获图。沿用0.6B的两卡BW1000配置、13条条件成本记录、LightOP executable_registry与成本查询路径。

| 检查项 | Qwen3-0.6B | Qwen2.5-3B |
| --- | ---: | ---: |
| 节点/边 | 591/730 | 687/866 |
| 结构SiLU→Mul位置 | 28 | 36 |
| 结构同层Add→RMSNorm位置 | 28 | 36 |
| 可执行多节点候选 | 56 | 0 |
| GPU0精确实测命中 | 253 | 0 |
| GPU1精确实测命中 | 140 | 0 |
| GPU0解析回退 | 310 | 651 |
| GPU1解析回退 | 423 | 651 |
| Nearest Shape（每卡） | 0 | 0 |
| 逻辑输入（每卡） | 28 | 36 |
| 不可用数值成本（每卡） | 0 | 0 |

成本统计按未融合节点逐卡查询，不是独立Profile条数或方案覆盖率。Nearest在当前查询链中未启用；未导入历史eager记录作为Graph标签。两个模型的物理布局都没有运行时捕获证据。0.6B仍有当前未测成本；3B的651项计算全部使用解析代理，不能把这一变化当作仅图规模变化的严格对照。

## 融合拒绝原因

- LightOP silu_mul Adapter只接受BF16 [1,256,3072]；3B实际为[1,256,11008]，36处均拒绝。
- LightOP add_rmsnorm Adapter只接受BF16 [1,256,1024]；3B实际为[1,256,2048]，36处均拒绝。
- KERV silu_mul的单Kernel宽度上限8192；11008超限，36处均拒绝。布局未知仍保持运行时连续性要求，没有擅自填为contiguous。

P0完成；P1/P2未启动。当前配置没有融合决策，三种模式若直接运行会退化为放置搜索，不能检验联合融合—放置优势。没有运行576次评分、扩展搜索树记录或新增GPU测量。树指标记null，不使用0冒充已观测结果。

后续若验证联合搜索，先核对真实LightOP实现能否接入至少一个3B Shape；不放宽现有规则、改FFN宽度或把planned规则当作可执行。如果只研究规模与运行耗时，可以另标为placement-only对照。

明细data/search_quality_qwen25_3b_audit_v1/report.json，含逐节点来源、72处拒绝原因与KERV guard。复现：

```bash
python -B -X utf8 tools/audit_search_quality_input.py --output /fresh/3b_input_audit
```
