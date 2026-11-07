# Qwen2.5-3B 静态契约联合搜索

2026-11-07。复用0.6B的静态执行声明与跨层条件匹配，不要求执行完整模型。

## 准入

- 687节点、866边，37条直接Add→RMSNorm连接。
- 36处层内候选符合KERV静态规则；final_norm一处在当前Adapter范围外。
- GPU0复用已有Native/KERV两条accepted记录；GPU1无对应成本对，保持Native。
- 原图Shape、dtype、层号、输入角色、权重大小、消费者和双输出依赖用于静态检查。
- 连续布局、独立激活分配、推理模式及无live alias是明确的目标执行契约，由原Adapter调用前检查；不是模型Tensor观测。未知布局没有写回原图。
- 与0.6B的LayerInvariantMeasuredProfile一致，依据执行声明查询条件；完整Key比较仅排除layout_provenance证据来源，保留Shape/stride/dtype/参数、实现和源码版本、设备、软件构建及测量契约。保存请求Key、原测量Key与条件ID，不伪称原始Key逐字相同。
- 已知Shape/stride/dtype/eps/别名冲突、非fresh-matmul生产者、静态元数据缺失或成本身份不符的位置不进入融合评分。

Native整链仅为同卡连续执行单元计价，不拆成单算子标签；两输出在链结束后发布。其余算子与通信保留解析来源。既有Graph解析1.5μs逻辑阶段开销仍为未标定代理，没有修改公式或重新拟合。

## 一次联合搜索

```bash
python -X utf8 tools/run_qwen25_static_search.py --output /fresh/qwen25_static_joint
```

只运行joint、seed0，评分预算上限64，无顺序搜索、Exact、新种子或GPU实验。

| 项目 | 结果 |
| --- | ---: |
| GPU0静态契约成本复用位置 | 36 |
| MCTS选择融合组 | 36 |
| 最终节点放置 | 全部GPU0 |
| 评分请求 / 实际评分 / 提前拒绝 | 64 / 32 / 32 |
| 不同请求方案 | 50 |
| Native全GPU0参考预测 | 25.414395447 ms |
| 最终方案预测 | 25.239290006 ms |
| 首次最终最佳评分 | 1 |
| 初始补全后改进 | 0 |

预测差值是36组局部实测成本差的聚合；579组解析、36组逻辑输入、36组融合实测条件复用。不是整图GPU耗时，也没有证明联合搜索优势。

## 交付

data/qwen25_static_joint_v1/：

- capability_cost_audit.json：逐位置、逐设备的规则与成本条件检查。
- execution_contract.json：目标执行契约，model_runtime_observed=false。
- search_trace.json、report.json：一次搜索、成本来源和一致性结果。
- joint_placed.json：原687节点及融合/设备标注。
- joint_fused_placed.json：651个组、36个融合组及依赖，供下游消费。

图fingerprint保持不变。逐个融合组的实现、测量Key/Contract和成本查询一致；导出往返后评分相同，双输出与所有原节点均保留。组图deployable字段沿用现有“规则可执行且显存不过限”的含义，未宣称下游运行时或完整模型执行通过。

46项CPU测试通过，覆盖已知不合法位置拒绝、GPU1不借GPU0、完整条件身份、Native链与导出依赖。没有新增测量或Profile，也没有开发完整执行器。原model_runtime证据路径保留；它不再是静态图搜索的全局前置条件。
## 同构设备共享成本（2026-11-07）

当前查询已按硬件类别匹配，GPU1不再因物理编号不同而被排除。已有记录证明两卡BW1000/gfx936及软件构建一致，36处×2设备共72个成本对可查；GPU1标记为GPU0测量的硬件类别迁移，不改成GPU1实测。一次重跑64实际评分、0设备成本拒绝，GPU1候选成本查询378次；最终仍36融合全GPU0、预测25.239290006ms。详见[PROFILE_HARDWARE_CLASS.md](PROFILE_HARDWARE_CLASS.md)，前面的GPU0专项结果保留为历史。