# 前向执行：融合＋放置联合 MCTS 是否找到更好的方案 v1

2026-11-09，CPU。目标为前向 makespan。第一层：解析模型＋LightOP Fusion Library（full，silu_mul 校准 c=2.417）。第二层：Proteus 式执行器 `cost_model/htae.py`。硬约束：不可行方案不进 Q 与结果。所有 MCTS 都以不融合方案为初始最优候选。

复现：

```bash
python tools/evaluate_forward_joint_search.py --scope medium --variant fixed --caps free,cap75,cap60 --budgets 16,64,256 --seeds 0,7,23 --output <dir>
python tools/evaluate_forward_joint_search.py --scope layer  --variant fixed --caps free,cap75 --budgets 64,256 --seeds 0,7,23 --output <dir>
python tools/evaluate_forward_joint_search.py --scope model_3b --variant fixed --caps free,cap75 --modes joint,sequential,placement_then_fusion,fusion_only --budgets 64 --seeds 0,7 --output <dir>
python tools/evaluate_forward_joint_search.py --scope all --references-only --caps free,cap75 --output <dir>
```

`--variant raw` 为原 MCTS 选择规则；`fixed` 加 `normalize_q` 与 `skip_exhausted`（见下）。

容量：free＝设备显存；capXX＝每卡 XX% × 不融合单卡前向峰值，强制切分。执行器：blocking＝发送阻塞发送方计算（硬件快照）；async＝拷贝与计算重叠、γ=0（BW1000 未实测）。随机基线：同一联合空间均匀合法补全，评估的不同方案数与 joint MCTS 相同。

## 文件

medium_fixed / medium_raw / layer_fixed / layer_raw / m3b_fixed.json：逐次运行。references.json：不融合单卡与"最佳单刀连续切分"参考（非搜索）。

## 结果（ms，3 种子中位数；3B 为 2 种子）

| 任务 | 不融合单卡 | 联合 Exact | 连续切分＋融合 | joint MCTS | sequential | 随机 |
|---|---:|---:|---:|---:|---:|---:|
| serial_tail free | 0.1030 | 0.0944 | 0.0944 | 0.0944 | 0.0944 | 0.0944 |
| serial_tail blocking cap75 | — | 0.1551 | 0.2128 | 0.2127 | 0.2127 | 0.1755 |
| mlp async cap75 | — | 0.1900 | 0.1900 | 0.1900 | 0.1900 | 0.1959 |
| attention_fork async cap75 | — | 0.1556 | 不可行 | 0.1556 | 0.1963 | 0.1650 |
| layer free（256 次） | 0.3169 | — | 0.3088 | 0.3088 | 0.3088 | 0.3169 |
| layer blocking cap75（256 次） | — | — | **0.3747** | 0.5424 | 0.4447 | 0.5644 |
| layer async cap75（256 次） | — | — | **0.3665** | 0.5413 | 0.4822 | 0.4976 |
| 3B free（64 次） | 29.377 | — | **29.070** | 29.377 | 29.364 | 29.377 |
| 3B blocking cap75（64 次） | — | — | **29.119** | 47.246 | 47.052 | 47.244 |
| 3B async cap75（64 次） | — | — | **28.997** | 44.246 | 44.498 | 44.656 |

（中等子图 14 组可行任务的 Exact：joint 与 sequential / placement_then_fusion 最优值全部相等。）

## 结论

1. 小图：联合空间不含比分阶段空间更好的方案（中等子图 14 组可行任务的 Exact 全部相等）。修正后 MCTS 在 256 次内多数达到 Exact，但不稳定优于随机（serial_tail blocking cap75 随机更好）。
2. 层与 3B：不受容量限制时，收益只来自单卡上融合 add_rmsnorm（3B 理论 −1.05%）。joint MCTS 64 次在 3B 上未找到任何融合（0%），staged 只拿到约 4%。
3. 受容量限制时，所有搜索都远差于一个简单参考：一刀连续切分＋融合 add_rmsnorm。3B 上搜索 44–47 ms，参考 29.0–29.1 ms（差 53–62%）；层图 sequential 差 19–32%，joint 差 45–48%。原因：动作逐节点选设备，补全产生大量跨卡边，每条 ≥55 μs。
4. 原 MCTS 选择有两处缺陷（本次加为可选修正，默认关闭）：
   - 原始 makespan（~1e-4 s）远小于探索项（~1），UCT 实际按访问次数选择；
   - 受容量限制时，最佳已评估终点被反复选中，迭代不再产生新方案（256 次迭代只评估 33 个方案）。
   `normalize_q` 把 Q 按已见可行范围归一化；`skip_exhausted` 跳过已全部评估的子树、全空间枚举完即停止。修正后评估数随迭代增长，中等子图多数达到 Exact。
5. 执行器：async 只在受限切分时有作用（3B cap75 −6%，参考切分 −0.4%）；该假设未在 BW1000 实测。

## 限制

前向、batch=1、两卡；第一层误差见 data/fusion_benefit_validation_v1（seq256 收益高估约 22–26%，布局转换未计入）。3B 只跑 64 次、2 种子。连续切分只是一个可达方案，不是最优；真实最优只会相等或更好。
