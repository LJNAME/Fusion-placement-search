# B4 run05 · v3回传审核

包SHA256：a9f304b76995304cb0f3d0b4215d121a0e8d2bacf60339ee19eb58a53b3c348f。归档transfer/ffn-b4-received-20261102-05；7/7源码/证据哈希匹配e888d72。机器可读审核：data/ffn_b4_run05_review/report.json。

## 执行与质量

| 项目 | 结果 |
|---|---:|
| 最小Smoke | 4/4通过 |
| 轮组进程正常退出 | 10/10 |
| 核心测量完成 | 18格，各7轮×50调用 |
| 原始正式样本 | 126轮，6300次函数体调用 |
| quality_passed | 9格 |
| 锚点漂移失败 | 9格 |
| 已知组合链阻断 | 1格，0调用 |
| 补轮/重启/运行时崩溃 | 0 / 0 / 0 |
| 新accepted / 搜索成本项 | 0 / 0 |

18格的identity、contract哈希、原始样本与汇总副本、ms→μs/call转换、Event始末设备、数值/不变量检查标记、后置路径证明及质量判据已逐项CPU核对。质量结果18/18与原记录一致；这是源代码与已记录证据复核，没有在本机重新执行GPU数值计算。

原始样本仍为pending_post_timing_trace；result单独保留after_all_group_timing及path_verified。18格预热/正式Event中位比范围0.977700～1.018078，没有run04的20～50倍突变。v3时序修复在本轮得到真机验证，不推广为所有Shape/设备的计时认证。

## 九项质量通过记录

单位均为Graph摊销μs/调用或μs/传输，不能视作普通Eager单次Launch耗时。当前仅为CPU导出/准入候选。

| 测量项 | GPU/方向 | 中位数 |
|---|---|---:|
| Gate GEMM | GPU0 | 16.620780 |
| Up GEMM | GPU0 | 16.601599 |
| Up GEMM | GPU1 | 16.550400 |
| SiLU | GPU1 | 7.049600 |
| Mul | GPU1 | 4.336000 |
| Native SiLU→Mul | GPU1 | 10.339220 |
| KERV Fused | GPU1 | 7.360000 |
| 0.5MiB Copy | GPU1→GPU0 | 33.782380 |
| 1.5MiB Copy | GPU1→GPU0 | 44.585581 |

GPU1 Native/Fused的pair_key及实际measurement_contract一致，耗时下降28.814743%。同轮Δ_add=SiLU+Mul−Native链的7个值均为正，中位+1.033599μs；不能将单算子简单加和视为Native链实测，也不据此拟合全局折扣或宣称B5完成。

## 九项锚点失败

| 轮组 | 受影响格数 | 作废轮（从0计） | 超阈值示例 |
|---|---:|---|---|
| activation_gpu0 | 4 | 2 | GPU0 6.713580→7.459160μs，11.1056% |
| link_quartet三个组件 | 3 | 0、1、2 | 两卡分别出现11.0293%～14.7030%漂移 |
| down_proj@gpu1 | 1 | 0 | GPU1 7.526400→6.588800μs，12.4575% |
| 1.5MiB Copy 0→1 | 1 | 2 | GPU0 12.4518%，GPU1 12.1916% |

每格全部7次尝试保留，按预注册任一锚点漂移>10%则该轮无效，固定7次不补轮，整格不能准入。失败不是数值或后置Trace错误；不放宽阈值、不挑选剩余轮、不拿旧批次或另一张卡替代。

## 当前边界

9项quality_passed尚未导出正式Canonical Kernel Key，未进入严格查询或Admission。已有运行时身份、布局和路径记录只作为导出依据；Copy的物理路由仍不能仅凭peer能力/DtoD Trace称为已认证P2P。Native链是组合比较项，不替代任意独立放置下的SiLU/Mul成本。

即使暂时把9项quality_passed当作可用标签，B1的48个合法FFN方案中也没有一个成本项齐全：两卡Down Projection都失败，部分正向通信也缺质量通过记录。report保存逐方案缺项与B1原文件哈希；该清点没有运行评分或MCTS，也不是正式准入覆盖率。

下一项仅CPU：为9项候选建立精确Key/contract导出与适用范围清单，再进行准入政策和负向测试审核。不会因为quality_passed自动改accepted。暂不新增GPU测量；组合链残差、完整FFN成本及B5.5/B6仍缺前置条件。

重放：

```powershell
python -m tools.review_ffn_b4_run /path/to/ffn_b4_run_20261102_05 --source-ref e888d72 --output /fresh/review.json
python -m tools.audit_search_subgraph_cost
```

新增4项CPU审计测试通过，覆盖单位换算、contract篡改和Event设备不一致的拒绝。未改v3运行时、原始测量、准入状态或搜索算法。
