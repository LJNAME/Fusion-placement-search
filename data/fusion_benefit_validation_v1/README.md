# 融合/不融合相对收益验证 v1

2026-11-09。只用已有同卡实测配对，CPU 计算，不采新 GPU。问题：代价模型预测的 r = 融合耗时 / 不融合耗时 是否与实测一致。

复现：`python -m calibration.fusion_benefit_validation --output <新目录>`

每个配对建局部夹具图（融合组 vs 其各算子单独执行，同卡、同外部输入输出，add→rmsnorm 保留双输出）。解析模型按契约取 launch：graph replay 用 1.5 μs（configs/ffn_profile_search.json），eager 用硬件快照 7.77 μs。偏好带 1%。

## 结果

| 配对 | 契约 | 实测 r | 预测 r | 偏好 |
|---|---|---:|---:|---|
| KERV silu_mul GPU0 seq256 | graph | 0.728 | 0.566 | 一致 |
| KERV silu_mul GPU0 seq1 | graph | 0.496 | 0.501 | 一致 |
| KERV silu_mul GPU1 seq1 | graph | 0.515 | 0.501 | 一致 |
| KERV silu_mul GPU1 seq256 | graph | 0.762 | 0.566 | 一致 |
| KERV add_rmsnorm GPU0 seq256（非同批） | graph | 0.524 | 0.670 | 一致 |
| LightOP silu_mul split+pack seq256 | eager | 1.275 | 0.527 | **相反** |

## 结论

- 偏好 5/6 一致，但不足以证明模型可靠：未校准解析模型对任何融合都预测更快（少一次 launch、少中间流量），"总是融合"同样 5/6。唯一的融合变慢样本被判反。
- 幅度：launch 主导的 seq1 误差 ≤3%；seq256 silu_mul 高估收益约 22–26%；add_rmsnorm 低估收益约 28%。
- 误差来源（绝对值拆分）：
  1. seq256 silu_mul 实测节省约 3.0 μs，约等于一个 kernel 的固定成本（seq1 推得约 3.5 μs/kernel）；模型另计约 2.3 μs 中间张量流量节省，实测未体现（中间结果可能留在缓存，未验证）。
  2. graph 契约 launch 1.5 μs 未标定；seq1 实测约 3.5 μs/kernel。两项误差在 seq256 部分抵消。
  3. split+pack 的融合路径多一次打包拷贝，模型不计 layout 转换成本，因此判反。
- 跨形状标量校准（用另一形状拟合系数）在 KERV silu_mul 上平均误差 27%，比未校准更差：误差随形状改变，单一系数无法修正。

## 范围限制

6 个配对、2 种融合、2 种实现、2 种契约；add_rmsnorm 只有 1 个形状且非同批；LightOP add_rmsnorm 没有实测。LightOP silu_mul 那一对就是现有校准来源，不能用于该校准的样本外检验。
