# LightOP 原版批测首批审核

回执：`lightop_vendor_suite_20261103_01.tar.gz`，SHA256 `0a0a68385eedc688b833eb70d0baa53c28e50f477559ee2440f9e2f312caa387`。解包位于工作区 `transfer/lightop-vendor-suite-received-20261103-01/`，原始回执保留。

## 结果

48 个任务全部尝试一次，30 个进程退出 0，18 个非零退出。审核结果按任务统计：

| 状态 | 任务数 |
| --- | ---: |
| 有时延且原版数值检查通过 | 22 |
| 有时延且量化/反量化打印的 8 项差异为零 | 1 |
| 只完成原版正确性检查 | 4 |
| Gemma Add+RMSNorm 结果依赖容差，未测时延 | 1 |
| 缺依赖，尚未进入测试 | 16 |
| 包内测试引用不存在的接口 | 1 |
| 数值检查失败，含两项退出 0 的脚本 | 3 |

共有 25 个任务打印时延，其中 2 个 Softmax Dropout 任务存在数值检查失败。时延保留为 253 条原文记录，单位和输入上下文分别保存；不跨脚本合并，不以进程退出 0 代替数值检查。

## 未通过项

依赖缺失：vLLM 12 项、tqdm 3 项、pytest 1 项。这是当前最先遇到的导入错误，补齐之后仍可能出现其他依赖或运行问题。优化版 SiLU/GELU/GELU-tanh 三项均被 vLLM 基准导入阻断；此前 trial02 的 SiLU 证据仍保留。

`test_fuse_rmsnorm_rope_qkv_no_quant` 导入的 `rms_rotary_embedding_fuse_with_kv_store_no_quant` 不在固定包的公开导出中。其余 RoPE 接口不能因此标成失败。

`test_rms_mrope` 在 FP16、interleaved=True、residual=False 的默认 Case 中触发容差 0.02 的异常，Q/K 最大误差分别为 175.0 / 173.625。执行路径、参考实现和输入契约还未定位，暂不使用该 Case 的成本。

两个 Softmax Dropout 脚本分别打印 150 次、6 次 False，却均退出 0。它们的 FP16 检查仍采用 rtol/atol=1e-5；随机数对齐、容差或实现是否为原因，尚未核实。保留原版结果，不事后修改容差使其通过。

Gemma Add+RMSNorm 的 1e-2 检查均为 True，1e-3/1e-4 对输出的检查共有 4 次 False。它没有输出时延，记录为容差相关的数值证据。

## 时延口径

Dropout、Bias 和部分 Norm 的原版计时包含前向、反向或其他 Python 操作。Root `fuse_silu_and_mul` 默认仅计时 10 次且没有预热，记录为 0.397658348 ms/次；不能拿它与此前优化绑定的稳态 Event 结果直接比较。

RoPE+Cache 的表格保留 mean/std（ms）；MoE SwiGLU 保留各 hidden/tokens 的原版 us 记录。两个 Softmax Dropout 的 `===fuse`/`===src` 是计时循环的秒数，代码在第 10 次迭代重置计时，覆盖剩余 990 次，未当作单次 ms。

CeluGN/GNSilu 原版源码第 120 行打印的 `speedup` 实际为 `T_torch/T_lightop - 1`。原文照存，不能将小于 1 的打印值解释为 LightOP 更慢。

## 已写入项目

- `data/lightop_vendor_suite01_review/report.json`：逐任务结果、错误原因及输入证据 SHA。
- `jobs.csv`：48 项审核表。
- `timings.jsonl`：253 条时延原文及单位、行号、上下文。
- `capability_status.json`：与实现 ID 绑定的本批观察。
- `integration.lightop.capabilities()`：返回原能力和 `vendor_suite01` 观察；原能力文件保持冻结，仍可复现原批测计划。

固定测试树的 121 文件 × 49 份副本共 5929 项 SHA 校验通过。计划、固定 wheel 和 op.so 身份匹配；运行器自身源码未随回执捕获，不能据此认证其运行时源码 SHA。Trace 文件已保留，本次未做逐 Kernel 解析或发射数量认证。

本批不自动增加 accepted Profile，不开放图重写或完整 FFN 定价。下一步优先处理导入依赖和包内 API 不一致，只补测受阻任务；数值失败项单独定位。保留当前已测数据，不重跑整批。

## CPU 重放

```bash
python tools/review_lightop_vendor_suite.py /path/to/lightop_vendor_suite_20261103_01 \
  --archive /path/to/lightop_vendor_suite_20261103_01.tar.gz \
  --output /fresh/review
python -m unittest calibration.tests.test_lightop_vendor_review calibration.tests.test_lightop_vendor_suite -v
```
