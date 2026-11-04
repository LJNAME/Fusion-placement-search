# LightOP 11.4 第二批完整回执

完整包 `lightop_vendor_suite_20261104_02_full.tar.gz`：3,820,468 字节，SHA256 `0b248b3fd1c9c6f6405d39212845a1e17b975d56794d8530ee328388a7e23bef`。gzip CRC 通过，2,126 个文件；已取代先前 524,288 字节截断包作为本批审核输入。

| 原版任务 | 结果 | 耗时记录 |
| --- | --- | --- |
| SiLU / GELU / GELU-Tanh × Mul | 各 10 个 Case 打印 Pass | 各 10 行，μs |
| Permute + Quant | 14 个 Case 打印 Pass | 14 行，ms |
| RMS + RoPE | 16 个 Case 检查通过 | 16 行，μs |
| Reshape + Cache | 17 个 Case 打印 PASS | 17 行，μs |
| BMM | 128/128；32 个不同 Shape 各出现 4 次 | Excel 128 行，μs |
| Fill + RMS × 2 + Concat | 10 个 Case，两项差值均为 0 | 无 |
| RMSNorm + RoPE + Contiguous | 1 个 Case，四项差值均为 0 | 无 |
| 剩余 7 项 | 缺依赖退出 | 未完成 |

共 16 项、各尝试 1 次：9 项退出 0，7 项退出 1。7 项有检查和计时证据，2 项只有零误差输出。终端 77 行计时、BMM 表格 128 行，共 205 行。报告、逐项状态、原值和来源 SHA 在 `data/lightop_vendor_20261104_02_review/`。

## 比较边界

- SiLU×Mul 的 10 个 Case 均为 BF16、d=4096，tokens=32…16384，打印加速比 1.57–1.96。基准是 `torch.ops._C.silu_and_mul`，即 vLLM 的另一种融合实现；不能写成对 Native SiLU＋Mul 的加速。原脚本 `test/test_act_and_mul.py:41-70`。
- `test_fuse_groupgemm_permute_quant.py` 实际测的是重排＋INT8 量化，未执行 GroupGEMM；原脚本 `:20-38`、`:49-115`。10 次预热、100 次调用、同步前后用 wall clock 计时，不能与其他任务的 GPU Event / Triton 计时合并。
- BMM 的 `fused_bmm_benchmark.xlsx` 表头写 FP8，入口实际设 `in_dtype=torch.int8`、`out_dtype=torch.bfloat16`（原脚本 `test/test_fused_trans_bmm_trans.py:910-912`）；`q_quant_time=0.0`（`:504-505`），基准为 BF16 Torch BMM（`:507-508`）。这组数据排除了量化成本且比较精度不同，不能据此单独归因融合收益。Shape 列和全部原始计时保留在 `timings.jsonl`，未把 128 条当成 128 个不同 Shape。
- 两个零误差任务只报告差值，保留为 correctness-only，未补造耗时或准入容差。

## 7 项依赖阻断

| 缺失模块 | 原版任务 |
| --- | --- |
| `matplotlib` | `test_fuse_silu_mul_quant`、`test_fuse_silu_mul_quant_ep`、`test_fused_moe_gate` |
| `numa` | `test_fuse_silu_mul_fp8_quant`、`test_gemma_rms_norm_fp8_quant`、`test_rms_norm_dynamic_per_token_quant`、`test_rms_norm_fp8_quant` |

前三项先打印精度检查通过，随后在 Triton 性能测试导入 `matplotlib.pyplot` 时退出；不能把部分通过当成整项完成。

后四项在 `RMSNorm` 构造时，经 `CustomOp.dispatch_forward → get_cached_compilation_config → VllmConfig → ParallelConfig → ExecutorBase → WorkerBase` 引入 `numa`。固定 vendor vLLM 的 `vllm/worker/worker_base.py:5` 无条件导入该模块，`:35-46` 使用 `get_max_node` / `bind`，`:286` 使用 `get_membind`。这些任务尚未完成数值验证；文件名含 `silu_mul_fp8` 的脚本实际调用 RMSNorm FP8，按真实 API 归类。

本批日志未出现数值失败。此前批次的数值失败记录不被这份结论覆盖。

## 环境与源码

- 完整预检退出 0、无超时，168.945 s；实际平台 `vllm.platforms.rocm.RocmPlatform`。此前双内置平台冲突在本批未复现。
- 安装前后 81 个包快照一致；Torch、Triton、LightOP、NumPy、vLLM、lmslim 未替换。
- 临时 SMI `.pth` 和平台注册目录均已清理。
- 原始测试树逐文件 SHA 核对 2,057 份（121 × 17）；原版测试未修改。
- 能力文件按批次追加 `vendor_20261104_02` 证据，保留 `vendor_suite01`。静态 48 接口目录不改写，未增加 PlacedOpGraph Adapter。

新增 accepted Profile 为 0，完整 FFN 可定价方案仍为 0/48。厂商脚本各自的耗时边界尚未转换成项目统一的 Canonical Key / Measurement Contract。

## 续测入口条件

`remaining_jobs.json` 固定 7 个依赖失败任务及各自源码 SHA，9 个完成任务保留，不重跑整批。

先补齐两项依赖，再做实际构造预检：除了导入 `matplotlib.pyplot` 和 `numa`，还要构造原 vLLM 的 `RMSNorm`（含 Gemma 对照脚本实际使用的同一类），不能只检查类是否存在。当前预检只检查 callable，漏掉了构造时的配置依赖。

`numa 1.4.6` 官方发布物只有源码包，CPython 安装会编译并链接 `libnuma`；SCNet 的头文件、编译环境及后续构造依赖尚未核验。暂未交付自动安装或新 GPU 运行命令，不能直接安装同名替代包或补整个 vLLM 推理环境。官方来源：[PyPI](https://pypi.org/project/numa/)、[构建脚本](https://github.com/smira/py-numa/blob/master/setup.py)。

本次仅 CPU 审核与证据解析，未重新运行 GPU。
