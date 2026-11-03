# LightOP 融合实现与原版批测

项目直接采用海光 LightOP 的已有实现。当前固定 wheel 的公开 Python 导出共 132 项，其中登记 45 个显式融合接口，另有包内测试明确调用的 3 个优化激活乘法接口，共 48 个接口及变体。这个数量不表示 48 种独立数学 Pattern，也不承诺每个接口只发射一个 Kernel。

清单：`data/lightop_fusion_catalog_v1/capabilities.json`。各项 `implementation_status=implemented`、`implementation_owner=Hygon LightOP`，记录绑定名称、源码位置和 SHA、原版测试引用。`public_exports` 保留全部公开导出；二进制扩展内部的全部接口尚未盘点。

项目可调用已有实现，不用重写 Kernel：

```python
from integration.lightop import resolve
op = resolve("op.silu_and_mul_opt")
op(out, packed_input)
```

调用时检查固定版本和源码，再返回真实函数或类。`fuse_silu_and_mul(input, output)` 与 `op.silu_and_mul_opt(output, input)` 分别登记，不互相替代。PlacedOpGraph 的匹配、输入映射和别名处理仍由对应 Adapter 承担；当前这些 LightOP 接口的 `placed_graph_adapter=false`。

## 在 SCNet 直接运行原版测试

```bash
cd ~/private_data/LJ/Fusion-placement-search
git pull --ff-only
export FLAGSCALE_ROOT="$HOME/private_data/LJ/FlagScale-main"
export FUSION_PYTHON="$HOME/private_data/LJ/dcu-kernel310-20261031/run_python.sh"
bash tools/run_lightop_vendor_suite.sh \
  "$HOME/private_data/LJ/lightop_vendor_suite_20261103_01" run
```

默认读取此前安装时缓存的固定 wheel：`~/private_data/LJ/lightop_wheels/<manifest.filename>`。位置不同可在命令末尾加 `--wheel /absolute/path/to/package.whl`。不重复安装 LightOP，不更新 Torch、Triton 或 vLLM。

48 个任务来自 46 个原版脚本；激活乘法脚本分别传入原版支持的 `--act silu/gelu/gelu_tanh`，BMM 脚本使用原版 `benchmark` 模式，其余按默认入口运行。Shape、dtype、预热、重复次数、同步、计时及正确性方法均保留。脚本包含的反向计算和 profiler 也保留，不把它们改写成前向耗时。整批运行长度取决于原版扫描范围，部分 Norm/量化测试的 Shape 集合很大。

每个脚本独立进程、独立工作目录、只尝试一次。失败后保留日志并继续下一项，整批有失败时最终退出码为 1。脚本缺依赖会照常记录，不自动安装依赖。默认不设超时；可显式加 `--timeout-seconds 1800`。只运行指定脚本可加 `--jobs test_fuse_silu_mul test_act_and_mul_silu`。

## 产物

- `plan.json`：原版脚本、参数、清单和矩阵 SHA。
- `identity.json`：实际安装版本、原生库 SHA、Python、Torch/Triton/vLLM 等版本及可见设备环境变量。SCNet 重启造成的跨批设备 ID 变化不阻断执行。
- `vendor_original/test/`：校验 SHA 后解出的原始测试树。
- `jobs/<job_id>/stdout.log`、`stderr.log`、`result.json`：原始输出、错误、时延行和正确性行。
- `jobs/<job_id>/vendor/test/`：每项原版运行目录，保留其生成的 CSV、Excel、trace 等文件。
- `results.json`、`summary.json`：逐项结果与完成情况。

`process_completed` 只表示进程退出 0。原版有些检查只打印 False/Fail 而不抛异常，所以不据此自动标正确或准入。原版脚本的结果可直接读日志和其表格，后续审核不需要重跑 GPU。不同脚本的测量边界各自保留，不混成一个统一时延。

## 当前缺口

44/48 个接口找到了包内测试中的静态调用引用，不代表所有分支和 Shape 已测。以下 4 个已实现接口尚未找到配套测试调用：

- `fast_topk_transform_fused`
- `fast_topk_transform_ragged_fused`
- `fuse_silu_mul_fp8_quant`
- `fuse_silu_mul_fp8_quant_ep`

包内 `test_fuse_silu_mul_fp8_quant.py` 实际调用 RMSNorm FP8 量化，不能按文件名算成 SiLU FP8 的测试。

此前 trial02 的真机证据仅属于 `op.silu_and_mul_opt`，记录在该接口的 `evidence` 字段。首批原版测试已返回，审核见 `LIGHTOP_VENDOR_SUITE01_REVIEW.md`。测量状态由 `data/lightop_vendor_suite01_review/capability_status.json` 追加记录，`capabilities()` 返回 `vendor_suite01` 观察；固定能力文件作为首版身份快照保持原 SHA，不增加正式 Profile 或搜索成本覆盖。

## 重建清单（CPU）

```bash
python tools/build_lightop_catalog.py /absolute/path/to/pinned.whl
python -m unittest calibration.tests.test_lightop_vendor_suite -v
```

生成过程只读取 wheel，不导入厂商二进制。固定版本、下载地址和 SHA 见 `data/hygon_lightop_wheel_v1/manifest.json`。
