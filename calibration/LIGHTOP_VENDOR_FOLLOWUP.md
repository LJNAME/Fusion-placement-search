# LightOP 原版测试：依赖修复与定向补测

> 2026-11-04：第三批安装的 amdsmi 与 HyHAL 符号不匹配，已阻断 Torch 导入。先按 [LIGHTOP_VENDOR_SUITE03_REVIEW.md](LIGHTOP_VENDOR_SUITE03_REVIEW.md) 恢复；下文保留原安装计划，当前入口已禁用该 SMI 安装项，暂不重启 16 项补测。

本次选择首批因缺依赖退出的 16 个任务。已完成的测量、3 个数值失败和缺失的 no_quant API 均保留原记录，不重跑整批。固定清单见 `data/lightop_bench_followup_v1/manifest.json`。

## 固定的基准依赖

| 包 | 选定版本 |
| --- | --- |
| vLLM | `0.11.0+das.opt1.dtk2604.torch290` |
| lmslim | `0.3.1+das.opt4.dtk2604.torch290.2604281437.g61fdfe` |
| amdsmi（仅缺失时安装） | `1.0.0+a788f784` |
| pytest | `8.3.5` |
| tqdm | `4.67.1` |

前三个包来自海光官方索引，wheel 已下载并在本地静态核验 SHA。vLLM 和 lmslim 均为 Python3.10、Linux x86_64、DTK2604、Torch2.9 构建；尚未在当前服务器验证导入、ABI 或 Kernel 执行。

同索引的 vLLM 0.15.1 将 `rms_norm_opt` 改为 `(input, weight, out, epsilon, training)`；原版测试按 `(out, input, weight, epsilon)` 调用。选定的 0.11 版本保留后一种顺序，安装后再次用实际导入对象的签名检查。

vLLM 0.11 的包元数据还要求 NumPy 1.25、Triton 3.3.0，与现有栈存在差异。本次保留已有 Torch、Triton、LightOP、NumPy，安装范围是微基准依赖，完整 vLLM 依赖树未安装；不能据此宣称整个 vLLM 推理栈兼容。

Host 依赖先执行 pip dry-run，检查解析计划，禁止安装或替换上述数值库及其他 GPU 运行包，再按解析出的 HTTPS URL 和 SHA 安装。解析结果、完整安装日志和安装前后版本都保留。厂商 wheel 使用 `--no-deps --no-index` 安装。已有不同版本的 vLLM/lmslim 会停止，不强制覆盖；已有 amdsmi 保留。

## 服务器执行

第二批在 pip 候选解析阶段停止。第三批按 [LIGHTOP_BENCH_OFFLINE.md](LIGHTOP_BENCH_OFFLINE.md) 上传离线包并设置 LIGHTOP_BENCH_WHEELHOUSE，随后执行同一入口。下面保留原联网入口。

```bash
cd ~/private_data/LJ/Fusion-placement-search
git pull --ff-only
export FLAGSCALE_ROOT="$HOME/private_data/LJ/FlagScale-main"
export FUSION_PYTHON="$HOME/private_data/LJ/dcu-kernel310-20261031/run_python.sh"

bash tools/run_lightop_vendor_followup.sh \
  "$HOME/private_data/LJ/lightop_vendor_suite_20261103_02" run
```

流程：准备依赖 → 导入及接口预检 → 16 项原版测试。预检包含 ROCm 平台识别、Torch 基准绑定、RMSNorm 参数顺序、RMSNorm 类、量化/BMM 基准、固定基准源码 SHA。预检不做性能计时。失败时保留准备目录并停止，不进入测量。

预检通过后，原版脚本的 Shape、参数、计时、校验和文件内容保持不变。逐项一次尝试，失败继续其余项。仅本次启动链关闭 core dump，Python traceback/日志仍保留。

输出目录：

- `preparation/`：依赖计划、pip dry-run、按 SHA 固定的安装清单、安装日志、before/after、probe 结果；失败看 `failure_install.json`、`failure_probe.json` 和 `probe.log`。
- `measurements/`：16 个原版测试的 plan、identity、逐项日志、表格和 trace；只有预检通过才创建。

成功或失败都返回整个输出目录。两个依赖缓存放在 `~/private_data/LJ/lightop_bench_wheels/`（还可能有缺失时安装的 amdsmi），不必打包缓存。

## API 缺失的处理

固定 LightOP 中没有 `rms_rotary_embedding_fuse_with_kv_store_no_quant` Python 定义或公开导出。存在的原生 `split_qkv_rms_rotary_embedding_fuse_with_kv_store_no_quant` 接受另一种输入组织，不能直接重命名充当原测试接口。因此这项仍记为 vendor API mismatch，不进入本批 16 项，不产生替代耗时。

## CPU 审核

```bash
python -m unittest calibration.tests.test_lightop_bench_followup \
  calibration.tests.test_lightop_vendor_review calibration.tests.test_lightop_vendor_suite -v

python tools/review_lightop_vendor_suite.py /path/to/output/measurements \
  --output /fresh/review --report-key lightop_vendor_suite02_review
```

审核器已支持冻结矩阵内的有序子集，并检查选中项无遗漏、无重复；首批 48 项重放仍与已提交的产物字节一致。本次未执行服务器安装或 GPU 测量，未新增正式 Profile。
