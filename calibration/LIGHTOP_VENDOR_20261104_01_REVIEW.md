# LightOP 11.4 首批：vLLM 平台检测冲突

| 项目 | 结果 |
| --- | --- |
| 宿主 SMI 接入 | 已通过，版本 `7.6.0+995313d1`，枚举 2 张设备 |
| 实际加载的 SMI 原生库 | SHA 与宿主固定库一致 |
| 基准预检 | 退出码 1，56.744 s，无超时 |
| GPU 测量任务 | 0 |
| 临时路径清理 | 已完成 |
| 安装前后包清单 | 字节一致 |

错误原文（`preparation/probe.log:26`）：

```text
RuntimeError: Only one platform plugin can be activated, but got: ['cuda', 'rocm']
```

固定 vLLM 0.11 wheel 的 `vllm/platforms/__init__.py:57-123` 分别执行 CUDA/NVML 与 ROCm/SMI 检测，`:210-213` 拒绝多个内置平台同时激活。本批证明两者都被激活，尚未确定 CUDA 检测在该环境命中的更深层原因，不能据此认定存在 NVIDIA 卡或指定某个兼容库。

该选择器不读取 `VLLM_TARGET_DEVICE`，`VLLM_PLUGINS` 只筛选外部插件，不筛选内置检测器。仅设置这两个变量不能直接消除当前的内置平台冲突。

原包：`lightop_vendor_suite_20261104_01.tar.gz`，3,218 字节，SHA256 `b7931fd0e163e88d0ded036a17c54884b0faabf7766ea1432899145b5f85563b`。审核记录见 `data/lightop_vendor_suite_20261104_01_review/report.json`。

## 修复

使用 vLLM 已提供的 `vllm.platform_plugins` 入口。`integration/lightop/hygon_platform.py` 仅在 Torch 报告 HIP 构建且 SMI 能枚举设备时返回原有的 `vllm.platforms.rocm.RocmPlatform`。不新增后端类，不改 vLLM、LightOP 或原版测试源码，不伪造设备信息。

原选择器 `:206-209` 优先使用唯一外部平台插件。入口在本次任务期间临时注册 `fusion_hygon_rocm`，源码位置加入同一临时搜索路径；基准子进程均能发现它。预检或补测结束后清理平台注册目录及路径配置。已有 `VLLM_PLUGINS` 若排除此插件则报错，保留用户配置。

完整预检仍须实际导入 vLLM、核对原算子绑定和签名，并确认最终类型为原 `RocmPlatform`。通过后才运行冻结的 16 项原版补测，仍各一次尝试、失败继续。

```bash
cd ~/private_data/LJ/Fusion-placement-search
git pull --ff-only
export FLAGSCALE_ROOT="$HOME/private_data/LJ/FlagScale-main"
export FUSION_PYTHON="$HOME/private_data/LJ/dcu-kernel310-20261031/run_python.sh"

bash tools/run_lightop_host_smi.sh \
  "$HOME/private_data/LJ/lightop_vendor_suite_20261104_02" run
```

回传整个 `lightop_vendor_suite_20261104_02`。不重新安装依赖，原 `_01` 回执保留。新输出额外记录 `platform_binding.json`、`platform_cleanup.json` 和 `preparation/platform_selection.json`。

49 项相关 CPU 测试通过，覆盖实际 entry point 发现、HIP/SMI 条件、失败清理及原 16 项分派。对固定 wheel 的原选择器函数进行 CPU 分支重放：两个内置检测器为真时复现异常，增加唯一外部插件后返回原 `RocmPlatform`；这是替代检测器的 CPU 测试，不是 GPU 运行证明。实际预检和补测待新回执，没有新增 Profile。
