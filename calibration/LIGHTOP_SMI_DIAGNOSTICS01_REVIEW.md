# SMI 诊断首批：恢复通过，复用宿主 SDK

| 检查 | 结果 |
| --- | --- |
| Torch 单独导入 | 退出码 0，9.938 s，无超时 |
| Torch + LightOP 导入 | 退出码 0，11.367 s，无超时 |
| 两项导入完成标记 | 均存在 |
| 宿主 SMI 初始化 | 退出码 0，枚举 2 张设备 |
| 包版本与卸载完成后的清单 | 一致 |

宿主 wrapper：`/opt/hyhal/share/amd_smi/amdsmi`，源码版本 `7.6.0+995313d1`。5 个 Python 源文件 SHA 全部与回执匹配。系统库 `/opt/hyhal/lib/libamd_smi.so.0.0` 的 SHA 为 `c2f663b92556039e328fda65bc13ea7cfc096ace9eda4310f6172c644ecf9686`；该库仍没有 `amdsmi_get_node_handle`，但自带 wrapper 能正常初始化及枚举设备。

上一批恢复子进程的异常未在本批复现，原退出码/超时记录缺失，原因仍未确定。本批证明导入及 SMI 基础接口恢复，不包含 vLLM 完整预检、算子计时或新增 Profile。

原包：`lightop_smi_diagnostics_20261104_01.tar.gz`，47,819 字节，SHA256 `99e4a2619819eaf62e7bf9753055853ae300e272b10168a6ce6c45428c72e91f`。审核记录见 `data/lightop_smi_diagnostics01_review/report.json`，接入来源见 `data/lightop_host_smi_v1/manifest.json`。

## 原版补测入口

不再安装 PyPI amdsmi。入口 `tools/run_lightop_host_smi.sh` 复用当前 HyHAL 的 wrapper：

1. 核对 5 个宿主源文件、系统库 SHA 和已有数值库版本。
2. 在私有 Python 的 site-packages 临时创建 `fusion_lightop_host_smi.pth`，仅加入 `/opt/hyhal/share/amd_smi`。不会覆盖已有同名配置。
3. 新子进程核对实际加载的 amdsmi 路径、源码版本及原生库 SHA，然后运行现有完整基准预检，包括真实 vLLM ROCm 检测、算子绑定及原 RMS 参数顺序。
4. 预检通过后，自动运行冻结的 16 项原版补测。脚本内容、参数和计时方法不变；逐项一次尝试，失败继续。
5. 预检或补测结束后在 finally 中移除本次路径配置，并记录清理及包状态。失败保留全部日志和退出状态；不会自动安装依赖或替换系统库。

Python 分发包清单中的 amdsmi 仍为空，实际 SDK 身份单独记录于 `preparation/host_smi.json`；不伪造分发包版本。临时路径配置会影响该私有 Python 在任务期间启动的子进程，任务结束后移除。

```bash
cd ~/private_data/LJ/Fusion-placement-search
git pull --ff-only
export FLAGSCALE_ROOT="$HOME/private_data/LJ/FlagScale-main"
export FUSION_PYTHON="$HOME/private_data/LJ/dcu-kernel310-20261031/run_python.sh"

bash tools/run_lightop_host_smi.sh \
  "$HOME/private_data/LJ/lightop_vendor_suite_20261104_01" run
```

沿用此前 LightOP wheel 缓存 `~/private_data/LJ/lightop_wheels/`；如果放在别处，可设置 `LIGHTOP_VENDOR_WHEEL` 为原固定 wheel 路径。不需要重新上传依赖包或设置 wheelhouse。

无论结果如何，回传整个 `lightop_vendor_suite_20261104_01`。`preparation/` 保存 SMI 身份、完整预检和进程状态；`measurements/` 只有预检通过后才生成。根目录 `path_cleanup.json` 记录临时配置移除，`before.json`/`after.json` 记录分发包状态。

45 项相关 CPU 测试通过，包含子进程路径可见性、预检失败禁止启动补测、成功后使用原冻结子集及路径清理。实际 vLLM/SDK 联合预检和原版补测仍待服务器执行；没有自动 Profile Admission。
