# LightOP 第三批：SMI 包与 HyHAL 不匹配

> 恢复首批已卸载 SMI，导入后的子进程结束状态仍待定位。已经执行过卸载的环境按 [LIGHTOP_SMI_RECOVERY01_REVIEW.md](LIGHTOP_SMI_RECOVERY01_REVIEW.md) 使用 resume，不重复运行卸载。

第三批完成离线依赖安装，基准预检在 `import torch` 处失败；没有创建 measurements，没有启动 16 项补测。

| 项目 | 回执结果 |
| --- | --- |
| 新增 Python 分发包 | 58 |
| 原有分发包版本变化 | 0 |
| amdsmi 安装前 | 无 |
| amdsmi 安装后 | `1.0.0+a788f784` |
| GPU 测量任务 | 0 |
| 新增 accepted Profile | 0 |

回执 SHA256：`55a64ddb0092ecc5aa214cfc803bb09595ea2ba3c8bbea586abd936ce295fba2`。

`preparation/install_vendor_baselines.log:6` 确认 vLLM、lmslim、amdsmi 安装完成。`preparation/probe.log:7-22` 记录导入链：Torch → amdsmi → ctypes → `/opt/hyhal/lib/libamd_smi.so`，在绑定 `amdsmi_get_node_handle` 时抛出 AttributeError。

选入的 amdsmi wheel 是 Python wrapper，导入时会绑定系统库符号。当前 HyHAL 缺少该符号，wrapper 不匹配。先前仅核对了 wheel 来源、SHA 和 Python 依赖，没有验证它与实际 HyHAL 的接口；这项选择需要撤回。Torch/Triton/LightOP/NumPy 版本保持原值，但 Torch 导入已经受到影响，不能把“版本未变”写成“运行环境可用”。

## 恢复入口

`tools/recover_lightop_smi.sh` 只处理这次新增的明确版本：

1. 核对原回执、私有 Python 路径、安装前不存在 amdsmi、当前版本及源码 SHA、原数值库版本。
2. 将该分发包全部文件备份到恢复目录，然后只卸载 amdsmi。
3. 子进程分别验证 Torch 和 Torch+LightOP 导入，保存完成标记、退出码及超时状态；异常仍保留后续只读检查。
4. 只读收集服务器已有 SMI wrapper 源码、系统库 SHA 与符号；独立子进程验证找到的纯 Python wrapper 能否导入、初始化和枚举设备。含其他 Python 版本扩展的候选不导入。

不安装其他 amdsmi 版本，不替换 HyHAL 系统库，不启动算子计时。SDK 检查失败也保留日志；不能将 SMI 设备枚举成功当成 vLLM 基准已通过。

```bash
cd ~/private_data/LJ/Fusion-placement-search
git pull --ff-only
export FLAGSCALE_ROOT="$HOME/private_data/LJ/FlagScale-main"
export FUSION_PYTHON="$HOME/private_data/LJ/dcu-kernel310-20261031/run_python.sh"

bash tools/recover_lightop_smi.sh \
  "$HOME/private_data/LJ/lightop_vendor_suite_20261103_03" \
  "$HOME/private_data/LJ/lightop_smi_recovery_20261104_01"
```

回传完整 `lightop_smi_recovery_20261104_01`。原第三批目录和离线包继续保留。

依赖安装入口已排除原 amdsmi wheel，并在任何 pip 安装前检查现有 Torch 导入。原 v1 manifest 和 v3 离线包保持字节不变，作为第三批安装来源；不再执行其中不兼容的 SMI 安装项。

本地 36 项相关 CPU 测试通过，覆盖误删前置包/其他环境/不同版本/不同回执的拒绝、Torch 导入失败时禁止安装及超时记录。服务器恢复和 SDK 兼容性待回执核验。
