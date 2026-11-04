# SMI 恢复首批：卸载完成，子进程结束状态未记录

不兼容的 amdsmi 已卸载，其他分发包版本未变，17 个备份文件 SHA 均匹配。

`torch_import.log:1-3` 已输出：

```text
2.9.0
6.3.26113
/root/private_data/LJ/dcu-kernel310-20261031/venv/lib/python3.10/site-packages/lightop/__init__.py
```

这三行位于 `import torch, lightop` 之后，两者已返回到打印语句。原脚本随后因子进程返回异常或超时而停止，但没有落盘退出码和 timed_out，不能确定是退出阶段异常还是超时，也不能据此称为 Torch 导入失败。SMI wrapper 和系统库检查均未开始。

原包：`lightop_smi_recovery_20261104_01.tar.gz`，150,017 字节，SHA256 `25a28eb4a18770a3787540f6130a4d60a2f0616ae85fcb62a3c6fdd2c320f014`。审核记录：`data/lightop_smi_recovery01_review/report.json`。本批没有算子计时或新增 Profile。

## 继续检查

新增 `resume` 阶段核对恢复回执、当前私有 Python 和包版本，不调用 pip、不重复卸载。两个独立子进程分别检查 Torch、Torch+LightOP，保存导入完成标记、退出码、信号名、超时和耗时。无论这两个子进程的退出状态如何，继续执行独立的只读 SMI 检查；失败日志保留，未正常退出不标为恢复通过。

```bash
cd ~/private_data/LJ/Fusion-placement-search
git pull --ff-only
export FLAGSCALE_ROOT="$HOME/private_data/LJ/FlagScale-main"
export FUSION_PYTHON="$HOME/private_data/LJ/dcu-kernel310-20261031/run_python.sh"

bash tools/recover_lightop_smi.sh \
  "$HOME/private_data/LJ/lightop_smi_recovery_20261104_01" \
  "$HOME/private_data/LJ/lightop_smi_diagnostics_20261104_01" resume
```

回传完整 `lightop_smi_diagnostics_20261104_01` 目录；原恢复目录保留。这一步不进行算子测量。

39 项相关 CPU 测试通过，覆盖信号/超时落盘、非零退出后继续系统库检查、续查不重复卸载及环境变化时拒绝继续。服务器退出问题和实际 SDK 兼容性仍待新回执。
