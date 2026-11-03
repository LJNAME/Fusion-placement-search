# LightOP 定向补测：离线依赖包

第二批停在 pytest 的 pip 候选解析，未安装、未测量，见 LIGHTOP_VENDOR_SUITE02_REVIEW.md。第三批改用离线包完成依赖解析和安装，仍只运行首批缺依赖的 16 项。

包名：`lightop_bench_deps_cp310_linux_v3.tar.gz`，84,184,701 字节（80.28 MiB），65 个 wheel。

SHA256：`4662f15216545c04f9faa95f4f01516d2e5962d416cda20bc368839d2e710d7f`。

Host 依赖从清华 PyPI 镜像下载，厂商 vLLM/lmslim/amdsmi 使用此前固定的官方 wheel。65 个文件的压缩包内 SHA、Python3.10 wheel tag 和 Requires-Python 已核验。Host 依赖闭包按回执中的已安装版本和 Linux/Python3.10 标记检查，共检查 932 条声明，无缺失；包含 exceptiongroup、tomli、async-timeout、hf-xet 等跨平台下载容易遗漏的依赖。未包含现有 Torch/Triton/LightOP/NumPy 的替换 wheel。

## 在服务器运行

先把压缩包上传到 `~/private_data/LJ/`，然后执行：

```bash
cd ~/private_data/LJ
printf '4662f15216545c04f9faa95f4f01516d2e5962d416cda20bc368839d2e710d7f  lightop_bench_deps_cp310_linux_v3.tar.gz\n' | sha256sum -c -
# 上一行必须显示 OK 后再解压。
tar -xzf lightop_bench_deps_cp310_linux_v3.tar.gz

cd Fusion-placement-search
git pull --ff-only
export FLAGSCALE_ROOT="$HOME/private_data/LJ/FlagScale-main"
export FUSION_PYTHON="$HOME/private_data/LJ/dcu-kernel310-20261031/run_python.sh"
export LIGHTOP_BENCH_WHEELHOUSE="$HOME/private_data/LJ/lightop_bench_deps_cp310_linux_v3"

bash tools/run_lightop_vendor_followup.sh \
  "$HOME/private_data/LJ/lightop_vendor_suite_20261103_03" run
```

离线模式会校验 bundle 与冻结计划及各文件的 SHA。pip 用 `--no-index --find-links` 从本地解析，按本地文件 SHA 锁定安装，再从本地安装厂商 wheel；这几步不访问包索引。已有数值库的版本约束和替换拒绝继续有效。

pip 子进程统一隔离继承的 PIP_* 配置，设置 PIP_CONFIG_FILE=/dev/null，并使用 --isolated；输出 pip_context.json，仅记录继承变量名，不输出可能含口令的配置值。

依赖安装后仍要通过实际基准导入、参数顺序和 ROCm 平台预检，才开始 16 项测量。离线包完整性检查不代表 vLLM ABI、原版数值检查或 Kernel 执行已经通过。原版计时、Shape 和容差不变，已有成功数据及数值失败项不重跑。

返回整个 `lightop_vendor_suite_20261103_03` 目录，包含 preparation 和实际生成的 measurements；失败也保留全部日志。

## 本地记录

- `data/lightop_vendor_suite02_review/report.json`：第二批失败事实及输入 SHA。
- `data/lightop_bench_offline_v1/bundle.json`：离线包完整清单及逐 wheel SHA。
- `data/lightop_bench_offline_v1/verification.json`：28 项 CPU 测试、离线包和目标平台核验。
- `tools/build_lightop_bench_bundle.py`：只下载和读取 wheel，不在 Windows 导入或安装 Linux 扩展。

本次未执行服务器安装或 GPU 测量，不新增正式 Profile。
