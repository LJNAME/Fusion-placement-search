# LightOP / hipprof 最小试验

入口 tools/run_hygon_silu_trial.sh，默认 plan。显式 run 只检查现有安装并尝试 GPU0 的 SiLU×Mul；不安装或升级包，不运行旧 Graph 捕获路径，不写 Profile/Admission，不启动搜索。首批检查回执已确认私有Python3.10没有LightOP，HIP6.3.26113与/opt/dtk/bin/hipprof已识别；未开始benchmark。该回执来源3d9b114，见data/hygon_silu_trial01_review/report.json。

## 执行顺序

1. 独立进程检查当前 Python、Torch/HIP、LightOP 版本、模块位置、相关导出、可取得的函数签名/说明、包内源码/二进制哈希、test/benchmark 文件位置及 hipprof 路径。
2. 仅当现有 lightop.op 模块公开 callable silu_and_mul_opt 时，按已查到的 out/input 调用形式尝试；接口不匹配或数值失败即停止，不猜另一种签名、不静默回退。v2按已校验wheel内test/test_act_and_mul.py确定真实命名空间；不是在lightop根模块查找该函数。运行兼容仍待目标环境确认。
3. GPU0 做四臂的正确性、10 次预热、100 次普通 eager 调用；每臂一个计时批，固定顺序，无重试。所有 Event 计时结束后，才分别用两个独立进程执行 hipprof --hip-trace，采集 Native/LightOP 的 packed 路径。
4. 保存全部输出。缺 LightOP/接口/HIP 时停在检查阶段；缺 hipprof 时保留已完成 benchmark，标记剖析未执行；崩溃或采集失败不重跑。

源码依据：[OpenDAS/vllm_cscc 固定提交 3842b316](https://developer.sourcefind.cn/codes/OpenDAS/vllm_cscc/-/commit/3842b316979a3ca0c56ceecb7d0faf7262653950)。已查到包装调用 op.silu_and_mul_opt(out, input)，由合并输入末维一分为二；尚未读到目标安装版底层实现。已取得海光wheel并校验索引SHA，包内有test/test_act_and_mul.py和test/test_fuse_silu_mul.py。前者明确使用from lightop import op、op.silu_and_mul_opt(out,input)、预分配输出、10次预热/100次普通Event调用。其baseline依赖vLLM且本身也是融合算子；本项目仍比较Native两算子，采用相同计时组织，不声称原版脚本或其baseline已经执行。固定wheel信息见data/hygon_lightop_wheel_v1/manifest.json。

## 比较边界

BF16，逻辑输入各 [1,256,3072]，调用视图各 [256,3072]；合并输入 [256,6144]。二维视图只改变维度解释，独立 gate/up 本身均连续。固定 seed1234。

| 臂 | 输入边界 | 计时内操作 |
|---|---|---|
| native_packed | 连续合并输入 | 切片 → F.silu → Mul |
| lightop_packed | 相同连续合并输入 | 写入预分配输出 → LightOP |
| native_split | 独立连续 gate/up | F.silu → Mul |
| lightop_split_with_pack | 相同独立连续 gate/up | cat 拼接 → 写入预分配输出 → LightOP |

四臂各自使用独立预分配输出，与库内测试一致；Native仍计入SiLU中间量分配。相对v1的输出分配策略已变更，v1计划原件保留，不拼接不同策略的时间。已合并输入只用于局部算子比较；split 对照在计时内包含拼接，不能用 packed 数字直接代表现有 FFN 两个 GEMM 输出的替代成本。两个边界分别比较，不与旧 KERV Graph 数据混用。

GPU Event 均值与同步 host wall 均值单独报告；前者包含 eager 提交间隙，后者还包含 Event/提交/同步开销，均不称纯 Kernel 指令耗时。只有一次批测，没有稳定性、失败率或显著性结论；本轮不采用此前正式七轮锚点协议，也不把探索数字升格 accepted。

每臂检查计时前及最后输出（未保留所有中间调用输出），与 Native 和 FP32 参考比较，rtol=.01、atol=.001；确认只读输入逐字节保持。split LightOP 检查实际拼接后的实参，不借其他 Tensor 元数据代替。失败保留 traceback、last_stage、已有显存快照和 worker.log；原生崩溃由 faulthandler 记录，父进程不补采 GPU。

## hipprof 证据

采用站内 [HIPTX Python 项目](https://developer.sourcefind.cn/codes/tsoc/hiptx-python/-/blame/cfcfe0b45187b3b0e69696b6bcf3f1fd61f536c9/README.md) 已展示的 hipprof --hip-trace 采集入口；本轮未依赖 HIPTX Python 扩展。每个 profile worker 调用指定臂 11 次，Trace 还会包含 fixture 准备与校验，不能把整个进程的 Kernel 总数当成一次融合调用的 Kernel 数。

返回码、worker 完成标志和原始 profiler 文件分别记录；必须回传核对实际 Kernel、API 和采集边界，pipeline_passed 不代表已认证单 Kernel 或无 native fallback。剖析进程不输出可消费的计时标签。

SCNet 重启后的跨批 UUID 变化不阻塞本试验；记录本次设备与架构、版本，不要求等于旧 pilot。此次不查询或占用 GPU1。

## 运行

```bash
cd ~/private_data/LJ/Fusion-placement-search
git pull --ff-only
export FLAGSCALE_ROOT="$HOME/private_data/LJ/FlagScale-main"
export FUSION_PYTHON="$HOME/private_data/LJ/dcu-kernel310-20261031/run_python.sh"
bash tools/run_hygon_silu_trial.sh ../hygon_silu_trial_20261103_01 run
```

FUSION_PYTHON 应指向实际装有兼容 LightOP 的海光 Python 启动器。上例沿用已验证的私有 3.10；首批已确认其中没有LightOP。安装前使用tools/install_hygon_lightop.sh，固定版本/哈希、检查ABI1并以--no-deps --no-index安装；缺依赖或ABI冲突保留日志停止，不直接 pip install 任意同名包，也不从系统 3.11 向私有 3.10 复制二进制。

无论成功或失败，打包全部目录：

```bash
tar -czf ../hygon_silu_trial_20261103_01.tar.gz -C .. hygon_silu_trial_20261103_01
```

本机 CPU 检查覆盖实际 packed/split 语义、拼接边界、只读检查、缺包/崩溃/缺工具/无 Trace 的停止顺序，不能作为 BW1000 或 LightOP 功能验收。

## 固定wheel安装与第二次试验

已静态验证的版本为0.6.0+das.dtk2604.torch290.2605220936.g6bfaca，cp310、manylinux2.27/2.28、x86_64、ABI1；包内METADATA要求torch==2.9.0，version.py声明DTK2604/Torch2.9。wheel共100433180字节，SHA256为61ef2672bbca7419a59ef14d41d1fcd883a972d57d51a2070df98e18d98a5795。来源为海光[release索引](https://pypi.sourcefind.cn/release/dtk/+simple/lightop/)，不采用镜像站任意同名包或Torch2.10构建。

```bash
cd ~/private_data/LJ/Fusion-placement-search
git pull --ff-only
export FLAGSCALE_ROOT="$HOME/private_data/LJ/FlagScale-main"
export FUSION_PYTHON="$HOME/private_data/LJ/dcu-kernel310-20261031/run_python.sh"
bash tools/install_hygon_lightop.sh ../hygon_lightop_install_20261103_01
bash tools/run_hygon_silu_trial.sh ../hygon_silu_trial_20261103_02 run
```

上述两个动作应在set -e子shell中顺序执行，安装失败不得继续试验。wheel缓存位于~/private_data/LJ/lightop_wheels，安装日志与前后版本在单独install目录，缓存wheel不必回传。脚本检查Python3.10/Linux x86_64、vendor Torch2.9/DTK2604、HIP和CXX11 ABI1；不同LightOP已安装时停止复核。只安装固定wheel，不解析其他依赖，安装后确认Torch/Triton版本未变且优化导出可用。

包内API有两条：根模块fuse_silu_and_mul(input, output)，以及本次选择的lightop.op.silu_and_mul_opt(output, input)。两者不能按名字相近自动替换。安装/导入仍可能遇到实际ABI、ROCM_HOME、rocminfo等运行环境问题；原件、GPUbenchmark、hipprof回执到达前不宣称功能或性能验收通过。
