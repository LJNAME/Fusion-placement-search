# LightOP / hipprof 最小试验

入口 tools/run_hygon_silu_trial.sh，默认 plan。显式 run 只检查现有安装并尝试 GPU0 的 SiLU×Mul；不安装或升级包，不运行旧 Graph 捕获路径，不写 Profile/Admission，不启动搜索。当前尚无服务器回执。

## 执行顺序

1. 独立进程检查当前 Python、Torch/HIP、LightOP 版本、模块位置、相关导出、可取得的函数签名/说明、包内源码/二进制哈希、test/benchmark 文件位置及 hipprof 路径。
2. 仅当现有 lightop 模块公开 callable silu_and_mul_opt 时，按已查到的 out/input 调用形式尝试；接口不匹配或数值失败即停止，不猜另一种签名、不静默回退。该公开导出是否存在仍待目标环境确认。
3. GPU0 做四臂的正确性、10 次预热、100 次普通 eager 调用；每臂一个计时批，固定顺序，无重试。所有 Event 计时结束后，才分别用两个独立进程执行 hipprof --hip-trace，采集 Native/LightOP 的 packed 路径。
4. 保存全部输出。缺 LightOP/接口/HIP 时停在检查阶段；缺 hipprof 时保留已完成 benchmark，标记剖析未执行；崩溃或采集失败不重跑。

源码依据：[OpenDAS/vllm_cscc 固定提交 3842b316](https://developer.sourcefind.cn/codes/OpenDAS/vllm_cscc/-/commit/3842b316979a3ca0c56ceecb7d0faf7262653950)。已查到包装调用 op.silu_and_mul_opt(out, input)，由合并输入末维一分为二；尚未读到目标安装版底层实现。当前是本项目调用厂商 API 的探索脚本，**不是声称已经取得并运行 LightOP 原版 benchmark**。

## 比较边界

BF16，逻辑输入各 [1,256,3072]，调用视图各 [256,3072]；合并输入 [256,6144]。二维视图只改变维度解释，独立 gate/up 本身均连续。固定 seed1234。

| 臂 | 输入边界 | 计时内操作 |
|---|---|---|
| native_packed | 连续合并输入 | 切片 → F.silu → Mul |
| lightop_packed | 相同连续合并输入 | 分配独立输出 → LightOP |
| native_split | 独立连续 gate/up | F.silu → Mul |
| lightop_split_with_pack | 相同独立连续 gate/up | cat 拼接 → 分配输出 → LightOP |

Native 保留真实中间量和输出分配，LightOP 包装每次分配独立输出。已合并输入只用于局部算子比较；split 对照在计时内包含拼接，不能用 packed 数字直接代表现有 FFN 两个 GEMM 输出的替代成本。两个边界分别比较，不与旧 KERV Graph 数据混用。

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

FUSION_PYTHON 应指向实际装有兼容 LightOP 的海光 Python 启动器。上例沿用已验证的私有 3.10；其中有没有 LightOP 尚未知，首次检查即会记录。缺包时返回整包后选择匹配版本，不直接 pip install 任意同名包，也不从系统 3.11 向私有 3.10 复制二进制。

无论成功或失败，打包全部目录：

```bash
tar -czf ../hygon_silu_trial_20261103_01.tar.gz -C .. hygon_silu_trial_20261103_01
```

本机 CPU 检查覆盖实际 packed/split 语义、拼接边界、只读检查、缺包/崩溃/缺工具/无 Trace 的停止顺序，不能作为 BW1000 或 LightOP 功能验收。
