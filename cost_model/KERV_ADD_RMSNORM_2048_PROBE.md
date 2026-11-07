# KERV Add＋RMSNorm 2048 单Case验证

2026-11-07。只验证固定源码的执行入口、数值和原地语义；无计时、Graph Capture、Profile准入或搜索规则变更。

## CPU核对

现有Qwen2.5-3B图的36处attention out_proj均仅由residual_add_1消费，图上允许覆盖；36处Add结果均还被residual_add_2使用，必须保留双输出。实际Tensor布局与别名未验证，图中unknown未改写为contiguous。

入口为FlagScale的adaptive_rms_norm.add_rms_norm_inference，固定源码SHA256 f9b3bafdd1a1c303627cf2dbe2561879155088f763c40c341cc997beed107f88。直接调用_add_rms_norm_inference_kernel，绕过公共Torch算子的auto/native分流。

Case：BF16 [1,256,2048]、weight [2048]、epsilon=1e-6。epsilon是与execution/reference.py:_rmsnorm一致的显式声明，未声称来自图元数据。输入为按真实图Shape生成的合成值，不能证明真实Projection输出布局。

CPU诊断（Torch 2.9.0+cpu、固定seed20261107）：手写BF16中间舍入与F.rms_norm输出有131839/524288个元素不同，最大绝对差0.03125。这只是两条CPU参考语义的差异，不是DCU Kernel误差，也不作为GPU通过结论。

数据：[plan.json](../data/kerv_add_rmsnorm_2048_probe_v1/plan.json)。8项CPU测试通过：图上覆盖安全、旁路/导出拒绝、未知布局保留、源码错配拒绝、参考舍入、dispatch失败恢复、设备trace入口约束、无GPU拒绝。

## 服务器执行

同步包含新入口的项目文件后，在项目根目录执行。使用新的输出目录；不需要升级LightOP或Torch。

```bash
export FLAGSCALE_ROOT="$HOME/private_data/LJ/FlagScale-main"
export FUSION_PYTHON="$HOME/private_data/LJ/dcu-kernel310-20261031/run_python.sh"
bash tools/run_kerv_add_rmsnorm_probe.sh ../kerv_add_rmsnorm_2048_01 run cuda:0
```

默认action为plan，只进行CPU计划检查；run才会访问HIP GPU，并要求显式设置FUSION_PYTHON，避免使用系统Python。首次回传在系统Python3.11的Triton读取AMD设备属性时UnicodeDecodeError，尚未进入Kernel；先换回既有Python3.10启动器，不能据此判定Kernel不支持2048。本机未运行run。

GPU检查固定为3次调用，每次精确恢复x；检查sum与x同址、normalized独立存储、residual/weight逐字节不变、sum等于BF16 Add，并分别对照手写舍入参考和本项目aten参考（rtol=0.02、atol=0.02，沿用已有AddRMS数值容差）。随后单独采集一次设备trace，要求唯一kernel为_add_rms_norm_inference_kernel。不根据耗时判断通过。

result.json只在全部通过后写入。checks.json保留数值检查；runtime_context.json记录实测布局、默认allocator和显存快照；failure.json保存Python异常，fatal_traceback.log保存进程崩溃栈。没有自动重试或Native回退。返回整个目录，包括失败文件。

通过后仍不代表Graph Replay可用、具备正式Profile、实际模型生产者布局已验证，或已进入搜索。先根据回传核对数值与真实执行证据，再决定图Adapter映射。

## 2026-11-07 回传复核

数据：data/kerv_add_rmsnorm_2048_02/raw（6个原文件逐字节保留），review/report.json（18项CPU复核）。压缩包4833字节，SHA256 2ecf7b6713bcb0b6beb263dd5e7ba4f862adcbe9ea0ccdbb07f67d3dfe0d89ae。collector、Reference executor、KERV源码SHA与现有文件匹配；FlagScale revision=f8e9a62c499d526a6ffa47f533bcf342330b7326，图fingerprint及36处liveness重算一致。

| 项目 | 回传结果 |
| --- | --- |
| 设备 | GPU0，gfx936:sramecc+:xnack- |
| 环境 | Torch2.9.0 / HIP6.3.26113 / Triton3.5.1 |
| Tensor | BF16 [1,256,2048]，stride [524288,2048,1]，连续 |
| 三次调用：手写参考max abs | 0 / 0 / 0 |
| 三次调用：项目aten参考max abs | 0.03125 / 0.03125 / 0.03125 |
| 数值容差 | rtol=0.02、atol=0.02，两参考全部通过 |
| 逐调用语义检查 | 21/21通过（恢复、sum别名、normalized独立存储、residual/weight只读、BF16 Add精确、有限输出） |
| 独立解析device trace | 唯一kernel=_add_rms_norm_inference_kernel |
| 正式Profile / 计时 / Capture | 0 / 未进行 / 未进行 |

本次核验的是绑定源码的GPU回执和trace，原始GPU Tensor未归档，未在本机重算GPU数值。两参考并非逐位等价；此次Kernel满足约定数值容差。连续布局是合成Tensor的实测属性，真实模型Projection生产者布局仍未捕获。36是图上可适配位置数，不是36个GPU测量Case；GPU1未验证。

换回既有启动器后同一源码成功执行；保留第一次系统Python3.11失败的历史记录，不修改驱动或Kernel。RO CTracer flow警告未阻断本次目标kernel trace核验。当前仍没有KERV AddRMS图Adapter或搜索注册，3B现有搜索候选数不因这次回传自动改变。

下一项为固定Shape的图Adapter：运行时检查布局与存储别名、消费attention输入、按原节点ID发布Add和Norm两个输出。先做CPU映射测试；本次不增加GPU测量，不扩大搜索。
