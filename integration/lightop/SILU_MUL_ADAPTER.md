# LightOP SiLU×Mul 单组执行 Adapter

首版执行映射与 CPU 验证完成，真实 GPU 执行、Graph Capture 和正式 Profile 查询待完成。新增 33 项 Adapter 测试，连同 KERV、厂商审核、旧 SiLU 探针及适用性审计共 87 项 CPU 测试通过。本轮未执行 GPU、未写入时延、未新增 accepted Profile。

## 输入与输出

`integration/lightop/silu_mul.py` 将一个 PlacedOpGraph 中的 `silu_mul` 组映射到 `lightop.op.silu_and_mul_opt`。首版仅覆盖已有证据对应的 BF16 `[1,256,3072]`、forward inference、层内 SiLU→Mul；图中 gate/up 必须来自两个明确的逻辑生产者。通过 SiLU 输入边识别 gate，通过 Mul 的另一条输入边识别 up，不依赖边列表顺序。

实际 gate/up 必须已位于该融合组对应的 HIP GPU，连续且无 autograd；不隐式搬卡或转换布局。图布局未知时只在准备阶段观测实际 Tensor，不修改原图；显式 Shape/dtype/stride/layout 声明若与实际值冲突则拒绝。输出独立分配，保持原逻辑 Shape，仅物化 Mul 节点；不生成 SiLU 中间输出、不使 gate 失效。

由于输入只读，gate 可有组外消费者或成为图输出，两个输入 Tensor 也允许共享存储；SiLU 中间值有组外消费者或被导出时不能删除，仍拒绝融合。该语义与 KERV 的 gate 原地覆盖不同，不复用 KERV 的原地安全约束。

## 单次执行边界

| 阶段 | 操作 | 进入候选调用的 GPU 计时 |
| --- | --- | --- |
| `bind` | 校验组、设备映射、显式实现标注 | 否 |
| `prepare` | 验证真实 Tensor，分配 packed/output，建立二维视图，记录布局及存储重叠 | 否 |
| `call()` | `torch.cat((gate_2d, up_2d), dim=-1, out=packed)` | 是 |
| `call()` | `lightop.op.silu_and_mul_opt(output_2d, packed)` | 是 |
| 调用方 | 将返回值写到 Mul 对应节点，管理依赖及同步 | 不增加额外 Kernel 操作 |

每个 prepared call 自持一份 `[256,6144]` packed 和 `[1,256,3072]` output，共 **4,718,592 字节（4.5 MiB）**，不计输入、其他存活 Tensor、框架池或库内部 workspace，不将其称为显存峰值。

重复调用每次都会重新打包，可更新原地址中的输入数据。gate/up 不被写回，因此无需恢复 gate。输入、缓冲区和 plan 在 Capture/Replay 期间保持地址及元数据稳定；调用方持有这些对象直到 GPU 完成。一个 prepared call 的输出会被下一次调用覆盖；需要多个同时存活的输出或并发流时分别准备，不能共用同一输出缓冲。

计时区间应覆盖整个 `call()` 的 GPU 操作。若成本模型引用这份完整调用时延，就不能再次追加打包成本或重复启动开销。旧 trial02 使用调用内分配的 `torch.cat`，本版使用预分配 `out=`，配方已不同，旧数字不能直接作为本版标签。

## 使用方式

调用方提供 `placed`、目标 `group_index` 和已就绪的边界 Tensor 字典 `values`。本接口只执行选中的融合组，不调度其他节点或通信。

```python
import copy
from integration.lightop.silu_mul import LightopSiluMulAdapter

adapter = LightopSiluMulAdapter()  # 校验固定 LightOP 版本及包文件 SHA
placed.plan.groups[group_index].extra["implementation"] = copy.deepcopy(adapter.implementation)
bound = adapter.bind(placed, group_index, {0: "cuda:0", 1: "cuda:1"})
call = bound.prepare(values)       # 在 Capture/计时前调用
values[call.output_node] = call()  # 返回异步 GPU 结果；同步和后续依赖由调用方负责
```

实现必须显式标注；不能将带 KERV 身份的组直接交给 LightOP 执行。`declared_binding()` 提供 Adapter 源码 SHA、入口、固定库版本及二进制 SHA，供组标注使用；这是执行声明，不是完整 Canonical Kernel Key，也不代替实际运行环境身份。

局部 `adapter_registry()` 的 `implemented` 仅表示有执行适配代码，原默认 registry 与 KERV 不变。`integration.lightop.capabilities()` 对 SiLU×Mul 单独返回 `placed_graph_adapter=true`、`validation=cpu_contract_only_gpu_pending`。固定厂商目录及 v1 静态适用性审计保持历史状态，未用后续代码改写历史证据。

## 已完成与缺项

CPU 测试验证真实 Shape 的打包顺序、每次重新打包、独立输出、输入字节保持、共享输入存储、布局观测、错误输入/设备/实现拒绝、异常不回退、工件往返及 28 组结构枚举。测试通过替代 callable 和 GPU guard 实现，记录的实际 Tensor 设备仍是 CPU，不视为新的 GPU 执行证据。

实际运行身份、Canonical Key、严格成本查询接口与最小 GPU Smoke 入口已完成 CPU 实现，见 [身份绑定与 Smoke](../../calibration/LIGHTOP_PROFILE_SMOKE.md)。GPU0 的 BF16 `[1,256,3072]` 首次 Capture/Replay Smoke 已通过，Key/Contract 经 CPU 重建一致；独立 dispatch 与正式 Profile 准入尚未完成。GPU 验证需要证明包含 `cat(out=...)` 的整条 Capture/Replay 路径、全部输出正确及输入只读；dispatch trace 应分别识别打包工作与 LightOP 融合 Kernel，不要求整个 Adapter 只有一个设备 Kernel。不得静默回退到 Native、eager 或其他实现。

Adapter 本身不采集计时轮次、不执行准入。独立 Smoke 入口记录契约与诊断计时，不产生时延标签；严格成本接口只消费已审核标签，目前没有本版正式标签。实际环境字段从服务器观测，SCNet 重启导致 UUID 改变不单独作为跨批阻断条件。同批设备和 Tensor 的对应关系仍分别核验。

```bash
python -m pytest integration/tests/test_lightop_silu_mul.py -q
```
