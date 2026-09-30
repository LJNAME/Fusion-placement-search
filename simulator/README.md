# 仿真器（方向 1 · 模块二）

对应 Obsidian 笔记《方向 1：》的模块二规格：

> 执行模型（roofline 式解析 + 事件队列）+ 内存模型 + 参数清单，
> 用于对代价模型筛选出的 top_k 方案精确模拟，结果更新代价模型。

零第三方依赖（纯标准库，Python 3.10+）。

## 从配置文件加载硬件参数

参数不写在代码里，放 `configs/` 目录（支持异构，见 `configs/README.md`）：

```python
from simulator import load_hardware

hw = load_hardware("configs/hetero_4cards.example.json")
```

配置里的 `flops` / `mem_bw` / `comm_bw` 是官方标称值，加载时会自动乘上
`efficiency` 系数（如 `{"flops": 0.7, "mem_bw": 0.7}`）得到有效值——
没有真卡时先用"标称 × 系数"替代，拿到实测数据后把标称值换成实测值、
系数改为 1.0 即可，代码无需改动。

配置字段与校准方法见 [configs/README.md](../configs/README.md)。

## 目录结构

```
simulator/
  schema.py       模块一：统一输入图格式（节点/边 schema、融合组、方案）
  hardware.py     参数清单：算力、显存带宽、延迟、启动/融合开销、n_devs、max_mem
  execution.py    模块二核心：roofline + 事件队列 + 设备竞争 + 共享总线 + 显存生命周期
  validate.py     方案合法性校验（覆盖 / 不重复 / 拓扑序 / 设备范围）
  generators.py   随机方案 + 启发式方案（all / chain / threshold）
  dataset.py      Stage 0 标签生成：(图, 方案, makespan, 显存峰值) -> JSONL
  examples/demo.py  三算子两 GPU 示例
  tests/          单元测试
```

## 快速开始

```bash
python examples/demo.py          # 三算子两 GPU 示例
python -m unittest discover -s tests -v   # 运行测试
```

## 执行模型

每个融合组（一个 kernel）的时间：

```
t = max(组计算量 / 设备有效算力, 边界读写字节 / 显存带宽) + 启动开销
    + 融合开销 × (组内算子数 - 1)
```

其中"设备有效算力"默认就是 `DeviceSpec.flops`（峰值算力）；若配置了
`shape_model`，则对 matmul 做形状相关的修正：

```
有效算力 = flops × eta,   eta = eta_max × K / (K + k_half)
```

即 GEMM 的规约维 K 越大越接近峰值算力，小 K 的 GEMM 有效算力显著低于峰值。
这让仿真器与解析式峰值 roofline（`cost_model/roofline.py`）不再同构，
学习式代价模型才有机会学到形状修正。曲线参数是占位假设，后续按
`configs/README.md` 的微基准流程逐算子校准。

- **融合组**：组内共享中间张量——中间张量不写全局内存、不产生通信，
  组时间按组内数据流（算力瓶颈 / 带宽瓶颈取大者）估算；
- **通信**：跨设备边字节 / 总线带宽 + 延迟，走**共享总线模型**（FIFO 排队，
  所有传输串行竞争总线）；
- **设备竞争**：每设备一个 FIFO 就绪队列，依赖（生产者结束 + 跨设备传输完成）
  满足后入队执行，端到端时间 = 最后完成时刻。

## 内存模型

- 张量生命周期：**产生（融合组开始）→ 消费（消费者组结束 / 跨设备传输完成）→ 释放**；
- 融合组内中间张量不落显存；
- 跨设备张量：源端保留到传输完成，目的端在传输完成后分配接收缓冲，
  消费者组结束时释放（同一时刻两台设备各持有一份）；
- 参数张量全程驻留所在设备；
- 逐设备累计峰值显存，超出 `max_mem` 标记溢出（供校验器做显存修复）。

## 硬件参数清单

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `DeviceSpec.flops` | 每台设备算力（FLOPs/s），支持异构 | 必填 |
| `DeviceSpec.mem_bw` | 显存带宽（Bytes/s） | 必填 |
| `DeviceSpec.max_mem` | 显存上限（Bytes） | 必填 |
| `HardwareSpec.comm_bw` | 共享总线带宽（Bytes/s） | 1e12 |
| `HardwareSpec.comm_latency` | 每次传输延迟（s） | 1e-6 |
| `HardwareSpec.launch_overhead` | 每个 kernel 启动开销（s） | 0 |
| `HardwareSpec.fusion_overhead` | 融合组内每多一个算子的额外开销（s） | 0 |
| `HardwareSpec.n_devs` | 设备数量（由 devices 推导） | — |

## 打标签（Stage 0）

```python
from simulator import OpGraph, OpNode, TensorEdge, HardwareSpec, DeviceSpec, generate_label_dataset

# 构建图（节点/边属性固定，extra 预留扩展字段）
g = OpGraph(name="demo")
g.add_node(OpNode("A", op_type="matmul", flops=2e12, out_bytes=512e6, param_bytes=256e6))
g.add_node(OpNode("B", op_type="gelu", flops=5e8, out_bytes=512e6))
g.add_edge(TensorEdge("A", "B"))

hw = HardwareSpec([DeviceSpec(flops=3e12, mem_bw=2e12, max_mem=16e9)])

records = generate_label_dataset([g], hw, "labels.jsonl", plans_per_graph=16)
```

每条记录：`graph`（图结构）+ `plan`（融合组 + 设备）+ `makespan` +
`peak_mem`（每设备）+ `mem_overflow`。这批数据即代价模型的训练集。

## 扩展点

- **关键路径**：`SimResult.op_intervals` 已含每个算子的 (start, end, device, group)，
  可直接在其上计算关键路径长度，供后续"关键路径图搜索"使用；
- **异构校准**：`DeviceSpec` 天然支持异构，接入国产卡实测微基准（算力/带宽/延迟）
  即可做国产卡代价模型校准（综述盲区 4）；
- **动态网络**：`HardwareSpec.comm_bw` 可在训练中途变化，仿真器按传输时刻读取，
  可用于动态拓扑场景。
