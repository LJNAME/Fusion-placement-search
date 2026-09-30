# configs：硬件参数配置

仿真器需要的全部硬件参数都在这里，按 JSON 配置加载，支持异构集群
（每台设备独立 `DeviceSpec`）。

## 快速使用

```python
from simulator import load_hardware, simulate

hw = load_hardware("configs/hetero_4cards.example.json")
# 之后把 hw 传给 simulate(graph, plan, hw) 即可
```

## 字段说明

### 集群级（顶层）

| 字段 | 含义 | 单位 |
| --- | --- | --- |
| `name` | 集群名 | — |
| `comm_bw` | 共享总线带宽（所有跨设备通信串行共享） | Bytes/s |
| `comm_efficiency` | 通信效率系数（默认 1.0），有效通信带宽 = `comm_bw` × 该系数 | — |
| `comm_latency` | 每次传输固定延迟 | s |
| `launch_overhead` | 每个 kernel（融合组）启动开销 | s |
| `fusion_overhead` | 融合组内每多一个算子的额外开销 | s |
| `devices` | 设备列表（见下） | — |
| 其他字段 | 任意扩展字段（说明、来源等），加载后存入 `extra` | — |

### 设备级

| 字段 | 含义 | 单位 |
| --- | --- | --- |
| `name` | 设备名（建议用厂商型号，如 HYGON-K100-AI） | — |
| `flops` | 算力（注意：取你训练实际用的精度，如 FP16/BF16） | FLOPs/s |
| `mem_bw` | 显存带宽 | Bytes/s |
| `max_mem` | 显存上限 | Bytes |
| `efficiency` | 效率系数：`{"flops": 0.7, "mem_bw": 0.7}`，有效值 = 标称 × 系数 | — |
| `class` / `dtype` / `source` / `calibrated` | 扩展字段（厂商类别/精度/数据来源/是否已实测校准） | — |

加载时自动折算：`DeviceSpec.flops = flops × efficiency.flops`，
`DeviceSpec.mem_bw = mem_bw × efficiency.mem_bw`，
`HardwareSpec.comm_bw = comm_bw × comm_efficiency`。
标称值与系数会保留在 `extra` 中（`nominal` / `efficiency`），不丢来源信息。

## 无真卡阶段：用系数替代（当前做法）

当前没有真实国产卡可跑微基准，配置里给的是**官方标称值 × 假设效率系数**：

| 设备 | 标称 flops | 效率 | 有效 flops | 标称带宽 | 效率 | 有效带宽 |
| --- | --- | --- | --- | --- | --- | --- |
| 海光 K100 AI | 196T | 0.7 | ~137T | 896 GB/s | 0.7 | ~627 GB/s |
| 沐曦 C500 OAM | 280T | 0.75 | 210T | 3.35 TB/s | 0.75 | ~2.51 TB/s |
| 天数 BI-V150 | 220T | 0.7 | 154T | 1.1 TB/s | 0.7 | 770 GB/s |
| 摩尔线程 S4000 | 25T | 0.6 | 15T | 768 GB/s | 0.7 | ~538 GB/s |
| 共享总线 | 50 GB/s | 0.6 | 30 GB/s | — | — | — |

这些系数是**占位假设**，不是实测。开发期结论必须配合敏感性分析使用
（在合理区间扫描系数，确认最优方案排名不翻转）。

## 四类国产卡起始值（hetero_4cards.example.json）

文件里给的是**官方标称规格**，只适合做初值/占位，不能直接当真实性能用：

| 设备 | flops（FP16） | 显存带宽 | 显存 | 备注 |
| --- | --- | --- | --- | --- |
| 海光 HYGON K100 AI | 196 TFLOPS | 896 GB/s | 64GB HBM2e | FP32 矩阵约 49T |
| 沐曦 MUXI C500 (OAM) | 280 TFLOPS | 3.35 TB/s | 64GB HBM2E | PCIe 版约 240T |
| 天数智芯 ILUVATAR BI-V150 | 220 TFLOPS | ≥1.1 TB/s | 64GB HBM2e | FP32 ≥56T |
| 摩尔线程 MTT S4000 | 25 TFLOPS（向量） | 768 GB/s | 48GB GDDR6X | TF32 张量 50T |

标称值普遍虚高：真实可达算力通常只有标称的 60%–80%，显存带宽受访存模式影响更大，
通信带宽/延迟必须实测（PCIe 与厂商自研互联差异很大）。

## 校准流程（综述盲区 4：国产卡代价模型校准）

每个参数对应一个微基准：

| 参数 | 实测方法 | 工具示例 |
| --- | --- | --- |
| `flops` | 大尺寸稠密 GEMM / roofline 测试 | rocBLAS、厂商 SDK |
| `mem_bw` | 大数组 copy / Stream triad | 自写 kernel |
| `comm_bw` | 双卡 ping-pong、环形 all-reduce | MPI / 厂商集合通信库 |
| `comm_latency` | 小消息 ping-pong | MPI |
| `launch_overhead` | 空 kernel 连续启动计时 | profiler |
| `fusion_overhead` | 同一算子对跑融合/未融合两版拟合差值 | profiler |

校准闭环：

1. 在真实集群上跑一组已知方案，记录真机 makespan / 显存峰值；
2. 用同一组方案跑仿真器，比较预测 vs 真实；
3. 按误差调整 `flops`（等效效率系数）和 `mem_bw`，必要时按算子类型细分；
4. 把校准后的数值写回配置，并把 `calibrated` 置为 `true`；
5. 保留校准日期与机型信息（可放 `extra` 或 `_说明` 字段）。

厂商运行环境：海光走 HIP/ROCm（rocprof、rocBLAS）；沐曦走 MACA；
天数智芯走 Corex；摩尔线程走 MUSA。性能数据要用各自生态的 profiler 采。

## 形状相关效率曲线（`shape_model`）

仿真器最初用的是"峰值 roofline"：每个融合组的时长 =
`max(计算量 / 峰值算力, 边界 IO / 峰值带宽) + 启动开销`。这与
`cost_model/roofline.py` 的解析式基线完全同构，因此解析式基线在仿真器上
几乎满分是"循环论证"，没有信息量。

为此加了 `shape_model`：让仿真器的有效算力取决于算子类型和输入形状。
目前只对 matmul 做一条饱和曲线——

```
有效算力系数 eta = eta_max * K / (K + k_half)
```

其中 K 是 GEMM 的规约维。K 越大，算术强度越高，越接近峰值算力；小 K 的
GEMM 访存受限，有效算力显著低于峰值。于是仿真器与解析式峰值 roofline
系统性分离，学习式代价模型（Ridge / GNN）有机会学到形状修正、在预测精度
和方案排名上真正超过解析式基线。

**字段（顶层 `shape_model`，可选）**

| 字段 | 含义 |
| --- | --- |
| `shape_model.<op_type>.compute.curve` | `const`（默认）或 `saturate_k` / `saturate_bytes` |
| `shape_model.<op_type>.compute.eta_max` | 曲线饱和上限（有效算力系数 ≤ 1） |
| `shape_model.<op_type>.compute.k_half` | 饱和曲线的半饱和 K（`saturate_k`） |
| `shape_model.<op_type>.compute.bytes_half` | 半饱和字节数（`saturate_bytes`） |

未列出的算子类型默认 `eta = 1.0`（无形状修正）。这些曲线参数和
`efficiency` 系数一样是**占位假设**，拿到国产卡后按"校准流程"逐算子用
微基准拟合，把曲线参数写回配置即可，代码不用改。

**文献依据**：Roofline（CACM 2009）是理论上界；nn-Meter（MobiSys 2021）
用 kernel 级形状相关预测优于解析式；Habitat（USENIX ATC 2021）做跨 GPU
逐迭代运行时预测；IEEE TPDS 2024 综述系统比较了解析式与学习式性能建模。
