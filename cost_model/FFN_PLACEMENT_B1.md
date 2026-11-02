# FFN Placement · B1

入口 `L000.pre_mlp_norm` 与出口消费者 `L000.residual_add_2` 均固定GPU0（用户确认）。五个FFN计算节点自由放置，KERV组不可拆分。来源为591节点Qwen3 Builder的第0层参考子图，不是运行时捕获图。

## T1：合法方案

| 划分 | 放置单元 | 带GPU标签方案 | 计算分组结构置换类 |
|---|---:|---:|---:|
| Native | 5 | 32 | 16 |
| Fused | 4 | 16 | 8 |
| 合计 | — | **48** | **24** |

`T1_legal_placements.json` 保留全部48个方案的融合状态、分组、GPU赋值、partition_id及结构类，不合并评分。实际成本评分调用为0；48表示后续需要保留的带标签待评分方案数。

24个结构类只对计算分组做0↔1置换，不包含固定GPU0的入口/出口。全局置换会破坏边界约束；只置换计算节点也会改变进出通信。历史局部Fused耗时GPU0/1分别7.251179814/7.232000232 μs，尚无完整成本对称证书。因此没有将48个方案压缩成24个评分对象，也不共享两卡Profile。

每个节点都能到达GPU0和GPU1；允许分组为5个singleton及1个SiLU×Mul组。枚举调用现有 `LegalFusionSearchSpace.actions/apply/to_plan`，不调用评分、Exact优化器、MCTS或模拟器；独立的两划分笛卡尔积用于检查无遗漏/重复。

已启用约束：覆盖、不重叠、DAG/拓扑、层内pattern、显式source-pinned KERV能力、形状/别名guard、设备范围、边界与可选节点设备限制。未宣称通过运行时布局、HBM峰值/workspace或Stage/Stream执行验收。KERV连续布局是计时前必须满足的运行条件，不通过把未知布局填成连续来扩大合法域。

## T2：逐方案边

每个方案记录6条子图边和1条出口边，共 **336行**；其中跨卡边实例 **160**。按计划ID与Edge ID可追溯生产者/消费者、两端GPU、Shape、原图逻辑bytes、实际transfer_bytes及融合内部边状态。

SiLU→Mul逻辑边原大小1,572,864 bytes：

- Native中0→1和1→0各有 **8个** 跨卡见证，共16个。
- 16个Fused方案中该边均为组内表达式，materialized_tensor=false，transfer_bytes=0。

T2保留被消除边的逻辑大小用于对照，不能将它当成真实传输。相同源Tensor的多个边实例也不直接等于多个物理复制或计费次数；复用/合并需后续明确执行契约。

## T3：自动生成的通信并集

| Tensor bytes | 方向 | T2跨卡实例 | 见证方案数 |
|---|---|---:|---:|
| 524,288（0.5 MiB） | GPU0→GPU1 | 48 | 36 |
| 524,288（0.5 MiB） | GPU1→GPU0 | 24 | 24 |
| 1,572,864（1.5 MiB） | GPU0→GPU1 | 44 | 34 |
| 1,572,864（1.5 MiB） | GPU1→GPU0 | 44 | 34 |

四项均由T2跨卡行自动归并，保存全部见证计划及边ID；并未在生成器中预填“四条”。去重包含大小、方向、dtype、layout、传输机制和同步语义；机制或同步不同不会被合并。

这是 **4种逻辑成本需求**，尚非4个可直接GPU测量的冻结契约。layout与传输实现保持未绑定，measurement_contract_id=null、measurement_ready=false。B2需核验实际机制、布局与同步契约是否要求拆分；本轮不开放通信成本准入。

## 产物与重放

`data/ffn_placement_b1/`：

- identity.json：图快照/哈希、能力与源码哈希、边界、约束、分组允许设备、节点可达设备及运行时前置条件。
- T1_legal_placements.json：两种划分与48个带标签方案、24个计算结构类、对称性限制。
- T2_cross_device_edges.json：逐方案全部7条原始逻辑边及传输判断。
- T3_communication_union.json：4种带完整见证的逻辑需求，B2契约仍待核验。
- verification.json：独立积空间核对、无重复/遗漏、T2与方案一致、T3精确重建、重复枚举字节一致。

```powershell
& 'C:/Users/LJ/miniconda3/envs/fusion-dev/python.exe' -m cost_model.ffn_placement_b1
& 'C:/Users/LJ/miniconda3/envs/fusion-dev/python.exe' -m pytest cost_model/tests/test_ffn_placement_b1.py -q
```

10项CPU测试通过，覆盖默认空间、边界变化、节点设备限制、禁止拆分的融合、未知/非法约束、T2/T3篡改、契约差异去重和重复结果。额外设备约束作为可选JSON由 `--restrictions` 传入；`--input-device/--output-device` 可生成其他边界场景，应使用不同 `--output` 保留各场景身份。

无新GPU测量、无成本评分、无搜索最优值，B2/B3未完成，B4仍关闭。B5.5后续可以在准入成本齐备后对这48个带标签方案全部做Before/After交叉评分；Before必须是目标Graph Replay契约下的解析模式，不沿用eager开销。
