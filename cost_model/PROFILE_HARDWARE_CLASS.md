# BW1000 同构硬件成本复用

2026-11-07。3B静态契约查询按硬件类别匹配，GPU1允许迁移GPU0的Native/KERV局部成本。

## 指纹与来源

已有data/gpu1_cost_completion_20261106_01/raw/runtime.json及fused_runtime.json，经原review/evaluation.json的SHA核验：

| 字段 | GPU0 | GPU1 |
| --- | --- | --- |
| 硬件类别 | BW1000 | BW1000 |
| 实际设备名 | BW | BW |
| 架构与特性 | gfx936:sramecc+:xnack- | gfx936:sramecc+:xnack- |
| 显存字节 | 68702699520 | 68702699520 |
| Torch/HIP/驱动/构建/Triton | 匹配原测量栈 | 匹配原测量栈 |

hardware_class_key相同：sha256:80afda64fe08dae7d6cf0b5280a8d33bf62eb0262f1c5cf8826cd540ca7c4247。这是既有记录支持的硬件类别匹配，没有新查询GPU或采集频率/温度/功耗状态。

查询保留完整Shape [1,256,2048]、dtype、stride、布局、权重、Kernel实现/源码、软件栈及各自完整测量契约。仅从匹配条件排除布局证据来源及硬件物理编号/UUID；型号、架构特性或软件构建不同仍拒绝。通信方向与链路成本不合并。

原始accepted记录及物理设备Key不改。查询迁移层保留measured_on=0、reuse_on=1、reuse_kind=homogeneous_hardware_class_transfer；GPU1成本标为transferred_hardware_class或transferred_native_chain_hardware_class，target_device_independently_measured=false。没有创建GPU1实测标签，迁移精度未独立标定。

## 一次严格联合搜索

```bash
python -X utf8 tools/run_qwen25_static_search.py --output /fresh/qwen25_hardware_class_joint
```

原三模式实验不扩展，只重跑joint、seed0、64次实际评分。

- 可查询成本对：36位置×2设备=72；其中GPU0同设备复用36、GPU1硬件类别迁移36。
- 非法设备成本拒绝：32→0；本次64请求、64实际评分。
- GPU1融合成本查询378次，非378个不同方案或GPU调用。报告的查询计数包含run_mode返回摘要的CPU评分；最终方案在GPU0，故这些GPU1查询来自搜索候选。
- 最终36融合、687节点全部GPU0，651个组，预测25.239290006ms；与前次相同。
- 第1次评分已达到最终最佳，后续改善0。没有双卡真机收益或联合搜索优势结论。
- 58项相关CPU测试通过，覆盖硬件类别/架构/驱动不匹配、GPU1迁移来源、原GPU0测量身份和Native链导出。

data/qwen25_hardware_class_joint_v1保存hardware_class_audit、逐位置成本审核、执行声明、一次搜索报告/轨迹以及joint_placed.json和joint_fused_placed.json。原图fingerprint、成本引用、双输出和导出往返评分保持一致。

同型号迁移是搜索的初始成本依据。频率、功耗、温度与干扰差异尚未校准；需要更高预测精度时再独立处理，不作为当前搜索前置条件。