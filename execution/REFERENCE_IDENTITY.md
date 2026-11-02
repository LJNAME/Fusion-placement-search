# Reference Singleton Identity · B0

正式阶段顺序、三种成本语义与测量入口见 `cost_model/STAGE_B_PROTOCOL.md`。

B0 本机 CPU 验证通过；BW1000 的执行环境身份尚未在服务器重新观测。无新时延 Profile、无准入升格、无 GPU 测量。

## 实现绑定

| 图中算子 | 注册键 | implementation.id | 实际 Reference 入口 |
|---|---|---|---|
| GEMM | unfused:matmul | reference.torch.matmul | flatten → torch.matmul → reshape，weight常驻 |
| SiLU | unfused:silu | reference.torch.nn.functional.silu | F.silu，非原地 |
| Mul | unfused:mul | reference.torch.Tensor.__mul__ | inputs[0] * inputs[1]，非原地 |

singleton 的 pattern_id 都是 unfused；Canonical Key 工厂优先选择 `pattern_id:op_type` 绑定，未提供时保留原 pattern 绑定行为。KERV 多节点 Key 和既有 P2 决策不变。（`cost_model/fusion_kernel_profile.py`）

`observe_reference_runtime` 从正在加载的 Torch 读取：distribution/module版本、Torch git revision、构建配置、执行设备、线程/数值策略/grad标志，以及实际 Reference executor、Torch extension、functional模块的源码文件哈希。id和version来自当前加载环境，不从 capability 或旧 GPU报告抄入。绝对路径只作 provenance，不进入实现哈希；文件内容变化会改变 Key。

`ReferenceKernelKeys.verify_runtime` 在描述 Key 前重新核对执行环境；版本、绑定、标志或设备变化会拒绝。torch.matmul/Tensor.__mul__ 被替换、F.silu不再是所声明入口时也拒绝。调用者必须在实际执行上下文中获取身份，例如 no_grad 的采集不能借用 grad_enabled=true 的 Key。（`execution/reference_identity.py`）

这绑定的是 Torch Reference 实现，不宣称识别了底层 ROCm GEMM算法，也不提供 KERV dispatch 证据。GPU测量时仍需实际环境与执行路径验证。

## 布局来源

`tensor_observation` 读取实际 Tensor，不从 Shape 推断连续性：

- shape、dtype、layout、stride；
- layout_provenance.kind=runtime_tensor；
- 观测设备和 observer。

`with_tensor_layouts` 在图副本中绑定观测，检查与原图 Shape/dtype 一致；缺少观测时保留未知。显式图声明允许 kind=graph_declaration，但必须有 source_id。当前 Builder 没有布局声明，不能凭此将两卡 GPU Tensor 填为 contiguous。

身份门禁同时检查输入、输出和 GEMM weight；未知布局、缺来源、连续性与stride冲突、设备来源不一致都拒绝。非连续 Tensor 记录为 strided，不做隐式 contiguous copy。CPU来源不能用于GPU，GPU0来源不能用于GPU1。

## 本地验收

`python -m calibration.close_reference_identity` 只执行一次本机 Reference FFN 以观测 Tensor，不计时。第0层子图为1个边界输入+5个计算节点，读取6个激活 Tensor 与3个weight布局。三个GEMM沿 `_ReferenceExecutor._matmul`，SiLU/Mul沿 `_execute_node` 原路径执行。

当前本机 Torch distribution=2.9.0、module=2.9.0+cpu，5个 singleton Key 的 identity_gaps均为0，JSON往返及重新构造后Key一致。输出在 `data/reference_identity_b0_v1/identity.json`；CPU硬件数值为identity-only schema占位，不能作为性能预测或BW1000校准数据。

严格测试包含：完整身份+accepted单元夹具通过、conditional与错Contract拒绝；缺id/version/source、未知布局、缺来源、stride冲突、缺weight、Shape/dtype不符、实现替换、运行上下文变化、错误实现/布局Key不命中。测试中的accepted耗时仅为单元夹具，不写入正式Profile。

身份完整不绕过 MeasurementLabelLookup：未实测、未审核或契约不同仍拒绝。当前 Search Cost Coverage仍是0，两个P2配对比较Case单独计Evidence Coverage。

## GPU入口状态

`gpu_identity_closed=false`、`gpu_measurement_entry_open=false`。当前本地结果不能证明服务器加载了相同Reference实现；BW1000应在实际环境读取版本/源码/设备及Tensor或显式图声明的布局，再通过相同检查。实际捕获安排在B4计时前；B1合法动作空间、B2跨卡边并集、B3配对矩阵也都未完成。

GPU0/1各自的实现/运行环境身份应独立保留。原始Builder draft仍有缺项，这是缺少实际观测，不用CPU Key将其填平。B0～B3全部通过前不补测，不生成搜索结果。

重放：

```powershell
& 'C:/Users/LJ/miniconda3/envs/fusion-dev/python.exe' -m calibration.close_reference_identity
& 'C:/Users/LJ/miniconda3/envs/fusion-dev/python.exe' -m pytest execution/tests/test_reference_identity.py -q
& 'C:/Users/LJ/miniconda3/envs/fusion-dev/python.exe' -m tools.audit_search_subgraph_cost
```
