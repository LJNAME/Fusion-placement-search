# KERV Add＋RMSNorm图Adapter

2026-11-07。固定BF16 [1,256,2048]、epsilon=1e-6，入口integration/flagscale/kerv_add_rmsnorm.py。仅前向推理；不执行整图、不提供耗时或Profile。

## 映射

PlacedOpGraph中的(residual_add_1, pre_mlp_norm)经显式implementation绑定，调用固定源码adaptive_rms_norm.add_rms_norm_inference。第0层单组GPU Adapter调用已通过kerv_add_rmsnorm_adapter_01回传复核；其余35处仍只有图匹配和CPU映射证据。

| 图角色 | Tensor行为 |
| --- | --- |
| self_attention.out_proj | 可消费输入x；无组外消费者且未导出，执行开始前从live values移除 |
| input | 原始residual，只读 |
| pre_mlp_norm的weight | 调用者提供BF16 [2048]连续权重，只读 |
| residual_add_1 | 发布sum，复用被消费的x缓冲区；保留给residual_add_2等消费者 |
| pre_mlp_norm | 发布独立分配的normalized输出 |

不复制图输入；输入所有权属于消费式接口。调用者需要保留原输入时，应先在外部建立独立缓冲区，其成本不能隐去。底层wrapper的normalized输出分配保留，不声称零分配。若原地调用发生异常或输出契约错误，不发布两个结果；旧x映射也不恢复，避免暴露可能已被改写的数据。

输入角色由local_name、层和依赖确定，不依赖边列表顺序。必须显式同层/forward/Shape/dtype及2048维权重；图布局unknown可保留，真正执行时要求连续stride [524288,2048,1]，拒绝隐式转换、梯度、跨设备输入、x与权重或其他live Tensor的存储别名。调用在输入所在设备上下文中执行，调用者负责流、前后依赖和GPU生命周期。

固定KERV源码SHA与先前GPU探针一致；缺Triton或加载失败直接报错，无Native fallback。BF16中间舍入与aten不逐位等价，现有单Case GPU证据满足rtol=atol=0.02，未将其描述为通用误差界。

## CPU检查

新增31项映射/拒绝测试；连同原KERV、LightOP及探针测试共56项通过。正向执行使用CPU mock，不能算新增GPU Adapter验证。

```bash
python -m pytest integration/tests/test_kerv_add_rmsnorm.py integration/tests/test_kerv_execution.py integration/tests/test_lightop_add_rmsnorm.py calibration/tests/test_kerv_add_rmsnorm_probe.py -q
```

现有687节点Qwen2.5-3B图在新adapter_registry中匹配36组AddRMS；默认registry与原LightOP registry均保持原状，该图多节点候选仍为0。implemented表示本地Adapter已经实现，不表示GPU1、Capture、Profile或成本评分已完成。

data/kerv_add_rmsnorm_adapter_v1/placed.json为CPU映射参考：全部AddRMS合并且放GPU0，未运行搜索或评价成本。36组均成功绑定，序列化往返一致，每组两个图输出合计2097152字节。report.json记录逐组输入/输出与来源。真实模型生产者布局未因本次CPU检查变成已知。

## 调用方式

先构造Adapter，再以其registry加载PlacedOpGraph。融合组必须携带declared_binding()返回的implementation；参考placed.json已包含该注解。values只提供已执行生产者的真实Tensor，不提前填入融合组输出。

```python
from integration.flagscale.kerv_add_rmsnorm import KervAddRmsnormAdapter
from simulator.placement_artifacts import load_placed_op_graph

adapter = KervAddRmsnormAdapter(flagscale_root)
placed = load_placed_op_graph(placed_path, registry=adapter.registry)
bound = adapter.bind(placed, group_index, {0: "cuda:0"})
call = bound(values, norm_weight)
# values[add_node] 和 values[norm_node] 已发布，旧attention节点已移除。
```

已有实测只覆盖GPU0 gfx936；GPU1及其他硬件没有这条路径的验证证据。新Adapter不接入默认整图搜索，新增Profile为0，没有复用LightOP或其他宽度的耗时。第0层GPU Adapter执行回传见下节；Graph Replay和耗时仍未验证，下一项只处理此实现的实际成本来源。

## 最小GPU Adapter检查

沿用原探针，第四个参数adapter选择PlacedOpGraph→Adapter路径；不另开矩阵。只执行已有参考图第0层的(residual_add_1, pre_mlp_norm)，输入仍为同一BF16 Shape的合成值。三次重复沿用原数值/只读/别名检查，增加旧输入失效、两个原节点输出发布及residual保持live的检查；再采一次唯一Kernel trace。无计时、性能比较、Capture、Profile准入或搜索实验。

```bash
export FLAGSCALE_ROOT="$HOME/private_data/LJ/FlagScale-main"
export FUSION_PYTHON="$HOME/private_data/LJ/dcu-kernel310-20261031/run_python.sh"
bash tools/run_kerv_add_rmsnorm_probe.sh ../kerv_add_rmsnorm_adapter_01 run cuda:0 adapter
```

该入口的CPU计划位于data/kerv_add_rmsnorm_adapter_smoke_v1/plan.json；新增3项探针测试后，相关测试共59项通过。该CPU计划本身不构成GPU通过证据；实际回传结果见下节。原direct参数默认行为保留，旧直接Kernel回传不重测、不改写；旧collector身份对应85ed889/a622331时的源码，新入口有独立源码SHA。

## 2026-11-07 GPU Adapter回传复核

data/kerv_add_rmsnorm_adapter_01/raw保留6原文件，review/report.json有22项离线复核；包5526字节，SHA256 80fcf5c0db7182ed05731f7221f9f3f28e3ead499b5cf3d0c271110af5124b64。collector和Adapter均对齐项目aaae3d2的Git源码；KERV源码SHA、已有placed.json的SHA与implementation一致。

| 项目 | 结果 |
| --- | --- |
| GPU Case | GPU0 gfx936，BF16 [1,256,2048]，第0层AddRMS一组 |
| 映射 | group_index=12；L000.residual_add_1 / L000.pre_mlp_norm |
| 环境 | Torch2.9.0 / HIP6.3.26113 / Triton3.5.1 |
| 三次布尔检查 | 36/36通过，其中图映射检查15项 |
| 双输出发布 | Add和Norm均按原节点ID发布，旧attention输入失效，residual保持live |
| 数值max abs | 手写参考0/0/0；项目aten参考0.03125/0.03125/0.03125 |
| 容差 | rtol=atol=0.02，两参考通过；并非逐位aten等价 |
| 设备trace | 唯一_add_rms_norm_inference_kernel |
| Profile / 耗时 / Graph Capture | 0 / 未测 / 未测 |

Adapter单Case正确性已闭合，不再增加冒烟Case。原始GPU Tensor未归档，离线审核核对的是绑定源码的回执和设备trace，未在本机重算GPU数值。记录中的gpu_mapping_previously_verified=false为运行前冻结状态，不能据此否认本次执行结果；原始文件未改写。UUID变化不作排除理由。

GPU证据仅覆盖这一组和GPU0；36处图匹配不等于36次GPU验证，也不代表完整模型运行。真实Projection生产者布局和GPU1仍未验证。默认搜索配置未变；后续只需要补当前实现的必要成本来源，不再重复当前正确性实验。
