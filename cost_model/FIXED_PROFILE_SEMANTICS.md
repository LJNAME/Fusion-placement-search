# 固定Profile计费与覆盖

当前主线使用固定Profile，不调用主动校准。工程解析模式与实测标签模式分别声明，不能把形状命中、同契约准入和正式融合能力当成同一件事。

## 两个入口

- --profile analytic：只读硬件快照，全部执行组/链路由既有解析公式预测；不读取raw标签，不需要allow-unbound，不接受测量契约准入参数。硬件参数仍是固定假设，不能称当前GPU真值。
- --profile measured（默认）：读固定raw表；Kernel标签须accepted且契约一致。--allow-unbound显式进入旧表低置信诊断模式，不能升级标签或部署资格。当前默认36条可用Kernel记录全为legacy_unbound，严格搜索会报错。

解析逻辑input单节点为零时间，与MeasuredHardwareProfile一致；它不应承担启动/访存费用。embedding当前仍有解析代价，GroupKernelKeys却排除其Key；这项覆盖口径需在正式组级Profile接入前明确，不自动把embedding视为零时间。

## 计费表

| 成本项 | 固定实测/诊断形状表 | 解析组/回退 | 排程消费方式 |
|---|---|---|---|
| duration | 使用标签总duration；插值另见下文 | max(compute,memory)+launch+按组内算子数计的fusion项 | 每组只消费duration一次；可再受显式干扰系数影响 |
| launch | min(硬件launch,duration)只是从总值拆出的展示字段 | 每组加一次硬件launch | 不把launch_time再加到duration |
| compute/memory字段 | compute=duration−launch、memory=0不代表物理纯计算/无访存 | FLOP效率模型和逻辑边界字节/带宽 | 原始分量用于解释，不将其相加替代duration |
| fusion额外项 | 不额外叠加到实测总值 | fusion_overhead×(组内节点数−1) | 当前常数0是模型假设，不是实测融合无额外成本 |
| link | 查方向消息表及插值/外推后的总时间 | bytes/bandwidth+comm_latency | 在依赖/链路时间线上计入，不再给消息表加comm_latency |
| host | 没有独立host组件，旧未绑定标签可能含harness影响 | 没有Python/driver提交模型 | 不把缺字段解释为真实host零成本 |
| 显存 | 参数、激活、通信buffer及workspace的结构估算 | 同左 | 约束和峰值，不是额外耗时项或allocator实测峰值 |

KeyedKernelProfile若将来命中完整组Key实测条目，也直接使用duration，launch字段为0；当前固定主入口是MeasuredHardwareProfile形状表路径，不是这个组Key路径。

当前硬件快照：launch7.770405267μs、fusion0μs、interference0、overlap_send=false。未因graph微测修改常数，未引入cache/pipeline修正。makespan由依赖和设备/链路时间线得到，不能用computation_time+communication_time+launch求总时间；前两项分别为原始组duration和消息duration之和，不等于关键路径耗时。

## 当前覆盖

审计枚举每图的本地合法候选×两个设备，不穷举完整放置图；planned候选只为开发。数据见data/fixed_profile_audit_v1。

| 尺度 | 候选×设备 | 逻辑input | 诊断精确形状命中 | 诊断近邻插值 | 解析回退 | 严格Kernel查询拒绝 |
|---|---:|---:|---:|---:|---:|---:|
| small | 12 | 0 | 10 | 0 | 2 | 10 |
| layer | 44 | 2 | 40 | 0 | 2 | 40 |
| model | 1238 | 56 | 1122 | 0 | 60 | 1122 |
| model_3b | 1446 | 72 | 288 | 144 | 942 | 432 |

诊断形状命中都不代表accepted。未绑定完整实现版本、runtime及layout/strides；本次生成的unbound_template_hash仅辅助去重，不重算或替换任何旧测量身份。各图独立模板数12/38/40/36，不能据此直接安排采集。融合planned组在当前形状表无实测分支，全部回退解析。

3B近邻MatMul采用requested_FLOPs/measured_FLOPs乘原总duration，包含固定开销在内的总值一起缩放，并非只缩放纯计算；这是当前模型假设，后续覆盖评估需单列。链路两方向各6点、4KiB–256MiB，本次图边消息尺寸查询均未外推；没有外推不代表契约已确认。

## 尚未解决的边界

1. Kernel闸门仅覆盖Kernel标签。链路表经source_id/方向检查，但不通过MeasurementLabelLookup；不能宣称整个实测Profile已统一执行契约。当前不扩闸门、不伪填契约。
2. GroupKernelKeys排除embedding，但形状Profile给它正的解析代价；model和model_3b各有2个此类设备使用。需先定义其角色，不能误算实测覆盖。
3. 加性组duration不能表达所有组合上下文；既有两/三核诊断不支持通用邻接修正。正式Profile须声明目标执行语义，当前下游仍未确认。
4. 正式融合能力文件后置；当前planned节点不等于KERV或下游实际可执行Kernel。不把所有解析缺口直接变成GPU任务。

## 重放

```bash
python tools/audit_fixed_profile.py --out-dir data/fixed_profile_audit_new
python -m cost_model.search_evaluation --profile analytic --scales small,layer \
  --mcts-iterations 16 --seed 23 --output data/analytic_new/search.json
```

主系统解析基线见data/mainline_analytic_v1，两次重放和严格旧标签拒绝记录均保留。SROP源/验证保持冻结；依赖旧共享源码hash的复验仍用90f623c，不更新旧记录追新源码。
