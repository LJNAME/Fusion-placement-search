# 测量契约与标签准入

Kernel Key及现有object_id/kernel_id不改。测量契约属于标签层，由`cost_model.measurement_labels.measurement_contract_id`调用既有`content_id("measurement-contract", contract)`生成ID。

## 标签结构

```json
{
  "key_id": "原ID",
  "measurement_contract_id": null,
  "statistics": {"median_us": 0, "mad_rel": null, "n_rounds": null, "n_calls": null},
  "admission": "legacy_unbound"
}
```

上例是结构示意，实际median_us必须为正。旧记录不改写，旁路标签保存原ID与统计；缺失契约写null及legacy_unbound，不凭现有数字猜测执行语义。已有Key中的implementation.measurement_contract是旧物理调用定义，保留原字节；新增的measurement_contract_id不进入Key或object_id哈希。

契约JSON包含executor、submission_included、allocation_included、granularity、unit及environment；environment含torch/hip/triton版本与device_model。值需明确声明。实际部署方式尚未确认，不把graph诊断的契约登记成正式下游契约。

## 唯一读取规则

两个模型入口均调用MeasurementLabelLookup.lookup(key_id, contract_id)，包括旧shape表的MatMul插值：

- accepted且契约ID匹配：返回。
- legacy_unbound、conditional_dev、缺失或不匹配契约：默认抛MeasurementContractError，不做解析或旧值的静默回退。
- 显式allow_unbound=True：发出RuntimeWarning，返回诊断值，source带diagnostic_unbound、OOD记录原因，confidence降到解析回退级别（单核≤.25、融合≤.15）。

synthetic E0数据仍是合成机制测试，不作为实测标签。旧shape表没有canonical组Key时，只生成内部legacy-shape-record查找地址，不把它升级成完整组级Key；原有key_id/object_id/kernel_id存在时原样采用。

KeyedKernelProfile逐次检查当前条目；SROP替换标签后不能沿用旧的已准入元数据。inventory也走同一读取规则计算measured_covered，不把conditional_dev计入完整实测覆盖。

现有搜索CLI默认严格；历史诊断可显式加`--allow-unbound`，报告同时带低置信度/OOD。不能用诊断模式结果声称正式准入。

## CPU机械检查

```powershell
python tools/verify_measurement_contract_gate.py --out-dir data/measurement_contract_gate_new
```

本地传入旧formal/verified_profile.json时还可加`--group-profile 路径`。工具导出旁路状态，核对112条E2原ID及可选12个group slot的canonical Key；检查原文件SHA未变；调用现有search_evaluation，要求非零退出且不生成结果。不可覆盖既有输出目录。

## graph诊断命令（scnet，由用户执行）

实现对应工作区graph预登记v1.1；单stream、50独立gate、仅保留末次未融合输出、7轮配对。GPU回传前不声称capture支持或融合收益。

```bash
cd ~/private_data/LJ/Fusion-placement-search
git pull --ff-only
GRAPH_OUT="$PWD/data/silu_mul_graph_$(date +%Y%m%d_%H%M%S)"
bash ~/private_data/LJ/dcu-kernel310-20261031/run_python.sh \
  -m calibration.probe_kerv_graph \
  --flagscale-root ~/private_data/LJ/FlagScale-main \
  --out-dir "$GRAPH_OUT"
echo "$GRAPH_OUT"
```

主字段graph_batch_amortized_per_call_us是批次重放摊销时间，不是单次隔离延迟。图内每次调用的Python派发被捕获；一次graph提交边界仍在Event内，故submission_included=true。allocator工作在capture而非replay计时内，allocation_included=false，GPU输出写入仍计时。

输出plan/environment、逐case partial/result。异常保存failure.json，包含完整traceback、失败阶段/case/方法、capture_attempted、attempted_pool_config、capture前及失败时可获得的allocated/reserved快照；无法读取的快照标不可用。无GPU只会记录环境失败，不能当作HIP capture失败证据。捕获失败不退回eager，不自动重试。

回传整个输出目录，含失败和已完成case。graph输赢不触发标签accepted；结果默认conditional_dev，无模型自动回填。旧组级采集草案仍锁定旧源码，新源码下load_draft会拒绝它；不重写旧草案哈希来绕过检查，本次graph入口不依赖该草案。

## launch与fusion参数的含义

当前解析路径按组使用max(compute,memory)+launch_overhead+fusion_overhead×额外组内节点数。7.770405μs是旧解析校准参数；fusion_overhead=0表示该项未额外调整，不证明融合额外成本不存在。

GroupCost.duration是排程使用的总成本；launch_time等字段是解释字段，不能再加一次。Keyed实测条目直接使用测量总值、launch_time=0；旧Measured精确命中只从duration拆出launch字段，duration未增加。当前路径没有已证实的实测加launch重复计费。

下游契约确认后，先明确参数到底代表GPU启动、host提交还是其他固定服务项，以及和实测总值/重叠的关系，再决定标签准入。不能直接将参数改成63μs/21.7μs，也不从现有值扣13μs。涉及下一次参数修改时重新核对计一次的约束；本次不改变两个数值。
