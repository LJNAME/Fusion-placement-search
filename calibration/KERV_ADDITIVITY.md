# KERV graph可加性诊断 v1

当前任务仅为条件性开发计费检查，不改变代价模型、launch/fusion参数、Kernel Key或旧标签。四臂同次复测：SiLU、Mul、未融合链、固定FlagScale KERV融合组。结果不自动作为SROP纠错用例。

## 固定协议

- case顺序：GPU0 seq256、GPU0 seq1、GPU1 seq1、GPU1 seq256；shape=[1,seq,3072]、BF16、连续、seed1234。
- 每臂单stream graph内50次调用，使用50份独立gate视图、同一个up；每次replay前恢复输入并同步，恢复不进Event。
- Mul的gate输入为GPU上F.silu(original)的结果，在capture/replay计时外准备。它与链中Mul的数值输入一致；不能用原gate直接替代该中间值。
- 四臂分别capture，沿用原生private pool；每臂只保留末次最终输出，SiLU中间输出不累计保留。capture前用side stream做一次完整50调用预热；capture后每臂10次warmup replay，每次恢复输入。
- 正式8轮。下面4行顺序重复两遍；每臂在每个位置出现2次，轮内每一种有向相邻关系出现2次。

| 轮mod4 | 顺序 |
|---|---|
| 0 | SiLU、Mul、融合、链 |
| 1 | Mul、链、SiLU、融合 |
| 2 | 链、融合、Mul、SiLU |
| 3 | 融合、SiLU、链、Mul |

正式共6400次graph内函数体调用、128个Event区间、32个四臂观测轮。链的每次函数体含SiLU与Mul两个算子，不把这些计数冒充硬件kernel trace或独立统计样本。

协议与源码hash在第一个GPU操作前写入plan.json，protocol_id使用既有content_id。支持--plan-only生成无GPU计划。本仓库data/kerv_additivity_protocol_v1/plan.json是预先落盘的版本，不能在看到数据后改顺序或预算。

## 新契约与适用域

继承graph_replay的executor/granularity、throughput_per_call单位；一次graph提交边界可能进入Event，submission_included=true。allocator工作在capture、replay计时外，allocation_included=false，设备输出写入仍包含在计时内。

新契约加serial_plan_only=true，因此有新的measurement_contract_id；旧98e382f8…及原始数据不改。该字段描述本诊断的单stream串行调用域，不代表模型已经实现通用plan域检查，也不证明这些组能直接相加或覆盖并行联合搜索。

同case四臂共享契约、输入池、runtime、采样计数，但各自的临时分配/缓存状态并不相同。所有结果保持conditional_dev，不绑定或重算旧Kernel Key，不调用标签准入升级。

## 数值与显存记录

融合检查50份gate输出；其余三臂检查末次最终输出，明确其余49个最终输出不逐一断言。Mul/链的参考在GPU上预先生成；融合仍对FP32计算后转BF16参考检查。rtol=.01、atol=.001，up保持。

每个Event只包一次graph.replay，总时间除50生成graph_batch_amortized_per_call_us。提交/完成wall另报，不相减解释为纯host成本。

memory_before_numeric_checks在数值断言及up复制检查前读取；它记录截至timed replay结束的全局allocated/reserved及峰值，不能称pool专属容量。该时点修正了原graph工具在数值检查后取峰值的限制；原工具及已回传数据不改。

当前轮的计时先以numeric_passed=null保存partial，再做数值校验，成功后更新为true。异常保存failure.json，含完整traceback、capture阶段、尝试pool配置与可用显存快照；已有partial/result保留，不自动重试，不退回eager。

## 结果口径

每轮按相同四臂观测计算，保留正负号：

```text
Δ_add = C_silu + C_mul - C_chain
B_hat = C_silu + C_mul - C_fused
B     = C_chain - C_fused
B_hat - B = Δ_add
```

Δ_add>0表示孤立加和高估未融合链代价/融合节省；Δ_add<0方向相反。报告逐轮值、median、inclusive IQR、未缩放MAD、range及相对链代价的残差。逐轮报告加和与链比较是否给出相反选择，平局单列，不设事后剔除或强迫残差非零。

缓存复用、graph边界摊销、工作集及调度均为候选解释；此实验不宣称缓存因果，不拟合cache_reuse_discount，不给统计显著性结论。可加性误差与决策间隔结合解释，不能把所有差异归为SROP选点问题。

## scnet执行

```bash
cd ~/private_data/LJ/Fusion-placement-search
git pull --ff-only
ADD_OUT="$PWD/data/silu_mul_additivity_$(date +%Y%m%d_%H%M%S)"
bash ~/private_data/LJ/dcu-kernel310-20261031/run_python.sh \
  -m calibration.probe_kerv_additivity \
  --flagscale-root ~/private_data/LJ/FlagScale-main \
  --out-dir "$ADD_OUT"
echo "$ADD_OUT"
```

成功后可在原源码版本下复验，无GPU操作：

```bash
bash ~/private_data/LJ/dcu-kernel310-20261031/run_python.sh \
  -m calibration.probe_kerv_additivity --verify "$ADD_OUT"
```

verify复算协议/源hash、四case、采样顺序、计数、契约ID、数值状态与全部残差汇总。失败bundle输出complete=false并返回非零；不能将失败或仅计划目录当成采集成功。

回传整个ADD_OUT，包含plan/environment、逐case partial/result及任何failure。旧单核标签不混用；不将当前结果自动升级accepted，也不把本次串行代理结果宣称为下游真实执行质量。
