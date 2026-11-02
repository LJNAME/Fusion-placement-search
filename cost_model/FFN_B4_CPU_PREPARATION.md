# B4 · CPU协议与身份导出

priceable_plans=0/48，新accepted=0，GPU入口关闭。本次只生成CPU产物，未改v3采集器、旧Profile或准入决策。

产物：data/ffn_b4_cpu_preparation_v1/profiles.json、scope_manifest.json、anchor_protocol.json、verification.json、cpu_checks.json。

## run05导出

9条quality_passed均导出Canonical Key及原始contract，沿用fusion-group-kernel.v1和GroupKernelKeys，不新建Key哈希算法。每条保留原始报告/样本/Trace/PlacedOpGraph证据哈希及record_id；FFNExportStore.load需要原始run05目录，核验文件并从原始行重新生成Key，拒绝不一致导出。

| 类型 | 条数 | 完整身份精确检查 | 当前用途 |
|---|---:|---:|---|
| Singleton | 5 | 4 | 身份检查，未开放成本准入 |
| KERV Fused | 1 | 1 | 选定组身份检查，旧P2决策不复用 |
| Native链 | 1 | 1 | comparison_only，禁止充当SiLU/Mul单算子 |
| Copy | 2 | 2 | 已观察driver-managed copy接口身份，禁止声称认证直连P2P |

完整身份指本导出schema所需的已记录边界Tensor与运行时字段；不等于所有性能条件已审核或Profile已准入。未记录的条件不从本机CPU环境补值。所有记录保持conditional_dev，严格成本查询9/9拒绝；8条完整身份可做精确audit inspection，不提供CostModel代价。

Canonical Key保留实际GPU UUID/架构、Torch发行版/构建与源码哈希、HIP/Triton/驱动、实现id/version、已记录flags、Shape/dtype/stride、alias语义。内存地址只做原始别名校验，不进入Key。Gate0与Up0沿用现有Factory的保守semantic_slice区分，因此本次9个Key不同；没有为了凑唯一ID把节点名或record_id加入Key，也没有强行合并同Shape记录。

接口：FFNExportStore.inspect_exact(key, contract_id, usage=...)仅做对应类型的身份检查；query_cost复用MeasurementLabelLookup严格闸门，当前全部拒绝。改Shape、dtype、stride、实现版本、UUID、flags或contract均不能错误命中；把accepted字段重哈希也不能获得准入权威。

## Mul的实际缺口

run05的Mul函数体使用left.unbind(0)，left来自预计算SiLU结果；身份字典却为所有激活臂统一记录gates和up。代码见calibration/ffn_b4_runtime.py:226、:253、:274。gates不是Mul实际左输入的Tensor观测，不能把它的连续布局直接搬给left。

因此gated_mul@gpu1的真实左输入保留layout/strides unknown，导出identity_gaps含mul_left.runtime_tensor_metadata_missing；完整身份查询拒绝。已通过的数值/时间质量判定不修改。现有数据不伪补；后续采集器必须记录实际left的元数据，或另做明确的源声明适用范围审核，不能以本机Tensor代替。

Native链内部SiLU临时量也未观测，Key明确标native_intermediate_not_observed，不宣称其contiguous；该记录仅按已观测的组合边界和执行实现做比较身份检查，不冒充单Kernel或两个singleton标签。Fused内部表达式不伪造中间物化Tensor。

Copy的两端UUID、传输方向、大小、实际源/目的/输出布局均绑定；输出与预分配目的缓冲的别名已核对。physical_route=not_certified保留，peer/DtoD Trace不升格成物理P2P路由认证。

## 冻结的锚点控制协议

协议名ffn-anchor-control.v1，pilot_only_not_profile_admission；不是已上线的B4 execution.v4。calibration.ffn_anchor_control_protocol只提供协议和CPU端点汇总校验，没有GPU collector，gpu_collector_ready=false、gpu_entry_open=false。

| 固定执行顺序 | 负载 | 每端点Replay |
|---|---|---:|
| 1 | Down0 | 1 |
| 2 | Up0 | 3 |
| 3 | Up0 | 1 |
| 4 | Down0 | 3 |

父进程只做CPU工作、不保留Smoke runtime；每case新进程、只在GPU0执行负载，只启动一次。这是新pilot的共同驻留条件，不与run05混比；现场重新捕获UUID/build并要求4项一致，不要求与run05物理卡相同，也不按卡型号复用旧测量。BF16、目标/锚点每Replay50调用、各10预热、固定7轮、不补轮不替换、原buffer/恢复/校验顺序、锚点预热仍在case准备之前，均保持v3定义。不能同时把预热位置挪到case准备后，避免混入第二种物理操作变化。

R1直接使用单值；R3使用全部3次的median，inclusive IQR/median>10%使端点失效。R1点内离散度为null/not-estimable。前后median漂移>10%使该轮失效；7轮必须全有效，目标IQR/median仍<=10%。这是预注册工程阈值，尚未验证为统计保证。R3端点策略包含重复、汇总和可估计的点内离散度规则；不能据结果单独归因哪一项起作用。

每个repeat和目标测量必须记录perf_counter_ns：reset_begin、timing_begin、event_wait_end、all_device_sync_end、validation_end，另记录case准备和anchor warmup结束时间。恢复/校验仍在Event外，Event内不写文件。尽量取得带来源/时间/单位的频率、利用率、温度；取不到写null和错误，不填正常值，环境因果结论保持inconclusive。

所有目标和锚点计时完成后才做Trace，此后不再计时。4项管道、身份、数值、时间记录均完整且两个R3 case通过预注册质量检查，才满足后续资格；R1质量失效原样保留，不重跑或事后选另一个协议。7项单卡核心矩阵仍待单独冻结和实现，后续还必须先解决Mul实际操作数元数据。v3/新协议数据不混用，旧9失败不重判。

## CPU验收

47项测试通过，覆盖新协议/导出以及现有Reference/KERV回归；四个派生产物重复导出逐字节一致，cpu_checks.json记录源码哈希。实际run05原件全文件SHA复核，未调用GPU。

```powershell
python -m calibration.prepare_ffn_cpu_stage /path/to/ffn_b4_run_20261102_05 --output /fresh/cpu_preparation
```

输出目录必须全新。原始证据不复制进新包，加载时显式传入run05根目录；bundle中的相对路径与哈希用于精确定位。普通身份检查不等于准入，也没有接入搜索器或解析回退。
