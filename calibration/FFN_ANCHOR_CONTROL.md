# B4 锚点控制采集器

入口 calibration.ffn_anchor_control，默认只写 CPU plan。当前 collector_implemented=true、gpu_validated=false；本轮 GPU 执行和新增准入均为 0。

冻结协议沿用 data/ffn_b4_cpu_preparation_v1/anchor_protocol.json，原文件及其来源不变。新实现身份、源码 SHA 和 plan_id 独立记录在 data/ffn_anchor_control_v2/plan.json。旧协议内 gpu_collector_ready=false 是上次冻结时的状态；当前实现状态读取新 plan，不重签旧协议。

## 四格控制实验

| 顺序 | 负载 | 每个锚点端点 Replay |
|---|---|---:|
| 1 | Down0 | 1 |
| 2 | Up0 | 3 |
| 3 | Up0 | 1 |
| 4 | Down0 | 3 |

每格一个新进程、只在 GPU0 执行；父进程不初始化 GPU、不保留 Smoke runtime。运行时不查询 peer，也不在 GPU1 建图、预热或同步。实际 GPU0 UUID、环境和 Torch build 在每格重新观察，父进程检查四格一致。

锚点 capture 和 10 次预热仍在 target 准备之前；target capture 后 10 次预热。正式固定 7 次尝试，每次 pre-anchor → target → post-anchor，每 Replay 含 50 次调用。保持原 fixture、恢复/校验位置、输出保留方式；不补轮、不替换。

R1 用单值，离散度为 null；R3 用全部三个值的 median，inclusive IQR/median 超过 10% 则端点失效。前后中位数漂移超过 10% 则该轮失效，七轮须全部有效，target IQR/median 不超过 10%。这些是冻结的工程阈值，尚未获得统计有效性证明。

这是锚点采样对照，不替代旧四类 Smoke。正的耗时较慢不会导致管道失败；锚点或 target 离散度越线标 quality_failed，仍继续固定四格。Capture、数值、不变量、Event、路径或身份失败则停止，后续标 blocked；进程崩溃也只启动一次，保留 worker.log、last_stage 和已有原始文件。普通异常另留 traceback、pool 和 GPU0 显存快照；父进程另留 parent_failure.json，包含自身 traceback、worker 故障日志及最后持久化 pool/显存。崩溃前快照标明时间范围，缺失写 unavailable，不伪造崩溃后观测。

## 时间与证据

每个锚点 repeat 和 target 留 reset_begin、timing_begin、event_wait_end、all_device_sync_end、validation_end，时钟为 perf_counter_ns。Event 只覆盖 Replay；恢复、同步完成后的数值检查在 Event 外。另留 anchor warmup 结束、case 准备起止时间。

频率、利用率、温度通过带来源的可用 Torch 接口尽力读取；失败写 null 和原始错误，不填 0。读取在 pre-anchor 前和 post-anchor 后，避免夹入本轮漂移窗口。环境因果结论保持 inconclusive，不能凭本实验唯一归因于机器或协议。

全部七轮和端点结束后才分别 Trace target、anchor，此后不再计时。原 samples.json 的 pending_post_timing_trace 不覆盖，result.json 单独附路径及 Trace SHA。父进程重新核对原始 Trace、样本哈希、契约、时序和质量结果，不仅检查 worker 退出码。

每格包含 identity.json、preparation.json、samples.json、两个 Trace、result.json、worker.log、parent_plan.json；全局 summary.json 保留 quality_failed 与 blocked。identity.json 同时绑定 measurement_contract_id，契约使用独立 pilot executor 和 protocol_id，不冒充 v3 或正式成本标签。

collector.v2 将两个结果分开：pilot_passed 只要求四格完成、无崩溃，锚点及逐阶段时间戳完整，端点读数表完整，且身份/契约/路径核验通过。R1/R3 质量失败不会改成管道失败；r3_quality_gate_passed 另保留原两个 R3 的冻结质量要求，pilot_gate_passed 只是该质量闸门的旧字段别名。CLI 返回码按 pilot_passed 判断，不代表数据可准入。

report.md 第一行固定为 Down0，随后所有表头/行标 pilot_only_not_profile_admission。每个端点保留 R1 单值、R3 中位、R3 IQR；策略不适用的格为 null/not_applicable，R1 点内 IQR 明确 not_estimable_single_replay。report.json 同时保存逐轮漂移、失败原因、target IQR 和质量状态。

每个“算子 × 策略”只有一次独立运行。七轮不是七次独立 pilot，不能估计越线率，也不能据此写“R3 更稳”或“R3 降低越线率”。报告只描述两次有固定先后顺序的运行所见差值；不能证明单次采样有偏。当前未冻结超过噪声的判据，noise_exceedance 写 not_assessed，差值和各自 IQR 并列，protocol_review_required=true、r3_adopted=false。

Down0 通过表示新单卡条件下未复现质量失败；仍失败表示新条件下复现质量失败。两种结果都不能单独证明 run05 的根因：窗口共享、驻留、运行时环境和执行时间均可能变化。先审阅 Down0 和 R1/R3 差值，再决定协议；不自动采用 R3。

无论 pilot 结果如何，followup_matrix_open=false。Mul0 是单独启动的元数据 Smoke，不属于四格 pilot；七项单卡矩阵仍未冻结和接通，不会自动继续、准入或启动搜索。

## Mul 实际操作数

新 ControlRuntime.activation 在实际函数体使用的 left_views 上逐一调用 tensor_meta，记录 inputs.left 的 50 份 Shape/dtype/stride/device/data_ptr 和 runtime provenance；另检查 left/up 逐字节不变。gate 恢复与 fused 输出别名继续检查。地址只供别名/操作数证据，不作为性能 Key。

该 hook 不在四格 pilot 中执行，make 仍只开放 Down0/Up0。独立入口 calibration.ffn_mul_metadata_smoke 在完整 pilot 之后验证实际 left 元数据落盘；只捕获 Mul，不运行 KERV probe，不执行 SiLU/Native/Fused 的测量臂。数值、只读、恢复及元数据使用布尔验收，耗时仅诊断。见 FFN_MUL_METADATA_SMOKE.md。run05 旧 Mul 缺口保留，旧导出/准入不变，不能用新代码补签旧记录。

## CPU 验收与重放

58 项测试通过：控制协议、采集器时序、崩溃/UUID 变化阻断、R1 质量失败保留、原始证据负向检查、单卡初始化、真实 Mul 操作数地址及 fused 不变量，以及旧 B4/导出回归。小 Tensor 和替身测试不是 HIP Capture 或真实 KERV 的验收。Bash 语法通过，CPU plan 重复生成逐字节一致；见 data/ffn_anchor_control_v2/verification.json。

CPU 重放（全新目录）：

```bash
python -m calibration.ffn_anchor_control --output /fresh/cpu_plan
```

显式 run 才启动四格 GPU pilot，须传 --flagscale-root。tools/run_ffn_anchor_control.sh 也默认 plan，解释器从 FUSION_PYTHON 读取，run 才要求 FLAGSCALE_ROOT；保持既有私有 Python 3.10 启动器和固定 FlagScale checkout。本轮没有执行 run。
