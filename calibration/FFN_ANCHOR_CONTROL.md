# B4 锚点控制采集器

入口 calibration.ffn_anchor_control，默认只写 CPU plan。当前 collector_implemented=true、gpu_validated=false；本轮 GPU 执行和新增准入均为 0。

冻结协议沿用 data/ffn_b4_cpu_preparation_v1/anchor_protocol.json，原文件及其来源不变。新实现身份、源码 SHA 和 plan_id 独立记录在 data/ffn_anchor_control_v1/plan.json。旧协议内 gpu_collector_ready=false 是上次冻结时的状态；当前实现状态读取新 plan，不重签旧协议。

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

这是锚点采样对照，不替代旧四类 Smoke。正的耗时较慢不会导致管道失败；锚点或 target 离散度越线标 quality_failed，仍继续固定四格。Capture、数值、不变量、Event、路径或身份失败则停止，后续标 blocked；进程崩溃也只启动一次，保留 worker.log、last_stage 和已有原始文件。普通异常另留 traceback、pool 和 GPU0 显存快照。

## 时间与证据

每个锚点 repeat 和 target 留 reset_begin、timing_begin、event_wait_end、all_device_sync_end、validation_end，时钟为 perf_counter_ns。Event 只覆盖 Replay；恢复、同步完成后的数值检查在 Event 外。另留 anchor warmup 结束、case 准备起止时间。

频率、利用率、温度通过带来源的可用 Torch 接口尽力读取；失败写 null 和原始错误，不填 0。读取在 pre-anchor 前和 post-anchor 后，避免夹入本轮漂移窗口。环境因果结论保持 inconclusive，不能凭本实验唯一归因于机器或协议。

全部七轮和端点结束后才分别 Trace target、anchor，此后不再计时。原 samples.json 的 pending_post_timing_trace 不覆盖，result.json 单独附路径及 Trace SHA。父进程重新核对原始 Trace、样本哈希、契约、时序和质量结果，不仅检查 worker 退出码。

每格包含 identity.json、preparation.json、samples.json、两个 Trace、result.json、worker.log、parent_plan.json；全局 summary.json 保留 quality_failed 与 blocked。identity.json 同时绑定 measurement_contract_id，契约使用独立 pilot executor 和 protocol_id，不冒充 v3 或正式成本标签。

四格管道完整、UUID/build 一致、两个 R3 通过，才记 pilot_gate_passed=true。R1 质量失败原样保留。无论 pilot 结果如何，followup_matrix_open=false；七项单卡矩阵仍未冻结和接通，不会自动继续、准入或启动搜索。

## Mul 实际操作数

新 ControlRuntime.activation 在实际函数体使用的 left_views 上逐一调用 tensor_meta，记录 inputs.left 的 50 份 Shape/dtype/stride/device/data_ptr 和 runtime provenance；另检查 left/up 逐字节不变。gate 恢复与 fused 输出别名继续检查。地址只供别名/操作数证据，不作为性能 Key。

该 hook 不在四格 pilot 中执行，make 只开放 Down0/Up0。未来开放 activation 矩阵时才能产生新 GPU 元数据。run05 的旧 Mul 缺口仍保留，旧导出/准入不变，不能用新代码把旧记录补签成完整身份。

## CPU 验收与重放

50 项测试通过：控制协议、采集器时序、崩溃/UUID 变化阻断、R1 质量失败保留、原始证据负向检查、单卡初始化、真实 Mul 操作数地址及 fused 不变量，以及旧 B4/导出回归。小 Tensor 和替身测试不是 HIP Capture 或真实 KERV 的验收。Bash 语法通过，CPU plan 重复生成逐字节一致；见 data/ffn_anchor_control_v1/verification.json。

CPU 重放（全新目录）：

```bash
python -m calibration.ffn_anchor_control --output /fresh/cpu_plan
```

显式 run 才启动四格 GPU pilot，须传 --flagscale-root。tools/run_ffn_anchor_control.sh 也默认 plan，解释器从 FUSION_PYTHON 读取，run 才要求 FLAGSCALE_ROOT；保持既有私有 Python 3.10 启动器和固定 FlagScale checkout。本轮没有执行 run。
