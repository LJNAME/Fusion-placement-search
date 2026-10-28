# BW1000 三方案执行诊断

原冻结实验保持不变。本轮分析已暴露的用例，定位单层 reference 执行与结构调度的差异，不作为独立测试或 SROP 收益实验。

## 原始执行日志审计

收到 `execution_logs.tar.gz`，其中仅包含 `execution_samples.jsonl`、`execution_environment.json`；原件在 `data/bw1000_gpu_holdout_v1/results_received_20261028/`。

44条记录、880次前向耗时；记录/环境hash、工件ID、完整执行槽位和原报告逐项重现通过。四组用例的单卡参考在两轮中均最快。3B两用例完整排序相同，0.6B两用例的接近候选有换位。候选轮间中位数绝对变化最大3.170%。

5个样本超过对应候选混合中位数2倍，全部保留。该阈值仅描述长尾，不用于清洗或重新验收。只有2个顺序执行的计时块，不能把880次前向当成880个独立任务，也不能声称已定位长尾原因。

复算（不需要GPU）：

```bash
python -m execution.audit_gpu_holdout \
  --samples data/bw1000_gpu_holdout_v1/results_received_20261028/execution_samples.jsonl \
  --environment data/bw1000_gpu_holdout_v1/results_received_20261028/execution_environment.json \
  --report data/bw1000_gpu_holdout_v1/results_received_20261028/execution_report.json \
  --output data/bw1000_gpu_holdout_v1/results_received_20261028/execution_log_audit.json
```

## 固定三个候选

Qwen2.5-3B、seq2048、micro batch1、BF16。选择单卡参考、原封存预测Top1，以及已测耗时最接近两者中点的候选；选择使用了旧GPU结果，属于事后诊断。

| 角色 | GPU1上的算子 | 原实测/ms | 冻结预测/ms | 跨卡张量 |
|---|---|---:|---:|---:|
| single_gpu_reference | 无 | 3.793061 | 3.740832 | 0 |
| predicted_top1 | mlp.up_proj | 4.182706 | 3.675852 | 2次，51MiB |
| intermediate | self_attention.k_rope | 3.975361 | 3.678040 | 2次，2MiB |

Top1中，GPU0→1传8MiB，GPU1→0传43MiB。原模型已经计入通信耗时合计0.409980ms；单卡与跨卡的算子总成本均为3.740832ms。跨卡makespan仍低于单卡0.064980ms，需核对预测调度中的重叠。这里是封存预测的静态重现，不是真实时间分解。

完整组起止、关键路径和传输区间在 `three_plan_prediction.json`。新入口复用原executor，不修改源码冻结清单内的文件，不调模型、阈值或并行流。

## 服务器采集

需要两张可见BW1000，仍使用原Torch/HIP环境。从仓库根目录执行：

```bash
git pull
python -m execution.diagnose_gpu_holdout \
  --out-dir data/bw1000_gpu_holdout_v1/diagnostics_3b_s2048
```

输出目录应为空；已有结果时换一个新目录。原冻结数据和报告不会覆盖。该入口硬要求GPU，不会回退到CPU。

执行顺序：

1. 共同CPU随机模板生成相同输入/权重，使用原正确性阈值。
2. 三方案各跑3轮，每轮5预热、20次无逐组插桩的同步前向，共180个性能样本。三轮循环轮换顺序，每个方案各处于一次前/中/后位置。
3. 完成全部无插桩测量后，再为每个方案采集1次独立的profiler轨迹；逐节点添加范围标记，观察实际to-copy的来源/目标、Shape、stride及字节数。与reference前驱循环预期的传输多重集合比较。
4. 保存原结构模型的逐组预测起止、通信和关键路径，供GPU时间线对照。

主性能看 `baselines`。`node_cpu_scopes`、`cpu_call_ms`和profiler耗时属于插桩诊断，可能包含提交、阻塞等待和工具开销；嵌套CPU/GPU时间不互斥，不能直接求和或用总时间相减分摊同步成本。

实际并发/等待要看GPU时间线。若当前Torch/HIP构建无法提供GPU活动，报告 `gpu_timeline_available=false` 或trace错误，同时保留全部无插桩数据；不得据此声称已验证GPU重叠。新脚本不增加逐节点同步，不把原执行改成串行计时。工具本身仍有观察开销，GPU时间线用于诊断，最终性能需无插桩复核。

回传整个诊断目录：

```bash
tar -czf data/bw1000_gpu_holdout_v1/diagnostics_3b_s2048.tar.gz \
  -C data/bw1000_gpu_holdout_v1 diagnostics_3b_s2048
```

包含 `diagnostic_report.json`、`diagnostic_environment.json`、`prediction_breakdown.json`、`baseline_samples.jsonl`和3份 `*.trace.json`。若trace报错，仍回传已产生的文件和终端错误；不必重跑原56对象的算子校准。
