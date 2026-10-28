# Reference主机提交校准

在已有训练任务上采CPU提交步长和reference算子body的GPU stream窗口，不采seq768/1280独立任务标签。

## 服务器执行

需要两张BW1000，使用原Torch/HIP环境，无新增依赖或模型权重下载。

```bash
cd ~/private_data/LJ/Fusion-placement-search
git pull
python -m calibration.calibrate_reference_host \
  --out-dir data/bw1000_reference_host_train
```

输出目录须为空。若报错，保留目录和终端错误，一并回传；入口不自动混合旧主机数据或续跑。GPU入口要求两张可见HIP/CUDA卡，无CPU回退。

## 训练范围和样本

Qwen3-0.6B/Qwen2.5-3B × seq256/1024，batch1、BF16，均为单层synthetic forward。非逻辑算子的完整Key逐项检查属于旧112条训练Key；新56条测试标签和规划的seq768/1280均不使用。

每个任务固定四种放置：全部GPU0、全部GPU1、K分支放GPU1、up_proj放GPU1。GPU0/1两种单卡用于测量重复，不按设备交换对称性去重。总计16用例。

每例四块：无探针前基准、CPU探针、设备窗口探针、无探针后基准。两个中间探针的顺序按用例奇偶交替；每块5次预热、20次正式前向，总64条块记录、1280次正式前向。共同输入/权重与原reference一致，检查输出有限性、rtol=.01、atol=.001。

## 测量口径

CPU探针不启用profiler、TorchDispatch或CUDA Event；它复用reference算子实现，并复现其提交/前驱复制/释放顺序。只在调用前后加CPU计时，元数据在预热时建立，操作明细在前向结束后整理。

每次操作分别记录：

- api_call_s：执行CPU调用本身的返回耗时，可能包含驱动阻塞，不能直接称为纯kernel启动开销。
- dispatch_step_s：本次API开始到下一次API开始的时间，包含随后executor bookkeeping，用于推进模型的主机时间线。
- bookkeeping_after_s：API返回后到下一次API开始的间隔。
- start_s：前向计时开始到首次API的间隔。
- idle_sync_s：已完成所有GPU工作后再次调用同步API的耗时，用作固定结束成本。

首次busy_final_sync_s_not_a_host_parameter包含GPU尾部等待，保留原值但不填入finish_s。

设备窗口探针单独为节点和复制记录stream上的开始/结束Event，直到整次前向结束才查询elapsed_time，不增加逐节点同步。窗口包含body内部GQA/布局/类型转换以及可能的CPU提交间隙；复制窗口还可能含目标stream等待。它们不是纯kernel活动求和，也不直接替换链路带宽/延迟。

body身份包含reference源码hash、环境、算子元数据、实际输入Shape/stride/dtype和输出信息，与旧torch.mm/bmm纯算子Key区分。窗口按body和用例分别保留，不把不同队列状态直接合并为精确延迟表。

## 参数导出和检查

主机参数先按用例取中位数，再跨用例取中位数；同一用例中多个MatMul不会获得额外任务权重。表中保留算子族fallback和local:算子body覆盖，后者优先；新模型增加独立start_s，原默认零参数预测数值不变。

预先固定探针开销目标：探针前向中位耗时相对前/后基准分别在±15%内，前后基准变化≤10%。CPU探针和设备窗口探针分别检查。

CPU探针全部用例通过并有复制观测时，host_submission_profile.json中的calibrated=true，否则保留候选值并标false；不删除失败用例追通过。该标记只描述本次宿主/运行时的采集检查，不等于整个代价模型已验证，也不保证可迁移到以前GPU报告的宿主。

body窗口一直标为diagnostic_candidates_only，是否完整及观察开销另看all_window_data_complete、all_window_probe_targets_pass。不会自动回填单核profile、清除operator-body OOD、替换默认DAG或修改SROP。

## 回传

```bash
tar -czf data/bw1000_reference_host_train.tar.gz \
  -C data bw1000_reference_host_train
```

回传整个压缩包，包含：protocol.json、environment.json、blocks.jsonl、calibration_report.json、host_submission_profile.json、body_window_candidates.json、collection_status.json。

先核对采集开销、API步长、body窗口及当前宿主上的训练用例重建误差，再决定profile如何接入。禁止把“本机主机参数已采集”当成旧宿主的同环境对照，或把训练重建当作独立验收；新任务仍待模型/候选冻结后再采。
