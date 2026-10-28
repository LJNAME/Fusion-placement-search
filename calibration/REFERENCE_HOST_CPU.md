# 无Event的主机提交校准

需要两张BW1000，沿用原Torch2.9/HIP环境。新入口只采CPU提交步长，原带Event版本的64块数据保留。

## 服务器执行

```bash
cd ~/private_data/LJ/Fusion-placement-search
git pull
python -m calibration.calibrate_reference_host_cpu \
  --out-dir data/bw1000_reference_host_cpu_v1
```

目录必须为空。程序拒绝CPU回退；这里的CPU探针表示用主机时钟计提交时间，模型计算仍在两张GPU上执行。无需下载权重或新增依赖。

## 固定采集协议

- Qwen3-0.6B/Qwen2.5-3B × seq256/1024，batch1、BF16、单层synthetic forward。
- 各任务uniform0、uniform1、K分支GPU1、up_proj GPU1，共16例。启用算子的完整Key全部属于旧112条训练Key。
- 固定3轮；第0轮原顺序，第1轮左移5例后反转，第2轮左移10例。不按本轮耗时调整顺序。
- 每例每轮baseline_before → cpu_probe → baseline_after；共48个三元组、144块。每块5次预热、20次正式前向，共2880次正式前向，初始化与预热不计入此数。
- 首个测量块前保存protocol.json，包含完整图/放置、顺序、参数、阈值、源码hash和freeze_id。
- CPU探针不创建CUDA/HIP Event，不启用profiler或TorchDispatch，不逐算子同步。基准使用原始reference执行器。

无Event指观察方式；本轮正式前向次数比旧版更多，用于观察跨轮变化。线程数沿用现有环境，记录Torch线程数和CPU亲和数量，不自动调整。

## 验收和导出

48个三元组分别检查：CPU探针中位耗时相对前/后基准都在±15%内；前后基准漂移≤10%。所有用例的所有轮次均通过且有复制观测，才导出calibrated=true。采集过程中不因耗时检查失败提前终止或删掉某轮。

跨轮基准中位数的max/min-1单独报告，不增加新的验收门槛。若任何三元组失败，仍保存所有候选参数，标complete_cpu_probe_perturbed/calibrated=false；不提高阈值或拼接其他采集轮次。

主机参数依次按case-round取中位、同case三轮取中位、全部case取中位。保留operator与local-body覆盖，start_s单列；finish_s只取GPU已经空闲后的同步API耗时，busy final sync不参与参数。

CPU API调用可能包含驱动等待；提交步长包含执行器bookkeeping，不能解释为纯kernel launch时间。不采设备body窗口，不改GPU服务、链路profile或默认搜索。即使本轮calibrated=true，也只表示这台宿主/运行时的采集检查通过，不等于方案级泛化或SROP收益成立。

## 文件与审计

输出protocol.json、environment.json、blocks.jsonl、host_submission_profile.json、calibration_report.json、collection_status.json、audit_report.json。每个完整块立即写入JSONL并带record_id。运行或数值错误保留已有块和错误状态，不自动续跑或跨宿主合并。

采集结束自动审计；也可在相同代码版本上重放，不需要GPU：

```bash
python -m calibration.calibrate_reference_host_cpu \
  --audit-dir data/bw1000_reference_host_cpu_v1
```

审计复核固定训练范围/源码、环境与记录hash、144个有序槽位、样本数量、复制多重集合、无设备窗口、CPU时间分解、汇总与host导出。审计通过与校准验收分别报告。

## 回传

```bash
tar -czf data/bw1000_reference_host_cpu_v1.tar.gz \
  -C data bw1000_reference_host_cpu_v1
```

回传整个压缩包。报错或calibrated=false也保留全部数据和终端输出，不反复重跑直到通过。seq768/1280独立任务仍未开放采集；本轮不读取新标签。
