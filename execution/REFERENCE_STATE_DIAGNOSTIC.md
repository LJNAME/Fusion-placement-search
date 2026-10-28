# Reference执行器状态诊断

固定训练用例Qwen3-0.6B、seq1024、batch1、BF16，比较GPU0单卡、GPU1单卡、up_proj跨卡。需要两张BW1000，沿用原Torch/HIP环境。

## 服务器执行

更新代码后运行：

```bash
cd ~/private_data/LJ/Fusion-placement-search
python -m execution.diagnose_reference_state \
  --out-dir data/bw1000_reference_state_v1
```

若继续手动上传，本轮新增运行文件只有execution/diagnose_reference_state.py，放到服务器项目的execution/目录即可。其余运行依赖均为前两批校准已有源码；测试文件和文档不需要上传。旧git pull若仍被本地产生的未跟踪数据阻止，保留数据，不用git clean或强制覆盖解决。

新输出目录必须为空，不自动续跑。入口验证两张可见GPU与原运行环境，不允许CPU回退。不下载模型权重，不采seq768/1280标签。

## 固定对照

| 因素 | 取值 |
|---|---|
| 放置 | uniform0 / uniform1 / mlp_up |
| 执行器生命周期 | 每块重建 / 同一三元组复用 |
| 中间块 | CPU计时探针 / 普通基准（空白探针对照） |
| 重复 | 3轮 |

12个条件，每轮各执行一次，每次before→middle→after，共36个三元组、108块。每块5次预热、20次正式前向，共2160次正式前向；初始化、预热和数值检查不在此数内。

第0轮原顺序，第1轮左移5个条件后反转，第2轮左移10个条件。采集前保存完整protocol和源码hash，顺序不随观测结果变化。

两个生命周期都使用SubmissionProbeExecutor：基准调用继承自原执行器的run_once(False)，仅真实探针中间块调用run_probe()。复用分支在同一三元组内保持同一执行器与输入/权重/Norm/RoPE缓存；重建分支在每块结束后销毁执行器。两者均预热后再计时，不清空GPU allocator cache、不锁频、不调线程，也不把重建执行器叫作GPU冷启动。

普通基准中间块与前后块使用同一执行方法，没有CPU逐操作探针。这样可检查没有探针时是否仍有漂移。所有路径不创建设备Event、不启用profiler，不增加逐节点同步。前后基准仍按原方式同步整个前向。

## 记录

每个完整块立即写JSONL，带record_id、条件、轮次、执行器实例编号、原始20个样本和首/末输出检查。输入/权重由同一canonical CPU模板生成，数值口径与原reference一致。

- 缓存状态：已预热的输入、权重、Norm和RoPE条目数及storage身份hash，分别在正式样本前后记录。复用条件另检查三块的身份hash一致。
- 真实探针：原始逐API调用、提交步长、bookkeeping与busy final sync；不导出设备窗口。
- 块外状态：当前进程CPU耗时、包含预热/检查的块墙钟、宿主load average、GPU allocated/reserved字节与allocator retry计数。它们不进入前向计时；宿主负载不是本进程专属负载。

allocator可能在重建后复用相同地址，所以实例编号和缓存hash一起核对。复用/重建比较同时改变张量生命周期和分配历史，不能直接等同于只改变某一级硬件缓存。

## 报告口径

输出protocol.json、environment.json、blocks.jsonl、diagnostic_report.json、collection_status.json、audit_report.json。自动审计源码与训练范围、实例生命周期、缓存、记录hash/槽位、数值与样本、复制多重集合、无设备窗口和报告复算。

报告各条件三轮的before/middle/after耗时、探针变化和漂移。沿用15%/10%数值作描述参照，不重新验收主机profile，不导出host_submission_profile.json，不改变旧calibrated=false。

- 空白探针条件也漂移：探针不是漂移出现的必要条件。
- 重建条件比复用更不稳定：执行器/分配历史相关，仍不能单独归因到allocator或硬件cache。
- 单卡对照也共同变化：继续检查更广的主机/运行时状态。

以上是对照的解释范围，不自动输出根因。只有3轮，不做显著性或泛化声明。线程、频率、驱动等待与GPU计算因果尚需其他证据。

## 回传

```bash
tar -czf data/bw1000_reference_state_v1.tar.gz \
  -C data bw1000_reference_state_v1
```

回传整包与终端输出；报错保留已写记录，不重复跑到满足某个阈值。需要重放审计时：

```bash
python -m execution.diagnose_reference_state \
  --audit-dir data/bw1000_reference_state_v1
```
