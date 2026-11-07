# KERV AddRMS2048单Case测时

2026-11-07。只采GPU0、第0层已有Adapter、BF16 [1,256,2048]。执行正确性已由kerv_add_rmsnorm_adapter_01验证；本次补Graph Replay成本，不做Native比较、GPU1、Shape扫描或搜索。

## 固定口径

| 项目 | 设置 |
| --- | --- |
| 一次Replay | 50次Adapter调用，50份独立可写x |
| 共享输入 | 原始source、residual、weight只读 |
| 输出 | 每次sum复用对应x，normalized分配自Graph private pool |
| 预热 | Capture前50次eager准备，Capture后10次Replay |
| 正式轮次 | 固定7轮，不补轮、不替换 |
| 计时 | 同设备/同stream的start Event→一次Replay→end Event |
| 恢复和验证 | 全部在Events外；每轮精确恢复50份x并将normalized置NaN |
| 结果校验 | 每个输出对照两参考，检查只读输入、sum别名、50份输出独立 |
| 统计 | μs/call，中位数；inclusive IQR/Median≤10% |
| profiler/锚点 | 不启动profiler；无跨窗口锚点，与既有isolated-stage规则一致 |

Graph Capture失败直接停止，无eager回退。Event读数失败保留该轮错误，仍固定七次尝试；质量失败不生成成本候选，不自动重测。数值错误立即停止，保留failure.json、progress.json、显存快照和fatal_traceback.log。

该数值覆盖Graph内Adapter的设备工作，排除输入恢复、Host提交及分配成本；并非eager完整调用时间或FFN边际耗时。50份独立可写x及其缓存/恢复语义属于契约，不能只凭同为Graph模式就与其他Profile视为等价。

## 身份和结果

采集前冻结代码SHA、FlagScale源码SHA/revision、已有正确性回传SHA与PlacedOpGraph绑定。运行时记录Torch/Triton/HIP/driver、Torch二进制/build、设备架构和真实Tensor布局。UUID仅保留溯源，不进入Key或契约身份。

沿用GroupKernelKeys生成含两个输出、权重布局及别名语义的Canonical Key，Measurement Contract独立编号。质量通过时只生成conditional_dev成本候选；不自动accepted，不修改默认搜索、不套用旧SiLU-Mul专项准入结论。回传后核对数据，再接严格成本查询。

代码calibration/kerv_add_rmsnorm_timing.py，入口tools/run_kerv_add_rmsnorm_timing.sh；复用既有stage_size_calibration.timed的Event计时，不启动新profiler。新增9项CPU测试，相关68项通过。data/kerv_add_rmsnorm_timing_plan_v1只有CPU计划，尚无GPU耗时。

## 服务器命令

```bash
git pull --ff-only origin main
export FLAGSCALE_ROOT="$HOME/private_data/LJ/FlagScale-main"
export FUSION_PYTHON="$HOME/private_data/LJ/dcu-kernel310-20261031/run_python.sh"
bash tools/run_kerv_add_rmsnorm_timing.sh ../kerv_add_rmsnorm_timing_01 run
```

使用新目录，返回整个目录，包括质量失败或异常记录。默认plan不执行GPU。一次只运行这个Case。
