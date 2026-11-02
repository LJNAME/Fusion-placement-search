# B4 run04 · 回传审核与定点诊断

回传包SHA256：8aa44e03168008eb2ba1ca439e83fc72254d390fec590dca30d46373fee06bd8。原件归档在transfer/ffn-b4-received-20261102-04；6/6采集源码匹配2fedc98。机器可读结果：data/ffn_b4_run04_review/report.json。

| 项目 | 结果 |
|---|---:|
| 最小Smoke | 4/4 passed |
| 正式网格状态 | 19/19 |
| 有完整7轮样本 | 15格 |
| quality_passed | 10格 |
| quality_failed | 9格 |
| 子进程native崩溃 | 1组，牵涉4格 |
| 锚点失效 | 5格 |
| 补轮/重启 | 0 |
| 新accepted / 搜索成本项 | 0 / 0 |

15格样本的质量状态已逐项CPU复算；5个锚点失败格仍保留全部7次尝试，不补轮。link_quartet退出码-11，最后阶段capture_body_returned，故障栈指向torch.cuda.graphs.CUDAGraph.capture_end；捕获体已返回，捕获结束时崩溃。该定位不等于驱动、allocator或跨设备Event的底层根因已确定。后续7组继续完成，轮组隔离在此次真实运行中生效。

## 计时变化

以下数值都是同一格、同一50调用Graph的Event批耗时；前列为Profiler前10次预热的中位数，后列为Profiler后7次正式尝试的中位数。不是把Profiler Trace跨度拿来充当Event标签。

| Case | 前 / ms | 后 / ms | 前÷后 |
|---|---:|---:|---:|
| Fused / GPU0 | 0.384478 | 0.013920 | 27.621 |
| Native / GPU0 | 0.539759 | 0.011520 | 46.854 |
| Fused / GPU1 | 0.366239 | 0.013920 | 26.310 |
| 1.5MiB Copy 0→1 | 2.334319 | 2.369279 | 0.985 |

计算侧12格的前÷后为20.992～46.854；3个有样本通信格为0.981～0.995。Event记录设备与目标设备一致，尚不能据此证明计时边界覆盖了实际Graph工作。当前记录只建立Profiler前后变化的关联；因果与底层机制均待控制实验。稳定IQR不能排除稳定的系统性偏差，10个quality_passed不升格准入，也不据此判断融合收益。

## 下一项：不再跑整批

入口：calibration.ffn_b4_diagnostics。每项一个新子进程、只启动一次，父进程不持有Smoke GPU runtime；该驻留条件与run04正式批次不同，已写入诊断plan，结果不跨条件拼接。无正式Profile、无准入。

1. timing_control：GPU0 KERV 50调用Graph，10预热，前后各3次；中间1次不带Profiler的Replay。
2. timing_profile：相同组织，中间改为1次带Profiler的Replay。
3. chain_allocating_1：原式跨卡Down→Copy→Gate，1调用，输出在捕获体中分配。
4. chain_preallocated_1：相同链与调用量，显式GEMM输出在捕获前预分配，使用matmul(out=...)。
5. chain_preallocated_50：仅第4项通过后执行；否则标blocked，不尝试。

每个定时样本同时保留GPU Event、Event等待返回的同步墙钟、所有参与设备同步完成的墙钟及额外同步时间。恢复和数值检查不进入这些区间。墙钟包含提交与同步开销，只是诊断proxy，不是可直接替换的GPU Profile。

预分配只覆盖显式Producer/Consumer输出，不宣称BLAS内部零分配；两种matmul执行接口分开标记，不默认实现身份或性能相同。跨卡链在Replay前毒化目标/输出，排除旧输出掩盖未执行。每次检查全部输出，保留Trace、实际环境、故障栈和last_stage；不替换run04任何样本。

```bash
cd ~/private_data/LJ/Fusion-placement-search
git pull --ff-only
export FLAGSCALE_ROOT="$HOME/private_data/LJ/FlagScale-main"
export FUSION_PYTHON="$HOME/private_data/LJ/dcu-kernel310-20261031/run_python.sh"

"$FUSION_PYTHON" -X faulthandler -m calibration.ffn_b4_diagnostics \
  --flagscale-root "$FLAGSCALE_ROOT" \
  --output ../ffn_b4_diagnostics_20261102_01
```

输出目录必须全新；回传整个目录。summary.json仅表示各诊断是否完成，不代表正式测量或准入通过。

CPU：54项相关测试通过（新增3项诊断边界/依赖/预分配语义检查，原51项回归）；run04数据复核通过。新诊断尚未GPU执行。原正式测量脚本及其policy/Recipe未改，B5～B6与MCTS保持未启动。
