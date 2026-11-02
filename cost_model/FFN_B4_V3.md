# B4 v3 · 计时与Trace时序

诊断回传：transfer/ffn-b4-diagnostics-received-20261102-01，包SHA256 8c033f30f81ca9dba7729ddf8e22ea0452f5e3e919904c00b7719c521ca7b0aa。4/4来源哈希匹配10c168c；审核记录data/ffn_b4_diagnostics01_review/report.json，CPU入口tools.review_ffn_b4_diagnostics。

run05已核验：18核心格均完成7轮、0进程崩溃，9质量通过/9锚点失败，1组合链按规则阻断；预热/正式比0.9777～1.0181。当前不继续GPU补测，下一项为9候选的CPU Key/Admission准备，见FFN_B4_RUN05_REVIEW.md。以下v3上卡前说明保留为历史。

## 控制结果

每列为3个样本的中位数，KERV / GPU0 / BF16[1,256,3072]，每Graph 50调用。

| 控制 | Event前 / μs每次 | Event后 / μs每次 | Event前÷后 | 同步墙钟前 / μs每批 | 同步墙钟后 / μs每批 |
|---|---:|---:|---:|---:|---:|
| 中间无Profiler | 8.412760 | 8.585560 | 0.979873 | 488.892 | 495.352 |
| 中间有Profiler | 7.238360 | 0.307200 | 23.562371 | 417.302 | 224.971 |

同一卡环境一致，Event始末均cuda:0，数值、gate恢复/别名、up只读均通过；Profiler Trace包含50个_silu_mul_kernel。该执行栈上，对既有Graph做Profiler Replay之后计时状态改变；同步墙钟只下降1.854915倍，不能把Event下降解释为融合加速，也不以墙钟替换GPU标签。硬件/驱动内部机制未确定；结论限定本次GPU0、软件栈与已存在Graph的路径。

| 跨卡组合 | 调用数 | 结果 |
|---|---:|---|
| 捕获体分配输出 | 1 | SIGSEGV，capture_end |
| 显式输出预分配 | 1 | SIGSEGV，capture_end |
| 显式输出预分配 | 50 | blocked，未尝试 |

两种单次链都在capture_body_returned后于torch.cuda.CUDAGraph.capture_end崩溃。显式输出预分配未解决，不再重复同一故障路径；仍未确定BLAS内部工作区、跨设备依赖或后端实现的责任。该阻断只覆盖Down0→Copy0.5MiB→Gate1的组合捕获，不扩大成所有HIP Graph或独立跨卡Copy不支持。

## v3执行

每个新轮组子进程依次完成：准备/编译初始化体1次→Capture→10次预热→7轮采样及前后锚点→各臂Trace→离线质量检查。Trace之前没有对这些Graph使用Profiler；第一个Trace开始后，该子进程不再做任何正式或锚点计时。保持原轮组、轮内顺序、调用量、阈值和不补轮规则。

原始samples.json保留pending_post_timing_trace，不在事后覆写计时或path字段；result.json单独保存后置Trace状态。质量检查结合真实Trace结果；Trace失败不准入。新policy/Recipe/contract明确记录before_formal=0、after_formal=1，不混用run03/04或诊断数据。

保留19格状态：18格核心测量照常；组合诊断在构造前标quality_failed、execution_status=blocked_backend_capture、attempted_calls=0，引用本次诊断证据。link轮组中的Down0、Copy0→1、Gate1仍独立测量，不计算缺失的链路残差，也不进行eager或host-staging回退。

B4核心数据仍需真实测量、Key导出与准入审核；B5链路残差未完成，完整FFN组合成本和B6搜索验证仍不开放。不能将18格采样成功写成完整成本覆盖。更慢的Fused仍允许作为可信数据审核，Native合法替代保留。

旧P2测量器在Eager Trace结束后才创建Graph（calibration/measure_kerv_profile_pair.py:142、:168），本次控制针对已创建Graph的Profiler Replay；原P2文件/决策保持，不因新实验改签或撤销。旧2/2对当前Recipe仍为superseded，不能直接复用。

CPU：56项相关测试通过，含“所有臂/锚点先于任何Trace且此后无计时”“原始样本不改写”“阻断链从不构造、其他组件继续”两项新增检查。v3 GPU尚未执行；没有新增accepted。

```bash
cd ~/private_data/LJ/Fusion-placement-search
git pull --ff-only
export FLAGSCALE_ROOT="$HOME/private_data/LJ/FlagScale-main"
export FUSION_PYTHON="$HOME/private_data/LJ/dcu-kernel310-20261031/run_python.sh"
bash tools/run_ffn_b4.sh ../ffn_b4_run_20261102_05 run
```

输出目录必须全新。保留所有旧数据，回传_05全目录；预期仍有1个明确阻断状态，其余格按实际质量记录，不承诺全部通过。
