# FFN B4 · Smoke 与正式网格

GPU入口已准备；本机只做CPU控制流验证，没有BW1000 Smoke或正式数据。旧B3产物保留，当前执行策略通过独立policy_id和active_recipe_id覆盖其失败处理规则。

## Smoke：布尔验收

最小集合固定四项，每项50调用、3次Replay：

| 类别 | 最小Case |
|---|---|
| Singleton | Down GEMM / GPU0 |
| Native链 | SiLU→Mul / GPU0 |
| Fused | KERV silu_mul / GPU0 |
| Link | 0.5 MiB GPU0→GPU1，预分配copy_捕获 |

通过条件仅为capture_ok、numeric_ok、invariants_ok、identity_written、contract_written全为true。不检查快慢、加速、IQR或锚点漂移来决定Smoke通过。耗时照常打印；计时诊断API失败或非有限数值记在diagnostic_timings，不以此判管道失败，也不会改用eager执行图。

数值沿契约rtol=.01/atol=.001，复制和源保持逐字节检查。KERV通过PlacedOpGraph Adapter调用真实公开Torch Operator，并在捕获前检查一次实际JIT派发；若观察到Host Staging或错误执行路径，属于管道契约错误，而非性能慢。

每次Replay前显式恢复全部50个gate并同步，检查其与原始Tensor逐字节一致；每次Replay后检查全部50个输出、共享up字节未变、Fused输出data_ptr与对应gate一致。Native不得修改gate。恢复、预热及输入上传均在计时外。

跨卡copy同时使用源/目的非默认流，并显式fork/join，避免把源侧复制工作遗留在默认流。1.5 MiB、反向Copy、GPU1激活以及完整诊断链不由最小四项自动认证；正式逐格准备仍可能失败。链中Producer→Copy→Consumer在相邻调用间也有完成栅栏，避免将批内流水重叠误当成简单串行加和残差。

Profiler不能暴露完整Graph子事件时保留原Trace或诊断错误，不把“观察不到”当成数值管道坏；正式证据仍需路径审核，缺少必要路径证明可标quality_failed。peer能力不自动等于物理P2P路由证书。

## 三条不变量

1. Fused输出与gate同址：逐个比较50个data_ptr；不是只看数值。
2. up共享且只读：与测前backup按uint8逐字节比较。
3. gate恢复精确：每次warmup/capture/replay入口恢复后逐字节比较，不能复用上次原地结果。

产物保留实际输入/输出Shape、dtype、stride、GPU、地址别名证据，实际UUID/架构/Torch/HIP/Triton构建、源码哈希、recipe_id和measurement_contract_id。地址用于运行时别名证据，不是Canonical Kernel Key字段。

## 两套失败策略

| 阶段 | 行为 |
|---|---|
| Smoke | fail-fast；保存失败Case完整traceback、pool配置、显存快照和上下文；停止，不启动正式网格 |
| 正式 | 固定19格，逐格捕获失败或质量失败标quality_failed；保留已有样本和错误，继续其余格；不重跑格、不替换样本 |

用户确认：正式固定 **7次尝试×50调用**，不再执行旧B3允许的额外2轮。任一作废轮使该格无法满足7有效轮，标quality_failed；IQR不合格同样标记。锚点仍按前后变化>10%作废，inclusive IQR/Median>10%判质量失败，融合快慢不参与准入。

全批保留19个格的状态，即使个别Capture失败也不丢掉后续格。配对格缺少可信对照时不能准入，链中缺某个组件时不能拼残差。原始轮组/顺序保持可追溯，当前数据不自动升格accepted。

每格Capture前有1次未捕获的编译/库初始化体，10次预热Replay之后另有1次路径Trace Replay；二者单列写入active Recipe和contract，不伪装为只有10次总执行。全部在正式7轮计时外。

正式格记录实际环境绑定的contract hash；Native/Fused同卡两臂使用相同候选/配方/环境，GPU0与GPU1保留独立身份。运行结果是待审核原始证据；精确Kernel-Key导出、路由证据复核及配对/链路一致性审核仍需离线完成，不能把quality_passed等同于完整方案准入。

## 旧2/2的状态

`data/ffn_b4_preflight/superseded_profiles.json` 保留两条旧Fused及其原accepted状态，增加 `superseded_by=c4fc415`，仅表示新B3/B4配方不复用。原始测量、旧决策、旧2/2比较证据均不删、不改签。当前新Recipe可复用数仍为0。

## 一次GPU作业内的运行

服务器工作目录为~/private_data/LJ/Fusion-placement-search；在当前终端显式export FLAGSCALE_ROOT="$HOME/private_data/LJ/FlagScale-main"，显式export FUSION_PYTHON="$HOME/private_data/LJ/dcu-kernel310-20261031/run_python.sh"，不升级FlagScale。脚本要求两项环境变量均显式设置，不默认调用系统python；系统3.11的Triton在读取gfx936属性时已有UnicodeDecodeError记录，不能替代已验收私有3.10栈。B4将--flagscale-root同步用于B1～B3冻结校验和实际Adapter；旧B1没有路径参数，仅在同步校验期间临时设置其目录查找值，正常或异常退出均恢复，不修改冻结源码/产物。输出目录必须不存在；没有超时上限，失败目录同样保留。

```bash
cd ~/private_data/LJ/Fusion-placement-search
export FLAGSCALE_ROOT="$HOME/private_data/LJ/FlagScale-main"
export FUSION_PYTHON="$HOME/private_data/LJ/dcu-kernel310-20261031/run_python.sh"

# 只检查最小管道，不开始正式计时
bash tools/run_ffn_b4.sh /fresh/output/ffn_b4_smoke smoke

# 同一进程先Smoke，成功后处理19格；Smoke失败直接停止
bash tools/run_ffn_b4.sh /fresh/output/ffn_b4_run run
```

返回整个目录：plan.json、superseded_profiles.json、runtime_identity.json、placed_gpu0/1、smoke/、formal/（若已启动）以及failure.json（若有）。正式formal/grid.json包含完整19格；每格目录保存身份、样本、Trace或失败资料。不要只返回一张成功截图或中位数。

只做CPU计划：

```powershell
& 'C:/Users/LJ/miniconda3/envs/fusion-dev/python.exe' -m calibration.ffn_b4 plan --output /fresh/output/ffn_b4_plan
```

`data/ffn_b4_preflight/verification.json` 是CPU计划检查，不是GPU冒烟报告。本机没有GPU执行结果；脚本及多设备Graph/捕获行为仍需实际HIP环境验证。Smoke通过只证明最小四类管道，不代表19项全部可测、数据质量合格或已准入。

CPU验证：48项相关测试通过（B4 15项，B1～B3 33项，含默认路径缺失时显式路径贯穿CLI校验、错误路径拒绝及异常后恢复）；Bash语法检查与计划/审计重放通过。GPU未执行。
