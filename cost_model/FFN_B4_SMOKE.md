# FFN B4 · Smoke 与正式网格

run03回传已核验：最小Smoke 4/4通过；正式8格完成，跨卡诊断链Capture边界发生native段错误，批次不完整，未新增准入。见data/ffn_b4_run03_review/report.json。当前v2按轮组隔离进程，只有CPU验证，尚无v2 GPU结果；旧B3和run03原始证据保留。

run04已回传：轮组隔离生效，19状态齐全（10 quality_passed/9 failed，15格有7次样本），跨卡诊断链定位capture_end段错误；计算侧Event在Profiler前后发生系统变化，当前不准入。下一项改为定点诊断，命令见FFN_B4_RUN04_REVIEW.md；不要继续整批重跑。

当前入口已转v3：定点控制确认既有Graph经Profiler Replay后Event计时状态改变；单次预分配链仍capture_end崩溃。新轮组先完成全部采样/锚点再做Trace，组合链构造前阻断，其余18核心格照常；新时序/计数/命令见FFN_B4_V3.md。以下v1/v2时序作为历史保留，不用于当前Recipe。

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

正式v2使用10个轮组子进程，顺序不变，每组只启动一次；同组各臂及锚点保持在同一子进程。父进程完成Smoke后保留其GPU驻留状态但不再执行GPU工作。某个子进程native崩溃或缺失结果时，该轮组全部标quality_failed，保留已有原始文件，继续后续轮组，绝不重启该组。每组结束即落盘progress.json，结束后生成19格grid.json。进程边界写入新policy/Recipe，不与run03拼接或替换旧样本。配对格缺少可信对照时不能准入，链中缺某个组件时不能拼残差。原始轮组/顺序保持可追溯，当前数据不自动升格accepted。

脚本和子进程均启用faulthandler；worker.log记录Python故障栈，last_stage.json在库初始化、capture体返回、capture结束及Replay边界落盘。该文件只能证明最后持久化阶段，不等于底层根因。原生崩溃无法保证取得异常后的显存快照，保留已有capture前快照，不伪造Python traceback。正式样本增加原始elapsed_ms及Event设备/流；preparation.json保留原有10次预热计时，便于比较Profiler前后变化，不额外执行或作为主样本。计时异常仍待定位，不自动修正或准入。

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

# 同一GPU作业先Smoke，成功后逐轮组子进程处理19格；Smoke失败直接停止
bash tools/run_ffn_b4.sh /fresh/output/ffn_b4_run run
```

返回整个目录：plan.json、superseded_profiles.json、runtime_identity.json、placed_gpu0/1、smoke/、formal/（若已启动）以及failure.json（若有）。正式formal/grid.json包含完整19格；formal/groups/<轮组>/保存worker.log、last_stage.json、worker_status.json，组内formal/保存身份、预热、样本、Trace或失败资料。不要只返回一张成功截图或中位数。

只做CPU计划：

```powershell
& 'C:/Users/LJ/miniconda3/envs/fusion-dev/python.exe' -m calibration.ffn_b4 plan --output /fresh/output/ffn_b4_plan
```

`data/ffn_b4_preflight/verification.json` 是CPU计划检查，不是GPU冒烟报告。run03证明最小管道通过；当前v2的进程隔离、故障定位和实际HIP行为仍待GPU验证。Smoke通过只证明最小四类管道，不代表19项全部可测、数据质量合格或已准入。

CPU验证：51项相关测试通过（B4 18项，B1～B3 33项），含真实子进程直接退出后继续全部网格、缺失结果、worker入口传参及旧路径回归；Bash语法、CPU计划/审计重放通过。v2 GPU未执行。
