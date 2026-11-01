# 三核组合与 graph 边界诊断 v1

单开发 case：GPU0、shape=[1,256,3072]、BF16、连续、seed1234。入口 `calibration.probe_sequence_composition`。只用 ATen，不加载 FlagScale/KERV，不需要 `--flagscale-root`。结果保持 `conditional_dev`，不改模型、Kernel Key、标签准入或 launch/fusion 常数。

## 固定采集

六臂：SiLU(S)、Mul(M)、Add(A)、SiLU→Mul(SM)、Mul→Add(MA)、SiLU→Mul→Add(SMA)。所有算子 out-of-place。Mul/MA 输入在 GPU 上预先准备为 SiLU(original)，Add 输入准备为 SiLU(original)*up；准备不进入 capture/replay 计时。up 和 add operand 固定同 shape。

每 graph 分别包含 50 或 100 次函数体调用。固定 50 份连续输入视图；100 次按相同顺序读取两遍，不把上次输出当下次输入。输入地址工作集固定，但重访频率、graph 大小及临时分配仍不同，N 对照不是完全因果隔离。

capture 顺序是 N50 六臂、N100 六臂，所有 12 个 graph 同时保留；每个 graph 独立默认 private pool，不共享 pool。只保留每个 capture 的末次最终输出，临时输出不累计。capture 前 side stream 跑一次完整函数体批次，capture 后每个 graph 做 10 次 warmup replay。每次 replay 前恢复输入并同步，恢复不进 Event。

12 个正式配对轮。以 [S,M,A,SM,MA,SMA] 为 0–5，Williams 基行 [0,1,5,2,4,3] 加 0–5 取 mod6，重复两遍。每种 N 内，各臂各位置两次，各有向轮内相邻关系两次。偶数轮先 N50 后 N100，奇数轮反过来；同轮两种 N 使用相同六臂顺序。

正式 **10800 次函数体调用、144 个 Event 区间**。一次 SMA 函数体含三个算子，调用数不等于设备实际 kernel trace 数或独立样本数。

## 契约和检查

Event 只包一次 graph.replay；总时间除当前 N，字段为 `graph_batch_amortized_per_call_us`。不命名为单次隔离延迟，不与 eager 相减分解纯 host 成本。提交/完成墙钟另列。

N 与 input_buffers 进入标签侧测量契约，因此 N50/N100 有不同 measurement_contract_id。旧契约/旧 Key 不回写。serial_plan_only=true 只描述本诊断域，不表示已实现模型通用 plan 域闸门或确认下游执行契约。

每轮计时先写 partial（numeric_passed=null），再检查末次最终输出，rtol=.01、atol=.001；其余最终输出不逐一断言。另检查全部输入缓冲、up、add operand 未改变。显存快照在数值检查之前，记录全局 allocated/reserved/峰值，不能当成 private pool 专属容量。

失败记录完整 traceback、stage、N、方法、尝试 pool 配置和可用显存快照；已有 partial 保留。不自动重试，不退回 eager。GPU0 不可用即失败，plan-only 不访问 GPU。

## 结果

同轮、同 N 的六臂配对计算：

```text
Δ_SM  = S + M - SM
Δ_MA  = M + A - MA
Δ_SMA = S + M + A - SMA
I     = Δ_SMA - Δ_SM - Δ_MA
```

分别报告逐轮值、median、inclusive IQR、未缩放 MAD、range，以及同轮 N100−N50 的残差差值。不裁剪，不拟合 c，不根据数据自动宣称“稳定”“近零”或机制成立；没有统计显著性声明。不能用各臂中位数的和替代配对残差中位数。

固定 graph 边界反例：若 T_L(N)=L*d+h/N，没有邻接收益也得到 Δ_SM=Δ_MA=h/N、Δ_SMA=2h/N、I=0。N 增长时残差下降可提示批次边界影响，但缓存/graph 大小仍可能共同变化；N 稳定且 I 小也只支持当前链的组合近似，不推广所有相邻对或整图。

若以后用正 c 表示节省，修正符号为 C_ctx=C_add−A*c。方案排名看邻接数差，不能把共同的总修正当成方案间误差。此次没有把这个参数写入代价模型。

## scnet 执行

使用原私有 Kernel 环境，由用户执行：

```bash
cd ~/private_data/LJ/Fusion-placement-search
git pull --ff-only
SEQ_OUT="$PWD/data/sequence_composition_$(date +%Y%m%d_%H%M%S)"
bash ~/private_data/LJ/dcu-kernel310-20261031/run_python.sh \
  -m calibration.probe_sequence_composition --out-dir "$SEQ_OUT"
echo "$SEQ_OUT"
```

采完复验（不访问 GPU）：

```bash
bash ~/private_data/LJ/dcu-kernel310-20261031/run_python.sh \
  -m calibration.probe_sequence_composition --verify "$SEQ_OUT"
```

回传整个 SEQ_OUT，包含 plan/environment/partial/result 或 failure。失败包同样保留。verify 检查采样顺序、Event 除数、两个契约、数值状态、调用数、源 hash，并重算全部汇总；失败包 complete=false，退出非零。

源码在首个 GPU 操作前写入 plan，protocol_id 绑定所用源码；`--plan-only --out-dir ...` 可生成无 GPU 协议。仓库冻结版本在 `data/sequence_composition_protocol_v1/plan.json`。复验必须使用同一源码版本，不修改旧登记使之追上新源码。
