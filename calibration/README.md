# 硬件校准套件

目标：把"差异化参数"从占位假设变成实测值——单卡有效算力与形状曲线、
四卡链路拓扑与并发性、计算-通信干扰、启动/融合开销——并生成
`calibrated.json` 直接喂给仿真器。整套逻辑与后端解耦：换摩尔卡只换
kernel 源文件（HIP -> MUSA）和编译命令，聚合/报告/验证脚本原样复用。

## 文件

| 文件 | 作用 |
| --- | --- |
| `kernels_hip.cpp` | HIP/ROCm 微基准：launch / gemm / copy / pingpong / concurrency / overlap / fusion，每个模式输出一行 JSON |
| `run_calibration.py` | 编排 + 聚合：跑测量 -> 生成 calibrated 配置 + 校准报告 |
| `validate_real.py` | 端到端验证：真机跑一组方案 vs 仿真器，算 MAPE/Spearman |

## BW1000 双卡校准

当前 Fusion 闭环只校准同构 mesh 内的算子、融合和放置，不测 TP / DP / PP。脚本要求 PyTorch 能看到两张同型号设备，默认使用 Qwen3-0.6B 的 BF16 shape，分别测两张卡，再测 0→1 和 1→0 链路。

```bash
cd ~/private_data/LJ/Fusion-placement-search
source ~/private_data/LJ/FlagScale-main/tools/install/hygon/env.sh

python calibration/calibrate_bw1000.py \
    --devices 0,1 \
    --dtype bf16 \
    --out-dir data/bw1000_2card \
    --quick

python calibration/calibrate_bw1000.py \
    --devices 0,1 \
    --dtype bf16 \
    --out-dir data/bw1000_2card
```

已有正式基础数据时，用增量模式补测 Q/K RMSNorm、Q/K RoPE 和 Softmax。Softmax 按实际路径计时：BF16 输入，内部转 FP32 计算，再转回 BF16 输出。该模式不重复 GEMM、显存、launch 和链路测试；同名补测会覆盖旧值：

```bash
python calibration/calibrate_bw1000.py \
    --devices 0,1 \
    --dtype bf16 \
    --out-dir data/bw1000_2card \
    --supplement \
    --base-raw data/bw1000_2card/raw_measurements.json

python -m cost_model.verify_measured_profile \
    --raw data/bw1000_2card/raw_measurements.json \
    --hardware data/bw1000_2card/bw1000_2card.calibrated.json \
    --expect-complete
```

完整验收应输出 20 个 `measured_exact_shape`、1 个 `logical_input`、0 个 `analytic_fallback`，confidence 0.95 且 OOD=false。

先跑 `--quick` 检查环境，成功后删除同名输出目录或直接用正式命令覆盖。输出：

- `raw_measurements.json`：每轮原始时延、Qwen3 matmul/reference op、显存拷贝和双向链路数据；
- `bw1000_2card.calibrated.json`：可以由 `load_hardware()` 读取的两卡 profile。

脚本不把 eager `SiLU × Mul` 当作 fused kernel，也不测计算通信干扰，所以输出配置中的 `real_fused_kernels` 和 `compute_communication_interference` 都是 `false`。这两项以后单独补测。

## 旧版海光 K100 运行步骤（历史）

```bash
# 1. 编译（ROCm 环境，需 rocblas）
mkdir -p build
hipcc -O2 -o build/kernels_hip calibration/kernels_hip.cpp -lrocblas

# 2. 跑校准（自动编译 -> 全部模式 -> 生成配置和报告）
python calibration/run_calibration.py \
    --nominal configs/hygon_4cards.example.json \
    --build-cmd "hipcc -O2 -o build/kernels_hip calibration/kernels_hip.cpp -lrocblas" \
    --binary build/kernels_hip \
    --out-config configs/hygon_k100_4cards.calibrated.json \
    --report calibration_report.json

# 3. 端到端验证（先按 validate_real.py 头注释的格式准备 measured.json）
python calibration/validate_real.py \
    --hw configs/hygon_k100_4cards.calibrated.json \
    --measured measured.json --out validation_report.json
```

## 每个测量对应仿真器哪个参数（差异化参数重点）

| 测量 | 模式 | 生成参数 | 意义 |
| --- | --- | --- | --- |
| GEMM 逐 K 扫描 | `gemm` | `shape_model.matmul.compute.points`（table 曲线） | 形状效率：小 K 低效是 roofline 公式层抓不到的（当前用 rocblas_sgemm FP32，DTK 兼容最稳；微基准/仿真器/真机执行器三者同为 FP32，闭环自洽） |
| 大数组拷贝 | `copy` | `devices[].efficiency.mem_bw` | 有效显存带宽 |
| 空 kernel 启动 | `launch` | `launch_overhead` | 每个融合组的启动成本 |
| 融合 vs 未融合 | `fusion` | `fusion_overhead` | 组内每多一个算子的开销 |
| 双卡 ping-pong | `pingpong` | `link_matrix` / `link_bw` / `comm_latency` | 逐链路拓扑（ASTRA-sim 参考） |
| 两对并发 vs 串行 | `concurrency` | 报告里的"互联判定" | 决定用逐链路模型还是共享总线模型 |
| 并发 GEMM+拷贝 | `overlap` | `interference` | 计算-通信干扰（Echo 参考），当前纯占位 0.1 |

## 本地无卡验证（dry-run）

```bash
python calibration/run_calibration.py \
    --nominal configs/hygon_4cards.example.json \
    --dry-run \
    --out-config /tmp/calib_dry.json --report /tmp/calib_report.json
```

用合成数据跑通"测量 -> 聚合 -> 生成配置"全链路，确认脚本没问题后再上机。

## 摩尔卡（MTT）移植清单

1. 复制 `kernels_hip.cpp` 为 `kernels_musa.cpp`，做机械替换：
   - `hipSetDevice/hipMalloc/hipMemcpy/hipEvent*/hipDeviceSynchronize` -> `musa*`；
   - `rocblas_gemm_ex` -> MUDNN 的 GEMM 接口（或厂商自带 benchmark），
     没有 MUDNN 时先用手写分块 GEMM 测形状曲线，峰值单独用官方工具测；
   - 编译命令换 `mcc`/`musacc`（按厂商 SDK 文档）。
2. `run_calibration.py` 只改 `--build-cmd` 和 `--binary`，聚合逻辑不动；
3. `validate_real.py` 完全复用；
4. 测量顺序按 `simulator/sensitivity.py` 的结论：mem_bw / interference /
   启动融合开销 / 链路 / 通信延迟 优先于 flops / 形状曲线 / max_mem。

## 校准闭环

微基准校准"零件"，验证集校准"整机"：跑 5~10 个真实 (图, 方案)，对比
仿真器与真机的 makespan / 峰值显存。若存在系统性偏差（如整体偏快 20%），
在 config 加全局效率修正，而不是逐个调参数。所有数值保留来源
（`source` 字段：机器、日期、方法），可追溯。

## 组合层真机验证（接近真实规模的图压到多卡）

微基准只标定叶子；组合层（调度/重叠/争用/显存生命周期）必须用
**真实组合出的工作负载**验证。做法：用仿真器的 (图, 方案) 格式自动
生成真机执行器，每个融合组 = 一台设备一条 stream 上的 kernel 序列，
组依赖用事件同步、跨设备边用 D2D 拷贝，量墙钟 makespan + 逐卡峰值显存。

```bash
# 1. 生成接近真实规模的图 + 方案（Transformer 2048 / 深 MLP / 12×GEMM 链 / 菱形）
python calibration/gen_validation_plans.py --out validation_plans.json

# 2. 生成执行器 -> 编译 -> 真机跑 -> 对比校准后的仿真器（一步到位）
python calibration/run_validation.py \
    --plans validation_plans.json \
    --build-cmd "hipcc -O2 -o build/plan_executor_hip build/plan_executor_hip.cpp -lrocblas" \
    --binary build/plan_executor_hip \
    --hw configs/hygon_k100_4cards.calibrated.json \
    --measured measured.json --report validation_report.json

# 本地无卡自检（合成真机数据）
python calibration/run_validation.py --plans validation_plans.json \
    --hw configs/hygon_4cards.example.json \
    --measured /tmp/measured.json --report /tmp/validation_report.json --dry-run
```

验证报告里 makespan MAPE / Spearman、逐卡峰值显存 MAPE 就是组合层的
"校准误差"——参考论文（Sailor/Echo/ASTRA-sim）都报告这类端到端指标。
误差归因规则：误差随某特征系统变化 = 结构问题（改模型，如加争用/重叠
模型）；常数偏移 = 数值问题（调参数/加全局效率修正）。

归因自动化：

```bash
python calibration/attribute_error.py \
    --measured measured.json --hw configs/hygon_k100_4cards.calibrated.json \
    --out attribution_report.json
```

脚本对每个方案提取组合层特征（组数 / 跨卡传输数 / 传输字节 / 通信占比 /
重叠度 / 显存利用率），算 log 误差与特征的 Spearman，并给出结论：
log std 小 -> "数值问题"，建议全局修正因子（median real/sim）；
某特征强相关 -> "结构问题"，指向对应模型机制（如 comm_share 强相关 ->
争用/重叠模型要改）。这是把"结构 vs 数值"的判断从人工变成自动化。

注意：执行器里的"融合"是同一 stream 上背靠背 kernel（近似），不是真正
的算子融合 codegen；真融合的额外开销差异会在对比中暴露出来，这正是要
校准/修正的对象。摩尔卡移植：kernels 与 rocblas 换成 MUSA 对应物，
生成/编排/验证脚本不动。
## E2 Shape Manifest 与严格划分

E2a 先从 0.6B / 3B、sequence length 128 / 256 / 512 / 1024 的逻辑图生成测量清单。对象按完整 Kernel Key 去重，图中节点只作为 usage；当前清单为 128 个对象，其中 112 个可测，16 个 input / embedding 逻辑节点不计时。此步骤不需要 GPU：

```bash
python -m calibration.e2_manifest \
    --output data/bw1000_e2/shape_manifest.json

python -m calibration.e2_contract \
    --manifest data/bw1000_e2/shape_manifest.json \
    --output data/bw1000_e2/split_contract.json

python -m calibration.verify_e2_interfaces
```

`model-qwen06-to-qwen25-3b` 包含 68 个训练 Key、44 个未见 Key和 12 个测试侧重复 Key；`sequence-short-to-1024` 包含 84 个训练 Key和 28 个未见 Key。重复 Key 记入 `shared_seen_object_ids`，不计作未见 Shape。

真机标签使用 `calibration.e2_records` 的追加式 JSONL 格式。每条记录引用 object id 和 environment id，保留全部计时样本；统计量会在读取时重新计算。Zero-shot 不允许揭示测试标签，Active calibration 只允许逐项揭示 `hidden_object_ids`，由隐藏标签派生的聚合值也保持隐藏。
### E2 真机采集

先运行 quick，输出到独立的 `raw_measurements.quick.jsonl` 和 `environments.quick.json`，不会混入正式标签：

```bash
python -m calibration.profile_e2_bw1000 \
    --manifest data/bw1000_e2/shape_manifest.json \
    --out-dir data/bw1000_e2 \
    --devices 0,1 \
    --assignment shard \
    --quick

python -m calibration.verify_e2_measurements \
    --manifest data/bw1000_e2/shape_manifest.json \
    --environments data/bw1000_e2/environments.quick.json \
    --raw data/bw1000_e2/raw_measurements.quick.jsonl \
    --expected-replicas 1 \
    --expect-complete \
    --max-sample-cv 0.20
```

正式采集：

```bash
python -m calibration.profile_e2_bw1000 \
    --manifest data/bw1000_e2/shape_manifest.json \
    --out-dir data/bw1000_e2 \
    --devices 0,1 \
    --assignment shard

python -m calibration.verify_e2_measurements \
    --manifest data/bw1000_e2/shape_manifest.json \
    --environments data/bw1000_e2/environments.json \
    --raw data/bw1000_e2/raw_measurements.jsonl \
    --expected-replicas 1 \
    --expect-complete \
    --max-robust-sample-cv 0.10
```

`shard` 将 112 个对象分到两张同型号卡，每个对象只测一次；`replicate` 在每张卡上都测一遍，共 224 项。默认读取已有 JSONL 并跳过同一环境下成功的对象；失败项会写入错误记录，下次运行继续补测。单项记录包含原始样本、设备、环境指纹、源码提交和 profiling wall time。

校验器同时报告普通 CV 和稳健 CV（`1.4826 × MAD / median`）。普通 CV 保留单次抢占或系统抖动的影响，只作诊断；正式验收使用稳健 CV，中位数仍作为该 Kernel Key 的标签。
