# 组级 Key 直接重测

入口：`python -m calibration.group_key_remeasurement`。旧标签保留；新数据在采集时直接绑定组级 Key。

## 当前状态

- 草案：`data/group_key_remeasurement_draft_v1/manifest.json`，12 个 slot，正式 Key 为 0。
- `check` 校验草案、源码和 KERV 临时能力，无需 GPU。
- `collect` 在实际 GPU 环境绑定 Key、保存冻结清单，再采集单 Kernel 原始记录。
- `verify` 独立复算记录身份、覆盖率和统计量；可显式导出离线 Profile。
- 本机无 GPU，只完成 CPU 协议验证。未采新 GPU 数据，未接在线 SROP，也未更新预测器。

## 首批范围

Qwen3-0.6B 最小 MLP，batch 1、seq256、BF16；微测输入为连续行主序，双卡各 6 项。

| 每卡项目 | 实现 | 物理 Shape |
|---|---|---|
| gate projection | `torch.mm` | M=256、N=3072、K=1024 |
| up projection | `torch.mm` | M=256、N=3072、K=1024 |
| SiLU | `torch.nn.functional.silu` | 1×256×3072 |
| Mul | `torch.mul` | 1×256×3072 |
| down projection | `torch.mm` | M=256、N=1024、K=3072 |
| SiLU-Mul 融合 | 固定源码的 KERV Triton | 1×256×3072 |

gate/up 虽然物理维度相同，当前 Key 仍保留不同语义属性，先分别记录。该清单覆盖这一个 5 节点子图的局部候选，不代表完整 Transformer Profile。

连续布局是微测输入条件，不是已核实的上游布局。MatMul 的图隐式输入与权重展开为 `[M,K]`、`[K,N]`，结果用零拷贝 view 对齐逻辑输出 Shape；这些约定写入最终 Key 的 `measurement_contract`。不同采集实现使用新版本身份。

## 环境绑定与冻结

`collect` 使用实际 Torch/HIP/Triton 版本、Torch 编译配置 hash、设备属性、UUID/PCI 信息（若可取得）、主机及可见设备环境变量生成环境身份。当前采集器要求草案涉及的两张 HIP 设备均可见。

逻辑设备序号不冒充物理卡号：`physical_index` 保留为 null，实际定位证据保存在 `uuid`、`pci_bus_id`、`locator`。没有 UUID/PCI 的环境只能按记录中的主机、可见性和 Torch 枚举定位，换环境后不能假定是同一物理卡。

冻结清单在首个被测算子调用前写入 `frozen_manifest.json`，包含完整 Key、环境、采集源码 hash、数值阈值、采样配方和具体 block 顺序。默认 `order_seed=0`；各 Shape 的双卡 block 相邻，隔轮反转设备及实现顺序。

草案的 `specification_id` 是 `key_id(kernel_key_template)`，不作为测量身份。正式 `kernel_id` 是绑定后的完整 Key hash；`draft_id`、`environment_id`、`manifest_id` 计算时排除各自 ID 字段。JSON 序列化后重新校验。

## BW1000 运行

scnet 上由用户执行。先分配至少 2 张 BW1000，激活已有 Torch/HIP/Triton 环境；当前开发清单只使用作业内可见的逻辑 GPU 0、1，不要求使用全部可见卡。

在 Fusion 仓库根目录执行一键流程，`FLAGSCALE_ROOT` 替换为 scnet 上的实际 FlagScale 源码路径：

```bash
git pull --ff-only
FLAGSCALE_ROOT=/path/to/FlagScale bash tools/run_group_key_remeasurement.sh
```

默认结果目录 `data/group_key_mlp_bw1000_v1`。脚本依次完成 check、smoke collect、smoke verify、formal collect、formal verify/export。任一步失败即停止；不会继续正式采样，也不会自动重试。已存在的结果目录拒绝覆盖。

如需指定 Python 或新结果目录：

```bash
PYTHON_BIN=python FLAGSCALE_ROOT=/path/to/FlagScale \
  bash tools/run_group_key_remeasurement.sh data/group_key_mlp_bw1000_v2
```

顶层 `workflow_plan.json` 固定命令及启动器源码 hash；逐步日志为 `check.log`、`smoke_collect.log`、`smoke_verify.log`、`formal_collect.log`、`formal_verify.log`，`workflow_summary.json` 保存状态和墙钟。父进程计时包含全部子进程启动、前置检查、采集、复验和日志 IO；不包含父进程自身启动及最后汇总文件写入，不能再叠加子层计时。回传整个结果目录；成功时测量表位于 `formal/verified_profile.json`。

以下为分步执行入口，和一键流程二选一，避免重复采样：

在 Fusion 仓库根目录执行。先通过正常 Git 流程同步当前改动；`FLAGSCALE_ROOT` 指向固定版本 `f8e9a62` 的 FlagScale 源码。默认查找兄弟目录 `../FlagScale-main/FlagScale-main`；目录不同时显式指定。

```bash
FLAGSCALE_ROOT=/path/to/FlagScale
python -m calibration.group_key_remeasurement check --flagscale-root "$FLAGSCALE_ROOT"

python -m calibration.group_key_remeasurement collect \
  --flagscale-root "$FLAGSCALE_ROOT" \
  --mode smoke --order-seed 0 \
  --out-dir data/group_key_mlp_smoke_v1

python -m calibration.group_key_remeasurement verify \
  --bundle data/group_key_mlp_smoke_v1
```

smoke 默认每 Key 10 次预热、3×20 次计时，共 36 个 block、720 次正式调用（不含预热、数值检查、编译）。数值或派发检查失败的 Key 不采时延，其余 Key 继续，失败原始记录保留；验证返回非零。缺 GPU 时保留 `failure.json`，不生成伪测量标签。

smoke 检查通过后，正式开发采样使用独立目录，避免覆盖或挑选重跑结果：

```bash
python -m calibration.group_key_remeasurement collect \
  --flagscale-root "$FLAGSCALE_ROOT" \
  --mode formal --order-seed 0 \
  --out-dir data/group_key_mlp_formal_v1

python -m calibration.group_key_remeasurement verify \
  --bundle data/group_key_mlp_formal_v1 \
  --out data/group_key_mlp_formal_v1/verified_profile.json
```

formal 默认每 Key 10 次预热、7×50 次计时，共 84 个 block、4,200 次正式调用。参数与数值阈值在新计时前冻结。smoke/formal 是开发采样，不是独立测试任务冻结；同 Shape 重采不构成新留出任务。

## 数值与派发

KERV 必须显式启用 Triton。先通过一次 Python JIT-launch 观察确认公共入口派发融合实现；这不等于 GPU profiler 的 Kernel 数量证据。CPU/native 回退、错误 dtype、非连续布局或超出宽度 guard 都不能产生融合时延标签。

所有算子与 CPU FP32 计算后转 BF16 的参考值核对；MatMul `rtol=.02/atol=.05`，逐元素及融合 `rtol=.01/atol=.001`。融合另与未融合 BF16 参考核对，检查 gate 输出别名、up 不变及有限值。阈值写在冻结配方，不按采样结果放宽。

KERV 每次调用前恢复原始 gate，包括预热。恢复排在 Event 起点之前，计入 profiling wall。未融合 SiLU/Mul 分别测量，不把两算子对照探针的总墙钟当作一次 Key 查询成本。

## 原始记录与成本

每次调用使用一对 GPU Event，保存每个调用的秒数。Event 区间排除 gate 恢复；该口径仍受主机提交与队列空隙影响，微小差异需结合样本波动判断。

每个 block 保留开始时间、完整 wall、设备已分配/保留显存，以及负载、温度、频率的前后观测。后 3 项通过当前 Torch 管理接口尝试读取，接口不可用时记录原因，不能当作已监控到稳定状态。

按 Key 保存输入准备、首次调用/数值验证、预热与采样 wall。`collection_wall_s` 从 `collect()` 入口计到复验结束，包含共享环境准备、采集内源码检查、文件 IO、失败项和复验；不含进程/模块启动、CLI 前置检查及最后汇总文件写入。若部署流程每次启动 CLI，需另计这些开销。嵌套计时不重复相加。单 Key 归属成本依赖本轮缓存和采集顺序，不能据此宣称在线 SROP 总成本收益。

| 文件 | 内容 |
|---|---|
| `frozen_manifest.json` | 环境、最终 Key、源码、配方、顺序、原草案 |
| `setup_records.jsonl` | 逐 Key 准备、数值/派发证据、阶段 wall、失败 |
| `raw_measurements.jsonl` | 按冻结顺序逐 block 的 Event 样本、wall、监控与状态 |
| `collection_summary.json` | 完整性、覆盖与总成本；不含时延表 |
| `failure.json` | 无法开始或异常中断；已有记录保留 |
| `verified_profile.json` | 显式 `verify --out` 后生成的离线测量表 |

`verify` 复算冻结清单和记录 hash、草案到正式 Key 的绑定、环境血缘、实际顺序、每轮样本数、数值/别名/融合派发证据；输出中位数、block 中位数 IQR 与首末变化，保留全部原始样本，不做异常值裁剪。完整覆盖前拒绝导出完整 Profile。

输出目录和导出文件必须是新路径；不自动续跑、覆盖旧结果或重试到成功。中断后保留目录，后续重跑使用新目录并记录原因，不能静默选取较好结果。回传整个采集目录，不能只传统计表。

## 数据用途

旧数据原样保留。新标签进入独立开发数据集，经显式校验导出后供固定评价器或后续实验使用；不会自动写入代价模型。当前工具不实现在线 SROP 选点、预测器更新、MCTS 重搜或完整算子图执行，也不将临时 KERV 能力提升为正式下游可执行能力。
