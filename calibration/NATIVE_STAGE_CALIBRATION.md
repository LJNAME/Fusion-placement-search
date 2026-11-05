# Native阶段校准入口

2026-11-04。新增SiLU与Mul独立阶段采集，先通过CPU计划和复核测试。本轮没有GPU测量，不重做LightOP Adapter、Dispatch或Native/Fused整链实验。解析公式、冻结规范与已有Profile均不改。

## 已有数据继续使用

| 数据 | 当前用途 |
| --- | --- |
| 正式LightOP完整调用17.8112 μs | 精确身份/契约匹配时，直接用于模式B的Fused成本 |
| Native整链10.09280 μs、完整Fused18.84796 μs | 保留原冻结实验的排序、误差与局部regret；新模型开发诊断 |
| 已有LightOP Dispatch | 保留目标Kernel执行证据，不重新采集 |
| 旧硬件率参数 | 保留历史参考，不等同当前Graph参数 |
| KERV结果 | 保留其实现与语义范围，不改名LightOP成本 |

17.8112与18.84796来自不同批次、不同数据角色。旧local_pair原来的独立评价结果不改；在新模型设计中使用后，不能再称新模型的独立测试集。

仅复核历史两候选的选择与regret，不需要本采集器。新阶段记录服务未测候选的成本约束；能否改善预测要在后续独立任务验证，不以获得新记录本身判定成功。

## 两条路径

| 阶段 | 计时区间 | 计时外 |
| --- | --- | --- |
| SiLU | `aten.silu.out(gate, out=output)` | 生成gate/up、预分配left与输出 |
| Mul | `aten.mul.out(left, up, out=output)` | `aten.silu.out(gate, out=left)`产生真实BF16左输入 |

固定GPU0/gfx936，BF16 `[1,256,3072]`。保存gate/up/left/output的实际Tensor shape、stride、dtype与布局来源，不用随机left代替SiLU结果。

每阶段50个独立预分配输出，两阶段合计150 MiB输出缓冲；输入、参考、快照和Graph pool另计。三个输入共享只读，50次eager准备、单Graph捕获、10次Graph预热、正式7×50。奇偶轮交替SiLU/Mul顺序，不补轮、不挑样本。每轮计时外填NaN并同步；同GPU同stream的Event包围Replay，不含输入生成、left生成、分配及数值检查。

两参考：SiLU输出对BF16/FP32 SiLU；Mul输出对BF16 left×up及其FP32计算。Mul参考使用真实已舍入的BF16 left，不偷换成整链FP32表达式。rtol=0.01、atol=0.001，逐轮检查全部50个输出、三个输入字节不变、left确为SiLU(gate)、输出不重叠且相互独立。

输入恢复无需执行：两阶段都不原地修改输入。已知profiler环境注入及Python profile/trace被拒绝；不调用profiler，也不声称检查覆盖所有外部剖析。

## 结果范围

`identity.json`绑定实际Torch构建、扩展二进制与采集源码哈希、HIP/驱动、架构、设备ordinal、算子schema、实现、实际布局、阶段Recipe与Graph契约。UUID和安装路径只作溯源，SCNet实例重启不会仅因UUID改变而拒绝。

计时失败或IQR/Median>10%标记该阶段quality_failed，仍保留其余固定尝试。数值、执行、Capture错误立即停止，保存traceback、pool、显存及last_stage；shell保存进程返回码和完整日志。

CPU review重建计划/身份、检查14次顺序和时间戳、逐条原始记录与summary一致、重算质量。计时数据只输出calibration_candidates，**不自动创建accepted Profile、不更新参数、不发布整链加和、纯Graph开销或新regret**。质量失败仍保留分阶段结果；不得把通过的某阶段推广为整链覆盖。

独立阶段的缓存/调度上下文与整链不同。相加只是后续可检验的近似，不用于反推10.09 μs的唯一分解，也不意味着已获得跨Shape或设备泛化能力。

## 当前CPU验证

仓库根目录：

```bash
python -X utf8 -m calibration.native_stage_calibration plan --output /fresh/native_stage_plan
python -X utf8 -m pytest calibration/tests/test_native_stage_calibration.py -q
```

shell入口默认为plan：

```bash
export FUSION_PYTHON=/path/to/validated/python-or-launcher
bash tools/run_native_stage_calibration.sh /fresh/native_stage_plan plan
```

plan不初始化GPU、不读取旧整链耗时作拟合。`run`为待GPU验证路径，本轮没有执行；不要求安装LightOP或重新提供FlagScale源码。若run时已设FLAGSCALE_ROOT，则沿用其厂商环境初始化脚本。

未来完整回执的CPU复核：

```bash
python -X utf8 -m calibration.native_stage_calibration review \
  --input /path/to/returned_directory --output /fresh/native_stage_review
```

保留所有历史GPU数据。只有当前独立阶段的身份与契约确实缺少可复用记录时才新增测量；Pack/LightOP阶段仍后置，不启动GPU1、通信或MCTS。

新数据使用规则与完整包操作见 [服务器交付](NATIVE_STAGE_DELIVERY.md)，包含独立阶段与整链Graph开销的差别；不直接将两阶段相加回填。

## 首批回传：2026-11-05

采集revision `5034028`，24个原始文件逐字节归档至 `data/native_stage_calibration_20261105_01/raw/`。冻结CPU reviewer在接收副本及仓库副本均返回 `calibration_candidates_ready`；复核结果逐项一致。原9个采集源SHA与Git对象匹配，统计另行从14条原始尝试重算。

| 阶段 | 中位数 / μs | 最小 / 最大 / μs | IQR/Median |
| --- | ---: | --- | ---: |
| SiLU.out | 6.5376001596 | 6.5343999863 / 6.7232000828 | 0.5384% |
| Mul.out，left由SiLU产生 | 3.6351999640 | 3.6191999912 / 3.6575999856 | 0.5722% |

两阶段各7轮×50次全部保留，回执的700个正式输出均完成双参考检查，三个输入字节保持、left派生与输出独立检查通过。CPU核对回执和冻结源，不声称重做未归档的GPU Tensor数值。

当前Torch构建、扩展binary、HIP/驱动、架构和执行flags与旧正式LightOP批次的可比字段均匹配；UUID变化仅作溯源。阶段/完整链的Recipe与缓存上下文不同，环境字段匹配不能替代链路验证。

阶段中位数之和10.1728001237 μs，减历史Native链10.0927996635 μs为+0.0800004601 μs（+0.7926%）。这是跨批次历史诊断，不是本批整链实测、纯Graph开销估计或加性模型验证。

原729探索场景在“独立阶段=core+h+c”的条件映射下全部高估Mul，预测最小4.1172235403 μs，仍比观测中位数高13.2599%。原网格无缓存等假设和开销范围没有覆盖本次观测；不能据此拒绝质量通过的数据，也不能挑选最好的一格当作校准结果。所有场景残差保存在 `review/hypothesis_residuals.csv`，没有筛选或拟合参数。

本批是两条有效阶段校准候选，新增accepted Profile仍为0。旧完整Profile、整链评价和模型参数保留。下一项仅CPU核对原假设范围与独立阶段/链路映射，不扩大GPU测量；先解释已有数据，再决定是否需要新的校准或独立验证。

## 实测数据使用更新

当前通过 [实测优先接口](../cost_model/NATIVE_MEASURED_FIRST.md) 直接查询已复核的两个独立阶段；未命中由调用者的解析Profile补缺。旧raw/review状态不改，不再以729网格分析或内部参数唯一识别作为成本使用前置。Mul只保留一次有限口径检查，后续转向方案级预测与搜索验证。
