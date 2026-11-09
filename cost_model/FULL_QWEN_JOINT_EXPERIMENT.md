# Qwen 完整图搜索对照（2026-11-09）

Qwen3-0.6B 与 Qwen2.5-3B，四种方法、两个显存场景、三个种子，共 48 次 CPU 搜索，3,072 次实际完整方案评分。

## 结果

下表为三个种子的最佳 makespan 中位数，单位 ms；括号为找到可行方案的运行数。

| 模型／场景 | 原始 Joint | 混合 Joint | Random 原空间 | Random 混合空间 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B／free | 10.12396（3/3） | 10.12396（3/3） | 10.18784（3/3） | 10.18784（3/3） |
| Qwen3-0.6B／cap75 | 10.24017（3/3） | 未找到（0/3） | 26.49267（3/3） | 未找到（0/3） |
| Qwen2.5-3B／free | 29.37739（3/3） | 29.37739（3/3） | 29.37739（3/3） | 29.37739（3/3） |
| Qwen2.5-3B／cap75 | 47.04910（3/3） | 30.23505（3/3） | 47.26194（3/3） | 30.50624（3/3） |

- 3B cap75：混合 Joint 比原始 Joint 中位预测成本低 **35.74%**，最佳方案的跨卡边中位数从 372 降为 70；同空间 Random 为 72 条。
- 混合 Joint 比同空间 Random 的中位成本低 **0.89%**，但逐种子只赢 1/3，不能据此认定树选择具有稳定优势。
- 3B free：混合 Joint 的 seed0 找到 29.19434 ms（第49次评分），其余两种子仍返回 29.37739 ms 的初始 Native 方案。中位数相同不代表三条轨迹完全相同。
- Qwen3 free：两种 Joint 均在第3次评分达到最终最佳，之后没有改进；Random 最佳仍是第1次 Native 初始化。

完整逐次轨迹、可行性和最优方案保存在 `data/full_joint_models_v1/`。汇总见 `summary.json`、`summary.csv`。

## 参考方案

单独评价八个预声明方案：单卡／半层切分、两种设备方向、Native／仅 AddRMS 融合。它们没有作为搜索种子注入。

| 模型／场景 | 最低已知合法参考 ms | 原始 Joint 相对参考 | 混合 Joint 相对参考 |
| --- | ---: | ---: | ---: |
| Qwen3／free | 9.95933 | +1.6530% | +1.6530% |
| Qwen3／cap75 | 10.07620 | +1.6273% | 无可行结果 |
| 3B／free | 29.06952 | +1.0591% | +1.0591% |
| 3B／cap75 | 29.19805 | +61.1378% | +3.5516% |

这些是有限参考，不是全图 Exact。相对参考差距允许为负；本次未计算全空间最优或真实 GPU regret。搜索结果仍未达到最低已知参考。

## Qwen3 cap75 的失败核对

六次混合域运行（Joint 与 Random，各三种子）均在64次评分内没有可行候选，原记录保留。

事后仅评价一个合法性见证：全部已识别 MLP 分支放 GPU1，其余主干放 GPU0，全部 Native。通过同一个混合动作接口逐步生成，峰值为 918,093,824／355,467,264 B，均低于每卡 952,811,520 B；预测 15.87973 ms，112条跨卡边。

因此混合域有解，本轮失败是预算内没有找到可行区域，不能写成空间无解。见证不加入原64次评分、不替换实验结果。

当前 embedding 与 lm_head 共享参数并强制同卡。混合域的主干只可留当前卡或启用未用卡，不能返回；尚未放置的绑定 lm_head 会阻止主干离开 embedding 所在卡（`legal_search.py::_stage_devices`）。半层参考在 `L014.input` 要求去 GPU1 时只允许 GPU0，因此该参考不属于当前混合域。分支仍可独立放置，不能将此解释为所有节点都被强制同卡。

见 `hybrid_feasibility_diagnostic.json`。本轮没有修改动作规则或 MCTS 核心算法。

## 冻结口径

- 完整静态前向图：Qwen3 591节点／730边／28个分支区域；Qwen2.5-3B 687节点／866边／36个分支区域，BF16、seq256、microbatch1。
- 两张 BW1000；每方法64次不同完整方案实际评分、种子0/7/23。两卡 Native 初始化各占一次，不可行方案也计费，重复缓存命中不计费。初始可行方案始终保留。
- 原始 Joint、混合 Joint、原空间 Random、混合空间 Random；沿用当前 rollout 与 Q 处理，不引入 Sequential 或其他新档位。
- 四方法均启用共享参数同卡约束，不把远程共享权重视作零开销。半层参考的 lm_head 返回 embedding 所在卡，不隐式假设权重复制。
- free 使用配置总显存；cap75 为单卡 Native 静态前向峰值的75%，Qwen3每卡952,811,520 B、3B每卡4,687,899,648 B。这是合成容量场景，不是物理显存75%、运行时空闲量或训练显存预算。
- 目标为当前 HTAE 前向 makespan，feature communication 阻塞、overlap_factor=0；分支计算仍可在依赖和资源允许时重叠。
- L1 使用现有解析模型及 LightOP 校准，SiLU×Mul校准系数2.417014088994849，AddRMS为未校准解析成本。当前库声明的多节点候选为 Qwen3各28、3B各36；这不是本轮新 GPU 适配或实测验收数量。
- 本轮不走此前严格 KERV AddRMS 实测查询模式；`fusion_calibrated` 也是解析预测乘校准系数，不是逐Key实测命中。完整模型与全部候选没有真机计时，结果不能与旧严格模式25.239 ms等数值直接比较。
- 两个模型进程并发运行；保存的墙钟仅作诊断，不比较算法加速比。

## 运行与检查

```powershell
python -B -X utf8 tools/evaluate_full_joint.py --model model --output data/full_joint_models_v1/qwen3_06b --budget 64 --seeds 0,7,23 --caps free,cap75
python -B -X utf8 tools/evaluate_full_joint.py --model model_3b --output data/full_joint_models_v1/qwen25_3b --budget 64 --seeds 0,7,23 --caps free,cap75
python -B -X utf8 tools/summarize_full_joint.py --directory data/full_joint_models_v1
python -B -X utf8 tools/inspect_full_hybrid.py --folder data/full_joint_models_v1/qwen3_06b
```

输出目录需为新目录；中断续跑可加 `--resume`，要求配置与源码哈希完全一致。每次完成的运行单独落盘，最佳图按候选SHA去重保存，`plan_path` 对应实际评分最优方案。

21项相关CPU测试通过。48条运行均为64次不同方案评分；逐次可行 incumbent、图指纹、冻结源码、最优方案ID、共享参数约束、manifest融合合法性及参考差距通过离线复核，复核没有额外调用代价模型。见 `verification.json`。评分量分开记录：搜索3,072次、八参考×两模型16次、事后见证1次；新增GPU测量为0。