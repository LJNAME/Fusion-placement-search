# 混合动作预算优势与真实耦合证据核对

2026-11-09。P0不改算法或成本，只检验已有混合动作的预算收益；P1只核对已有真实测量。无完整3B搜索或新增GPU测量。

## P0：主实验固定8／16次评分

仍用三个3B中等子图、blocking HTAE、free/cap75和种子0／7／23。8与16均小于最小混合域24，排除完整覆盖空间带来的优势。目标在实验前固定为Exact（1e-9ms浮点容差）和Exact Gap≤1%。

比较Joint Node、Joint Hybrid、Random Node、Random Hybrid。后两者使用与MCTS随机补全一致的融合概率0.7，不声称均匀采样完整方案。所有方法计入相同的两卡未融合初始化；不可行候选计费，缓存命中不计费；每条轨迹都恰好消耗8或16次独立实际模型调用。Random Hybrid用于区分空间缩小与树选择的收益。

四个Exact可行任务×两预算×四方法×三种子＝96次冷缓存运行。两个已知Exact无解的cap75任务保留无解状态，不重复统计搜索达标率。成功分母保留所有种子；未达标次数记null／右删失，不能当作0次或仅对成功者算总体平均。

### 主实验结果

表内为三种子中位Exact Gap。

| MLP场景／预算 | Joint Node | Joint Hybrid | Random Node | Random Hybrid |
| --- | ---: | ---: | ---: | ---: |
| free／8 | 6.5288% | 6.5288% | 5.6503% | 6.5288% |
| free／16 | 5.6341% | 6.5288% | 5.6503% | 5.6503% |
| cap75／8 | 27.4989% | 14.9864% | 5.6503% | 17.4636% |
| cap75／16 | 17.4667% | 14.9864% | 5.6503% | 5.6503% |

MLP四方法在8／16预算内均为0/3达到1%目标，全部找到可行解。串行尾部free的两种Joint和Random Hybrid均3/3达Exact，Random Node为0/3，Gap8.4736%；注意力free四方法均在初始化第1次已达Exact。主实验没有证明混合Joint在很小预算下同时优于原Joint和Random。

## P0补充：已冻结64预算轨迹的达标曲线

这是看到8／16负面结果后明确追加的探索性分析，不替换主表，不重选目标或修改参数。复用branch_search_medium_v1的两种Joint轨迹及stage_search_medium_v1的Random Node轨迹，只补12条同混合域Random冷缓存轨迹。旧／新曲线必须逐项匹配独立8／16运行的已暴露前缀；图、容量、Exact成本也须一致。

### 首次达到Gap≤1%的实际评分数

按种子0／7／23列出；>64表示该旧轨迹到64次仍未达到目标，不当作成功值参与中位数。

| 场景 | Joint Node | Joint Hybrid | Random Node | Random Hybrid |
| --- | --- | --- | --- | --- |
| MLP free | 48／52／51 | 21／22／18 | >64／>64／>64 | 44／30／26 |
| MLP cap75 | >64／44／>64 | 23／32／37 | >64／>64／>64 | 44／30／26 |

free场景中位达标数51→21，比原Joint少58.82%；同域Random中位为30，混合Joint少30%。这些时点都早于60项空间耗尽。cap75中混合Joint比原Joint更稳定达标，但其1%目标中位32次，未优于同域Random的30次；逐种子只赢1/3。

### 首次达到Exact的评分数

| 场景 | Joint Node | Joint Hybrid | Random Node | Random Hybrid |
| --- | --- | --- | --- | --- |
| MLP free | 48／52／51 | 49／22／44 | >64／>64／>64 | 48／30／59 |
| MLP cap75 | >64／>64／>64 | 38／32／37 | >64／>64／>64 | 48／30／59 |

结论依赖质量目标与场景。当前只支持“MLP free的已有种子中存在有限预算优势”，不支持所有预算或任务上混合Joint都更好；补充曲线是探索性结果，3种子不能给出普适或统计显著性结论。

## P1：已有测量中的耦合候选

六条既有比较记录按实现与执行契约分组：KERV silu_mul Graph四例始终Fused更快，KERV AddRMS Graph一例Fused更快，LightOP split+pack eager一例Native更快。KERV与LightOP、eager与Graph的胜负变化不算同一实现／契约下的融合偏好翻转。

另核对kerv_profile_admission_v1原始两卡、两臂的7×50样本。输入、实现源码和完整测量契约一致，跨卡只忽略物理UUID差异；原测量身份不改。

| 指定KERV silu_mul，BF16 [1,256,3072] | GPU0中位数 | GPU1中位数 | 点估计偏好 |
| --- | ---: | ---: | --- |
| Native完整SiLU→Mul | 10.316780μs | 11.254359μs | GPU0 |
| Fused | 7.251180μs | 7.232000μs | GPU1 |

存在“融合状态变化后点估计最优卡改变”的真实数据候选。但Fused两卡仅差0.019180μs；GPU0样本范围7.200000～7.273560μs，GPU1为7.219160～7.951960μs，两者重叠。Native跨卡样本范围也重叠。min/max不是置信区间，当前不能据此确认稳定设备排序；也不能反向断言真实差异必为0。

原准入仅接受指定Fused组的比较代理，Native仍comparison_only，不是完整方案成本。宽度3072也不能直接复用到3B的11008 FFN。因此标记point_estimate_candidate_only、confirmed_coupling=false，不修改Profile准入或当前搜索模型。若后续追踪，只需先验证这一固定实现／Shape的跨卡排序，之后才能讨论图级影响；本轮未创建新GPU任务。

当前已确认的真实同实现融合偏好翻转为0，已确认的真实完整方案最优放置变化为0。上一轮模型反事实中的数值交互仍保留，但不升级为真机耦合证明。

## 交付与复现

```bash
python -B -X utf8 tools/evaluate_hybrid_budget.py --output data/hybrid_budget_v1 --budgets 8,16 --seeds 0,7,23 --followup-existing64
python -B -X utf8 tools/audit_coupling_evidence.py --output data/coupling_evidence_audit_v1
```

输出目录必须为新目录。P0 report.json保存主实验与单独标注的补充曲线，summary.csv只含96条主实验结果，curve_summary.csv保存48条达标曲线摘要（36条复用、12条新Random）。P1 report.json保存测量矩阵、分组结果、范围与来源SHA。

18项CPU测试通过；P0完整冷缓存重放除墙钟外一致，所有8／16前缀与补充曲线一致；P1重放逐字节相同。未修改代价公式、MCTS选择／回传、融合能力或原始Profile。
