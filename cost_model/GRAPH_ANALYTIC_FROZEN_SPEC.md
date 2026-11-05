# Graph解析模型冻结规范v1

2026-11-04。冻结CPU解析公式、固定Case、模式A/B、参数来源和标定设计；运行参数尚未校准，预测精度与MCTS接入待验证。实现不再增加参数，`graph_analytic.py`与场景文件按SHA256固定在 `data/graph_analytic_freeze_v1/freeze.json`。

## 适用范围与执行契约

GPU0，BF16，gate/up均为连续 `[1,256,3072]`，只读、分离输入与独立预分配输出。Native为`aten.silu.out → aten.mul.out`；Fused为预分配Pack＋LightOP完整调用。固定同卡局部候选，不覆盖GEMM、GPU1、通信或完整FFN，不能据此声明融合—放置搜索已通过验证。

目标时间为同设备、同stream Event包围一次Graph Replay的批摊销设备时间，50调用/轮，10次Graph预热，7轮全部保留；预分配、输入生成、数值检查和同步位于计时边界外。沿用已冻结执行路径的50次eager准备；eager准备不作为计时标签。IQR/Median≤10%，不补轮、不替换，数值或执行错误停止。没有独立锚点，不声明跨窗口漂移已认证。

旧同步eager循环的Host提交/等待时间不计入目标Graph成本。`graph_stage_overhead_us`是阶段有效残差参数，不宣称等于Host launch或纯硬件调度时延。一次逻辑阶段不自动等于已观测单Kernel。

## 冻结公式

N=786432，S=2N B。阶段(read,write)为SiLU(S,S)、Mul(2S,S)、Pack(2S,2S)、LightOP(2S,S)。算术乘法数量分别N、N、0、2N；SiLU与LightOP各有N个非线性服务单元。逻辑总IO为Native7.5 MiB、Fused10.5 MiB，不是实测HBM流量。

设U=vector_flops_per_s×compute_utilization，V=silu_elements_per_s×compute_utilization，m=lightop_compute_multiplier。

- 计算服务（秒）：SiLU=N/U+N/V，Mul=N/U，Pack=0，LightOP=2N/(mU)+N/(mV)。
- 阶段读（秒）：read_bytes×read_utilization×[(1-p)/(B×u×k)+p/(Bc×k)]。
- 阶段写（秒）：write_bytes/(B×u)。p为读缓存比例，k为访问合并效率，u为带宽利用率。
- 阶段核心（μs）：10^6×max(计算服务,阶段读＋阶段写)。
- 调用总成本（μs）：串行阶段核心之和＋阶段数×h＋c；Fused另加d。

h、c、d分别为Graph阶段、调用及Fused额外开销。算术与非线性服务串行相加、读写串行相加、计算与访存取max均为v1近似。p>0且Bc缺失时保持未知；不假设小Tensor自动命中缓存，不把历史矩阵率作为向量率。

参考OpenXLA的计算/访存重叠与访问修正；实际Recipe、未知传播与Graph项是本项目适配。没有移植其编译器分析或验证海光缓存/调度行为。[官方方法说明](https://github.com/openxla/xla/discussions/10065)

## 两种使用模式

| 模式 | Native | Fused | 当前数值状态 |
| --- | --- | --- | --- |
| A，两臂解析 | C_N＋2h＋c | C_F＋2h＋c＋d | 未校准，缺项不输出正式排名 |
| B，Fused完整Profile | C_N＋2h＋c | 精确匹配后的正式完整耗时 | 旧匹配Case为17.811180353164673 μs，Native仍未校准 |

完整Profile按原reviewer重放审核与严格Key/Contract查询；缺失或不匹配时不得继续使用17.8112这个常量。模式B失败必须显式报告，若另行使用模式A，保留其名称与来源。整臂命中后不再追加Pack、缓存或Graph参数。SCNet重启导致UUID变化只作溯源；实现、布局、架构、软件构建和执行契约仍检查。

模式B是混合来源评价，不承诺比A更可靠。正式Profile的accepted状态不赋予Native预测实测身份。

## 参数来源与范围收敛

本轮检查 `data/bw1000_2card/{raw_measurements,bw1000_2card.calibrated}.json`、旧采集源码、LightOP正式runtime、run05与conditioning审核报告；没有对整个设备资料库作“所有资料均不存在”的声明。

| 参数/事实 | 可用证据 | 冻结处理 |
| --- | --- | --- |
| Shape、BF16元素大小、Recipe逻辑读写 | 原执行代码与已归档runtime/Recipe | 直接确定；布局只适用于已声明/观测Case |
| 架构、设备显存 | 正式runtime为gfx936、68702699520 B | 身份与资源事实，不推出带宽或向量率 |
| B，带宽基准 | 2026-10-19，GPU0 Copy 1341567274866.83 B/s | 仅历史有效率代理，不是峰值或当前Graph约束 |
| 矩阵吞吐 | 历史GEMM率 | 不导入向量/SiLU参数 |
| U、V、m | 无当前独立阶段服务率 | 保留未知，旧网格仅探索范围 |
| u、k、p、Bc | 无当前访问与缓存证据 | 保留未知；p=0、k=1只是无缓存/理想访问场景 |
| h、c、d | 旧launch为同步eager Host wall | 拒绝转用Graph；保留未知 |
| 正式完整Fused | 原严格Profile | 保留完整值及身份，不拆参数 |

旧Copy为256 MiB，读写合计536870912 B，7轮中每轮25次，来自`time_loop`的同步Host wall计时。正式LightOP runtime的HIP为6.3.26113，旧硬件采集为6.3.26093。即使Torch主版本都为2.9.0，也不足以认证相同执行条件。

当前没有证据支持把旧729场景的取值区间收缩成“当前DCU可信区间”；原假设文件不改、结果不追溯调参。物理域只给正率、0<u/k/eta≤1、0≤p≤1和非负开销。有效带宽代理也不是严格硬件上界。新测量前必须明确区分探索域与已获证据的范围。

run05的旧KERV、原地恢复与共享窗口规则，以及conditioning的diagnostic_only记录，不直接提升成当前LightOP分离只读输入的阶段参数。排除原因不是UUID。

## 可辨识性与标定目标

固定两阶段的两臂只依赖G=2h+c，不能分别识别h和c。U、V也只识别为有效率，不能分别识别基础吞吐与compute_utilization。模式A共同G抵消，模式B不抵消；Fused完整Profile替代后无需单独校准其m/d。

729探索场景的翻转证明这些组合项值得核查，不证明其整个范围合理，也不证明单独参数一定需要新增GPU测量。缓存、访问合并、向量率和调用级项未全扫，不声称其无影响。新增数据以约束有效组合和候选差值为目标，不要求唯一还原所有内部成本。

## 最小起步标定设计

| Case | 测量路径 | 适用模式 | 作用 |
| --- | --- | --- | --- |
| C_N1 | 独立预分配SiLU.out | A/B | 约束Native非线性＋访存＋Graph组合服务 |
| C_N2 | 独立预分配Mul.out，左输入由计时外SiLU产生 | A/B | 约束真实Mul输入路径的计算＋访存＋Graph组合服务 |
| C_F1 | 仅预分配cat(out=packed) | 仅需A无Profile预测时 | 约束Pack组合服务 |
| C_F2 | 已打包输入上的LightOP调用 | 同上 | 约束厂商融合组合服务；不伪装成完整Adapter Profile |

Case统一GPU0、BF16、目标Shape、7×50与上述Graph计时语义，分别记录输入布局、实现和Recipe。C_F2可使用现有已验证接口，但原完整调用Profile及Adapter源不改；阶段记录有独立身份，不自动准入为完整组成本。

固定Case模式B最小起步为2个Native阶段Case；模式A若需要补足无Profile预测，再增加2个Fused阶段Case。**2/4是起步测量路径数，不是已证明足够的参数识别集，实际GPU清单仍需采集入口与输入契约核验。** 两阶段观测不能唯一拟合所有隐变量。数据不足时保留可行区间与不确定排序，不能强行点估计。

两组均为待实现的标定设计，本轮没有新增采集器或服务器命令。执行前核验同Case历史记录是否满足身份与契约；能合法复用则减少Case。无法得到合理范围且排序持续不确定时，先约束Native有效组合；不自动扩GPU1、通信、更多算子或训练数据集。

不另测空Graph并将其当作通用每Kernel调度开销，不用旧local_pair整链减预测值反解h；也不为729个假设逐格测量。

## 校准与独立验证

- 旧local_pair保持原冻结结果；因参与模型设计，标记development/diagnostic。正式Fused作为模式B输入，不作该模式的独立验证标签。
- 新C_*归calibration。拟合前冻结公式、参数组合、约束方式及原始数据身份；不按同Case的重复轮次拆训练/测试。
- 首轮验证用同固定Shape的新独立批次，两臂重测，输入seed与校准批次分离；冻结A/B的预测、来源、选择及参数版本后才读取标签。此验证只支持该Shape跨批次结论，不证明Shape泛化。
- 新Shape验证后置到明确扩展范围时，不因当前协议自动修改固定Shape Adapter。
- 报告逐臂绝对/相对误差、差值误差、两候选排序、选择是否正确、局部绝对regret。若没有整FFN共同成本，不报整FFN相对regret。单个Case只记0/1选择结果，不宣称统计选择正确率或MCTS收益。
- 没有匹配Profile或可用参数时返回unknown，不将缺项计0，也不把探索网格baseline作为正式搜索成本。

本轮冻结规格和证据边界；下一项是按C_N1/C_N2设计实现独立阶段采集入口并做CPU计划校验，再决定GPU执行。预测误差不设未经验证的合格阈值；先检查排序、成本来源一致性和误差对后续放置的影响。
