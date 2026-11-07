# Qwen2.5-3B Add＋RMSNorm图级接线

2026-11-07。扫描现有687节点、866边逻辑图，接入既有MCTS；无GPU测量，不改默认0.6B搜索入口或代价模型公式。

## 候选与约束

| 项目 | 数量 |
| --- | --- |
| 直接Add→RMSNorm连接 | 37 |
| 符合当前Adapter静态结构 | 36 |
| 原图信息充分、可直接启用 | 0 |
| 信息不足 | 36（物理布局、别名和执行契约未提供） |
| 当前Adapter不支持 | 1：L035.residual_add_2→final_norm |

最后一处不是宽度超限：Shape仍为2048，但输入是MLP down_proj、norm为final_norm，不符合v1 Adapter的attention输入和同层约束。不把它误记成已支持，也不跨逻辑input节点构造三节点融合。

36处层内节点的Shape/dtype和图上可消费条件符合已实现Adapter；这些事实不证明物理Tensor连续、无存储别名或当前执行为Graph Replay。原图unknown保留，5.35356μs的已接受benchmark没有自动变成图上第0层或其他层的命中。

## Profile复用

cost_model/qwen25_add_rmsnorm.py的qualify_context按组和设备检查上下文。需要明确model_runtime来源/source_id、原图fingerprint、节点ID和设备，以及真实Tensor布局、x与其他live Tensor的别名、实现、runtime和测量契约。规划声明不能冒充运行时观测。

先逐项检查Shape/dtype/stride、推理环境、实现和别名约束，再以原Canonical Key工厂重建完整Key，并匹配完整Measurement Contract。仅全部一致时，才通过既有KeyedKernelProfile读取那一条accepted成本；不新增测量记录、不按层号复制数值，也不替GPU1借GPU0数据。原有单组准入文件保持不变；扩展用途有独立的逐组等价性结果。

本次没有模型runtime上下文文件：图上实测复用0，已有benchmark记录仍1；匹配的原生Add＋RMSNorm实测链0，可靠实测对照0。CPU测试中的跨层相同元数据是mock，只验证等价性门槛，没有当作真实模型证据。

## 成本覆盖与模式

入口tools/evaluate_qwen25_add_rmsnorm.py默认严格模式：未满足能力/契约的融合不进入动作集合；有明确设备冲突的方案在评分前拒绝。原生算子选项始终保留。显式--conditional-planning才允许未知前置条件的融合，用于CPU接线，不表示可部署。

本次保存的是该条件规划场景，原生和融合均采用相同Graph解析假设：

| 查询范围 | 成本来源 |
| --- | --- |
| 每卡687个Singleton | 651解析、36逻辑输入、实测0 |
| 37处×2算子×2设备 | 148项原生解析成本 |
| 36个静态融合组×2设备 | 72项融合解析成本 |
| final_norm融合×2设备 | 2项不可用 |
| 融合实测复用 | 0 |

复用现有AnalyticHardwareProfile公式；仅在独立profile副本里把旧eager launch 7.770405267μs替换为既有FFN配置的1.5μs探索假设，fusion_overhead=0。1.5μs现在按逻辑op/group使用，未经BW1000标定，也不是论文原始参数；Native RMSNorm实际dispatch数量未验证。历史硬件/shape曲线和通信仅作代理，均标OOD，不混入旧eager Profile。

原生Add和RMSNorm是两个逻辑阶段，Fused是一个双输出组；同为Graph目标口径不代表已经有可信Native实测。当前不能据此做真实融合收益结论。

## 最小搜索检查

三种原有模式，seed0、每次64个实际评分；不增加种子/预算矩阵，不改MCTS选择、回传或rollout算法。调度沿用dependency_level_node_id_v1。

| 条件规划方案 | 预测ms | 融合组 | 首次最终最佳评分 |
| --- | --- | --- | --- |
| 无融合、全GPU0参考 | 25.295361 | 0 | — |
| Joint | 25.213223 | 36 | 1 |
| 先融合后放置 | 25.213223 | 36 | 1 |
| 先放置后融合 | 25.213223 | 36 | 33 |

三者最终均全GPU0，实际不同候选分别50/38/38。第33次对应先放置后融合进入第二阶段后的评分，不能当作联合树搜索优势。Joint没有初始补全之后的改进。该成本口径与0.6B旧混合Profile实验不同，不作为严格的0.6B/3B性能对照。

Exact使用现有第0层图的6节点切片，保留Add旁路消费者；4个边界只表示GPU0驻留，不生成新计算任务。96个组合中6个满足边界约束；Exact与MCTS同为0.004629474ms、model gap=0。其成本独立使用解析代理，不套用可选整图runtime上下文，不是GPU真实最优值。

新增17项、相关68项CPU测试通过：严格模式无未知融合、完整等价性门槛、GPU1不借用、别名违规不评分、图布局不被改写，以及融合更慢时MCTS保留Native。两种快慢数值是单元测试fixture，不属于实验测量。

## 复现与交付

```bash
# 默认严格能力模式；无runtime证据时只开放原生选项
python -X utf8 tools/evaluate_qwen25_add_rmsnorm.py --output /fresh/strict
# 本次CPU条件规划接线结果
python -X utf8 tools/evaluate_qwen25_add_rmsnorm.py --conditional-planning --output /fresh/conditional
# 已有真实逐层runtime证据时才提供，不会从benchmark自动生成
python -X utf8 tools/evaluate_qwen25_add_rmsnorm.py --contexts /path/to/model_contexts.json --output /fresh/qualified
```

上下文以Lxxx.pre_mlp_norm@gpuN为键；字段见qualify_context及测试fixture。source_id须指向该组实际模型运行证据，不能把benchmark元数据换层号后当成model_runtime。

data/qwen25_add_rmsnorm_graph_v1包含逐位置/逐设备审计、三种搜索轨迹、报告及joint_placed.json。导出保留原节点ID和双输出依赖，携带实现身份、规划契约和deployable=false；未把未知布局改成已观测。

执行链路和MCTS接口已经接通。正式跨层复用仍缺模型布局/契约证据，可信Native对照仍缺对应成本；本轮不重测Kernel，不新增Profile，也不声称完整方案实测覆盖或真实加速。

## 单条 Native 对照入口（2026-11-07）

见 [NATIVE_ADD_RMSNORM_COMPARISON.md](NATIVE_ADD_RMSNORM_COMPARISON.md)。只新增GPU0第0层完整原生链的7×50测量，不重测KERV。--native-profile读取复核后的完整链成本，--audit-only只检查其余位置的复用条件，不启动新搜索。尚未得到Native新读数；未知模型布局不由benchmark补齐。