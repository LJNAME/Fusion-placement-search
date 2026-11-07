# Qwen2.5-3B 显存模型审计

2026-11-07。只复核现有687节点前向图及A/B/C三个固定放置，未运行MCTS或GPU，未修改生产评分器与搜索算法。四基线初始化暂缓。

## 三个方案

显存单位GiB（2^30字节），时间单位ms。每卡配置容量为68,702,699,520字节，即63.984375GiB；不是当前空闲显存，也没有扣除运行时预留。下表的“通过”仅表示现有前向估算值低于该配置容量。

| 方案（均36组KERV融合） | GPU0估算峰值 | GPU1估算峰值 | 当前模型/配置容量检查 | 训练及实际可用预算检查 | 预测时间 |
| --- | ---: | ---: | --- | --- | ---: |
| A：整图GPU0 | 5.821262 | 0 | 通过 | 未知 | 25.239290 |
| B：0–17层GPU0，18–35层GPU1 | 3.180435 | 2.657551 | 通过；共享权重驻留有遗漏 | 未知 | 25.308637 |
| C：整图GPU1 | 0 | 5.821262 | 通过 | 未知 | 25.255440 |

B的embedding在GPU0，final norm和lm_head在GPU1。0表示模型未在该卡分配图中对象，不代表运行时占用为0。数值来自data/qwen25_memory_audit_v1/report.json:43、66、110，与上次放置审计完全一致。

## 六项核对

| 核对项 | 结果 | 源码或数据 |
| --- | --- | --- |
| Shape、dtype | 687/687节点、866/866边完整；均可从元数据计算字节数，差异0；节点均为BF16/forward | report.json:3；dense_layer.py:59 |
| 参数 | 图声明参数6,171,693,056字节，按组求和并全程驻留；是静态配置值，不是实际checkpoint统计 | structure.py:133；structural.py:61 |
| 生命周期 | 评分时由调度和消费者推导；组结束时发布输出，同卡保留至最后消费者组结束，跨卡源保留至Copy结束；末端输出保留到makespan | structural.py:75–108 |
| 融合 | 支持不计完全在组内消费的输出；本次36组的Add和Norm输出都有组外消费者，实际消除的独占内部输出为0 | structural.py:90；report.json各方案internal_only_outputs_eliminated |
| 训练、Workspace、通信 | 训练状态未计；Workspace接口存在但本次651组均未声明，模型按0；通信接收缓冲只从Copy完成开始计 | structure.py:108；structural.py:64、107 |
| 搜索显存约束 | 已有：预测峰值超过capacity时加有限惩罚，最终只保留显存检查通过的候选；capacity可显式传入，当前3B入口默认整卡容量 | search_evaluation.py:245、279、424；evaluate_search_quality.py:48 |

生命周期逐张量记录在tensor_lifetimes.json，包含生产/最后消费拓扑位置及三个方案的发布/释放时间。它记录现有模型的逻辑区间，不是运行时分配轨迹。因峰值依赖调度，现有实现先得到时间线再检查容量，并非独立的调度前过滤。

## 尚未覆盖的对象

- `saved_activation_bytes`、`bwd_flops`只是图元数据；687节点全部为forward，没有反向、梯度、优化器或主权重节点。saved_activation元数据合计1,434,124,288字节，不能当作训练激活峰值，更不能直接加到表中作为训练显存。
- KERV Adapter要求no-grad，且Add输出与消费掉的attention缓冲别名。显存估算不读取物理storage/别名信息，仍按逻辑输出处理；本次全Native GPU0与全KERV GPU0的估算峰值相同，不能据此判定原地融合没有显存收益。
- B跨卡共享权重遗漏：lm_head的`shared_param_with=embedding`且param_bytes=0；embedding权重622,329,856字节（593.5MiB）。估算未读取共享参数关系；将两者放到不同GPU后，没有复制、远程访问或权重传输策略。不能直接把该大小追加到现有峰值并称为修正后的真实峰值。
- 输出从组结束才开始计，未覆盖执行中分配输出与临时空间共存；目的缓冲从Copy结束才开始计，未覆盖传输进行时的驻留；同一时刻先分配后释放，可能重复计算别名或零时长对象。该估算不是已证明的显存上界。
- 未声明Workspace按0处理，不代表库Kernel无需Workspace；allocator缓存、Graph池、上下文、其他进程及运行时预留也没有计入。

上述别名与共享权重限制分别见integration/flagscale/kerv_add_rmsnorm.py:189、203、210，simulator/model_graphs/transformer.py:150、165，cost_model/structural.py:103、107、114。

## 后续

暂不修改MCTS。若要把“模型容量检查通过”用于部署准入，先明确前向/训练范围和每卡显式预算，并处理跨卡共享权重、已知别名及必要缓冲区。训练范围需要下游提供状态类型、精度、分片和激活保留策略；不在此处假设优化器或开发完整模型执行器。四基线只能按同一范围与预算筛选，可行的才进入初始化；不能将当前三方案预测当作训练可行性证明。

入口：`python -B -X utf8 tools/audit_qwen25_memory.py --output <新目录>`。已有17项容量/结构测试通过；CPU重放得到相同评分、峰值和逐张量生命周期。verification.json保存命令与工件哈希。没有新增GPU测量或搜索实验。
