# FFN Measurement · B3

B3 CPU配方与矩阵冻结；7项CPU检查完成，HIP Graph/跨卡路径Smoke仍待B4，不能记为8/8。无新GPU测量，无新Profile准入。

## 核心矩阵与复用审核

| 类型 | 证据槽位 | 本轮安排 |
|---|---:|---|
| FFN singleton × GPU | 10 | 新测 |
| Native SiLU→Mul × GPU | 2 | 新测 |
| KERV Fused × GPU | 2 | 旧证据审核后判定重测 |
| 通信大小 × 方向 | 4 | 新测 |
| 核心合计 | **18** | — |
| Down0→0.5MiB Copy→Gate1诊断 | **1** | 新测，不作为独立Kernel标签 |
| 总计 | **19** | 当前19条新测任务，尚未执行 |

旧Fused确实记录warmup=10、7×50、全部输出保留与计时外gate恢复，up为共享只读Tensor。它们缺少本轮锚点/有效轮次证据和新recipe_id，且新实验的四臂驻留、顺序与命名输入fixture发生变化。因此两条都不满足复用条件。旧P2 accepted状态及用途保留不改，不能补写缺失锚点或改签Recipe来复用。17条只是两条Fused均可复用时的假设预算，不是当前计划。

完整逐行条目见 `measurement_matrix.json`：每行含candidate_id、设备、Recipe、轮组、输入复用策略和计时前待绑定项。统计的是计划槽位，不是已经取得19份实测证据。

## Pair Key与实际身份

Native/Fused的配对字段为 `(candidate_id, device_id, shape, dtype, layout, measurement_contract, recipe_id)`，同卡两臂字段完全一致，GPU0与GPU1的配对键不同。具体实现身份在各臂Kernel Key里分别记录，不能把Native绑定为KERV实现。

CPU阶段冻结pair_key模板及其哈希。实际UUID、Torch/HIP构建、布局观测和环境相关measurement_contract_id由B4计时前绑定，两臂必须精确一致；当前runtime_contract_id=null，不冒充完整实测身份。各named fixture独立CPU generator种子和缩放已写入Recipe，上传与fixture生成都在计时外。

## 配对轮组

19行组织成10个轮组，每行恰好出现一次，不重复计算测量任务：

- 每卡一个激活四臂组：SiLU、Mul、Native chain、Fused；共享attempt_id和前后锚点。使用四种固定顺序按attempt_id循环，作废轮也消耗序号。
- 一个链路四臂组：矩阵中的Down/GPU0、0.5MiB 0→1 Copy、Gate/GPU1，外加诊断链。四臂使用匹配输入/权重/Y数据、同轮锚点与主样本集合。
- 其余4个GEMM singleton和3个Copy各自组成单臂组。

这样可逐轮计算加和残差，不把不同批次或不同有效轮次的中位数相减当成同轮结果。GPU0/GPU1分别保留数据，不共享Profile。

## 7×50、预热与质量门禁

每个捕获体50次调用，每臂10次预热Replay。正式主样本需要7个有效轮次，最多尝试9轮（用户确认的保守规则）。每个参与GPU均使用同一固定只读SiLU Graph锚点，在整个轮组前后测量；锚点不作为算子Profile，也不从被测时延中扣除。

| 规则 | 冻结行为 |
|---|---|
| 锚点漂移 | abs(post−pre)/pre >10% 时整组本轮作废；等于10%不作废 |
| 主样本 | 时间顺序中的前7个锚点合格轮，所有臂用相同索引，不挑选或裁剪 |
| 离散度 | 使用inclusive IQR，(Q3−Q1)/Median >10% 时停止准入 |
| 最多补2轮 | 可用于锚点作废后的有效轮次补足；前7主样本的IQR失败不能靠替换救回，剩余仅诊断 |
| 数值、路径、Capture错误 | 立即停止整个受影响轮组并保留原始证据，不能因该轮锚点不合格而隐藏错误 |
| 不稳定 | 第9次尝试仍不足7有效轮，或7主样本离散度不合格，停止该Case；不改配方延长 |
| 准入 | 质量检查仅给出eligible_for_later_admission；仍须身份/来源/执行路径审核，不自动accepted，不以融合快慢筛选 |

原始轮次、锚点、顺序、全部输出检查、无效原因与额外诊断均须保存。10%是预注册工程阈值，不是已经验证的统计保证。未达到7有效轮不能以较少轮数声称完成本配方。

## 链路诊断与残差

Producer取Down形状GEMM/GPU0，Copy传输[1,256,1024] BF16至GPU1，Consumer取Gate形状GEMM/GPU1。这是形状兼容的诊断替身，**不是原FFN里Down→Gate的一条真实边，也不注册跨层融合**。

`Y`在计时外由固定V/W_down通过GPU0 Reference生成，并作为孤立Copy与Gate的匹配输入。链中Producer使用同样的V/W_down，Consumer使用同样W_gate；所有调用的输入/输出按50份独立范围管理。完整链Graph包含跨设备Copy；隔离Kernel Graph不含输入上传或设备搬运。

统一 `R_link = T_chain − (T_producer + T_copy + T_consumer)`，先按四臂共同主轮次计算，再汇总；同时报告 `R_link/T_chain`。负值表示单项加和高估链路，正值表示低估。abs(R_link)<0.5μs仅允许在该诊断Case内暂用加性模型，达到或超过0.5μs检查依赖/同步/组合开销；不自动回写邻接修正，也不因残差大而否认可信原始测量。

链路使用同一GPU1上的起止Event，显式起始栅栏到GPU0 Producer，并将最终Consumer完成纳入GPU1终止栅栏。不能相减两张GPU的Event。该依赖图及HIP捕获可用性尚未验证，Smoke失败不回退同步eager墙钟。这一条链不能证明反向、1.5MiB或完整FFN的组合精度。

## 原地输入与缓冲区

KERV每次调用覆盖一个gate，Fused输出别名该gate。50次调用使用一个pool内50个不相交gate视图；每次Replay前从gate_original恢复整个pool并同步，恢复在Event计时外。Capture后和每次预热前也恢复。不能把上次Fused输出继续当作下一次原始gate。

up保留 **1份共享只读Tensor**，两臂一致，并检查未修改；不宣称有50份独立up。Native的50个最终输出独立保留，Fused的50个输出句柄指向gate pool。所有50个输出在每次尝试（包括作废/诊断轮）均检查。

| 显式保留缓冲范围 | 声明字节合计 |
|---|---:|
| 一个激活四臂组（5个50份pool + 4个基础/备份Tensor；Fused别名不新增存储） | 381 MiB |
| 每个参与GPU的锚点（1份只读输入+50份输出） | 76.5 MiB |
| 链路四臂组的列明输入/权重/输出及备份，双卡合计 | 389 MiB |

详见buffer_audit.json。以上为显式存储下界，不是实际峰值HBM；不含Graph私有临时区、allocator余量和未列出的数值oracle。只保留当前轮组与锚点，不能把不同组的工作集混算。GEMM默认50份独立输入+50份输出、权重常驻；Copy为50份只读source与50份destination。实际GPU的地址、范围、布局和生命周期还要在B4检查。

## B4 Smoke与B6 sanity边界

freeze_checklist.json提供静态Smoke计划：每个捕获体50次调用、重复3次Replay；两卡激活四臂、四种Copy（特别是1.5MiB双向）、以及链路四臂均需检查。改变Copy source数据在计时外进行，以识别旧输出；实际route/双端身份和依赖证据不能由peer能力或接口存在代替。

目的流上的消费guard须在复制后、终止Event前建立因果依赖。仅在host synchronize后检查Tensor正确，不能单独证明Event覆盖了复制。受影响轮组任一臂失败时保留失败资料，不静默删臂、放宽阈值或换eager计时。

当前检查清单为 **7项CPU核验+1项GPU待验证**，formal_gpu_timing_allowed=false；本轮没有执行Smoke。之后先完成B4 Smoke与实际身份绑定，再进入正式计时。

两个零通信方案只属于入口/出口GPU0场景，其plan_id已记录。B6应单独报告Native全GPU0与Fused全GPU0；若评价计算侧预测误差，还需这两个完整FFN子图的真机测量。它们是后续另列任务，不包含在当前19槽位中。Exact是当前成本模型下的穷举最优，不是物理耗时真值。

## 重放

`data/ffn_measurement_b3/` 保存identity、recipe、measurement_matrix、reuse_audit、buffer_audit、freeze_checklist和verification。当前工具只生成声明及CPU核验，不是B4采集器；旧run_kerv_profile_stage3.sh没有新四臂/锚点门禁，不能直接当作本轮采集入口。

```powershell
& 'C:/Users/LJ/miniconda3/envs/fusion-dev/python.exe' -m cost_model.ffn_measurement_b3
& 'C:/Users/LJ/miniconda3/envs/fusion-dev/python.exe' -m pytest cost_model/tests/test_ffn_measurement_b3.py cost_model/tests/test_ffn_communication_b2.py cost_model/tests/test_ffn_placement_b1.py -q
```

相关CPU测试33项通过，覆盖19行/轮组唯一性、同卡配对、旧证据拒绝复用、内存别名声明、锚点作废与边界、9轮预算、IQR不挑样本、fatal不隐藏、R_link符号/0.5μs阈值、未执行GPU的状态及冻结篡改拒绝。
