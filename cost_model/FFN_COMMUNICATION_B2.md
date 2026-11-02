# FFN Communication · B2

B2完成CPU侧通信需求覆盖与契约声明核验；B3配对矩阵未完成，B4计时入口关闭。无GPU路径验证、无新时延数据、无通信Profile准入。

## 覆盖结果

| 大小 | Shape | 方向 | B1跨卡边实例 |
|---|---|---|---:|
| 524,288 bytes | [1,256,1024] | GPU0→GPU1 | 48 |
| 524,288 bytes | [1,256,1024] | GPU1→GPU0 | 24 |
| 1,572,864 bytes | [1,256,3072] | GPU0→GPU1 | 44 |
| 1,572,864 bytes | [1,256,3072] | GPU1→GPU0 | 44 |

共 **4个声明请求**，覆盖全部 **160个跨卡边实例**。48个带标签方案中46个有通信、2个无通信，均有核验记录。每个请求保留B1 requirement_id、全部plan_id/edge_id见证，coverage.json将每个跨卡边唯一映射到请求。不是为某个参考放置单独生成清单。

B1五份文件的SHA256保留并核对，原始产物不改写。每次重放重新运行B1覆盖核验，再检查B2请求/映射及零通信方案，结果字节一致。

## 现有实现的口径差异

| 来源 | 实际行为 | 与本次Graph契约关系 |
|---|---|---|
| execution/reference.py:255 | value.to(target)，目标Tensor由to分配；逐组通信按双端同步墙钟计时 | eager、包含分配，不直接复用 |
| calibration/calibrate_bw1000.py:582 | 预分配target.copy_(source, non_blocking=True)，双端同步墙钟循环 | eager，不能改名为Graph Replay标签 |
| cost_model/measured_profile.py:480 | 按方向插值旧link表 | 尚无本次通信身份/契约准入；本轮未接入或升格 |

旧表两个方向各6个尺寸点，仅作为历史来源记录；不导入新Profile，不因尺寸在插值范围内就自动接受。

## 选定的声明契约

实现为 `torch.Tensor.copy_.preallocated_cross_gpu.v1`，入口 `execution/ffn_transfer.py:BoundTransfer`。这是单次复制绑定，不是完整FFN执行器或Graph采集器。

- source与destination预分配，调用体只做destination.copy_(source, non_blocking=True)。不隐式分配、打包、跨设备to搬运或创建流。
- 绑定要求显式且不同的GPU序号、目的设备上的流，以及peer能力。CPU Tensor或设备不符直接拒绝；不自动使用host staging或eager计时。
- 目标口径为Graph Replay，单位us/transfer；同一目的GPU的一对Event包围一次重放，再除以B3规定的调用量。不计算跨GPU两个Event的时间差。
- 孤立通信测量中，生产者完成及双端准备同步在计时外，source在整轮保持只读；该时延只描述ready-to-ready通信proxy，不含生产者计算。
- 消费者必须在目的Tensor就绪后运行，使用同一目的流或明确完成等待。B4必须证明目的Event确实覆盖实际复制完成，不能只凭API返回或peer能力声称完成。
- 每次调用拥有独立预分配destination，保留并检查全部输出；source保持到重放完成。分配/capture排除在计时外，实际设备写入在计时内。
- 每条跨卡图边对应一次动作；不引入跨消费者缓存或合并。B1同源的多个边实例不自动减免计费。该政策尚须B5代表链路检验，不能称为已验证完整方案成本。

calls/rounds、warmup、顺序、统计由B3冻结，实际Torch/HIP/UUID/route等由B4捕获。因此当前measurement_contract_id为null，尚无可独立查询的通信标签。B2 request_id只是声明请求身份，不冒充Canonical实测Profile Key。

## 布局声明与运行观测

carrier由具体 `torch.empty_strided` 构造声明定义：

| Shape | dtype | source/destination strides |
|---|---|---|
| [1,256,1024] | BF16 | [262144,1024,1] |
| [1,256,3072] | BF16 | [786432,3072,1] |

provenance.kind=constructor_declaration，producer_layout_observed=false。这里声明的是新通信载体要求，没有将原始Builder的未知生产者布局填为contiguous。

B4必须读取真实source/destination的Shape、dtype、stride和连续性，与声明一致后才绑定。非连续、未知或不符时拒绝，不添加未计费的contiguous copy；必要时回到动作空间/Case定义，不先计时再补身份。双端UUID、Torch构建及观察到的实际复制路径当前均为空。

## B4前置验证

B0～B3完成后，B4先验证双端身份、实际peer路径、目标Graph捕获/重放可用性、目的Event覆盖完成、复制正确性与source保持、输出保留。peer能力不是实际路由证据。捕获失败、路径不明、host staging、目的Event不能覆盖完成或布局不符时停止该Case，不回退eager或复用旧链路表。

本轮只有CPU元数据与数值夹具测试，不能证明BW1000上的跨卡Graph捕获或上述Event行为可用。B4应保留失败报告；B5再检查真实生产者→通信→消费者链路及资源争用，不把孤立传输proxy直接当成整图真值。

## 产物与重放

`data/ffn_communication_b2/`：identity.json、communication_requests.json、coverage.json、verification.json。

```powershell
& 'C:/Users/LJ/miniconda3/envs/fusion-dev/python.exe' -m cost_model.ffn_communication_b2
& 'C:/Users/LJ/miniconda3/envs/fusion-dev/python.exe' -m pytest cost_model/tests/test_ffn_communication_b2.py cost_model/tests/test_ffn_placement_b1.py -q
```

20项CPU测试通过。包括完整覆盖/源文件未变、零通信方案、声明与观测区分、设备/Shape/dtype/stride拒绝、CPU不能绑定GPU路径、旧eager数据不可复用、契约/覆盖/ready状态篡改拒绝及重放。10项为新增B2测试，10项为B1回归。

下一项B3；当前4个请求的measurement_ready均为false，accepted_link_profiles=0，成本评分与GPU测量均为0。
