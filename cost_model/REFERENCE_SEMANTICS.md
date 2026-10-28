# Reference执行语义代价对照

新增ReferenceEagerCostModel，保留原StructuralCostModel和默认搜索。当前仅支持reference已经能执行的单节点组，原GPU冻结清单和源码不改。

## 调度规则

原DAG模型在生产者完成时排传输，允许依赖独立的组按设备空闲时间并行。新模式按plan.groups顺序推进一条主机提交时间线，每设备只有一个current stream：

- 到消费者时才为各跨卡前驱发起复制，不预取、不缓存复制结果；重复前驱也重复复制。
- 复制等待源、目标两侧已经排队的工作及链路资源，完成后推进两侧stream。
- 算子等待当前设备stream与数据依赖。提前提交的独立分支仍可并行，不强行串行整个图。
- 主机提交与GPU服务窗口并行推进；主机不因每次GPU复制而强制等待DMA完成，最终等待所有stream和主机提交结束。
- 原reference的缓存输入不产生GPU kernel，新模式输入GPU服务成本为0，评测单列这一相对旧DAG的共同修正。
- 实际复制完整前驱tensor，按output_shape和dtype算字节，不把边上的部分字节数当作已实现的张量切片。

跨设备copy的current stream和双向barrier依据[PyTorch v2.9.0 Copy.cu:220–278](https://github.com/pytorch/pytorch/blob/v2.9.0/aten/src/ATen/native/cuda/Copy.cu#L220-L278)。这是上游实现；BW1000 vendor源码是否逐项一致尚未核实，已有HIP事件仅提供行为证据。新模式是待验证的执行契约，不是vendor运行时的精确模拟器。

HostSubmissionProfile将start_s、node_submit_s、copy_submit_s、finish_s及分算子operator_submit_s与GPU服务成本分开；local:body覆盖优先于算子族fallback。默认全部为0、calibrated=false；未用已暴露候选的实际耗时反向拟合。复制/算子GPU服务窗口从主机提交开始时释放，不能把一次完整CPU算子调用耗时和GPU事件窗口简单首尾相加。后续校准需要明确CPU调用、设备等待和服务窗口口径。

主机成本未校准、GQA/布局等算子body覆盖未确认时，输出带OOD原因与较低confidence。显存仍是图级生命周期估计，不包含全部eager临时张量/工作区。

## 已有GPU数据对照

```bash
python -m cost_model.evaluate_reference_semantics \
  --output data/bw1000_gpu_holdout_v1/reference_semantics_evaluation.json
```

该命令只读已有GPU测量，不产生新GPU实测；可在CPU运行。覆盖原4组、22个候选，另保留3B seq2048的详细/轻量前/后重复快照，不能把7个快照当7个独立任务。新模式主机成本全部为0；GPU算子/链路成本沿用原封存预测。

| 原GPU用例 | 原DAG选择regret | 新模式选择regret | 原DAG MAPE | 新模式MAPE |
|---|---:|---:|---:|---:|
| 3B seq1536 | 6.425% | 0% | 13.781% | 11.085% |
| 3B seq2048 | 10.273% | 4.806% | 5.966% | 3.358% |
| 0.6B seq1536 | 13.417% | 13.417% | 22.703% | 20.209% |
| 0.6B seq2048 | 9.802% | 9.802% | 16.921% | 14.894% |

3B seq1536新模式前两名只差0.069微秒，seq2048只差2.154微秒；不能据这两个选择就认定排序稳定。新模式仍偏好0.6B的k_proj/k_norm/k_rope跨卡支路，尚不能解释当前reference中因主机提交而损失的重叠。3B轻量前/后快照regret仍为9.194%/8.130%。

串行工作量相加保留为事后算术消融，不是新模式实现，不是实际时间上界，也不替换默认搜索。当前没有完成独立GPU验收、完整训练或SROP收益验证。

## 后续顺序

先在旧训练任务范围采主机/API提交成本，并核对算子内部GQA/布局服务口径；不得用这4组暴露测试标签追调参数后宣称独立成功。

独立任务规划为两模型族各seq768/1280、batch1、BF16单层，共4任务。整模型静态清单64个对象、56可测/8逻辑（最终单层测量池需按实际执行图裁剪）；56个完整Kernel Key与旧112+新56个已暴露Key无交集。仅核对完整Key，尚未核对预测特征别名，不能宣称全为物理新颖任务。

规划保存在reference_semantics_validation_plan.json，标记planning_only_do_not_collect_before_predictor_freeze。先完成训练侧校准及实现，再冻结预测、候选和标签访问规则，之后才采这些新任务。候选只从公开模型/结构探针产生，保留单卡对照，不使用Oracle搜索候选。

训练侧主机校准入口现为calibration.calibrate_reference_host，操作见calibration/REFERENCE_HOST_CALIBRATION.md，需要两张BW1000。独立任务采集尚未启动，不重跑旧诊断轨迹。校准结果不可无条件用于旧宿主的性能验收。
