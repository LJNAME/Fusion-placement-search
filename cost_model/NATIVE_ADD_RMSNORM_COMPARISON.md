# Add＋RMSNorm：一次 Native 完整链对照

2026-11-07。只测 GPU0、第0层 Native Add→RMSNorm；已有 KERV 5.353559851646423 μs/call 保持原记录，不重测。

## 执行

```bash
cd ~/private_data/LJ/Fusion-placement-search
git pull --ff-only
export FLAGSCALE_ROOT="$HOME/private_data/LJ/FlagScale-main"
export FUSION_PYTHON="$HOME/private_data/LJ/dcu-kernel310-20261031/run_python.sh"
bash tools/run_native_add_rmsnorm_timing.sh ../native_add_rmsnorm_timing_01 run
```

成功或失败均回传整个输出目录。只有这一条 GPU 任务，无逐层测量、GPU1、Profiler、锚点或重试。

固定 BF16 [1,256,2048]、eps=1e-6。每次调用执行 out-of-place Add，再执行 F.rms_norm，并保留原 Add 和 Norm 两个输出。50份独立 x、共享只读 residual/weight/source，50次 eager 准备、10次预热 Replay、7次正式 Replay。恢复、输出失效、数值与只读校验均在 Events 外；同一 stream 包围完整 Graph Replay。IQR/Median 超过10%不准入，不补轮或替换。

Native 不覆盖 x，sum 独立分配；KERV 的 sum 覆盖 x。两臂各自完整执行配方不同，公共计时边界、输入复用、缓存场景、环境必须一致。Native 的 sum/normalized 分配由私有 Graph pool 捕获；不在完整链成本外再次追加 launch、分配或打包成本。实例重启造成的 UUID 改变只用于溯源；GPU0序号、架构、型号、容量及软件构建仍核对。

## 回传处理

```bash
python -X utf8 -m cost_model.native_add_rmsnorm_profile review /path/to/native_add_rmsnorm_timing_01
```

显式 CPU 复核逐轮记录、完整链 Canonical Key、契约和采集源码；写入 review/profile.json，给出 Native/KERV 成本、差值和选择。Fused 慢于 Native 也不影响准入。旧 KERV 不重新审核或测量；两批比较不是同批随机对照。

## 代价模型接线

```bash
# 只核验，不新增搜索实验
python -X utf8 tools/evaluate_qwen25_add_rmsnorm.py \
  --native-profile /path/to/native_add_rmsnorm_timing_01 \
  --contexts /path/to/real_model_contexts.json \
  --audit-only --output /fresh/add_rmsnorm_reuse_audit
```

--contexts 沿用原有逐组证据格式；没有实际文件时省略该参数，未知项仍为 insufficient_info。采集脚本不会把合成输入伪装成模型生产者布局。

原图其余35处逐一检查 Shape/dtype、同层、输入角色、外部消费者、布局、别名、实现、运行环境和完整契约。静态结构一致不自动补齐运行时布局。符合条件的 GPU0 位置才查询原 KERV 及本次 Native 完整链，GPU1不借GPU0数据；final_norm仍在v1映射范围外。

Native 整链仅在两个节点均为原生、同卡且连续保留执行时使用。调度视图将该链作为一个执行单元，双输出在链结束后可用；原搜索/导出图仍保留两个原生节点，并注明连续执行约束。跨卡拆分或未通过等价性检查时不命中整链成本，继续标明原成本来源。没有将整链时间拆成两个单算子标签。

Native回传已复核并准入；原图36处运行时布局/别名/契约仍未知，未自动开放跨层融合。
## native_add_rmsnorm_timing_01 回传

| GPU0完整调用 | Graph摊销 μs/call |
| --- | ---: |
| Native Add→RMSNorm | 10.217599869 |
| 已有KERV | 5.353559852 |
| KERV减少耗时 | 4.864040017（47.604526%） |

Native固定7轮×50次，350次输出双参考/只读/恢复检查回执通过；inclusive IQR/Median=1.424699%。12个采集源码SHA对应ff343bd，完整Key与Contract重建通过，runtime/构建/硬件类与KERV一致。UUID变化仅溯源。两者来自不同批次、相同公共Graph边界；该局部比较不等于整图加速。

原13文件保存在data/native_add_rmsnorm_timing_01/；review/profile.json为accepted记录，review/cost_query.json为实际严格Canonical查询，测得整链不追加访存/launch成本，GPU1查询拒绝。review/summary.json与strict_reuse_audit记录模型复用仍0，未将benchmark改写成模型布局证据。Native单算子标签新增0，无新GPU或搜索。

这一个测时Case结束。剩余工作是已有3B位置的实际布局、别名和执行契约；无需再测本组Native/KERV。