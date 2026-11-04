# LightOP × Qwen3 图适用性审计（2026-11-04）

9 项已完成厂商任务：**直接映射 0、需要适配 1、证据不足 1、不适用 7**。本轮只做 CPU 审计，未新增 Adapter、GPU 测量或 accepted Profile；7 项依赖失败任务继续标记未完成、暂缓。

## 图与证据范围

对象是 `cost_model.search_evaluation.build_evaluation_graph("model")` 生成的 Qwen3-0.6B **逻辑参考图**，不是上游捕获图：28 层、591 节点、730 边，BF16，batch=1、seq=256、hidden=1024、FFN=3072，Q/K heads=16/8、head_dim=128。图指纹与 B1 相同：`sha256:fc718571cb1f7e3d17d32e50401d81f6d465d410dd8f493e3e81305be3ba79ea`。

图只声明 Shape 和 dtype，没有运行时 stride、storage alias、RMS epsilon、RoPE theta/旋转维数/neox 信息。相同的 `implementation_group` 不证明 gate/up 共享一块打包存储；未知布局保留 `unknown`。

本轮逐项采用 11.4 第二批完整回执的实际主调用接口，复核固定 wheel 与原测试源码。原文节选、文件 SHA 和源行号保存在 `data/lightop_qwen3_applicability_v1/source_evidence.json`。分类由源码语义审核给出，节点集合由 CPU 程序枚举；没有按接口名称自动认定可融合，也未核验闭源 Kernel 的全部支持范围。

## 逐项结果

| 已完成接口 | 结果 | 当前图与已测路径的差别 |
| --- | --- | --- |
| `op.silu_and_mul_opt` | 需要适配；28 组 SiLU→Mul | 图为两个投影产生的 gate/up，均 `[1,256,3072]`；API 接收 `[tokens,2d]` 打包输入、独立输出。本批测 d=4096，不能直接定价 d=3072 |
| `op.gelu_and_mul_opt` | 不适用 | 当前激活是 SiLU，不能换成 GELU |
| `op.gelu_tanh_and_mul_opt` | 不适用 | 当前激活是 SiLU，不能换成 GELU-Tanh |
| `fuse_fill_rms_x2_concat` | 不适用 | 需要位置置零、两份状态各自原地 RMSNorm、Concat；当前图没有该计算组合。厂商 hidden=7168，只有零差值证据 |
| `fuse_groupgemm_permute_quant` | 不适用 | 实际是专家路由重排＋INT8 量化，没有执行 GroupGEMM；当前 dense 图没有 routing/quant/scales |
| `fused_rms_norm_rope_contiguous` | 不适用 | 已测为 MLA 的 compressed-KV RMSNorm、RoPE 与缓存写入；当前 Qwen GQA 图没有 MLA/KV-cache 存储节点 |
| `rms_rotary_embedding_fuse` | 证据不足；28 组结构对应 | 图中每层有 Q/K 两条 RMSNorm→RoPE 分支，但不是现有线性融合规则。已测 FP16、heads=8/1、rotary_dim=64；图为 BF16、heads=16/8，epsilon/旋转配置及别名约束未闭合 |
| `fused_bmm` | 对当前图不适用 | 已测 INT8 输入＋scale、BF16 输出，量化成本被设为 0；图中 attention matmul 为 BF16，不能静默引入量化。未据此断言 API 不支持其他未测类型 |
| `reshape_and_cache_cuda` | 不适用 | 已测 FP16、heads=1、head_dim=128 的缓存写入；当前图没有 slot_mapping、KV-cache 写入节点 |

`node_matches.json` 列出全部 28 组 SiLU→Mul 的节点、边界输入、Shape/dtype 和布局来源，以及 28 组 Q/K 分支。前者通过现有 SiLU→Mul 的层内、同形状和内部消费者结构检查；后者只记录结构对应，不赋予融合合法性。两种数量均不代表新增可执行候选。

SiLU 的厂商基准是 vLLM 的另一种融合实现；RMS＋RoPE 的计时 lambda 内包含 Q/K clone；BMM 的表头虽写 FP8，实际入口为 INT8。三者的时间不能互换，也不能直接改名为 Graph Replay Profile。具体原文及源行号见每行 `source_refs`。

## 宽度 3072 的旧证据

`data/hygon_silu_trial02_review/report.json` 已有 BF16 `[1,256,3072]` 的四臂实验，无需为确认“3072 是否测过”再开 GPU。

| 路径 | GPU μs/call |
| --- | ---: |
| Native 分离 gate/up | 17.111990 |
| LightOP 分离输入，调用内含 `torch.cat` 打包 | 21.812770 |
| LightOP 输入预先打包，仅测调用 | 7.806390 |

同一分离输入边界下，LightOP 路径比 Native **慢 27.4707%**；不能只取 7.806 μs 给当前图评分。该实验为 eager Event 批计时，每臂 10 次预热＋100 次调用，仅 1 个独立批次；数值和输入字节检查通过，dispatch 来自独立 profiling 进程。保留为开发证据，不转换成正式成本，也不据此断言 Graph Replay 或不同打包策略必然更慢。打包位置见 `calibration/hygon_silu_trial.py:123`。

## 最小缺项与下一步

首个候选仍是 SiLU×Mul，但 Adapter 设计需先明确三件事：

1. gate/up 的真实 stride、存储关系、设备及生存期；不能由逻辑模块名假定已经打包。
2. 打包、输出分配与复制的执行边界和成本归属；完整路径同时保留 Native 与现有 KERV，允许 LightOP 更慢。
3. Adapter 版本、实际二进制、输入布局、设备和测量契约如何进入 Canonical Key。已有 eager 单批数据只作参照，不复用为正式 Graph 标签。

Q/K RMS＋RoPE 暂不开发 Adapter，先补图的 epsilon、位置/缓存构造、旋转约定与权重语义，并取得 BF16、16/8 GQA 的支持证据。其余 7 个已完成接口在当前图中没有合法替代对象，不扩展图去凑厂商接口。

本轮新增 accepted Profile 为 0。此前审核记录的完整 FFN 可定价方案仍为 0/48；本轮没有重新运行方案成本查询，不以 28 组结构匹配替代成本覆盖率。

## CPU 重放

```bash
python -m tools.audit_lightop_qwen3_applicability
python -m pytest calibration/tests/test_lightop_qwen3_applicability.py -q
```

重放只依赖仓库中的冻结报告、目录和原文节选，不需 wheel、回执解包目录或 GPU。输出 `applicability.json`、`node_matches.json`、`verification.json`；相同输入重复生成字节一致。11 项 CPU 测试覆盖非 SiLU、跨层、内部外部消费者、Shape/dtype 缺失或不一致、未知布局、Q/K 缺边、图版本变化拒绝，以及旧打包负收益证据不升格。
