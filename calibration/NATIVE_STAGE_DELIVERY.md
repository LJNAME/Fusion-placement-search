# Native阶段数据使用规则与服务器交付

2026-11-05。采集范围仍只有GPU0的独立SiLU/Mul。先写出新数据可约束什么，再收集数据。原正式LightOP Profile、历史整链评价和Dispatch证据继续保留。

## 接收后处理

1. 运行`calibration.native_stage_calibration review`，核对计划、来源、实际输入、运行环境、14次尝试和质量。保留失败阶段及原始记录，不替换样本。
2. 通过阶段只得到该身份/Graph契约下的有效服务成本候选。记录中位数、逐轮值、离散度和Key；不自动准入正式成本，不修改解析参数。
3. 对已有显式参数场景计算阶段残差和候选排序的敏感性；报告全部结果。旧729场景是探索假设，不按新数据筛出最好的一格后称已校准，更不将其区间称置信区间。
4. 如果需要正式校准，另冻结参数更新方式及误差容忍规则，再使用这些calibration数据约束组合参数；旧local_pair继续作为开发诊断。更新后预测在新独立批次上评价。

## 独立阶段与链路的开销

令C_s、C_m为两个阶段的核心计算/访存近似。若额外假设独立阶段也服从当前h/c分解，则阶段预测分别为C_s+h+c、C_m+h+c；当前整链公式为C_s+C_m+2h+c。独立阶段相加会多一次c，不能默认c=0。

这个h/c映射本身也是待验证假设。独立阶段与链路的缓存、依赖和调度上下文可能不同，不能仅靠两条阶段标签保证可迁移。当前模型结构不改，未建立映射前保留整链预测的不确定性，不用旧10.09 μs减预测项反解纯Graph参数。

阶段实测能直接约束独立阶段的有效服务；对整链成本的约束是条件性的。新数据不保证能唯一确定带宽、缓存、非线性吞吐或Graph开销，未改善决策时停止扩大测量。

## 旧数据角色

- 17.811180353164673 μs：匹配原Key/Contract时的正式完整Fused成本，含Pack，不追加开销。
- Native10.092799663543701 μs与完整Fused18.847960233688354 μs：原冻结实验评价保留；新方法设计中为开发诊断，不再次充当新方法独立验证集。
- 旧硬件、KERV和Dispatch数据：保留原实现与语义范围。阶段Case的复用必须分别满足来源和契约，不能由总时间任意分解。

模式B后续使用时仍检查Native与正式Fused的架构、Torch构建、HIP/驱动及执行语义是否可比较，不因它们都在GPU0就认定环境一致。实例UUID只作溯源。

## 服务器操作

完整bundle与校验文件放在工作区`transfer/`，最终文件名和SHA256以本次交付清单为准。完整bundle携带历史，无增量前置要求；服务器工作区有本地修改或历史分叉时，保留Git报错，不用reset强制覆盖。

上传bundle到`~/private_data/LJ/`，在项目根目录执行交付消息中的`git pull --ff-only`。然后先预览计划：

```bash
export FUSION_PYTHON="$HOME/private_data/LJ/dcu-kernel310-20261031/run_python.sh"
bash tools/run_native_stage_calibration.sh ../native_stage_plan_20261105_01 plan
```

一次两阶段测量，输出目录必须不存在：

```bash
export FLAGSCALE_ROOT="$HOME/private_data/LJ/FlagScale-main"
export FUSION_PYTHON="$HOME/private_data/LJ/dcu-kernel310-20261031/run_python.sh"
bash tools/run_native_stage_calibration.sh ../native_stage_calibration_20261105_01 run
```

如无需FlagScale环境初始化，可以不设FLAGSCALE_ROOT；Native路径不加载LightOP厂商库。实际计时保持原厂商Python/运行环境，失败不切换Python重试。

完成或失败均回传整个目录：

```bash
cd "$HOME/private_data/LJ"
tar -czf native_stage_calibration_20261105_01.tar.gz native_stage_calibration_20261105_01
```

本次交付仅在本地执行CPU测试和恢复验证；GPU结果待服务器回传。不重测已有整链或Dispatch，不启动Pack、GPU1、通信或MCTS。
