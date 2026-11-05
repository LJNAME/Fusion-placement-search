# GPU0 SiLU/Mul 尺寸标定、Native整链核对与四组Copy

2026-11-05。本轮只完成CPU准备：采集器、拟合/留出评价、48方案复核入口和合成回执测试。没有GPU测量，解析模型 `graph_analytic.py` 与已有Profile、原始记录均未修改。

## 范围

| 项 | 内容 |
| --- | --- |
| 局部标定 | 每阶段 `T(N)=a+bN`，N=S×3072输出元素；作为GPU0、当前Graph契约下的局部层，不替换OpenXLA风格解析模型 |
| 尺寸 | `[1,S,3072]` BF16，S=64、128、256、512、1024 |
| 拟合集 / 留出集 | S=64、256、1024 / S=128、512 |
| 整链核对 | 同批SiLU→Mul整链，五个尺寸；只做验证，不拟合折扣 |
| 通信 | 0.5 MiB、1.5 MiB × GPU0→GPU1、GPU1→GPU0，共四组 |
| 不做 | GEMM、GPU1完整标定、Fused多尺寸预测、跨GPU验证；缓存读取惩罚的代码不一致留待拟合p/k时处理 |

S=256也在本批重测，以便同批拟合和整链核对；`native_stage_calibration_20261105_01`的6.5376/3.6352 μs保留为跨批对照，不混入拟合。

## 采集契约

与 `native_stage_calibration` 一致：预分配输出，每Graph 50次调用，eager准备50次、10次预热、正式7轮；同stream Event包围单次Replay，计时外填NaN、同步、数值检查。19个Case按轮交替正序/逆序，共133次尝试，不补轮、不挑样本。IQR/Median>10%标记该Case `quality_failed`，其余Case照常保留。

- SiLU、Mul：与已归档阶段相同的计时体；Mul的left由真实BF16 SiLU在计时外生成。
- 整链：`silu.out(gate,out=mid); mul.out(mid,up,out=output)`，每次调用独立mid和输出，检查mid与BF16 SiLU逐字节一致。
- Copy：`execution/ffn_transfer.py` 的 `BoundTransfer`，目标卡stream计时，捕获内fork/join源侧stream；要求peer能力，缺失直接失败，不走host中转。peer能力和DtoD不等于已认证P2P路由。

身份按Case绑定Torch构建、扩展、HIP/驱动、两卡架构、Recipe、实际布局与契约；UUID仅溯源。

## 预注册的拟合与评价（已写入plan.json）

1. `fit`只能读取拟合尺寸标签（代码按role过滤，测试验证篡改留出/整链/Copy标签不改变拟合结果），用三点中位数OLS得 a、b，并对7³种逐轮组合给出范围（敏感性，不是置信区间）。斜率范围全正才可预测；a范围跨0则标“未单独识别”，只以组合直线做预测。拟合点任一失败则该阶段不拟合。
2. `fit`写出留出预测及其prediction_id，再由 `validate` 读留出标签；预测被改动即拒绝。报告逐条APE、signed bias、全部留出Case的两两排序，不预设误差阈值。冻结解析假设（core+h+c）在同一留出集上作为基线，不重拟合。
3. 整链：同批 Δ_add = SiLU+Mul−整链，以及拟合和与整链实测的误差；S=256另列历史10.0928 μs。
4. 参数按 GPU身份（设备+软件栈）× 契约ID × Recipe 存储；查询不命中返回空，不跨GPU/契约回退；拟合范围外标 `extrapolated`，留出前 `validated_elements_range` 为空。
5. 预测评价之后才运行48方案复核：GPU0 SiLU/Mul用同批S=256实测，质量通过的Copy替换四类链路代理，其余（GEMM、GPU1激活/Fused、GPU0 Fused Profile）不变。报告最优方案是否变化、通信翻转阈值相对实测Copy的位置，以及实测Copy与旧临界值13.99/15.42 μs的比较。最优方案不变不作为预测准确的证据。

## 服务器操作

提交同步到服务器后，在项目根目录：

```bash
export FUSION_PYTHON="$HOME/private_data/LJ/dcu-kernel310-20261031/run_python.sh"
bash tools/run_stage_size_calibration.sh ../stage_size_plan_20261106_01 plan
```

确认计划显示19个Case、133次尝试后，执行一次测量（输出目录必须不存在，失败不换Python重试）：

```bash
export FLAGSCALE_ROOT="$HOME/private_data/LJ/FlagScale-main"   # 可选，仅用于厂商环境初始化
bash tools/run_stage_size_calibration.sh ../stage_size_calibration_20261106_01 run
cd "$HOME/private_data/LJ" && tar -czf stage_size_calibration_20261106_01.tar.gz stage_size_calibration_20261106_01
```

需要两张卡都可见；显存峰值约2.5 GiB（GPU0）。完成或失败都回传整个目录。

## 回传后的CPU步骤

```bash
R=data/stage_size_calibration_20261106_01
python -X utf8 -m calibration.stage_size_calibration review --input $R/raw --output $R/review
python -X utf8 -m cost_model.stage_linear_calibration fit --review $R/review --plan $R/raw/plan.json --output $R/fit
python -X utf8 -m cost_model.stage_linear_calibration validate --review $R/review --plan $R/raw/plan.json --fit $R/fit/fit.json --output $R/validation
python -X utf8 -m cost_model.ffn_calibrated_search --validation $R/validation/validation.json --output $R/ffn_search
```

CPU测试：`python -X utf8 -m pytest calibration/tests/test_stage_size_calibration.py -q`（17项，均为合成回执）。

## 首批结果：2026-11-05（stage_size_calibration_20261106_01）

服务器HIP 6.3.26113，两卡gfx936，peer能力双向为真。133次尝试全部完成，无执行或数值错误；原始目录逐字节归档到 `data/stage_size_calibration_20261106_01/raw/`。采集时仓库HEAD仍为cab527d8，新采集源码未提交，溯源以plan.json中逐文件SHA256为准。

### 质量

15个SiLU/Mul/整链Case全部quality_passed（IQR/Median 0.27%～3.52%）。Copy仅0.5 MiB 0→1通过（8.76%）；另三组失败，原样保留：

| Copy | 中位数 / μs | IQR/Median | 7轮 / μs |
| --- | ---: | ---: | --- |
| 0.5 MiB 1→0 | 39.65 | 25.81% | 1201.95, 35.75, 35.94, 43.34, 52.56, 39.50, 39.65 |
| 1.5 MiB 0→1 | 50.24 | 23.73% | 45.44, 50.85, 50.24, 123.83, 64.66, 45.31, 46.22 |
| 1.5 MiB 1→0 | 55.88 | 14.94% | 1369.79, 46.75, 55.88, 52.29, 64.97, 52.15, 56.16 |

两组1→0的第0轮超过1 ms，属于首次正式Replay的异常值；按规则不剔除、不补轮。

### 拟合（S=64、256、1024）

| 阶段 | a / μs | b / (μs/元素) | a敏感性范围 | S=64/256/1024残差 / μs |
| --- | ---: | ---: | --- | --- |
| SiLU | 2.3557 | 5.7799e-6 | 2.30～2.92 | +0.27 / −0.33 / +0.07 |
| Mul | 0.5509 | 5.0987e-6 | 0.46～0.95 | +0.76 / −0.95 / +0.19 |

斜率和截距在全部7³种轮次组合下均为正，按预注册规则均算已识别。Mul在S=256处比直线低0.95 μs（约26%），线性形式对Mul拟合较差。

### 留出预测（先于搜索）

| Case | 线性预测 | 实测 | 误差 | 冻结解析假设 | 误差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SiLU S=128 | 4.6285 | 4.6848 | −1.20% | 10.1508 | +116.67% |
| SiLU S=512 | 11.4468 | 11.3312 | +1.02% | 36.1030 | +218.62% |
| Mul S=128 | 2.5558 | 2.7616 | −7.45% | 4.0123 | +45.29% |
| Mul S=512 | 8.5704 | 9.4208 | −9.03% | 11.5492 | +22.59% |

线性：MAPE 4.67%、signed bias −4.16%、最大9.03%，6对排序全对。冻结解析：MAPE 100.79%，全部高估，排序也全对。只有4个留出点、一个GPU、一个契约，均在拟合范围内；Mul两点同向低估，说明误差带有系统性，并非随机散布。

### 整链核对（同批，未拟合）

| S | 整链 | SiLU+Mul | Δ_add | 拟合和误差 |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 5.7792 | 6.0768 | +0.30（+5.15%） | −12.70% |
| 128 | 7.1136 | 7.4464 | +0.33（+4.68%） | +0.99% |
| 256 | 10.1664 | 10.1727 | +0.01（+0.06%） | +12.74% |
| 512 | 19.2896 | 20.7519 | +1.46（+7.58%） | +3.77% |
| 1024 | 37.2736 | 37.3856 | +0.11（+0.30%） | −0.39% |

单测相加均不低于整链，偏差0.06%～7.58%，不随尺寸单调变化。S=256整链10.1664 μs与历史10.0928 μs相差+0.73%。拟合和误差在S=64、256达到±12.7%，主要来自Mul直线的偏差。

### 48方案（预测验证之后）

GPU0 SiLU/Mul改用本批S=256实测（6.5664/3.6064 μs），只有0.5 MiB 0→1的Copy替换为实测34.46 μs，其余三类仍用代理。最优方案不变，仍为全GPU0 Native，78.8304 μs；最优跨卡方案由170.69降到147.43 μs。

对当前混合通信成本整体缩放，翻转阈值为α≈0.3003（原为0.2425）。由于三类Copy仍是代理，这个数字不是对实测通信的阈值。描述性比较：四组Copy全部28个轮次值都在33 μs以上，均超过原临界成本13.99/15.42 μs两倍以上；但三组质量不合格，不作为准入成本。

最优方案不变不能证明预测准确；完整FFN实测参考仍为0/48。
