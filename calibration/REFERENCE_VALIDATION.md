# 独立序列任务：实验版方案代价验证

冻结两族模型的seq768/1280单层参考图、预测与候选，主指标为预先固定物理选择距已测候选最优的regret≤1%。这轮验证实验版代价模型，不验证SROP主动测量收益或停止规则。

## 冻结范围

- Qwen3-0.6B/Qwen2.5-3B × seq768/1280，batch1、BF16、单层synthetic forward，4任务。
- Manifest只保留单层实际节点的测量对象，共56个：52项可测、4项逻辑输入。不测embedding、最终模型Norm或词表head。
- 52个完整Kernel Key与此前暴露的168项无重合；按预测器实际使用的列比较，49项特征不同、3项有特征别名。全部52项保留，结果另报49项口径。
- 仍用原112标签拟合nearest/distance-floor q90；未加入原56个开发对象的真实时间。旧开发结果用于方法选择，新的4任务才是独立序列任务；不是未知模型架构或跨GPU泛化。
- 主方案为ReferenceEagerCostModel + 原未验收host候选；另冻结StructuralCostModel、零host模型作为基线。host.calibrated=false和算子body/GQA/布局未核实均保留为实验条件，不宣称可部署。

正式freeze_id为sha256:92193b748c7440a4528f249e207dd77b3fadcf9e0d25f0be118900997ec4e9fb；manifest、输入、源码、预测及选择已锁定。check只校验，不测GPU、不读取新标签。

## 候选和指标

共同候选池由原结构MCTS（seed0/1/7/23、64次、rollouts3）及结构探针提出，再由三种模型重评分。这不是ReferenceEager引导的重新搜索，也没有Oracle候选。

每种模型Top4交换类，加单卡/半图跨卡控制；所有入选类都执行两个物理方向。4任务共66个物理候选；各2轮、每轮5预热+20正式，合计132块/2640正式前向。数值使用共同canonical CPU输入/权重，原rtol=.01、atol=.001及有限性检查。

每种模型的物理选择在标签出现前按完整预测并列集合的最小ID固定，两个方向均实测。epsilon_report.json报告：

- 该预选物理方案的真实候选子集regret，主ε=1%、敏感性0.1%/0.5%；
- 两轮分别的regret、全部原始时间及数值/运行失败；
- 预测并列集合中真实最好/最坏regret；
- 所选交换类的最佳成员（只作事后诊断，不作为部署选择）；
- 单算子全部52项及特征不同49项的预测误差和q90覆盖。

不能看完标签再挑快卡或重排候选。主regret以两轮合并中位为准，各轮另报，不把有限重复当1%统计置信上界。“最优”仅限实际执行的冻结候选，不是全搜索空间Oracle。

旧execution_report.json保留原5%辅助指标；本实验的主指标在新增epsilon_report.json，不能用旧5%通过替代新1%口径。

## 服务器执行（两张BW1000）

仓库同步完整时：

```bash
cd ~/private_data/LJ/Fusion-placement-search
python -m calibration.reference_validation check
bash tools/run_reference_validation.sh
```

若继续手动上传，使用reference_validation_v1.deployment.zip。它含冻结代码、输入和工件，不含服务器原始采集输出。把压缩包上传到服务器后，在项目根目录解压：

```bash
cd ~/private_data/LJ/Fusion-placement-search
python -m zipfile -e /实际上传路径/reference_validation_v1.deployment.zip .
python -m calibration.reference_validation check
bash tools/run_reference_validation.sh
```

不用在服务器运行prepare，也不用重新训练或下载模型权重。原Torch2.9/HIP环境即可；实际采集入口拒绝CPU回退。

脚本先校验，测52对象×GPU0/1=104条正式kernel记录（10预热、7×50 Event计时），评估冻结zero-shot预测，然后执行66候选并计算ε结果。不在过程中更新模型、候选或阈值；Active calibration禁用，未来需单独冻结访问规则。

采集结果写data/reference_validation_v1_results/。可在同一宿主环境恢复；不要混入旧结果或跨宿主接续。质量或regret未达目标也保留数据；运行/数值失败中止后回传已有记录，不反复重跑到通过。

## 回传

```bash
tar -czf data/reference_validation_v1_results.tar.gz \
  -C data reference_validation_v1_results
```

回传整包及终端输出，包含raw_measurements.jsonl、environments.json、zero_shot_report.json、execution_samples.jsonl、execution_environment.json、execution_report.json、epsilon_report.json。成功采集不等于1%目标通过；失败结果同样保留。
