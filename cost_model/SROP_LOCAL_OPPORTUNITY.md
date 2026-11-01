# 四臂局部八掩码审计

入口cost_model.srop_local_opportunity，只读取已暴露四臂开发数据，不采GPU、不调用SROP评分、不更新模型。三Key为SiLU/Mul/fused；chain只评价。全部8个可见掩码、12条单次揭示边完整列出，不按结果筛状态。

A用现有解析模型及旧硬件配置，明确为迁移先验；B没有独立校准依据，输出32行pending，不伪造graph一致先验。C只是假设launch3.2μs，并对隐藏融合Key乘0.5/1/1.5，不当真实校准或概率模型。

主选择严格用SiLU+Mul对fused；参考为每case固定chain/fused中位数，所有掩码与逐轮标签敏感性共享同一参考。单次补测只替换一个Key，无refit或MCTS。旁表上下文修正使用隐藏参考，显式标oracle/posthoc，不回流主选择或动作收益，不拟合邻接常数。数据仍conditional_dev，工具不绕过模型入口的标签闸门。

```bash
python -m cost_model.srop_local_opportunity \
  --bundle data/silu_mul_additivity_20261101_150818 \
  --out-dir data/local_opportunity_new_run
```

输出plan/report/verification.json，128行主状态、192条动作、1024逐轮状态、1536逐轮动作、32行B待建立、128行oracle旁表。计数来自4case×A和3个C情景；不是独立测试样本数。plan注明事后开发审计，源/输入hash保留；输出必须为新目录。

首次运行结果摘要与主CSV见data/srop_local_opportunity_v1；完整本地工件在工作区transfer/srop-local-opportunity-20261101-v2，含逐轮CSV和独立重算脚本。新7项测试覆盖隐藏参考隔离、全掩码、纠错/有害动作、单有效多可选动作、旧先验复现、B空结果和固定参考。
