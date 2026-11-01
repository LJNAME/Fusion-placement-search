# 固定Profile只读审计（2026-11-01）

当前主入口读取的36条Kernel记录全部legacy_unbound，没有accepted记录。2740条本地候选×设备使用中，1604条形状精确/插值查询会被严格Kernel闸门拒绝；不能将诊断覆盖率写成正式Profile覆盖。

| 尺度 | 节点 | 本地候选 | 使用数 | 逻辑input | 诊断精确 | 诊断近邻 | 解析 |
|---|---:|---:|---:|---:|---:|---:|---:|
| small | 5 | 6 | 12 | 0 | 10 | 0 | 2 |
| layer | 21 | 22 | 44 | 2 | 40 | 0 | 2 |
| model | 591 | 619 | 1238 | 56 | 1122 | 0 | 60 |
| model_3b | 687 | 723 | 1446 | 72 | 288 | 144 | 942 |

只审本地候选与设备组合，不是完整图空间。当前未绑定完整组Key的实现/runtime/布局，各图去重模板数12/38/40/36，identity_complete使用数均0；模板hash不替换原标签ID。GroupKernelKeys排除embedding但该项仍有解析成本，两个完整模型各2条设备使用，单列不吞入logical_input。

回评mainline_four_mode_v1/run1全部20个最佳方案，预测与原报告一致：213组中190诊断精确命中、15解析回退、8逻辑input；10条跨设备消息均查链路实测表。group duration只消费一次，拆出的launch未被额外相加。详细计费表和源码位置见cost_model/FIXED_PROFILE_SEMANTICS.md。

两方向链路各6点，4KiB–256MiB；26个图尺寸/方向查询未外推，但链路不走Kernel测量契约闸门。source_id相符不能替代执行语义确认。没有单独host调度成本字段，也不能推断旧标签不含host/harness。

report.json保留来源/严格查询/身份缺口/参数/告警/20方案回评。candidate_groups.csv有2740行，link_queries.csv26行，selected_groups.csv213行，selected_transfers.csv10行。validation.json复核逐尺度CSV计数、实测duration拆分与20方案数，输入及源码hash不变；冻结SROP的41证据/11未提交文件hash也保持。

未采GPU、未回写模型/标签、未拟合参数。新解析入口与逻辑input零时间修正在独立主系统基线中验证；本审计不把旧标签升accepted，不以缺口数量决定新测量任务。
