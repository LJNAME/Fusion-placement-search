# 3B融合后端源码与发布包审计

2026-11-07。目标为Qwen2.5-3B的BF16 SiLU×Mul宽度11008、Add＋RMSNorm [1,256,2048]。只读取KERV固定源码与LightOP官方包，未改注册规则、安装环境或执行GPU。

## KERV Add＋RMSNorm

| 入口 | 2048形状 | 输出及副作用 | 路由 |
| --- | --- | --- | --- |
| torch.ops.flagos_embodied.kerv_add_rms_norm | 满足width≤8192 | hidden原地变为归一化结果；原residual只读；没有post-add残差输出 | auto默认Native；必须显式triton且算子在enabled集合内 |
| KERVRuntimeOptimization.adaptive_rms_norm.add_rms_norm_inference | 满足当前源码的形状/dtype算术条件 | 返回(normalized, post-add residual)；post-add覆盖contiguous_input，原residual只读 | 直接调用_add_rms_norm_inference_kernel；缺Triton报错 |

公共算子源码：embodied_ops/kerv_ops.py:362。它只写normalized，不能独自替代当前双输出融合组；FlagScale现有flagos_kerv hook在调用前另算attention_output＋residual_input以保留残差（adaptive_linear_fusion.py:507），该Add必须计入完整调用。

优先核对的双输出入口是adaptive_rms_norm.py:247。其内部会对三个输入调用contiguous()，并empty_like分配输出（:277），这些操作有条件产生转换/分配，不能默认为零成本。连续BF16 [1,256,2048]的契约stride为[524288,2048,1]，weight为[2048]，256行，block2048、num_warps4。

36层对应的out_proj只由该层Add消费，具备覆盖attention输出的图上前提；Add结果还被residual_add_2消费，双输出不可省略。完整逻辑图物理布局未知，真实Tensor的连续性、设备、别名与生命周期仍未验证；epsilon未写入图，需要明确模型声明。

舍入有差异：公共kernel在FP32求和后直接归一化；adaptive kernel先将和转回BF16，归一化后也先转BF16再乘weight（adaptive_rms_norm.py:163/178）。后者针对手写Llama/Qwen路径，不能未经核对就当作当前aten RMSNorm参考的逐位等价实现。

本项目未找到2048的KERV GPU执行、dispatch或Profile证据。现有公共算子CPU测试是2×8；benchmark的AddRMS是8×4096。benchmark里虽出现SiLU宽度11008，但会受8192 guard回退，不能用测试名称证明融合发生。当前Fusion项目没有KERV AddRMS图Adapter。

## LightOP版本与目标实现

检查范围为官方release/dtk索引：https://pypi.sourcefind.cn/release/dtk/+simple/lightop/ 。索引原文和wheel官方SHA均保存。

| 项目 | 当前固定包 | 索引中最新日期构建 |
| --- | --- | --- |
| 版本 | 0.6.0+das.dtk2604.torch290.2605220936.g6bfaca | 0.6.0+das.dtk2604.torch2100.2608121706.g09a4f3 |
| 发布集合 | DAS1.8.2 | DAS1.8.5 |
| METADATA要求 | torch==2.9.0 | torch==2.10.0 |
| wheel内测试Python文件 | 121 | 0 |
| 目标算子的C++/HIP源码 | 未附带 | 未附带 |

最新wheel SHA256：4a31521919e4c76405e8dcb5035b55cdc92b2ad6dc5e9f2e72373510835df159。当前Torch2.9分支在所查索引中未发现比已固定包更新的构建。下载仅供离线检查，未安装。

最新接口仍指向二进制扩展：lightop/activation/existing.py:25的silu_and_mul_opt、lightop/norm/existing.py:28的fused_add_rms_norm_opt。最新rmsnorm.py:25保留原地双输出接口，新增可选limit=-1.0。包内不含目标kernel的C++/HIP函数体；匿名公开项目索引未取得与该构建对应的完整源码，不能据此核实11008/2048的底层dispatch条件。

当前包的直接测试：test_act_and_mul.py:53为BF16输出宽度4096；test_rms_add.py:91为tokens128/288/352、宽度4096，FP16/BF16。均不是目标11008或2048。量化测试里的vLLM基准或其他RMSNorm接口不算这两个入口的目标验证。

结论：两个LightOP目标Shape目前均未核实；这不表示厂商Kernel不支持。上一轮3072/1024是本项目Adapter的证据范围，不能当作厂商Kernel上限。最新Torch2.10包也不能拿来直接替换当前Torch2.9已准入实现。

当前最直接的源码候选为KERV adaptive双输出AddRMS2048；还需核对参考舍入及真实布局/别名，再进行单Case验证。没有增加搜索实验或将任何候选改成GPU已验证。

明细：data/3b_fusion_backend_audit_v1/report.json；离线包及相关Python源码位于E:/Chatgpt_work/transfer/lightop-3b-source-audit-20261107/。未把wheel二进制加入项目数据目录。
