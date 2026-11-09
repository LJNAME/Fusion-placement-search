# 三类融合调用测时扩充

固定当前LightOP 0.6.0／Torch2.9栈，GPU0、BF16，各取一个真实图上的典型形状。已有Native与KERV记录不重测。

| Case | 典型形状 | 实际入口 | 完整计时体 |
| --- | --- | --- | --- |
| SiLU×Mul | Qwen2.5-3B `[1,256,11008]` | `lightop.op.silu_and_mul_opt` | 分离gate/up打包＋厂商调用 |
| Add＋RMSNorm | Qwen2.5-3B `[1,256,2048]` | `lightop.fused_add_rms_norm` | 两次私有输入复制＋原地融合调用，保留Add与Norm双输出 |
| Q/K RMSNorm＋RoPE | Qwen3 Q=`[1,256,16,128]`，K=`[1,256,8,128]` | `lightop.rms_rotary_embedding_fuse` | Q/K私有复制＋联合归一化旋转调用，保留Q/K双输出 |

三个API均有此前厂商测试通过记录，但本表的目标Shape／BF16组合尚未在本轮GPU验证。SiLU×Mul与RMSRoPE证据来自 `data/lightop_vendor_20261104_02_review/capability_status.json`，AddRMS来自 `data/lightop_vendor_suite01_review/capability_status.json`；历史七项未完成任务不重跑。

当前图不记录RoPE旋转约定。本轮显式采用head_dim=rotary_dim=128、theta=1e6、NeoX半分旋转、eps=1e-6、无residual的局部实验声明；只确认Q/K形状对应图，不认证完整模型RoPE语义，也不注册搜索规则。此前厂商测试的FP16、部分旋转维度结果不代替该BF16目标验证。

## 采集口径

- 每Case独立子进程、独立Graph私有池，50份可写缓冲，输入只读；最大一项的私有buffer约806.25 MiB（SiLU×Mul）。
- 先50次eager准备，再capture；3次smoke仅检查capture、数值、只读输入与独立输出，计时值不作smoke通过条件。
- 10次预热Replay，正式固定7次尝试，每Replay50次调用；同卡同stream Event计时，取μs/call中位数，IQR/Median≤10%。不补轮、不挑选样本，不以融合更快作为通过条件。
- 打包与保护输入的复制包含在计时内；分配、输入生成、数值校验、host提交不包含。这里采的是完整局部调用，不声称分离出的纯Kernel时间。
- 每次Replay前将全部私有缓冲置NaN，防止未真正执行却使用预热旧输出。每次测后检查50份结果、两种数值参考、全部只读输入逐字节不变。
- 已知profiler环境变量开启时拒绝计时；本轮不启动profiler，不认证唯一物理Kernel数量。直接调用固定厂商入口，无CPU／Native／eager计时回退。
- Smoke失败即停止该Case、不进入正式计时；一个子进程失败不跳过其余Case。完整日志、进度、显存快照、traceback及返回码保留。正式离散度不合格只标quality_failed，不重跑。

## 服务器运行

```bash
cd ~/private_data/LJ/Fusion-placement-search
git pull --ff-only
export FLAGSCALE_ROOT="$HOME/private_data/LJ/FlagScale-main"
export FUSION_PYTHON="$HOME/private_data/LJ/dcu-kernel310-20261031/run_python.sh"

bash tools/run_fusion_timing_expansion.sh ../fusion_timing_expansion_20261109_01 run
```

输出目录必须不存在。脚本使用已安装且哈希匹配的LightOP，不安装新包；FLAGSCALE_ROOT只用于加载现有海光环境。无论通过还是失败，回传整个目录：

```bash
cd ~/private_data/LJ
tar -czf fusion_timing_expansion_20261109_01.tar.gz fusion_timing_expansion_20261109_01
```

输出含总 `plan.json`、`summary.json`；各Case下有 `process.log`、`runtime.json`、`identity.json`、`smoke.json`、`round_0..6.json`、`result.json`，失败时保留已生成文件。SIGSEGV由父进程返回码记录，不因已有result文件而认定成功；不生成大体积core文件。

## CPU准备及回传审核

本地仅CPU版Torch2.9，未运行BW1000。已生成 `data/fusion_timing_expansion_plan_v1/plan.json`，三组图上Shape/dtype、厂商历史证据及源码哈希通过检查。29项CPU测试通过，Shell语法检查通过；pytest缓存目录不可写产生一个缓存警告，不影响测试结果。

```bash
python -m calibration.fusion_timing_expansion review --input /path/to/returned_directory --output /fresh/review
```

审核重算7轮统计，检查实际输入／布局、库版本／二进制哈希、设备架构、全部输出检查回执、计时边界及子进程退出状态。运行设备UUID只作溯源，不绑定成本复用。

记录的execution_key属于本轮局部调用身份，不冒充现有accepted搜索Profile。自动准入、搜索规则变更、MCTS修改均关闭；回传前没有新增时延数值或已验证目标能力。