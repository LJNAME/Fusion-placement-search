# Mul0 元数据 Smoke

独立于四格 pilot 和七项正式矩阵。scope=mul_metadata_smoke_only_not_profile_admission；不产生 Profile 标签或准入决策。

先运行四格 pilot，审阅 Down0、R1/R3 描述性差值与质量失败原因，再单独启动本入口。CPU 父进程要求 pilot_passed=true，并从当前版本的原始四格重算身份、contract、Trace、样本和质量；不能只改 summary 标志或用旧 collector.v1 的回执替代。R3 质量状态另列，不与 pilot 完整性混淆。

默认 plan，不访问 GPU：

```bash
python -m calibration.ffn_mul_metadata_smoke --output /fresh/mul_cpu_plan
```

显式 run 要求 --pilot-input、--flagscale-root 和全新输出目录。父进程只启动一次新 worker；无重试，无 pilot 运行时驻留。每次现场 GPU0 UUID/build 必须与 pilot 一致。

GPU worker 调用 ControlRuntime.activation(0, mul_smoke_only=True)，只创建 Mul 测量 Case。SiLU 仅用于在计时外生成 left fixture，不运行 KERV 探针或其他测量臂。Capture 含 50 次独立 Mul，随后三次诊断 Replay；所有恢复、数值与只读校验保持在 Event 外。

布尔验收：

- Capture 成功，50 个输出数值一致。
- 实际 inputs.left 含 50 份运行时 Shape/dtype/stride/device/data_ptr，地址与 gate/up 区分，输出不与输入别名。
- left/up 逐字节不变，gate 恢复精确。
- identity.json、measurement_contract_id、元数据、三次 Replay 的检查记录落盘，CPU 父进程复核通过。

计时值正常、过慢或不可用均不参与 pass。它不能验证计时质量，也不证明完整 activation 矩阵或 Kernel 性能已通过。通过后仍 followup_matrix_open=false，正式七项矩阵需另行冻结。

失败立即停止。普通异常保存完整 traceback、pool、GPU0 显存及 context；native worker 崩溃时保留 faulthandler 日志、last_stage、已写原件和 parent_failure.json。父进程只读取最后持久化的显存快照，不初始化 GPU 补采。

CPU 验证见 data/ffn_mul_metadata_smoke_v1/verification.json。当前没有 GPU Smoke 回执，旧 run05 Mul 身份缺口不变。
