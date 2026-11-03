# LightOP 已有融合实现清单

固定 LightOP 0.6.0 / DTK2604 / Torch2.9，实际实现由海光库提供。按接口及变体登记 48 项，非独立 Pattern 数。公开导出共 132 项，完整索引在 capabilities.json 的 public_exports；未声称完成 C++ 扩展所有接口的盘点。

原版批测说明见 ../../calibration/LIGHTOP_VENDOR_SUITE.md，首批审核见 ../../calibration/LIGHTOP_VENDOR_SUITE01_REVIEW.md。capabilities.json 保留首版身份快照及当时的测量状态，最新观察追加在 ../lightop_vendor_suite01_review/capability_status.json，capabilities() 会一并返回。全部接口为 implemented；图适配状态单独记录，尚未接入的不自动开放图重写。

| 接口 ID | 类别 | 首批原版测试观察 | 此前项目 GPU 证据 |
| --- | --- | --- | --- |
| `AddDropout` | activation_elementwise | 时延与原版检查通过 | 无 |
| `AddLn` | layernorm | 时延与原版检查通过 | 无 |
| `CeluGN` | groupnorm | 时延与原版检查通过 | 无 |
| `FuseBiasDropoutAdd` | activation_elementwise | 时延与原版检查通过 | 无 |
| `FuseBiasGelu` | activation_elementwise | 时延与原版检查通过 | 无 |
| `FuseBiasSwiGLU` | activation_elementwise | 时延与原版检查通过 | 无 |
| `FusedScaleMaskSoftmax` | softmax | 仅正确性 | 无 |
| `FusedScaleMaskSoftmaxDropout` | softmax | 打印校验失败 | 无 |
| `GNSilu` | groupnorm | 时延与原版检查通过 | 无 |
| `LnAddDropout` | layernorm | 时延与原版检查通过 | 无 |
| `LnAddDropoutBias` | layernorm | 时延与原版检查通过 | 无 |
| `LnDropPath` | layernorm | 时延与原版检查通过 | 无 |
| `LnDropout` | layernorm | 时延与原版检查通过 | 无 |
| `RMSNormAdd` | rmsnorm | 时延与原版检查通过 | 无 |
| `RMSNormAddDropout` | rmsnorm | 时延与原版检查通过 | 无 |
| `RMSNormAddDropoutBias` | rmsnorm | 时延与原版检查通过 | 无 |
| `RMSNormDropout` | rmsnorm | 时延与原版检查通过 | 无 |
| `ReluDropout` | activation_elementwise | 时延与原版检查通过 | 无 |
| `blockwise_scaled_cast_transpose` | data_movement_quantization | 时延与原版检查通过 | 无 |
| `destindex_copy_quantize_int8_cuda` | data_movement_quantization | 时延及零差异报告 | 无 |
| `fast_topk_transform_fused` | moe_routing | 本批无配套测试 | 无 |
| `fast_topk_transform_ragged_fused` | moe_routing | 本批无配套测试 | 无 |
| `fuse_fill_rms_x2_concat` | data_movement_quantization | 缺依赖，未执行 | 无 |
| `fuse_groupgemm_permute_quant` | data_movement_quantization | 缺依赖，未执行 | 无 |
| `fuse_rms_mrope_cuda` | norm_rope_cache | 数值检查失败 | 无 |
| `fuse_rmsnorm_rope_quant_qkv` | norm_rope_cache | 仅正确性 | 无 |
| `fuse_silu_and_mul` | activation_elementwise | 时延与原版检查通过 | 无 |
| `fuse_silu_mul_fp8_quant` | activation_quantization | 本批无配套测试 | 无 |
| `fuse_silu_mul_fp8_quant_ep` | activation_quantization | 本批无配套测试 | 无 |
| `fuse_silu_mul_quant` | activation_quantization | 缺依赖，未执行 | 无 |
| `fuse_silu_mul_quant_ep` | activation_quantization | 缺依赖，未执行 | 无 |
| `fused_add_rms_norm` | rmsnorm | 时延与原版检查通过 | 无 |
| `fused_bmm` | bmm | 缺依赖，未执行 | 无 |
| `fused_rms_norm_rope_contiguous` | norm_rope_cache | 缺依赖，未执行 | 无 |
| `gammarms_rotary_embedding_fuse_with_kv_store` | norm_rope_cache | 时延与原版检查通过 | 无 |
| `gemma_fused_add_rmsnorm` | rmsnorm | 容差相关，无时延 | 无 |
| `gemma_rms_norm_per_token_fp8_quant` | norm_quantization | 缺依赖，未执行 | 无 |
| `moe_fused_gate` | moe_routing | 缺依赖，未执行 | 无 |
| `moe_swiglu_dynamic_quant` | activation_quantization | 时延与原版检查通过 | 无 |
| `op.gelu_and_mul_opt` | activation_elementwise | 缺依赖，未执行 | 无 |
| `op.gelu_tanh_and_mul_opt` | activation_elementwise | 缺依赖，未执行 | 无 |
| `op.silu_and_mul_opt` | activation_elementwise | 缺依赖，未执行 | trial02 探索性已测 |
| `reshape_and_cache_cuda` | data_movement_quantization | 缺依赖，未执行 | 无 |
| `rms_norm_dynamic_per_token_quant` | norm_quantization | 缺依赖，未执行 | 无 |
| `rms_norm_per_token_fp8_quant` | norm_quantization | 缺依赖，未执行 | 无 |
| `rms_rotary_embedding_fuse` | norm_rope_cache | 缺依赖，未执行 | 无 |
| `rms_rotary_embedding_fuse_with_kv_store` | norm_rope_cache | 时延与原版检查通过 | 无 |
| `split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant` | norm_rope_cache | 仅正确性 | 无 |
