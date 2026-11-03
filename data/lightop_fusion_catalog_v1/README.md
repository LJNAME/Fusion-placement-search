# LightOP 已有融合实现清单

固定 LightOP 0.6.0 / DTK2604 / Torch2.9，实际实现由海光库提供。按接口及变体登记 48 项，非独立 Pattern 数。公开导出共 132 项，完整索引在 capabilities.json 的 public_exports；未声称完成 C++ 扩展所有接口的盘点。

原版批测入口与说明见 ../../calibration/LIGHTOP_VENDOR_SUITE.md。全部项目已注册为 implemented；PlacedOpGraph Adapter 状态单独记录，尚未接入的不自动开放图重写。

| 接口 ID | 类别 | 包内测试任务 | 项目 GPU 证据 |
| --- | --- | --- | --- |
| `AddDropout` | activation_elementwise | test_add_dropout | 待测 |
| `AddLn` | layernorm | test_ln_add | 待测 |
| `CeluGN` | groupnorm | test_celu_groupnorm | 待测 |
| `FuseBiasDropoutAdd` | activation_elementwise | test_bias_dropout_add | 待测 |
| `FuseBiasGelu` | activation_elementwise | test_bias_gelu | 待测 |
| `FuseBiasSwiGLU` | activation_elementwise | test_bias_swiglu | 待测 |
| `FusedScaleMaskSoftmax` | softmax | test_scaled_masked_softmax, test_scaled_masked_softmax_dropout, test_scaled_upper_traing_masked_softmax, test_scaled_upper_traing_masked_softmax_dropout | 待测 |
| `FusedScaleMaskSoftmaxDropout` | softmax | test_scaled_masked_softmax_dropout, test_scaled_upper_traing_masked_softmax_dropout | 待测 |
| `GNSilu` | groupnorm | test_gn_silu | 待测 |
| `LnAddDropout` | layernorm | test_ln_add_dropout | 待测 |
| `LnAddDropoutBias` | layernorm | test_ln_add_dropout_bias | 待测 |
| `LnDropPath` | layernorm | test_ln_droppath | 待测 |
| `LnDropout` | layernorm | test_ln_dropout | 待测 |
| `RMSNormAdd` | rmsnorm | test_rn_add | 待测 |
| `RMSNormAddDropout` | rmsnorm | test_rn_add_dropout | 待测 |
| `RMSNormAddDropoutBias` | rmsnorm | test_rn_add_dropout_bias | 待测 |
| `RMSNormDropout` | rmsnorm | test_rn_dropout | 待测 |
| `ReluDropout` | activation_elementwise | test_relu_dropout | 待测 |
| `blockwise_scaled_cast_transpose` | data_movement_quantization | test_blockwise_scaled_1d_cast_transpose | 待测 |
| `destindex_copy_quantize_int8_cuda` | data_movement_quantization | test_quantization_cuda | 待测 |
| `fast_topk_transform_fused` | moe_routing | 尚未找到 | 待测 |
| `fast_topk_transform_ragged_fused` | moe_routing | 尚未找到 | 待测 |
| `fuse_fill_rms_x2_concat` | data_movement_quantization | test_fuse_fill_rms_x2_concat | 待测 |
| `fuse_groupgemm_permute_quant` | data_movement_quantization | test_fuse_groupgemm_permute_quant | 待测 |
| `fuse_rms_mrope_cuda` | norm_rope_cache | test_rms_mrope | 待测 |
| `fuse_rmsnorm_rope_quant_qkv` | norm_rope_cache | test_fuse_rmsnorm_rope_quant_qkv, test_fused_trans_bmm_trans_benchmark | 待测 |
| `fuse_silu_and_mul` | activation_elementwise | test_fuse_silu_mul | 待测 |
| `fuse_silu_mul_fp8_quant` | activation_quantization | 尚未找到 | 待测 |
| `fuse_silu_mul_fp8_quant_ep` | activation_quantization | 尚未找到 | 待测 |
| `fuse_silu_mul_quant` | activation_quantization | test_fuse_silu_mul_quant | 待测 |
| `fuse_silu_mul_quant_ep` | activation_quantization | test_fuse_silu_mul_quant_ep | 待测 |
| `fused_add_rms_norm` | rmsnorm | test_rms_add | 待测 |
| `fused_bmm` | bmm | test_fused_trans_bmm_trans_benchmark | 待测 |
| `fused_rms_norm_rope_contiguous` | norm_rope_cache | test_fuse_rmsnorm_rope_contiguous, test_fused_trans_bmm_trans_benchmark | 待测 |
| `gammarms_rotary_embedding_fuse_with_kv_store` | norm_rope_cache | test_fuse_gammarmsnorm_rope_qkv_kernel, test_fuse_rmsnorm_rope_qkv_no_quant, test_split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant | 待测 |
| `gemma_fused_add_rmsnorm` | rmsnorm | test_gemma_fused_add_rmsnorm | 待测 |
| `gemma_rms_norm_per_token_fp8_quant` | norm_quantization | test_gemma_rms_norm_fp8_quant | 待测 |
| `moe_fused_gate` | moe_routing | test_fused_moe_gate | 待测 |
| `moe_swiglu_dynamic_quant` | activation_quantization | test_moe_swiglu_dynamic_quant | 待测 |
| `op.gelu_and_mul_opt` | activation_elementwise | test_act_and_mul_gelu | 待测 |
| `op.gelu_tanh_and_mul_opt` | activation_elementwise | test_act_and_mul_gelu_tanh | 待测 |
| `op.silu_and_mul_opt` | activation_elementwise | test_act_and_mul_silu | trial02，探索性已测 |
| `reshape_and_cache_cuda` | data_movement_quantization | test_reshape_cache_cuda | 待测 |
| `rms_norm_dynamic_per_token_quant` | norm_quantization | test_rms_norm_dynamic_per_token_quant | 待测 |
| `rms_norm_per_token_fp8_quant` | norm_quantization | test_fuse_silu_mul_fp8_quant, test_rms_norm_fp8_quant | 待测 |
| `rms_rotary_embedding_fuse` | norm_rope_cache | test_fuse_rmsnorm_rope_qkv_no_quant, test_fused_rms_rope, test_split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant | 待测 |
| `rms_rotary_embedding_fuse_with_kv_store` | norm_rope_cache | test_fuse_rmsnorm_rope_qkv_kernel, test_fuse_rmsnorm_rope_qkv_no_quant, test_split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant | 待测 |
| `split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant` | norm_rope_cache | test_split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant | 待测 |
