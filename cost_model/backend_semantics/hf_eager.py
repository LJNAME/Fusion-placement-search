"""HuggingFace transformers Qwen2 eager attention, plain torch autograd.

Scope: transformers==4.57.6 modeling_qwen2 with ``_attn_implementation =
"eager"``, torch 2.9.0 autograd, ``ForCausalLMLoss``, wgrad through ``.grad``
(post-accumulate hooks copy into an FP32 main_grad). This is the harness of
calibration/training_memory_validation.py, not FlagScale/Megatron.

Evidence (BW1000, HIP 6.3):
- per-op saved tensors: saved_tensors_hooks audit, one decoder layer at seq
  256 saves 41,420,800 B (data/training_memory_p1d_v1_audit/
  saved_tensors_seq256.json); these rules reproduce it exactly.
    rmsnorm: x.to(fp32) kept by pow/mul, rsqrt kept, BF16 normalized kept by
             the weight mul; the BF16 input itself is not kept.
    softmax: FP32 result kept by SoftmaxBackward; the BF16 .to() copy is a
             graph edge kept by the av bmm.
    rope:    keeps only cos/sin, shared by all layers (not counted).
    add / input passthrough / embedding (indices): keep no activation.
    qk / av bmm keep K / V after repeat_kv (GQA-expanded).
- tied lm_head / embedding wgrad: allocator trace of the backward
  (data/training_memory_p1d_v1_audit/backward_trace): the lm_head wgrad is
  held in autograd's InputBuffer until the embedding backward, which adds its
  own wgrad and an out-of-place sum (3x weight live).
- loss: ForCausalLMLoss upcasts logits, so only the FP32 copy is kept; in its
  backward NllLossBackward and LogSoftmaxBackward each hold an FP32
  logits-sized grad at once (trace data/training_memory_p1d_v2_audit/
  backward_trace_seq1024); both are freed before the producer backward.
"""

from __future__ import annotations

from ..training_semantics import (
    LOSS_RELEASE_AT_LOSS_BACKWARD,
    ExecutionSemantics,
    LossSemantics,
    OpSaveRule,
    WeightGradSemantics,
)

HF_QWEN2_EAGER = ExecutionSemantics(
    name="hf_qwen2_eager_saved_v1",
    ops={
        "matmul": OpSaveRule(saves_inputs=True),
        "silu": OpSaveRule(saves_inputs=True),
        "mul": OpSaveRule(saves_inputs=True),
        "add": OpSaveRule(saves_inputs=False),
        "input": OpSaveRule(saves_inputs=False),
        "embedding": OpSaveRule(saves_inputs=False),
        "rope": OpSaveRule(saves_inputs=False),
        "rmsnorm": OpSaveRule(
            saves_inputs=False,
            internal=(("output", "fp32"), ("output", "output"), ("row", "fp32")),
        ),
        "softmax": OpSaveRule(saves_inputs=False, internal=(("output", "compute"),)),
    },
    expanded_inputs={
        ("self_attention.qk_matmul", "self_attention.k_rope"): "gqa_repeat_factor",
        ("self_attention.av_matmul", "self_attention.v_proj"): "gqa_repeat_factor",
    },
    loss=LossSemantics(
        saves_logits=False,
        fp32_copy_factor=1.0,
        backward_fp32_grads=2.0,
        release=LOSS_RELEASE_AT_LOSS_BACKWARD,
        verified=True,
    ),
    weight_grad=WeightGradSemantics(
        accumulate_into_main_grad=False,
        shared_held_across_users=True,
        shared_extra_copies_per_later_user=2.0,
    ),
    verified=True,
    source="transformers==4.57.6 modeling_qwen2 eager + torch 2.9.0 autograd; BW1000 audits",
)

__all__ = ["HF_QWEN2_EAGER"]
