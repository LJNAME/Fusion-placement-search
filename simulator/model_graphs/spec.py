"""Framework-independent model specifications used by logical graph builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


_DTYPE_BYTES = {
    "bf16": 2,
    "fp16": 2,
    "fp32": 4,
}


@dataclass(frozen=True)
class TransformerModelSpec:
    """Static decoder-only Transformer description.

    The specification deliberately contains no device, rank, or parallelism
    fields.  Those belong to the parallel-lowering layer.
    """

    model_type: str
    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    num_query_groups: int
    head_dim: int
    seq_length: int
    micro_batch_size: int
    vocab_size: int
    dtype: str = "bf16"
    normalization: str = "RMSNorm"
    activation: str = "silu"
    use_swiglu: bool = True
    qk_layernorm: bool = False
    tied_embeddings: bool = True
    attention_backend: str = "unfused"
    attention_softmax_in_fp32: bool = True
    use_rotary_position_embeddings: bool = True
    linear_bias: bool = False
    num_experts: Optional[int] = None
    moe_layer_pattern: Tuple[bool, ...] = ()

    def __post_init__(self) -> None:
        dtype = self.dtype.lower()
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "attention_backend", self.attention_backend.lower())
        if dtype not in _DTYPE_BYTES:
            raise ValueError(f"unsupported dtype: {self.dtype}")

        positive_fields = (
            "num_layers",
            "hidden_size",
            "ffn_hidden_size",
            "num_attention_heads",
            "num_query_groups",
            "head_dim",
            "seq_length",
            "micro_batch_size",
            "vocab_size",
        )
        for field_name in positive_fields:
            value = int(getattr(self, field_name))
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}")

        if self.num_attention_heads % self.num_query_groups != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_query_groups: "
                f"{self.num_attention_heads} vs {self.num_query_groups}"
            )
        if self.num_experts is not None and self.num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {self.num_experts}")
        if self.moe_layer_pattern and len(self.moe_layer_pattern) != self.num_layers:
            raise ValueError(
                "moe_layer_pattern length must match num_layers: "
                f"{len(self.moe_layer_pattern)} vs {self.num_layers}"
            )

    @property
    def dtype_bytes(self) -> int:
        return _DTYPE_BYTES[self.dtype]

    @property
    def softmax_dtype_bytes(self) -> int:
        return 4 if self.attention_softmax_in_fp32 else self.dtype_bytes

    @property
    def q_projection_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_projection_size(self) -> int:
        return self.num_query_groups * self.head_dim

    @property
    def tokens_per_micro_batch(self) -> int:
        return self.micro_batch_size * self.seq_length


__all__ = ["TransformerModelSpec"]
