"""Adapters from FlagScale/Megatron configuration to a neutral model spec."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Iterable

from simulator.model_graphs import TransformerModelSpec


_MISSING = object()
_NO_DEFAULT = object()


def _read(obj: Any, key: str, default: Any = _NO_DEFAULT) -> Any:
    if obj is None:
        if default is _NO_DEFAULT:
            raise KeyError(key)
        return default
    if isinstance(obj, Mapping) and key in obj:
        return obj[key]
    try:
        return getattr(obj, key)
    except (AttributeError, TypeError):
        pass
    try:
        return obj[key]
    except (KeyError, TypeError, IndexError):
        if default is _NO_DEFAULT:
            raise KeyError(key)
        return default


def _path(obj: Any, dotted: str, default: Any = _MISSING) -> Any:
    value = obj
    for part in dotted.split("."):
        try:
            value = _read(value, part)
        except KeyError:
            return default
    return value


def _first(obj: Any, paths: Iterable[str], default: Any = _MISSING) -> Any:
    for path in paths:
        value = _path(obj, path, _MISSING)
        if value is not _MISSING and value is not None:
            return value
    if default is _MISSING:
        raise KeyError(" | ".join(paths))
    return default


def _flagscale_root(config: Any) -> Any:
    train = _read(config, "train", _MISSING)
    if train is not _MISSING and _read(train, "model", _MISSING) is not _MISSING:
        return train
    return config


def _dtype_name(value: Any) -> str:
    text = str(value).lower().replace("torch.", "")
    aliases = {
        "bfloat16": "bf16",
        "bf16": "bf16",
        "float16": "fp16",
        "half": "fp16",
        "fp16": "fp16",
        "float32": "fp32",
        "float": "fp32",
        "fp32": "fp32",
    }
    if text not in aliases:
        raise ValueError(f"unsupported dtype value: {value}")
    return aliases[text]


def _dtype_from_flagscale(root: Any) -> str:
    if bool(_first(root, ("system.precision.bf16", "model.bf16"), default=False)):
        return "bf16"
    if bool(_first(root, ("system.precision.fp16", "model.fp16"), default=False)):
        return "fp16"
    raw = _first(root, ("model.params_dtype", "system.params_dtype"), default="fp32")
    return _dtype_name(raw)


def model_spec_from_flagscale(
    config: Any,
    model_type: str = "qwen3",
) -> TransformerModelSpec:
    """Translate a composed FlagScale config or its ``train`` section.

    Both dict/OmegaConf-style mappings and attribute objects are accepted.  A
    composed config is expected so that tokenizer vocabulary fields inherited
    through Hydra defaults are visible.
    """

    root = _flagscale_root(config)
    hidden_size = int(_first(root, ("model.hidden_size",)))
    attention_heads = int(_first(root, ("model.num_attention_heads",)))
    head_dim = int(
        _first(
            root,
            ("model.kv_channels", "model.head_dim"),
            default=hidden_size // attention_heads,
        )
    )
    query_groups = int(
        _first(root, ("model.num_query_groups",), default=attention_heads)
    )
    vocab_size = int(
        _first(
            root,
            (
                "model.padded_vocab_size",
                "model.vocab_size",
                "data.tokenizer.vocab_size",
                "data.vocab_size",
            ),
        )
    )
    use_swiglu = bool(_first(root, ("model.swiglu", "model.use_swiglu"), default=False))
    activation = str(
        _first(root, ("model.activation", "model.activation_func"), default="silu" if use_swiglu else "gelu")
    ).lower()

    return TransformerModelSpec(
        model_type=str(_first(root, ("model.model_type",), default=model_type)),
        num_layers=int(_first(root, ("model.num_layers",))),
        hidden_size=hidden_size,
        ffn_hidden_size=int(_first(root, ("model.ffn_hidden_size",))),
        num_attention_heads=attention_heads,
        num_query_groups=query_groups,
        head_dim=head_dim,
        seq_length=int(_first(root, ("model.seq_length",))),
        micro_batch_size=int(_first(root, ("model.micro_batch_size", "system.micro_batch_size"))),
        vocab_size=vocab_size,
        dtype=_dtype_from_flagscale(root),
        normalization=str(_first(root, ("model.normalization",), default="LayerNorm")),
        activation=activation,
        use_swiglu=use_swiglu,
        qk_layernorm=bool(
            _first(root, ("system.qk_layernorm", "model.qk_layernorm"), default=False)
        ),
        tied_embeddings=not bool(
            _first(root, ("model.untie_embeddings_and_output_weights",), default=False)
        ),
        attention_backend=str(
            _first(root, ("model.attention_backend",), default="unfused")
        ),
        attention_softmax_in_fp32=bool(
            _first(
                root,
                ("system.precision.attention_softmax_in_fp32", "model.attention_softmax_in_fp32"),
                default=False,
            )
        ),
        use_rotary_position_embeddings=bool(
            _first(root, ("model.use_rotary_position_embeddings",), default=False)
        ),
        linear_bias=not bool(
            _first(root, ("system.disable_bias_linear", "model.disable_bias_linear"), default=False)
        ),
        num_experts=_first(root, ("model.num_experts",), default=None),
    )


def model_spec_from_megatron(
    args: Any,
    transformer_config: Any,
    model_type: str = "qwen3",
) -> TransformerModelSpec:
    """Build the same neutral spec from live Megatron argument objects."""

    def pick(name: str, default: Any = _MISSING) -> Any:
        for source in (transformer_config, args):
            value = _read(source, name, _MISSING)
            if value is not _MISSING and value is not None:
                return value
        if default is _MISSING:
            raise KeyError(name)
        return default

    def pick_any(names: Iterable[str], default: Any = _MISSING) -> Any:
        for name in names:
            try:
                return pick(name)
            except KeyError:
                pass
        if default is _MISSING:
            raise KeyError(" | ".join(names))
        return default

    hidden_size = int(pick("hidden_size"))
    attention_heads = int(pick("num_attention_heads"))
    use_swiglu = bool(pick_any(("swiglu", "gated_linear_unit"), False))
    dtype = _dtype_name(pick("params_dtype", "bf16" if pick("bf16", False) else "fp32"))
    return TransformerModelSpec(
        model_type=str(pick("model_type", model_type)),
        num_layers=int(pick("num_layers")),
        hidden_size=hidden_size,
        ffn_hidden_size=int(pick("ffn_hidden_size")),
        num_attention_heads=attention_heads,
        num_query_groups=int(pick("num_query_groups", attention_heads)),
        head_dim=int(pick("kv_channels", hidden_size // attention_heads)),
        seq_length=int(pick("seq_length")),
        micro_batch_size=int(pick("micro_batch_size")),
        vocab_size=int(pick_any(("padded_vocab_size", "vocab_size"))),
        dtype=dtype,
        normalization=str(pick("normalization", "LayerNorm")),
        activation=str(pick("activation", "silu" if use_swiglu else "gelu")).lower(),
        use_swiglu=use_swiglu,
        qk_layernorm=bool(pick("qk_layernorm", False)),
        tied_embeddings=not bool(pick("untie_embeddings_and_output_weights", False)),
        attention_backend=str(pick("attention_backend", "unfused")),
        attention_softmax_in_fp32=bool(pick("attention_softmax_in_fp32", False)),
        use_rotary_position_embeddings=bool(pick("use_rotary_position_embeddings", False)),
        linear_bias=not bool(pick("disable_bias_linear", False)),
        num_experts=pick("num_experts", None),
    )


__all__ = [
    "TransformerModelSpec",
    "model_spec_from_flagscale",
    "model_spec_from_megatron",
]
