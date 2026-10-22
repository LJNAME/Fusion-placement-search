"""Logical Qwen3-style dense decoder layer graph."""

from __future__ import annotations

from typing import Sequence

from simulator.schema import OpGraph, OpNode, TensorEdge

from .spec import TransformerModelSpec


def _numel(shape: Sequence[int]) -> int:
    value = 1
    for dim in shape:
        value *= int(dim)
    return value


def _backward_flops(op_type: str, forward_flops: float) -> float:
    """Store an explicit first-order estimate without expanding a backward DAG."""

    if op_type in {"input", "embedding"}:
        return 0.0
    if op_type in {"matmul", "rmsnorm", "softmax", "silu", "mul", "rope"}:
        return 2.0 * forward_flops
    return forward_flops


class _LayerBuilder:
    def __init__(self, spec: TransformerModelSpec, layer_id: int):
        self.spec = spec
        self.layer_id = int(layer_id)
        self.prefix = f"L{self.layer_id:03d}"
        self.graph = OpGraph(
            name=f"{spec.model_type}-dense-layer-{self.layer_id}",
            extra={"kind": "logical", "layer_start": self.layer_id, "layer_end": self.layer_id + 1},
        )

    def node_id(self, local_name: str) -> str:
        return f"{self.prefix}.{local_name}"

    def add_node(
        self,
        local_name: str,
        op_type: str,
        output_shape: Sequence[int],
        flops: float = 0.0,
        param_elements: int = 0,
        module_path: str = "",
        matmul_shape: dict | None = None,
        output_dtype: str | None = None,
        recompute_eligible: bool = True,
        extra: dict | None = None,
    ) -> str:
        spec = self.spec
        nid = self.node_id(local_name)
        dtype = output_dtype or spec.dtype
        dtype_bytes = spec.softmax_dtype_bytes if dtype == "fp32" else spec.dtype_bytes
        out_bytes = _numel(output_shape) * dtype_bytes
        metadata = {
            "layer": self.layer_id,
            "local_name": local_name,
            "module_path": module_path,
            "output_shape": tuple(int(x) for x in output_shape),
            "dtype": dtype,
            "phase": "forward",
            "fwd_flops": float(flops),
            "bwd_flops": _backward_flops(op_type, float(flops)),
            "saved_activation_bytes": float(out_bytes),
            "recompute_eligible": bool(recompute_eligible),
            "collective": None,
        }
        if matmul_shape is not None:
            metadata["shape"] = dict(matmul_shape)
        else:
            metadata["shape"] = {"n": _numel(output_shape)}
        if extra:
            metadata.update(extra)
        self.graph.add_node(
            OpNode(
                id=nid,
                op_type=op_type,
                flops=float(flops),
                out_bytes=float(out_bytes),
                param_bytes=float(param_elements * spec.dtype_bytes),
                extra=metadata,
            )
        )
        return nid

    def add_edge(self, src: str, dst: str, shape: Sequence[int], dtype: str | None = None) -> None:
        tensor_dtype = dtype or self.spec.dtype
        dtype_bytes = self.spec.softmax_dtype_bytes if tensor_dtype == "fp32" else self.spec.dtype_bytes
        self.graph.add_edge(
            TensorEdge(
                src=src,
                dst=dst,
                bytes=float(_numel(shape) * dtype_bytes),
                extra={
                    "tensor_id": f"{src}:output",
                    "shape": tuple(int(x) for x in shape),
                    "dtype": tensor_dtype,
                },
            )
        )


def build_decoder_layer(spec: TransformerModelSpec, layer_id: int = 0) -> OpGraph:
    """Build one pre-norm dense decoder layer as a framework-neutral DAG."""

    if spec.num_experts is not None or any(spec.moe_layer_pattern):
        raise NotImplementedError("MoE layers are not supported by the dense-layer builder")
    if spec.normalization.lower() not in {"rmsnorm", "rms_norm"}:
        raise NotImplementedError(f"unsupported normalization: {spec.normalization}")
    if not spec.use_swiglu:
        raise NotImplementedError("the first graph builder version supports SwiGLU only")
    if spec.activation.lower() not in {"silu", "swish"}:
        raise NotImplementedError(f"unsupported SwiGLU activation: {spec.activation}")
    if spec.attention_backend != "unfused":
        raise NotImplementedError(
            f"the explicit attention DAG requires the unfused backend, got {spec.attention_backend}"
        )

    b = spec.micro_batch_size
    s = spec.seq_length
    h = spec.hidden_size
    a = spec.num_attention_heads
    qg = spec.num_query_groups
    d = spec.head_dim
    f = spec.ffn_hidden_size
    m = b * s
    hidden_shape = (b, s, h)
    q_shape = (b, s, a, d)
    kv_shape = (b, s, qg, d)
    score_shape = (b, a, s, s)
    ffn_shape = (b, s, f)
    bias = spec.linear_bias
    builder = _LayerBuilder(spec, layer_id)
    layer_path = f"decoder.layers.{layer_id}"

    inp = builder.add_node(
        "input", "input", hidden_shape, module_path=layer_path, recompute_eligible=False
    )
    input_norm = builder.add_node(
        "input_norm",
        "rmsnorm",
        hidden_shape,
        flops=5.0 * m * h,
        param_elements=h,
        module_path=f"{layer_path}.input_layernorm",
    )
    builder.add_edge(inp, input_norm, hidden_shape)

    q_proj = builder.add_node(
        "self_attention.q_proj",
        "matmul",
        q_shape,
        flops=2.0 * m * h * spec.q_projection_size,
        param_elements=h * spec.q_projection_size + (spec.q_projection_size if bias else 0),
        module_path=f"{layer_path}.self_attention.linear_qkv.q",
        matmul_shape={"m": m, "n": spec.q_projection_size, "k": h},
        extra={
            "sharding_rule": "column",
            "implementation_group": f"{layer_path}.self_attention.linear_qkv",
            "semantic_slice": "q",
        },
    )
    k_proj = builder.add_node(
        "self_attention.k_proj",
        "matmul",
        kv_shape,
        flops=2.0 * m * h * spec.kv_projection_size,
        param_elements=h * spec.kv_projection_size + (spec.kv_projection_size if bias else 0),
        module_path=f"{layer_path}.self_attention.linear_qkv.k",
        matmul_shape={"m": m, "n": spec.kv_projection_size, "k": h},
        extra={
            "sharding_rule": "column",
            "implementation_group": f"{layer_path}.self_attention.linear_qkv",
            "semantic_slice": "k",
        },
    )
    v_proj = builder.add_node(
        "self_attention.v_proj",
        "matmul",
        kv_shape,
        flops=2.0 * m * h * spec.kv_projection_size,
        param_elements=h * spec.kv_projection_size + (spec.kv_projection_size if bias else 0),
        module_path=f"{layer_path}.self_attention.linear_qkv.v",
        matmul_shape={"m": m, "n": spec.kv_projection_size, "k": h},
        extra={
            "sharding_rule": "column",
            "implementation_group": f"{layer_path}.self_attention.linear_qkv",
            "semantic_slice": "v",
        },
    )
    for projection in (q_proj, k_proj, v_proj):
        builder.add_edge(input_norm, projection, hidden_shape)

    q_source = q_proj
    k_source = k_proj
    if spec.qk_layernorm:
        q_norm = builder.add_node(
            "self_attention.q_norm",
            "rmsnorm",
            q_shape,
            flops=5.0 * _numel(q_shape),
            param_elements=d,
            module_path=f"{layer_path}.self_attention.q_layernorm",
        )
        k_norm = builder.add_node(
            "self_attention.k_norm",
            "rmsnorm",
            kv_shape,
            flops=5.0 * _numel(kv_shape),
            param_elements=d,
            module_path=f"{layer_path}.self_attention.k_layernorm",
        )
        builder.add_edge(q_proj, q_norm, q_shape)
        builder.add_edge(k_proj, k_norm, kv_shape)
        q_source = q_norm
        k_source = k_norm

    if spec.use_rotary_position_embeddings:
        q_rope = builder.add_node(
            "self_attention.q_rope",
            "rope",
            q_shape,
            flops=4.0 * _numel(q_shape),
            module_path=f"{layer_path}.self_attention.rotary_pos_emb.q",
        )
        k_rope = builder.add_node(
            "self_attention.k_rope",
            "rope",
            kv_shape,
            flops=4.0 * _numel(kv_shape),
            module_path=f"{layer_path}.self_attention.rotary_pos_emb.k",
        )
        builder.add_edge(q_source, q_rope, q_shape)
        builder.add_edge(k_source, k_rope, kv_shape)
        q_source = q_rope
        k_source = k_rope

    softmax_compute_dtype = "fp32" if spec.attention_softmax_in_fp32 else spec.dtype
    scores = builder.add_node(
        "self_attention.qk_matmul",
        "matmul",
        score_shape,
        flops=2.0 * b * a * s * s * d,
        module_path=f"{layer_path}.self_attention.core_attention.qk",
        matmul_shape={"m": b * a * s, "n": s, "k": d},
        output_dtype=spec.dtype,
        extra={"gqa_repeat_factor": a // qg, "compute_dtype": spec.dtype},
    )
    builder.add_edge(q_source, scores, q_shape)
    builder.add_edge(k_source, scores, kv_shape)

    softmax = builder.add_node(
        "self_attention.softmax",
        "softmax",
        score_shape,
        flops=5.0 * _numel(score_shape),
        module_path=f"{layer_path}.self_attention.core_attention.softmax",
        output_dtype=spec.dtype,
        extra={"compute_dtype": softmax_compute_dtype},
    )
    builder.add_edge(scores, softmax, score_shape, dtype=spec.dtype)

    context = builder.add_node(
        "self_attention.av_matmul",
        "matmul",
        q_shape,
        flops=2.0 * b * a * s * s * d,
        module_path=f"{layer_path}.self_attention.core_attention.av",
        matmul_shape={"m": b * a * s, "n": d, "k": s},
    )
    builder.add_edge(softmax, context, score_shape, dtype=spec.dtype)
    builder.add_edge(v_proj, context, kv_shape)

    out_proj = builder.add_node(
        "self_attention.out_proj",
        "matmul",
        hidden_shape,
        flops=2.0 * m * spec.q_projection_size * h,
        param_elements=spec.q_projection_size * h + (h if bias else 0),
        module_path=f"{layer_path}.self_attention.linear_proj",
        matmul_shape={"m": m, "n": h, "k": spec.q_projection_size},
        extra={"sharding_rule": "row"},
    )
    builder.add_edge(context, out_proj, q_shape)

    add1 = builder.add_node(
        "residual_add_1",
        "add",
        hidden_shape,
        flops=float(_numel(hidden_shape)),
        module_path=f"{layer_path}.self_attention_residual",
    )
    builder.add_edge(out_proj, add1, hidden_shape)
    builder.add_edge(inp, add1, hidden_shape)

    pre_mlp_norm = builder.add_node(
        "pre_mlp_norm",
        "rmsnorm",
        hidden_shape,
        flops=5.0 * m * h,
        param_elements=h,
        module_path=f"{layer_path}.pre_mlp_layernorm",
    )
    builder.add_edge(add1, pre_mlp_norm, hidden_shape)

    gate_proj = builder.add_node(
        "mlp.gate_proj",
        "matmul",
        ffn_shape,
        flops=2.0 * m * h * f,
        param_elements=h * f + (f if bias else 0),
        module_path=f"{layer_path}.mlp.linear_fc1.gate",
        matmul_shape={"m": m, "n": f, "k": h},
        extra={
            "sharding_rule": "column",
            "implementation_group": f"{layer_path}.mlp.linear_fc1",
            "semantic_slice": "gate",
        },
    )
    up_proj = builder.add_node(
        "mlp.up_proj",
        "matmul",
        ffn_shape,
        flops=2.0 * m * h * f,
        param_elements=h * f + (f if bias else 0),
        module_path=f"{layer_path}.mlp.linear_fc1.up",
        matmul_shape={"m": m, "n": f, "k": h},
        extra={
            "sharding_rule": "column",
            "implementation_group": f"{layer_path}.mlp.linear_fc1",
            "semantic_slice": "up",
        },
    )
    builder.add_edge(pre_mlp_norm, gate_proj, hidden_shape)
    builder.add_edge(pre_mlp_norm, up_proj, hidden_shape)

    silu = builder.add_node(
        "mlp.silu",
        "silu",
        ffn_shape,
        flops=4.0 * _numel(ffn_shape),
        module_path=f"{layer_path}.mlp.activation",
    )
    builder.add_edge(gate_proj, silu, ffn_shape)

    gated = builder.add_node(
        "mlp.gated_mul",
        "mul",
        ffn_shape,
        flops=float(_numel(ffn_shape)),
        module_path=f"{layer_path}.mlp.gated_linear_unit",
    )
    builder.add_edge(silu, gated, ffn_shape)
    builder.add_edge(up_proj, gated, ffn_shape)

    down_proj = builder.add_node(
        "mlp.down_proj",
        "matmul",
        hidden_shape,
        flops=2.0 * m * f * h,
        param_elements=f * h + (h if bias else 0),
        module_path=f"{layer_path}.mlp.linear_fc2",
        matmul_shape={"m": m, "n": h, "k": f},
        extra={"sharding_rule": "row"},
    )
    builder.add_edge(gated, down_proj, ffn_shape)

    add2 = builder.add_node(
        "residual_add_2",
        "add",
        hidden_shape,
        flops=float(_numel(hidden_shape)),
        module_path=f"{layer_path}.mlp_residual",
    )
    builder.add_edge(down_proj, add2, hidden_shape)
    builder.add_edge(add1, add2, hidden_shape)

    builder.graph.extra.update(
        {
            "input_node": inp,
            "output_node": add2,
            "dtype": spec.dtype,
            "model_type": spec.model_type,
        }
    )
    builder.graph.topological_order()
    return builder.graph


__all__ = ["build_decoder_layer"]
