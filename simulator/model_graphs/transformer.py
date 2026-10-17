"""Full logical decoder graph construction and pipeline-stage slicing."""

from __future__ import annotations

from typing import Sequence

from simulator.schema import OpGraph, OpNode, TensorEdge

from .dense_layer import build_decoder_layer
from .spec import TransformerModelSpec


def _numel(shape: Sequence[int]) -> int:
    value = 1
    for dim in shape:
        value *= int(dim)
    return value


def _copy_node(node: OpNode) -> OpNode:
    return OpNode(
        id=node.id,
        op_type=node.op_type,
        flops=node.flops,
        out_bytes=node.out_bytes,
        param_bytes=node.param_bytes,
        extra=dict(node.extra),
    )


def _copy_edge(edge: TensorEdge) -> TensorEdge:
    return TensorEdge(src=edge.src, dst=edge.dst, bytes=edge.bytes, extra=dict(edge.extra))


def _add_tensor_edge(
    graph: OpGraph,
    src: str,
    dst: str,
    shape: Sequence[int],
    dtype: str,
    dtype_bytes: int,
) -> None:
    graph.add_edge(
        TensorEdge(
            src=src,
            dst=dst,
            bytes=float(_numel(shape) * dtype_bytes),
            extra={
                "tensor_id": f"{src}:output",
                "shape": tuple(int(x) for x in shape),
                "dtype": dtype,
            },
        )
    )


def build_model_graph(spec: TransformerModelSpec) -> OpGraph:
    """Build embedding, all decoder layers, final norm, and LM head."""

    b = spec.micro_batch_size
    s = spec.seq_length
    h = spec.hidden_size
    v = spec.vocab_size
    hidden_shape = (b, s, h)
    graph = OpGraph(
        name=f"{spec.model_type}-{spec.num_layers}L-logical",
        extra={
            "kind": "logical",
            "model_type": spec.model_type,
            "num_layers": spec.num_layers,
            "dtype": spec.dtype,
            "attention_backend": spec.attention_backend,
            "layer_start": 0,
            "layer_end": spec.num_layers,
        },
    )

    embedding_bytes = v * h * spec.dtype_bytes
    graph.add_node(
        OpNode(
            id="embedding",
            op_type="embedding",
            flops=0.0,
            out_bytes=float(_numel(hidden_shape) * spec.dtype_bytes),
            param_bytes=float(embedding_bytes),
            extra={
                "layer": -1,
                "local_name": "embedding",
                "module_path": "embedding.word_embeddings",
                "output_shape": hidden_shape,
                "shape": {"n": _numel(hidden_shape)},
                "dtype": spec.dtype,
                "phase": "forward",
                "fwd_flops": 0.0,
                "bwd_flops": 0.0,
                "saved_activation_bytes": float(_numel(hidden_shape) * spec.dtype_bytes),
                "recompute_eligible": False,
                "collective": None,
            },
        )
    )

    previous = "embedding"
    for layer_id in range(spec.num_layers):
        layer = build_decoder_layer(spec, layer_id=layer_id)
        for node in layer.nodes.values():
            graph.add_node(_copy_node(node))
        for edge in layer.edges:
            graph.add_edge(_copy_edge(edge))
        current_input = layer.extra["input_node"]
        _add_tensor_edge(
            graph, previous, current_input, hidden_shape, spec.dtype, spec.dtype_bytes
        )
        previous = layer.extra["output_node"]

    final_norm_flops = 5.0 * b * s * h
    graph.add_node(
        OpNode(
            id="final_norm",
            op_type="rmsnorm",
            flops=float(final_norm_flops),
            out_bytes=float(_numel(hidden_shape) * spec.dtype_bytes),
            param_bytes=float(h * spec.dtype_bytes),
            extra={
                "layer": spec.num_layers,
                "local_name": "final_norm",
                "module_path": "decoder.final_layernorm",
                "output_shape": hidden_shape,
                "shape": {"n": _numel(hidden_shape)},
                "dtype": spec.dtype,
                "phase": "forward",
                "fwd_flops": float(final_norm_flops),
                "bwd_flops": float(2.0 * final_norm_flops),
                "saved_activation_bytes": float(_numel(hidden_shape) * spec.dtype_bytes),
                "recompute_eligible": True,
                "collective": None,
            },
        )
    )
    _add_tensor_edge(graph, previous, "final_norm", hidden_shape, spec.dtype, spec.dtype_bytes)

    logits_shape = (b, s, v)
    lm_head_flops = 2.0 * b * s * h * v
    graph.add_node(
        OpNode(
            id="lm_head",
            op_type="matmul",
            flops=float(lm_head_flops),
            out_bytes=float(_numel(logits_shape) * spec.dtype_bytes),
            param_bytes=0.0 if spec.tied_embeddings else float(v * h * spec.dtype_bytes),
            extra={
                "layer": spec.num_layers,
                "local_name": "lm_head",
                "module_path": "output_layer",
                "output_shape": logits_shape,
                "shape": {"m": b * s, "n": v, "k": h},
                "dtype": spec.dtype,
                "phase": "forward",
                "fwd_flops": float(lm_head_flops),
                "bwd_flops": float(2.0 * lm_head_flops),
                "saved_activation_bytes": float(_numel(logits_shape) * spec.dtype_bytes),
                "recompute_eligible": False,
                "collective": None,
                "sharding_rule": "column",
                "shared_param_with": "embedding" if spec.tied_embeddings else None,
            },
        )
    )
    _add_tensor_edge(graph, "final_norm", "lm_head", hidden_shape, spec.dtype, spec.dtype_bytes)

    graph.extra.update({"input_node": "embedding", "output_node": "lm_head"})
    graph.topological_order()
    return graph


def slice_pipeline_stage(
    graph: OpGraph,
    layer_start: int,
    layer_end: int,
) -> OpGraph:
    """Select a contiguous decoder-layer range from a full logical graph.

    The first stage retains the embedding; the last stage retains final norm
    and the LM head.  Intermediate stages begin at their first layer input and
    end at their last residual output, ready for later P2P lowering.
    """

    num_layers = int(graph.extra.get("num_layers", -1))
    if num_layers <= 0:
        raise ValueError("slice_pipeline_stage expects a full model graph")
    if not (0 <= layer_start < layer_end <= num_layers):
        raise ValueError(
            f"invalid layer range [{layer_start}, {layer_end}) for {num_layers} layers"
        )

    keep: set[str] = set()
    for node_id, node in graph.nodes.items():
        layer_id = int(node.extra.get("layer", -2))
        if layer_start <= layer_id < layer_end:
            keep.add(node_id)
    if layer_start == 0:
        keep.add("embedding")
    if layer_end == num_layers:
        keep.update({"final_norm", "lm_head"})

    stage = OpGraph(
        name=f"{graph.name}-stage-{layer_start}-{layer_end}",
        extra={
            **graph.extra,
            "kind": "logical_stage",
            "layer_start": layer_start,
            "layer_end": layer_end,
            "is_first_stage": layer_start == 0,
            "is_last_stage": layer_end == num_layers,
        },
    )
    for node_id in graph.topological_order():
        if node_id in keep:
            stage.add_node(_copy_node(graph.nodes[node_id]))
    for edge in graph.edges:
        if edge.src in keep and edge.dst in keep:
            stage.add_edge(_copy_edge(edge))

    stage.extra["input_node"] = (
        "embedding" if layer_start == 0 else f"L{layer_start:03d}.input"
    )
    stage.extra["output_node"] = (
        "lm_head" if layer_end == num_layers else f"L{layer_end - 1:03d}.residual_add_2"
    )
    stage.topological_order()
    return stage


def build_stage_graph(
    spec: TransformerModelSpec,
    layer_start: int,
    layer_end: int,
) -> OpGraph:
    """Convenience wrapper for building and slicing a model graph."""

    return slice_pipeline_stage(build_model_graph(spec), layer_start, layer_end)


def build_pipeline_stages(
    spec: TransformerModelSpec,
    layer_split: Sequence[int],
) -> list[OpGraph]:
    """Build all PP stage graphs described by a FlagScale layer split."""

    split = [int(count) for count in layer_split]
    if not split or any(count <= 0 for count in split):
        raise ValueError(f"layer_split must contain positive counts, got {split}")
    if sum(split) != spec.num_layers:
        raise ValueError(
            f"sum(layer_split)={sum(split)} does not match num_layers={spec.num_layers}"
        )

    full_graph = build_model_graph(spec)
    stages: list[OpGraph] = []
    layer_start = 0
    for stage_id, count in enumerate(split):
        layer_end = layer_start + count
        stage = slice_pipeline_stage(full_graph, layer_start, layer_end)
        stage.extra.update(
            {
                "global_stage_id": stage_id,
                "layer_split": tuple(split),
            }
        )
        stages.append(stage)
        layer_start = layer_end
    return stages


__all__ = [
    "build_model_graph",
    "build_pipeline_stages",
    "build_stage_graph",
    "slice_pipeline_stage",
]
