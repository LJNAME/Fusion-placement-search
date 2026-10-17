"""Structural validation and summaries for logical Transformer graphs."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from simulator.schema import OpGraph

from .spec import TransformerModelSpec


_DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4}


@dataclass(frozen=True)
class ModelGraphSummary:
    node_count: int
    edge_count: int
    parameter_elements: int
    forward_flops: float
    backward_flops: float
    saved_activation_bytes: float
    layer_count: int


def expected_dense_layer_node_count(spec: TransformerModelSpec) -> int:
    count = 17
    if spec.qk_layernorm:
        count += 2
    if spec.use_rotary_position_embeddings:
        count += 2
    return count


def _expected_layer_node_ids(spec: TransformerModelSpec, layer_id: int) -> set[str]:
    prefix = f"L{layer_id:03d}"
    local_names = {
        "input",
        "input_norm",
        "self_attention.q_proj",
        "self_attention.k_proj",
        "self_attention.v_proj",
        "self_attention.qk_matmul",
        "self_attention.softmax",
        "self_attention.av_matmul",
        "self_attention.out_proj",
        "residual_add_1",
        "pre_mlp_norm",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.silu",
        "mlp.gated_mul",
        "mlp.down_proj",
        "residual_add_2",
    }
    if spec.qk_layernorm:
        local_names.update(
            {"self_attention.q_norm", "self_attention.k_norm"}
        )
    if spec.use_rotary_position_embeddings:
        local_names.update(
            {"self_attention.q_rope", "self_attention.k_rope"}
        )
    return {f"{prefix}.{name}" for name in local_names}


def _expected_layer_edges(
    spec: TransformerModelSpec,
    layer_id: int,
) -> set[tuple[str, str]]:
    prefix = f"L{layer_id:03d}"

    def node(local_name: str) -> str:
        return f"{prefix}.{local_name}"

    edges = {
        (node("input"), node("input_norm")),
        (node("input_norm"), node("self_attention.q_proj")),
        (node("input_norm"), node("self_attention.k_proj")),
        (node("input_norm"), node("self_attention.v_proj")),
        (node("self_attention.qk_matmul"), node("self_attention.softmax")),
        (node("self_attention.softmax"), node("self_attention.av_matmul")),
        (node("self_attention.v_proj"), node("self_attention.av_matmul")),
        (node("self_attention.av_matmul"), node("self_attention.out_proj")),
        (node("self_attention.out_proj"), node("residual_add_1")),
        (node("input"), node("residual_add_1")),
        (node("residual_add_1"), node("pre_mlp_norm")),
        (node("pre_mlp_norm"), node("mlp.gate_proj")),
        (node("pre_mlp_norm"), node("mlp.up_proj")),
        (node("mlp.gate_proj"), node("mlp.silu")),
        (node("mlp.silu"), node("mlp.gated_mul")),
        (node("mlp.up_proj"), node("mlp.gated_mul")),
        (node("mlp.gated_mul"), node("mlp.down_proj")),
        (node("mlp.down_proj"), node("residual_add_2")),
        (node("residual_add_1"), node("residual_add_2")),
    }

    q_source = node("self_attention.q_proj")
    k_source = node("self_attention.k_proj")
    if spec.qk_layernorm:
        q_norm = node("self_attention.q_norm")
        k_norm = node("self_attention.k_norm")
        edges.update({(q_source, q_norm), (k_source, k_norm)})
        q_source, k_source = q_norm, k_norm
    if spec.use_rotary_position_embeddings:
        q_rope = node("self_attention.q_rope")
        k_rope = node("self_attention.k_rope")
        edges.update({(q_source, q_rope), (k_source, k_rope)})
        q_source, k_source = q_rope, k_rope
    edges.update(
        {
            (q_source, node("self_attention.qk_matmul")),
            (k_source, node("self_attention.qk_matmul")),
        }
    )
    return edges


def expected_parameter_elements(spec: TransformerModelSpec) -> int:
    """Return the dense decoder parameter count represented by the graph."""

    h = spec.hidden_size
    f = spec.ffn_hidden_size
    q = spec.q_projection_size
    kv = spec.kv_projection_size

    per_layer = 2 * h
    per_layer += h * q + 2 * h * kv + q * h
    per_layer += 3 * h * f
    if spec.qk_layernorm:
        per_layer += 2 * spec.head_dim
    if spec.linear_bias:
        per_layer += q + 2 * kv + h + 2 * f + h

    embedding_and_head = spec.vocab_size * h
    if not spec.tied_embeddings:
        embedding_and_head += spec.vocab_size * h
    return spec.num_layers * per_layer + embedding_and_head + h


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_model_graph(
    graph: OpGraph,
    spec: TransformerModelSpec,
) -> ModelGraphSummary:
    """Validate the first-version dense decoder graph and return its summary."""

    order = graph.topological_order()
    _require(len(order) == len(graph.nodes), "topological order does not cover all nodes")
    _require(graph.extra.get("kind") == "logical", "expected a full logical model graph")
    _require(
        int(graph.extra.get("num_layers", -1)) == spec.num_layers,
        "graph num_layers does not match model spec",
    )
    _require(graph.extra.get("input_node") == "embedding", "invalid graph input node")
    _require(graph.extra.get("output_node") == "lm_head", "invalid graph output node")
    _require("embedding" in graph.nodes, "embedding node is missing")
    _require("final_norm" in graph.nodes, "final_norm node is missing")
    _require("lm_head" in graph.nodes, "lm_head node is missing")

    expected_nodes = spec.num_layers * expected_dense_layer_node_count(spec) + 3
    _require(
        len(graph.nodes) == expected_nodes,
        f"node count mismatch: got {len(graph.nodes)}, expected {expected_nodes}",
    )
    expected_node_ids = {"embedding", "final_norm", "lm_head"}
    for layer_id in range(spec.num_layers):
        expected_node_ids.update(_expected_layer_node_ids(spec, layer_id))
    _require(
        set(graph.nodes) == expected_node_ids,
        "graph node identifiers do not match the model specification",
    )

    expected_edges = {("final_norm", "lm_head")}
    for layer_id in range(spec.num_layers):
        expected_edges.update(_expected_layer_edges(spec, layer_id))
        previous = "embedding" if layer_id == 0 else f"L{layer_id - 1:03d}.residual_add_2"
        expected_edges.add((previous, f"L{layer_id:03d}.input"))
    expected_edges.add((f"L{spec.num_layers - 1:03d}.residual_add_2", "final_norm"))
    actual_edges = [(edge.src, edge.dst) for edge in graph.edges]
    _require(
        len(actual_edges) == len(expected_edges) and set(actual_edges) == expected_edges,
        "graph edges do not match the model specification",
    )

    required_node_metadata = (
        "layer",
        "local_name",
        "module_path",
        "output_shape",
        "dtype",
        "phase",
        "fwd_flops",
        "bwd_flops",
        "saved_activation_bytes",
        "recompute_eligible",
        "collective",
    )
    for node_id, node in graph.nodes.items():
        for key in required_node_metadata:
            _require(key in node.extra, f"node {node_id} is missing metadata field {key}")
        shape = tuple(int(dim) for dim in node.extra["output_shape"])
        _require(shape and all(dim > 0 for dim in shape), f"node {node_id} has invalid shape")
        dtype = str(node.extra["dtype"])
        _require(dtype in _DTYPE_BYTES, f"node {node_id} has unsupported dtype {dtype}")
        expected_bytes = prod(shape) * _DTYPE_BYTES[dtype]
        _require(
            int(node.out_bytes) == expected_bytes,
            f"node {node_id} out_bytes mismatch: got {node.out_bytes}, expected {expected_bytes}",
        )

    for edge in graph.edges:
        for key in ("tensor_id", "shape", "dtype"):
            _require(key in edge.extra, f"edge {edge.src}->{edge.dst} is missing {key}")
        _require(
            edge.extra["tensor_id"] == f"{edge.src}:output",
            f"edge {edge.src}->{edge.dst} has an invalid tensor_id",
        )
        shape = tuple(int(dim) for dim in edge.extra["shape"])
        dtype = str(edge.extra["dtype"])
        _require(dtype in _DTYPE_BYTES, f"edge {edge.src}->{edge.dst} has invalid dtype")
        expected_bytes = prod(shape) * _DTYPE_BYTES[dtype]
        _require(
            int(edge.bytes or 0) == expected_bytes,
            f"edge {edge.src}->{edge.dst} bytes mismatch: got {edge.bytes}, expected {expected_bytes}",
        )
        source_shape = tuple(int(dim) for dim in graph.nodes[edge.src].extra["output_shape"])
        _require(
            shape == source_shape,
            f"edge {edge.src}->{edge.dst} shape {shape} does not match source {source_shape}",
        )

    for layer_id in range(spec.num_layers):
        prefix = f"L{layer_id:03d}"
        inp = f"{prefix}.input"
        add1 = f"{prefix}.residual_add_1"
        add2 = f"{prefix}.residual_add_2"
        out_proj = f"{prefix}.self_attention.out_proj"
        down_proj = f"{prefix}.mlp.down_proj"
        _require(
            set(graph.predecessors(add1)) == {inp, out_proj},
            f"layer {layer_id} has an invalid first residual",
        )
        _require(
            set(graph.predecessors(add2)) == {add1, down_proj},
            f"layer {layer_id} has an invalid second residual",
        )
        previous = "embedding" if layer_id == 0 else f"L{layer_id - 1:03d}.residual_add_2"
        _require(
            previous in graph.predecessors(inp),
            f"layer {layer_id} is not connected to {previous}",
        )

    _require(
        f"L{spec.num_layers - 1:03d}.residual_add_2" in graph.predecessors("final_norm"),
        "last decoder layer is not connected to final_norm",
    )
    _require("final_norm" in graph.predecessors("lm_head"), "lm_head input is invalid")

    parameter_elements = int(
        sum(float(node.param_bytes) for node in graph.nodes.values()) / spec.dtype_bytes
    )
    expected_parameters = expected_parameter_elements(spec)
    _require(
        parameter_elements == expected_parameters,
        f"parameter count mismatch: got {parameter_elements}, expected {expected_parameters}",
    )

    return ModelGraphSummary(
        node_count=len(graph.nodes),
        edge_count=len(graph.edges),
        parameter_elements=parameter_elements,
        forward_flops=sum(float(node.extra["fwd_flops"]) for node in graph.nodes.values()),
        backward_flops=sum(float(node.extra["bwd_flops"]) for node in graph.nodes.values()),
        saved_activation_bytes=sum(
            float(node.extra["saved_activation_bytes"]) for node in graph.nodes.values()
        ),
        layer_count=spec.num_layers,
    )


__all__ = [
    "ModelGraphSummary",
    "expected_dense_layer_node_count",
    "expected_parameter_elements",
    "validate_model_graph",
]
