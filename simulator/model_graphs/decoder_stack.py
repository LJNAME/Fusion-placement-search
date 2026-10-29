"""Continuous decoder body, excluding embedding, final norm and LM head."""

from simulator.schema import OpGraph
from .dense_layer import build_decoder_layer
from .transformer import _add_tensor_edge, _copy_edge, _copy_node


def build_decoder_stack(spec):
    graph = OpGraph(name=f"{spec.model_type}-{spec.num_layers}L-decoder-body", extra={
        "kind": "logical-decoder-body", "model_type": spec.model_type,
        "num_layers": spec.num_layers, "layer_start": 0, "layer_end": spec.num_layers,
        "dtype": spec.dtype, "attention_backend": spec.attention_backend})
    previous = None
    for layer_id in range(spec.num_layers):
        layer = build_decoder_layer(spec, layer_id=layer_id)
        for node in layer.nodes.values():
            graph.add_node(_copy_node(node))
        for edge in layer.edges:
            graph.add_edge(_copy_edge(edge))
        if previous is not None:
            _add_tensor_edge(graph, previous, layer.extra["input_node"],
                (spec.micro_batch_size, spec.seq_length, spec.hidden_size), spec.dtype, spec.dtype_bytes)
        else:
            graph.extra["input_node"] = layer.extra["input_node"]
        previous = layer.extra["output_node"]
    graph.extra["output_node"] = previous
    graph.topological_order()
    return graph
