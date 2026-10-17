"""FlagScale adapters for context, model specification, and logical graphs."""

from .context import HeteroExecutionContext
from .graph_builder import (
    build_transformer_graph,
    build_transformer_layer,
    build_transformer_stages,
)
from .model_spec import (
    TransformerModelSpec,
    model_spec_from_flagscale,
    model_spec_from_megatron,
)

__all__ = [
    "HeteroExecutionContext",
    "TransformerModelSpec",
    "build_transformer_graph",
    "build_transformer_layer",
    "build_transformer_stages",
    "model_spec_from_flagscale",
    "model_spec_from_megatron",
]
