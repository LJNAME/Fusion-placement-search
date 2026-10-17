"""FlagScale-facing facade for framework-neutral logical graph construction."""

from __future__ import annotations

from typing import Any

from simulator.model_graphs import (
    TransformerModelSpec,
    build_decoder_layer,
    build_model_graph,
    build_pipeline_stages,
    build_stage_graph,
)
from simulator.schema import OpGraph

from .model_spec import model_spec_from_flagscale


def _as_spec(config_or_spec: Any) -> TransformerModelSpec:
    if isinstance(config_or_spec, TransformerModelSpec):
        return config_or_spec
    return model_spec_from_flagscale(config_or_spec)


def build_transformer_layer(config_or_spec: Any, layer_id: int = 0) -> OpGraph:
    """Build one logical decoder layer from a FlagScale config or neutral spec."""

    return build_decoder_layer(_as_spec(config_or_spec), layer_id=layer_id)


def build_transformer_graph(
    config_or_spec: Any,
    layer_start: int | None = None,
    layer_end: int | None = None,
) -> OpGraph:
    """Build a full logical graph or one contiguous pipeline stage."""

    spec = _as_spec(config_or_spec)
    if layer_start is None and layer_end is None:
        return build_model_graph(spec)
    if layer_start is None or layer_end is None:
        raise ValueError("layer_start and layer_end must be provided together")
    return build_stage_graph(spec, int(layer_start), int(layer_end))


def build_transformer_stages(config_or_spec: Any, layer_split: list[int]) -> list[OpGraph]:
    """Build every logical PP stage from FlagScale's layer-split vector."""

    return build_pipeline_stages(_as_spec(config_or_spec), layer_split)


__all__ = [
    "build_transformer_graph",
    "build_transformer_layer",
    "build_transformer_stages",
]
