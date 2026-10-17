"""Versioned JSON artifacts for logical model graphs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from simulator.dataset import graph_from_dict, graph_to_dict
from simulator.schema import OpGraph

from .spec import TransformerModelSpec
from .validation import ModelGraphSummary, validate_model_graph


MODEL_GRAPH_ARTIFACT_VERSION = "1.0"


def model_graph_artifact(
    graph: OpGraph,
    spec: TransformerModelSpec,
) -> dict[str, Any]:
    summary = validate_model_graph(graph, spec)
    return {
        "format": "fusion-placement-search.model-graph",
        "version": MODEL_GRAPH_ARTIFACT_VERSION,
        "model_spec": asdict(spec),
        "summary": asdict(summary),
        "graph": graph_to_dict(graph),
    }


def save_model_graph_artifact(
    path: str | Path,
    graph: OpGraph,
    spec: TransformerModelSpec,
) -> ModelGraphSummary:
    artifact = model_graph_artifact(graph, spec)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    return ModelGraphSummary(**artifact["summary"])


def load_model_graph_artifact(
    path: str | Path,
) -> tuple[OpGraph, TransformerModelSpec, ModelGraphSummary]:
    data: Mapping[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format") != "fusion-placement-search.model-graph":
        raise ValueError(f"unsupported artifact format: {data.get('format')}")
    if data.get("version") != MODEL_GRAPH_ARTIFACT_VERSION:
        raise ValueError(f"unsupported artifact version: {data.get('version')}")

    spec_data = dict(data["model_spec"])
    spec_data["moe_layer_pattern"] = tuple(spec_data.get("moe_layer_pattern") or ())
    spec = TransformerModelSpec(**spec_data)
    graph = graph_from_dict(dict(data["graph"]))
    actual_summary = validate_model_graph(graph, spec)
    stored_summary = ModelGraphSummary(**dict(data["summary"]))
    if actual_summary != stored_summary:
        raise ValueError("artifact summary does not match graph contents")
    return graph, spec, actual_summary


__all__ = [
    "MODEL_GRAPH_ARTIFACT_VERSION",
    "load_model_graph_artifact",
    "model_graph_artifact",
    "save_model_graph_artifact",
]
