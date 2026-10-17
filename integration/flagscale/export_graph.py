"""Export a FlagScale model configuration as a logical model-graph artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from simulator.model_graphs import save_model_graph_artifact

from .graph_builder import build_transformer_graph
from .model_spec import model_spec_from_flagscale


def load_flagscale_config(
    model_config: str | Path,
    data_config: str | Path | None = None,
) -> Any:
    """Load the model YAML and optionally merge the data YAML below it."""

    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise RuntimeError(
            "OmegaConf is required to load FlagScale YAML; install requirements.txt"
        ) from exc

    model = OmegaConf.load(Path(model_config))
    if data_config is None:
        return model
    data = OmegaConf.load(Path(data_config))
    return OmegaConf.merge(data, model)


def export_flagscale_graph(
    model_config: str | Path,
    output: str | Path,
    data_config: str | Path | None = None,
    model_type: str = "qwen3",
):
    config = load_flagscale_config(model_config, data_config)
    spec = model_spec_from_flagscale(config, model_type=model_type)
    graph = build_transformer_graph(spec)
    summary = save_model_graph_artifact(output, graph, spec)
    return spec, summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and validate a framework-neutral logical Transformer graph."
    )
    parser.add_argument("--model-config", required=True, help="FlagScale model YAML")
    parser.add_argument(
        "--data-config",
        help="FlagScale data YAML containing tokenizer vocabulary settings",
    )
    parser.add_argument("--out", required=True, help="Output JSON artifact")
    parser.add_argument("--model-type", default="qwen3")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    spec, summary = export_flagscale_graph(
        model_config=args.model_config,
        data_config=args.data_config,
        output=args.out,
        model_type=args.model_type,
    )
    print(f"model: {spec.model_type}")
    print(f"layers: {spec.num_layers}")
    print(f"graph nodes: {summary.node_count}")
    print(f"graph edges: {summary.edge_count}")
    print(f"parameters: {summary.parameter_elements}")
    print(f"artifact: {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["export_flagscale_graph", "load_flagscale_config", "main"]
