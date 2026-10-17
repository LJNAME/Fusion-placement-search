"""Validation and artifact round-trip tests for logical model graphs."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from simulator.dataset import graph_from_dict, graph_to_dict
from simulator.model_graphs import (
    TransformerModelSpec,
    build_model_graph,
    load_model_graph_artifact,
    save_model_graph_artifact,
    validate_model_graph,
)


def _qwen3_spec() -> TransformerModelSpec:
    return TransformerModelSpec(
        model_type="qwen3",
        num_layers=28,
        hidden_size=1024,
        ffn_hidden_size=3072,
        num_attention_heads=16,
        num_query_groups=8,
        head_dim=128,
        seq_length=256,
        micro_batch_size=1,
        vocab_size=151936,
        dtype="bf16",
        normalization="RMSNorm",
        activation="silu",
        use_swiglu=True,
        qk_layernorm=True,
        tied_embeddings=True,
        attention_backend="unfused",
        attention_softmax_in_fp32=True,
        use_rotary_position_embeddings=True,
        linear_bias=False,
    )


class TestModelGraphValidation(unittest.TestCase):
    def test_qwen3_summary(self):
        summary = validate_model_graph(build_model_graph(_qwen3_spec()), _qwen3_spec())
        self.assertEqual(summary.node_count, 591)
        self.assertEqual(summary.edge_count, 730)
        self.assertEqual(summary.parameter_elements, 596_049_920)
        self.assertEqual(summary.layer_count, 28)
        self.assertGreater(summary.forward_flops, 0)
        self.assertGreater(summary.backward_flops, summary.forward_flops)

    def test_invalid_edge_bytes_are_rejected(self):
        spec = _qwen3_spec()
        graph = build_model_graph(spec)
        graph.edges[0].bytes = 1.0
        with self.assertRaisesRegex(ValueError, "bytes mismatch"):
            validate_model_graph(graph, spec)

    def test_missing_attention_dependency_is_rejected(self):
        spec = _qwen3_spec()
        graph = build_model_graph(spec)
        graph.edges = [
            edge
            for edge in graph.edges
            if not (
                edge.src == "L000.self_attention.softmax"
                and edge.dst == "L000.self_attention.av_matmul"
            )
        ]
        with self.assertRaisesRegex(ValueError, "graph edges"):
            validate_model_graph(graph, spec)


class TestModelGraphArtifact(unittest.TestCase):
    def test_graph_metadata_survives_generic_round_trip(self):
        graph = build_model_graph(_qwen3_spec())
        restored = graph_from_dict(graph_to_dict(graph))
        self.assertEqual(restored.extra, graph.extra)
        self.assertEqual(restored.nodes["lm_head"].extra, graph.nodes["lm_head"].extra)

    def test_versioned_artifact_round_trip(self):
        spec = _qwen3_spec()
        graph = build_model_graph(spec)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "qwen3.json"
            saved_summary = save_model_graph_artifact(path, graph, spec)
            restored, restored_spec, loaded_summary = load_model_graph_artifact(path)
            raw = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(raw["format"], "fusion-placement-search.model-graph")
        self.assertEqual(raw["version"], "1.0")
        self.assertEqual(restored_spec, spec)
        self.assertEqual(loaded_summary, saved_summary)
        self.assertEqual(restored.extra, graph.extra)
        self.assertEqual(len(restored.nodes), 591)
        self.assertEqual(len(restored.edges), 730)


if __name__ == "__main__":
    unittest.main()
