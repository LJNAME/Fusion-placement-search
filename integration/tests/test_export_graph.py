"""FlagScale YAML to logical graph artifact tests."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from integration.flagscale.export_graph import export_flagscale_graph
from simulator.model_graphs import load_model_graph_artifact


MODEL_YAML = """
system:
  disable_bias_linear: true
  qk_layernorm: true
  precision:
    bf16: true
    attention_softmax_in_fp32: true
model:
  num_layers: 2
  hidden_size: 128
  ffn_hidden_size: 384
  kv_channels: 32
  num_attention_heads: 4
  num_query_groups: 2
  attention_backend: unfused
  seq_length: 64
  micro_batch_size: 1
  normalization: RMSNorm
  swiglu: true
  use_rotary_position_embeddings: true
  untie_embeddings_and_output_weights: false
"""


DATA_YAML = """
data:
  tokenizer:
    vocab_size: 4096
"""


class TestExportGraph(unittest.TestCase):
    def test_two_yaml_files_are_merged_and_exported(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model.yaml"
            data_path = root / "data.yaml"
            output_path = root / "graph.json"
            model_path.write_text(MODEL_YAML, encoding="utf-8")
            data_path.write_text(DATA_YAML, encoding="utf-8")

            spec, summary = export_flagscale_graph(
                model_config=model_path,
                data_config=data_path,
                output=output_path,
            )
            graph, restored_spec, restored_summary = load_model_graph_artifact(output_path)

        self.assertEqual(spec.vocab_size, 4096)
        self.assertEqual(spec.num_layers, 2)
        self.assertEqual(summary.node_count, 45)
        self.assertEqual(graph.extra["kind"], "logical")
        self.assertEqual(restored_spec, spec)
        self.assertEqual(restored_summary, summary)


if __name__ == "__main__":
    unittest.main()
