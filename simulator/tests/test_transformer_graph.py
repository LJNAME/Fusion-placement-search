"""Logical Qwen3 graph regression tests; no torch/GPU is required."""

from __future__ import annotations

import unittest

from simulator.model_graphs import (
    TransformerModelSpec,
    build_decoder_layer,
    build_model_graph,
    build_pipeline_stages,
    build_stage_graph,
)


def _qwen3_spec(**overrides) -> TransformerModelSpec:
    values = {
        "model_type": "qwen3",
        "num_layers": 28,
        "hidden_size": 1024,
        "ffn_hidden_size": 3072,
        "num_attention_heads": 16,
        "num_query_groups": 8,
        "head_dim": 128,
        "seq_length": 256,
        "micro_batch_size": 1,
        "vocab_size": 151936,
        "dtype": "bf16",
        "normalization": "RMSNorm",
        "activation": "silu",
        "use_swiglu": True,
        "qk_layernorm": True,
        "tied_embeddings": True,
        "attention_backend": "unfused",
        "attention_softmax_in_fp32": True,
        "use_rotary_position_embeddings": True,
        "linear_bias": False,
    }
    values.update(overrides)
    return TransformerModelSpec(**values)


class TestQwen3DenseLayer(unittest.TestCase):
    def test_shapes_branches_and_residuals(self):
        graph = build_decoder_layer(_qwen3_spec(), layer_id=0)
        self.assertEqual(len(graph.topological_order()), len(graph.nodes))
        self.assertEqual(
            graph.nodes["L000.self_attention.q_proj"].extra["output_shape"],
            (1, 256, 16, 128),
        )
        self.assertEqual(
            graph.nodes["L000.self_attention.k_proj"].extra["output_shape"],
            (1, 256, 8, 128),
        )
        self.assertEqual(
            graph.nodes["L000.self_attention.v_proj"].extra["output_shape"],
            (1, 256, 8, 128),
        )
        self.assertEqual(
            graph.nodes["L000.mlp.gate_proj"].extra["output_shape"],
            (1, 256, 3072),
        )
        self.assertEqual(
            set(graph.predecessors("L000.residual_add_1")),
            {"L000.input", "L000.self_attention.out_proj"},
        )
        self.assertEqual(
            set(graph.predecessors("L000.residual_add_2")),
            {"L000.residual_add_1", "L000.mlp.down_proj"},
        )
        self.assertIn("L000.self_attention.q_norm", graph.nodes)
        self.assertIn("L000.self_attention.k_norm", graph.nodes)
        self.assertEqual(
            graph.nodes["L000.self_attention.q_proj"].extra["implementation_group"],
            graph.nodes["L000.self_attention.k_proj"].extra["implementation_group"],
        )
        input_norm_edges = graph.out_edges("L000.input_norm")
        self.assertEqual(len({edge.extra["tensor_id"] for edge in input_norm_edges}), 1)
        self.assertEqual(len(graph.nodes), 21)
        qk = graph.nodes["L000.self_attention.qk_matmul"]
        softmax = graph.nodes["L000.self_attention.softmax"]
        self.assertEqual(qk.extra["dtype"], "bf16")
        self.assertEqual(qk.extra["compute_dtype"], "bf16")
        self.assertEqual(softmax.extra["dtype"], "bf16")
        self.assertEqual(softmax.extra["compute_dtype"], "fp32")
        self.assertEqual(graph.out_edges(qk.id)[0].extra["dtype"], "bf16")
        self.assertEqual(graph.out_edges(softmax.id)[0].extra["dtype"], "bf16")
        for edge in graph.edges:
            self.assertIn("tensor_id", edge.extra)
            self.assertIn("shape", edge.extra)
            self.assertIn("dtype", edge.extra)

    def test_qk_norm_switch_changes_structure(self):
        graph = build_decoder_layer(_qwen3_spec(qk_layernorm=False), layer_id=3)
        self.assertNotIn("L003.self_attention.q_norm", graph.nodes)
        self.assertNotIn("L003.self_attention.k_norm", graph.nodes)
        self.assertEqual(
            graph.predecessors("L003.self_attention.q_rope"),
            ["L003.self_attention.q_proj"],
        )
        self.assertEqual(len(graph.nodes), 19)

    def test_layer_parameter_count(self):
        graph = build_decoder_layer(_qwen3_spec(), layer_id=0)
        parameter_elements = int(sum(node.param_bytes for node in graph.nodes.values()) / 2)
        self.assertEqual(parameter_elements, 15_730_944)

    def test_invalid_gqa_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            _qwen3_spec(num_query_groups=3)

    def test_fused_attention_backend_is_not_silently_expanded(self):
        with self.assertRaisesRegex(NotImplementedError, "unfused"):
            build_decoder_layer(_qwen3_spec(attention_backend="flash"))


class TestQwen3FullModel(unittest.TestCase):
    def test_full_model_connectivity_and_parameter_count(self):
        graph = build_model_graph(_qwen3_spec())
        self.assertEqual(len(graph.nodes), 28 * 21 + 3)
        self.assertEqual(len(graph.topological_order()), len(graph.nodes))
        self.assertIn("L000.input", graph.successors("embedding"))
        self.assertIn("L001.input", graph.successors("L000.residual_add_2"))
        self.assertIn("final_norm", graph.successors("L027.residual_add_2"))
        self.assertIn("lm_head", graph.successors("final_norm"))
        self.assertEqual(graph.nodes["lm_head"].param_bytes, 0.0)
        parameter_elements = int(sum(node.param_bytes for node in graph.nodes.values()) / 2)
        self.assertEqual(parameter_elements, 596_049_920)

    def test_untied_lm_head_counts_separate_weight(self):
        graph = build_model_graph(_qwen3_spec(tied_embeddings=False))
        self.assertEqual(graph.nodes["lm_head"].param_bytes, 151936 * 1024 * 2)

    def test_pipeline_stage_slice_18_10(self):
        first = build_stage_graph(_qwen3_spec(), 0, 18)
        last = build_stage_graph(_qwen3_spec(), 18, 28)
        self.assertIn("embedding", first.nodes)
        self.assertNotIn("final_norm", first.nodes)
        self.assertNotIn("L018.input", first.nodes)
        self.assertNotIn("embedding", last.nodes)
        self.assertIn("L018.input", last.nodes)
        self.assertIn("final_norm", last.nodes)
        self.assertIn("lm_head", last.nodes)
        self.assertEqual(first.extra["output_node"], "L017.residual_add_2")
        self.assertEqual(last.extra["input_node"], "L018.input")
        self.assertEqual(len(first.topological_order()), len(first.nodes))
        self.assertEqual(len(last.topological_order()), len(last.nodes))

    def test_layer_split_builds_all_stages(self):
        stages = build_pipeline_stages(_qwen3_spec(), [7, 7, 7, 7])
        self.assertEqual(len(stages), 4)
        self.assertEqual(
            [(stage.extra["layer_start"], stage.extra["layer_end"]) for stage in stages],
            [(0, 7), (7, 14), (14, 21), (21, 28)],
        )
        self.assertEqual(
            [stage.extra["global_stage_id"] for stage in stages],
            [0, 1, 2, 3],
        )
        with self.assertRaisesRegex(ValueError, "num_layers"):
            build_pipeline_stages(_qwen3_spec(), [14, 13])


if __name__ == "__main__":
    unittest.main()
