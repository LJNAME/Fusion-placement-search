"""FlagScale model-spec adapter tests; no FlagScale installation is required."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from integration.flagscale.graph_builder import (
    build_transformer_graph,
    build_transformer_stages,
)
from integration.flagscale.model_spec import (
    model_spec_from_flagscale,
    model_spec_from_megatron,
)


def qwen3_flagscale_config() -> dict:
    return {
        "system": {
            "disable_bias_linear": True,
            "qk_layernorm": True,
            "precision": {
                "bf16": True,
                "attention_softmax_in_fp32": True,
            },
        },
        "model": {
            "num_layers": 28,
            "hidden_size": 1024,
            "ffn_hidden_size": 3072,
            "kv_channels": 128,
            "num_attention_heads": 16,
            "group_query_attention": True,
            "num_query_groups": 8,
            "attention_backend": "unfused",
            "seq_length": 256,
            "micro_batch_size": 1,
            "normalization": "RMSNorm",
            "swiglu": True,
            "use_rotary_position_embeddings": True,
            "untie_embeddings_and_output_weights": False,
        },
        "data": {"tokenizer": {"vocab_size": 151936}},
    }


class TestModelSpecAdapter(unittest.TestCase):
    def test_qwen3_hygon_config_fields(self):
        spec = model_spec_from_flagscale(qwen3_flagscale_config())
        self.assertEqual(spec.model_type, "qwen3")
        self.assertEqual(spec.num_layers, 28)
        self.assertEqual(spec.hidden_size, 1024)
        self.assertEqual(spec.ffn_hidden_size, 3072)
        self.assertEqual(spec.num_attention_heads, 16)
        self.assertEqual(spec.num_query_groups, 8)
        self.assertEqual(spec.head_dim, 128)
        self.assertEqual(spec.seq_length, 256)
        self.assertEqual(spec.micro_batch_size, 1)
        self.assertEqual(spec.vocab_size, 151936)
        self.assertEqual(spec.dtype, "bf16")
        self.assertEqual(spec.dtype_bytes, 2)
        self.assertEqual(spec.softmax_dtype_bytes, 4)
        self.assertTrue(spec.qk_layernorm)
        self.assertTrue(spec.use_swiglu)
        self.assertTrue(spec.tied_embeddings)
        self.assertEqual(spec.attention_backend, "unfused")
        self.assertFalse(spec.linear_bias)

    def test_composed_train_section_is_accepted(self):
        spec = model_spec_from_flagscale({"train": qwen3_flagscale_config()})
        self.assertEqual(spec.num_layers, 28)
        self.assertEqual(spec.vocab_size, 151936)

    def test_missing_required_field_fails_early(self):
        config = qwen3_flagscale_config()
        del config["model"]["ffn_hidden_size"]
        with self.assertRaisesRegex(KeyError, "ffn_hidden_size"):
            model_spec_from_flagscale(config)

    def test_megatron_object_adapter(self):
        args = SimpleNamespace(
            num_layers=28,
            seq_length=256,
            micro_batch_size=1,
            padded_vocab_size=151936,
            untie_embeddings_and_output_weights=False,
            disable_bias_linear=True,
            use_rotary_position_embeddings=True,
        )
        transformer_config = SimpleNamespace(
            hidden_size=1024,
            ffn_hidden_size=3072,
            num_attention_heads=16,
            num_query_groups=8,
            kv_channels=128,
            params_dtype="torch.bfloat16",
            normalization="RMSNorm",
            gated_linear_unit=True,
            qk_layernorm=True,
            attention_softmax_in_fp32=True,
            attention_backend="unfused",
        )
        spec = model_spec_from_megatron(args, transformer_config)
        self.assertEqual(spec.dtype, "bf16")
        self.assertEqual(spec.q_projection_size, 2048)
        self.assertEqual(spec.kv_projection_size, 1024)
        self.assertFalse(spec.linear_bias)

    def test_graph_builder_facade_accepts_flagscale_config(self):
        config = qwen3_flagscale_config()
        first_layer_stage = build_transformer_graph(config, layer_start=0, layer_end=1)
        self.assertIn("embedding", first_layer_stage.nodes)
        self.assertIn("L000.residual_add_2", first_layer_stage.nodes)
        self.assertNotIn("final_norm", first_layer_stage.nodes)

        stages = build_transformer_stages(config, [14, 14])
        self.assertEqual(len(stages), 2)
        self.assertEqual(stages[0].extra["layer_end"], 14)
        self.assertEqual(stages[1].extra["layer_start"], 14)


if __name__ == "__main__":
    unittest.main()
