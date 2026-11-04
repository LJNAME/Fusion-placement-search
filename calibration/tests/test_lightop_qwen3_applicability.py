"""Negative checks for the CPU applicability audit, without a GPU runtime."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cost_model.search_evaluation import build_evaluation_graph
from simulator.schema import TensorEdge
from tools.audit_lightop_qwen3_applicability import (
    build, encoded, enumerate_matches, layout_state, main,
)


class LightopQwenApplicabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = build_evaluation_graph("model")

    def changed(self):
        return copy.deepcopy(self.graph)

    def layer_zero_pairs(self, graph):
        return [m for m in enumerate_matches(graph)["silu_mul"] if m["layer"] == 0]

    def test_gelu_is_not_a_silu_replacement(self):
        graph = self.changed()
        graph.nodes["L000.mlp.silu"].op_type = "gelu"
        self.assertEqual(self.layer_zero_pairs(graph), [])

    def test_internal_external_consumer_prevents_silu_group(self):
        graph = self.changed()
        graph.add_edge(TensorEdge(src="L000.mlp.silu", dst="L000.residual_add_2", bytes=1))
        self.assertEqual(self.layer_zero_pairs(graph), [])

    def test_cross_layer_pair_is_not_counted(self):
        graph = self.changed()
        graph.nodes["L000.mlp.gated_mul"].extra["layer"] = 1
        self.assertEqual(self.layer_zero_pairs(graph), [])

    def test_unknown_or_mismatched_shape_and_dtype_do_not_match(self):
        for field, value in (("output_shape", None), ("output_shape", (1, 256, 4096)),
                             ("dtype", None), ("dtype", "int8")):
            with self.subTest(field=field, value=value):
                graph = self.changed()
                graph.nodes["L000.mlp.up_proj"].extra[field] = value
                self.assertEqual(self.layer_zero_pairs(graph), [])

    def test_common_projection_module_does_not_prove_packed_storage(self):
        nodes = self.graph.nodes
        self.assertEqual(nodes["L000.mlp.gate_proj"].extra["implementation_group"],
                         nodes["L000.mlp.up_proj"].extra["implementation_group"])
        row = self.layer_zero_pairs(self.graph)[0]
        self.assertFalse(row["packed_storage_proven"])
        self.assertEqual(row["input_layout"]["gate"]["state"], "unknown")
        self.assertEqual(layout_state(nodes["L000.mlp.gated_mul"])["state"], "unknown")

    def test_rope_group_requires_both_normalization_edges(self):
        graph = self.changed()
        graph.edges = [e for e in graph.edges if (e.src, e.dst) !=
                       ("L000.self_attention.k_norm", "L000.self_attention.k_rope")]
        self.assertNotIn(0, [m["layer"] for m in enumerate_matches(graph)["qk_rms_rope"]])

    def test_rope_structure_does_not_grant_semantic_or_legality_proof(self):
        result = build()
        groups = result["node_matches.json"]["qk_rms_rope"]
        self.assertEqual(len(groups), 28)
        self.assertTrue(all(not g["semantic_equivalence_proven"] and
                            not g["fusion_group_legality_proven"] for g in groups))
        row = next(r for r in result["applicability.json"]["rows"] if r["capability_id"] == "rms_rotary_embedding_fuse")
        self.assertEqual(row["status"], "evidence_insufficient")
        self.assertFalse(row["direct_mapping_ready"])

    def test_unreviewed_graph_change_requires_new_audit(self):
        graph = self.changed()
        graph.nodes["L000.mlp.silu"].extra["dtype"] = "fp32"
        with patch("tools.audit_lightop_qwen3_applicability.build_evaluation_graph", return_value=graph):
            with self.assertRaisesRegex(ValueError, "graph changed"):
                build()

    def test_no_new_admission_and_deferred_jobs_remain_excluded(self):
        report = build()["applicability.json"]
        self.assertEqual(len(report["rows"]), 9)
        self.assertFalse(report["deferred_jobs_resumed"])
        self.assertEqual(report["new_accepted_profiles"], 0)
        self.assertTrue(all(not r["profile_reusable_for_search"] for r in report["rows"]))
        self.assertEqual(report["historical_ffn_cost_coverage"]["priceable_plans"], "0/48")
        bmm = next(r for r in report["rows"] if r["capability_id"] == "fused_bmm")
        self.assertEqual(bmm["status"], "not_applicable")
        self.assertIn("INT8", bmm["verified_case"])

    def test_split_packing_regression_is_preserved_without_profile_promotion(self):
        row = build()["applicability.json"]["existing_width_3072_evidence"]
        self.assertTrue(row["numeric_checks_passed"])
        self.assertGreater(row["lightop_split_with_pack_us"], row["native_split_us"])
        self.assertAlmostEqual(row["relative_duration_change_percent"],
                               (row["lightop_split_with_pack_us"] / row["native_split_us"] - 1) * 100)
        self.assertFalse(row["formal_profile_reuse"])

    def test_replay_is_byte_identical_including_saved_artifacts(self):
        self.assertEqual(encoded(build()), encoded(build()))
        with tempfile.TemporaryDirectory() as temp:
            args = ["audit", "--output", temp]
            with patch("sys.argv", args):
                main()
                first = {p.name: p.read_bytes() for p in Path(temp).iterdir()}
                main()
                second = {p.name: p.read_bytes() for p in Path(temp).iterdir()}
            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
