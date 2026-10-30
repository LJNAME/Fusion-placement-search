"""E0 independent controls: coupling, identities, missing labels and artifacts."""

import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from cost_model.fusion_kernel_profile import (
    GroupKernelKeys, KernelLatency, KeyedKernelProfile, candidate_structure, inventory, key_id,
)
from cost_model.legal_search import LegalFusionSearchSpace
from cost_model.search_evaluation import build_evaluation_graph
from cost_model.srop_e0 import coupling_fixture, run_coupling_checks, run_mode
from cost_model.structural import StructuralCostModel
from simulator import default_fusion_registry, validate_fusion_plan
from simulator.dataset import graph_from_dict, graph_to_dict, plan_from_dict
from simulator.placement_artifacts import (
    graph_fingerprint, load_placed_op_graph, plan_fingerprint,
)


class TestGroupKernelKeys(unittest.TestCase):
    def setUp(self):
        self.graph, self.hardware, self.registry, self.profile = coupling_fixture()
        self.space = LegalFusionSearchSpace(self.graph, 2, require_executable=False)
        self.candidate = next(c for c in self.space.candidates if c.pattern_id == "silu_mul")
        self.group = candidate_structure(self.graph, self.candidate, 0)

    def key(self, graph=None, hardware=None, bindings=None):
        factory = GroupKernelKeys(graph or self.graph, hardware or self.hardware,
                                  bindings or self.profile.keys.bindings, self.profile.keys.runtime)
        key, gaps = factory.describe(self.group)
        self.assertFalse(gaps)
        return key_id(key)

    def test_kernel_identity_tracks_device_version_dtype_shape_and_layout(self):
        original = self.key()
        self.assertNotEqual(original, key_id(self.profile.keys.describe(replace(self.group, device=1))[0]))
        bindings = copy.deepcopy(self.profile.keys.bindings)
        bindings["silu_mul"]["version"] = "other-implementation"
        self.assertNotEqual(original, self.key(bindings=bindings))
        for field, replacement in (("dtype", "fp32"), ("output_shape", [8, 4]),
                                   ("layout", {"contiguity": "strided", "order": "column_major"})):
            changed = copy.deepcopy(self.graph)
            for node in changed.nodes.values():
                node.extra[field] = replacement
            self.assertNotEqual(original, self.key(graph=changed), field)

    def test_layer_renames_and_tensor_names_do_not_duplicate_measurements(self):
        original = self.key()
        data = graph_to_dict(self.graph)
        for node in data["nodes"]:
            node["id"] = "L007." + node["id"]
            node["extra"].update(layer=7, module_path="decoder.layers.7", local_name=node["id"])
        for edge in data["edges"]:
            edge["src"] = "L007." + edge["src"]
            edge["dst"] = "L007." + edge["dst"]
            edge["extra"]["tensor_id"] = edge["src"] + ":output"
        renamed = graph_from_dict(data)
        group = replace(self.group, node_ids=tuple("L007." + node for node in self.group.node_ids))
        factory = GroupKernelKeys(renamed, self.hardware, self.profile.keys.bindings,
                                  self.profile.keys.runtime)
        self.assertEqual(original, key_id(factory.describe(group)[0]))

    def test_real_transformer_layers_share_identical_fused_keys(self):
        graph = build_evaluation_graph("model")
        candidates = LegalFusionSearchSpace(graph, 2, require_executable=False).candidates
        fused = [c for c in candidates if c.pattern_id == "silu_mul"]
        factory = GroupKernelKeys(graph, self.hardware, self.profile.keys.bindings,
                                  self.profile.keys.runtime)
        self.assertEqual(len(fused), 28)
        self.assertEqual(len({key_id(factory.describe(candidate_structure(graph, c, 0))[0])
                              for c in fused}), 1)

    def test_constituent_measurements_do_not_cover_a_fused_kernel(self):
        fused_id = self.key()
        partial = {key: row for key, row in self.profile.latencies.items() if key != fused_id}
        profile = KeyedKernelProfile(self.profile.keys, partial, self.profile.base)
        with self.assertRaisesRegex(KeyError, "missing group Kernel Key"):
            profile.estimate_group(self.graph, self.group)
        audit = inventory(profile.keys, self.registry, latencies=partial)
        self.assertFalse(audit["complete_measured_kernel_profile"])
        self.assertEqual(audit["measured_key_count"], 0)
        self.assertEqual(audit["latency_kinds"]["missing"], 1)

    def test_measured_entries_require_identity_metadata(self):
        factory = GroupKernelKeys(self.graph, self.hardware)
        key, gaps = factory.describe(self.group)
        self.assertIn("implementation.version", gaps)
        entry = KernelLatency(.001, "measured", "test-provenance")
        profile = KeyedKernelProfile(factory, {key_id(key): entry}, self.profile.base)
        with self.assertRaisesRegex(ValueError, "incomplete measured Kernel identity"):
            profile.estimate_group(self.graph, self.group)
        audit = inventory(factory, self.registry, latencies={key_id(key): entry})
        self.assertFalse(audit["complete_measured_kernel_profile"])

    def test_fused_latency_changes_prediction_and_wrong_graph_is_rejected(self):
        row = run_mode(self.graph, self.profile, self.registry, "joint", method="exact")
        plan = plan_from_dict(row["plan"])
        baseline = row["makespan_s"]
        durations = dict(self.profile.latencies)
        # The optimum uses the device-1 fused implementation, independently of the scorer.
        fused_id = key_id(self.profile.keys.describe(replace(self.group, device=1))[0])
        durations[fused_id] = KernelLatency(.101, "synthetic", "test-perturbation")
        changed = KeyedKernelProfile(self.profile.keys, durations, self.profile.base)
        self.assertAlmostEqual(StructuralCostModel().predict(self.graph, plan, changed).makespan,
                               baseline + .1)
        with self.assertRaisesRegex(ValueError, "different graph snapshot"):
            changed.estimate_group(copy.deepcopy(self.graph), self.group)


class TestE0Search(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_coupling_checks()

    def test_exact_spaces_and_analytic_witness(self):
        exact = self.report["exact"]
        self.assertEqual(exact["placement_only"]["stages"][0]["space_size"], 32)
        self.assertEqual(exact["fusion_only"]["stages"][0]["space_size"], 2)
        self.assertEqual(exact["joint"]["stages"][0]["space_size"], 48)
        self.assertAlmostEqual(exact["placement_only"]["makespan_s"], .011)
        self.assertAlmostEqual(exact["sequential"]["makespan_s"], .011)
        # Two 64-byte transfers on a 1e12-B/s link plus two 0.1-ms latencies.
        self.assertAlmostEqual(exact["joint"]["makespan_s"], .0042 + 128 / 1e12)
        self.assertEqual(exact["sequential"]["partition"], exact["fusion_only"]["partition"])
        self.assertNotEqual(exact["joint"]["partition"], exact["sequential"]["partition"])

    def test_mcts_is_seed_repeated_with_equal_total_iteration_budget(self):
        self.assertEqual(len(self.report["mcts"]), 16)
        for row in self.report["mcts"]:
            self.assertAlmostEqual(row["exact_gap_s"], 0.)
            self.assertEqual(sum(stage["iterations"] for stage in row["stages"]), 128)
            self.assertEqual(sum(stage["cost_model_evaluations"] for stage in row["stages"]),
                             row["cost_model_evaluations"])

    def test_fusion_only_optimum_reacts_to_fused_cost(self):
        graph, hardware, registry, profile = coupling_fixture()
        values = dict(profile.latencies)
        for candidate in LegalFusionSearchSpace(graph, 2, require_executable=False).candidates:
            if candidate.pattern_id == "silu_mul":
                key, _ = profile.keys.describe(candidate_structure(graph, candidate, 0))
                values[key_id(key)] = KernelLatency(.0001, "synthetic", "test-cheap-local-fusion")
        changed = KeyedKernelProfile(profile.keys, values, profile.base)
        row = run_mode(graph, changed, registry, "fusion_only", method="exact")
        self.assertTrue(any(len(group["node_ids"]) == 2 for group in row["plan"]["groups"]))

    def test_deployable_filter_excludes_synthetic_planned_pattern(self):
        graph, _, registry, profile = coupling_fixture()
        row = run_mode(graph, profile, registry, "joint", method="exact", require_executable=True)
        self.assertAlmostEqual(row["makespan_s"], .011)
        self.assertTrue(all(len(g["node_ids"]) == 1 for g in row["plan"]["groups"]))

    def test_artifact_roundtrip_preserves_original_graph_and_rejects_tampering(self):
        artifact = self.report["representative_output"]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "placed.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            restored = load_placed_op_graph(path)
            self.assertEqual(graph_fingerprint(restored.graph), artifact["graph_hash"])
            self.assertEqual(plan_fingerprint(restored.plan), artifact["candidate_id"])
            self.assertFalse(restored.extra["deployable"])
            validate_fusion_plan(restored.graph, restored.plan, 2)
            altered = copy.deepcopy(artifact)
            altered["fusion_groups"][0]["device_id"] = 99
            path.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_placed_op_graph(path)

    def test_output_checker_rejects_memory_overflow(self):
        graph, hardware, registry, profile = coupling_fixture()
        hardware.devices[0].max_mem = 1.
        hardware.devices[1].max_mem = 1.
        with self.assertRaisesRegex(ValueError, "no feasible plan"):
            run_mode(graph, profile, registry, "joint", method="exact")


if __name__ == "__main__":
    unittest.main()
