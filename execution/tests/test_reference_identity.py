"""B0 identities and strict rejection; accepted labels below are unit fixtures."""
import copy
import json
import unittest
from unittest.mock import patch

import torch

from calibration.close_reference_identity import prepare_cpu_identity
from cost_model.fusion_kernel_profile import KernelLatency, KeyedKernelProfile, key_id
from cost_model.hardware_profile import AnalyticHardwareProfile
from cost_model.measurement_labels import MeasurementContractError, measurement_contract_id
from execution.reference_identity import ReferenceKernelKeys, _layout_gaps, tensor_observation, with_tensor_layouts


class TestReferenceIdentity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.keys, cls.groups, cls.report = prepare_cpu_identity()

    def setUp(self):
        context = torch.no_grad()
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        self.group = next(g for g in self.groups if g.node_ids[0].endswith("mlp.silu"))
        self.contract = measurement_contract_id({"executor": "unit_fixture_only", "submission_included": True,
            "allocation_included": False, "granularity": "graph_replay", "unit": "us/call",
            "environment": {"torch": str(torch.__version__), "hip": None, "triton": None, "device_model": "CPU"}})

    def changed_keys(self, node_id, **updates):
        graph = copy.deepcopy(self.keys.graph)
        graph.nodes[node_id].extra.update(updates)
        return ReferenceKernelKeys(graph, self.keys.hardware, self.keys.observation)

    def profile(self, keys, group, *, admission="accepted", contract=None):
        key, _ = keys.describe(group)
        entry = KernelLatency(1e-6, "measured", "unit-fixture-not-a-measurement", contract or self.contract,
                              {"median_us": 1.0}, admission)
        return KeyedKernelProfile(keys, {key_id(key): entry}, AnalyticHardwareProfile(keys.hardware),
                                  measurement_contract_id=self.contract)

    def test_five_real_cpu_keys_complete_without_admitting_profiles(self):
        self.assertEqual(len(self.report["records"]), 5)
        for group in self.groups:
            key, identifier = self.keys.require_complete(group)
            self.assertEqual(identifier, key_id(json.loads(json.dumps(key))))
        self.assertEqual(self.report["accepted_profiles_created"], 0)
        self.assertFalse(self.report["gpu_identity_closed"])

    def test_unfused_operators_select_three_distinct_actual_bindings(self):
        keys = [self.keys.require_complete(g)[0] for g in self.groups]
        self.assertEqual({key["pattern_id"] for key in keys}, {"unfused"})
        self.assertEqual(len({key["implementation"]["id"] for key in keys}), 3)
        self.assertTrue(all(key["implementation"]["version"] == str(torch.__version__) for key in keys))
        by_node = {g.node_ids[0]: self.keys.require_complete(g)[0] for g in self.groups}
        gate, up = by_node["L000.mlp.gate_proj"], by_node["L000.mlp.up_proj"]
        self.assertEqual(gate["implementation"], up["implementation"])
        self.assertEqual(gate["external_inputs"], up["external_inputs"])
        self.assertNotEqual(key_id(gate), key_id(up))  # Semantic slices remain distinct.

    def test_roundtrip_reconstructs_identical_keys(self):
        observation = json.loads(json.dumps(self.keys.observation))
        rebuilt = ReferenceKernelKeys(copy.deepcopy(self.keys.graph), copy.deepcopy(self.keys.hardware), observation)
        for group in self.groups:
            self.assertEqual(self.keys.require_complete(group), rebuilt.require_complete(group))

    def test_unknown_layout_stays_unknown_and_strict_label_rejects(self):
        keys = self.changed_keys(self.group.node_ids[0], layout="unknown", strides=None, layout_provenance=None)
        key, gaps = keys.describe(self.group)
        self.assertEqual(key["operators"][0]["output"]["layout"], "unknown")
        self.assertIn("tensor.layout_or_strides", gaps)
        with self.assertRaises(MeasurementContractError):
            keys.require_complete(self.group)
        with self.assertRaisesRegex(ValueError, "incomplete measured Kernel identity"):
            self.profile(keys, self.group).estimate_group(keys.graph, self.group)

    def test_missing_external_layout_provenance_rejected(self):
        source = self.keys.graph.in_edges(self.group.node_ids[0])[0].src
        keys = self.changed_keys(source, layout_provenance=None)
        with self.assertRaises(MeasurementContractError):
            keys.require_complete(self.group)

    def test_invalid_contiguous_strides_rejected(self):
        keys = self.changed_keys(self.group.node_ids[0], strides=[1, 1, 1])
        with self.assertRaisesRegex(MeasurementContractError, "layout_stride_mismatch"):
            keys.require_complete(self.group)

    def test_cross_device_layout_evidence_rejected(self):
        keys = self.changed_keys(self.group.node_ids[0], layout_provenance={"kind": "runtime_tensor", "device": "cuda:0", "observer": "fixture"})
        with self.assertRaises(MeasurementContractError):
            keys.require_complete(self.group)

    def test_gpu1_tensor_source_is_not_gpu0_layout_evidence(self):
        observed = copy.deepcopy(self.report["layout_observations"][self.group.node_ids[0]])
        observed["layout_provenance"]["device"] = "cuda:1"
        self.assertIn("tensor.layout_provenance", _layout_gaps(observed, {"execution_device": "cuda:0"}))

    def test_graph_declaration_needs_source_and_unknown_is_not_accepted(self):
        node = self.group.node_ids[0]
        keys = self.changed_keys(node, layout_provenance={"kind": "graph_declaration"})
        with self.assertRaises(MeasurementContractError):
            keys.require_complete(self.group)
        declared = self.changed_keys(node, layout_provenance={"kind": "graph_declaration", "source_id": "unit-fixture-artifact"})
        declared.require_complete(self.group)
        unknown = self.changed_keys(node, layout="unknown", layout_provenance={"kind": "graph_declaration", "source_id": "unit-fixture-artifact"})
        with self.assertRaises(MeasurementContractError):
            unknown.require_complete(self.group)

    def test_missing_or_changed_implementation_is_not_current_runtime(self):
        for field in ("id", "version", "source_sha256"):
            with self.subTest(field=field):
                keys = ReferenceKernelKeys(self.keys.graph, self.keys.hardware, self.keys.observation)
                keys.bindings["unfused:silu"].pop(field)
                with self.assertRaisesRegex(MeasurementContractError, "actual loaded runtime"):
                    keys.require_complete(self.group)

    def test_runtime_flag_change_is_rejected(self):
        with torch.enable_grad():
            with self.assertRaises(MeasurementContractError):
                self.keys.require_complete(self.group)

    def test_replaced_public_operator_cannot_reuse_original_identity(self):
        with patch("torch.matmul", lambda *args: None):
            with self.assertRaisesRegex(MeasurementContractError, "another implementation"):
                self.keys.require_complete(self.group)

    def test_cpu_runtime_cannot_be_relabelled_as_gpu(self):
        keys = ReferenceKernelKeys(self.keys.graph, self.keys.hardware, self.keys.observation)
        keys.runtime["accelerator"] = "HIP"
        with self.assertRaises(MeasurementContractError):
            keys.require_complete(self.group)

    def test_missing_weight_layout_rejected(self):
        group = next(g for g in self.groups if g.node_ids[0].endswith("gate_proj"))
        keys = self.changed_keys(group.node_ids[0], reference_parameters={})
        with self.assertRaises(MeasurementContractError):
            keys.require_complete(group)

    def test_shape_dtype_observation_mismatch_rejected(self):
        node = self.group.node_ids[0]
        for field, value in (("shape", [1]), ("dtype", "fp32")):
            observed = copy.deepcopy(self.report["layout_observations"][node])
            observed[field] = value
            with self.assertRaises(MeasurementContractError):
                with_tensor_layouts(self.keys.graph, {node: observed})

    def test_strided_tensor_is_observed_without_contiguous_copy(self):
        tensor = torch.empty((3, 4), dtype=torch.bfloat16).t()
        observed = tensor_observation(tensor)
        self.assertEqual(observed["layout"], "strided")
        self.assertEqual(observed["strides"], list(tensor.stride()))

    def test_complete_identity_passes_only_the_reviewed_fixture_contract(self):
        profile = self.profile(self.keys, self.group)
        cost = profile.estimate_group(self.keys.graph, self.group)
        self.assertEqual(cost.duration, 1e-6)
        self.assertEqual(cost.ood_reasons, ())
        with self.assertRaises(MeasurementContractError):
            self.profile(self.keys, self.group, admission="conditional_dev").estimate_group(self.keys.graph, self.group)
        with self.assertRaises(MeasurementContractError):
            self.profile(self.keys, self.group, contract="different-contract").estimate_group(self.keys.graph, self.group)

    def test_different_implementation_or_layout_cannot_hit_original_label(self):
        profile = self.profile(self.keys, self.group)
        original, _ = self.keys.require_complete(self.group)
        for change in ("implementation", "layout"):
            changed = copy.deepcopy(original)
            if change == "implementation":
                changed["implementation"]["id"] = "different-execution-path"
            else:
                changed["operators"][0]["output"]["strides"][0] += 1
            self.assertNotEqual(key_id(original), key_id(changed))
            with self.assertRaises(KeyError):
                profile.measurement_labels.lookup(key_id(changed), self.contract)


if __name__ == "__main__":
    unittest.main()
