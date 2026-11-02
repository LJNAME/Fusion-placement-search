"""CPU B2 obligation coverage and fail-closed transfer binding tests."""
import copy
import unittest
from unittest.mock import patch

import torch

from cost_model.ffn_communication_b2 import B2_FILES, build, derive, load_b1, verify
from execution.ffn_transfer import BoundTransfer, allocate_carrier, declared_carrier, validate_carrier


class TestFfnCommunicationB2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = build()

    def test_full_b1_coverage_reproducible_and_sources_unchanged(self):
        result = verify(self.bundle)
        self.assertEqual(result["request_count"], 4)
        self.assertEqual(result["covered_cross_device_edge_occurrences"], 160)
        self.assertEqual((result["b1_total_plans"], result["plans_requiring_communication"], result["zero_transfer_plans"]), (48, 46, 2))
        self.assertTrue(result["B1_sources_unchanged"])
        self.assertTrue(result["repeat_byte_equivalent"])

    def test_source_destination_declarations_are_not_gpu_observations(self):
        for request in self.bundle[B2_FILES[1]]["requests"]:
            spec = request["source_carrier"]
            self.assertEqual(spec, request["destination_carrier"])
            self.assertEqual(spec["layout_provenance"]["kind"], "constructor_declaration")
            self.assertFalse(spec["producer_layout_observed"])
            self.assertIsNone(request["runtime_identity"]["source_uuid"])
            self.assertIsNone(request["runtime_identity"]["observed_route"])
            self.assertIsNone(request["measurement_contract_id"])
            self.assertFalse(request["measurement_ready"])
            self.assertEqual(request["admission"], "not_requested")

    def test_all_directions_and_payload_sizes_come_from_witnesses(self):
        requests = self.bundle[B2_FILES[1]]["requests"]
        self.assertEqual({(r["source_carrier"]["tensor_bytes"], r["source_gpu"], r["destination_gpu"]) for r in requests},
                         {(size, src, dst) for size in (524288, 1572864) for src, dst in ((0, 1), (1, 0))})
        self.assertTrue(all(r["witnesses"] for r in requests))

    def test_old_eager_profiles_not_relabelled_as_graph_measurements(self):
        identity = self.bundle[B2_FILES[0]]
        self.assertTrue(all(not row["compatible_with_selected_contract"] for row in identity["existing_paths"]))
        self.assertTrue(all(not row["new_contract_admission"] for row in identity["historical_link_tables"]))
        self.assertFalse(identity["gpu_measurement_allowed"])
        self.assertEqual(identity["accepted_link_profiles"], 0)

    def test_cpu_layout_constructor_and_copy_semantics_only(self):
        spec = declared_carrier([2, 4], "bf16")
        source, destination = allocate_carrier(spec, "cpu"), allocate_carrier(spec, "cpu")
        source.copy_(torch.arange(8).reshape(2, 4))
        before = source.clone()
        destination.copy_(source, non_blocking=True)
        validate_carrier(source, spec)
        validate_carrier(destination, spec)
        torch.testing.assert_close(destination, before)
        torch.testing.assert_close(source, before)

    def test_noncontiguous_dtype_and_shape_mismatch_rejected(self):
        spec = declared_carrier([2, 4], "bf16")
        for value in (torch.empty((4, 2), dtype=torch.bfloat16).t(), torch.empty((2, 4)), torch.empty((1, 8), dtype=torch.bfloat16)):
            with self.assertRaisesRegex(ValueError, "no implicit packing"):
                validate_carrier(value, spec)

    def test_inconsistent_declaration_is_not_a_layout_observation(self):
        spec = declared_carrier([2, 4], "bf16")
        value = allocate_carrier(spec, "cpu")
        for field, replacement in (("layout", "unknown"), ("tensor_bytes", 1), ("strides", [1, 2])):
            changed = {**spec, field: replacement}
            with self.assertRaises(ValueError):
                validate_carrier(value, changed)
            with self.assertRaises(ValueError):
                allocate_carrier(changed, "cpu")

    def test_cpu_binding_cannot_fall_back_or_probe_peer(self):
        spec = declared_carrier([2, 4], "bf16")
        source, destination = allocate_carrier(spec, "cpu"), allocate_carrier(spec, "cpu")
        with patch("torch.cuda.can_device_access_peer") as probe:
            with self.assertRaisesRegex(ValueError, "fallback is forbidden"):
                BoundTransfer(source, destination, spec, 0, 1, None)
            probe.assert_not_called()

    def test_tampered_coverage_contract_or_readiness_rejected(self):
        for change in ("missing_edge", "duplicate_edge", "mechanism", "layout", "readiness", "zero", "b1_source"):
            with self.subTest(change=change):
                bundle = copy.deepcopy(self.bundle)
                if change == "missing_edge":
                    bundle[B2_FILES[2]]["assignments"].pop()
                elif change == "duplicate_edge":
                    bundle[B2_FILES[2]]["assignments"].append(bundle[B2_FILES[2]]["assignments"][0])
                elif change == "zero":
                    bundle[B2_FILES[2]]["zero_transfer_plans"].pop()
                elif change == "b1_source":
                    bundle[B2_FILES[0]]["b1_source_sha256"]["identity.json"] = "wrong-source"
                else:
                    request = bundle[B2_FILES[1]]["requests"][0]
                    if change == "mechanism":
                        request["transfer_semantics"]["operation"] = "Tensor.to"
                    elif change == "layout":
                        request["source_carrier"]["layout"] = "unknown"
                    else:
                        request["measurement_ready"] = True
                with self.assertRaises(ValueError):
                    verify(bundle)

    def test_existing_different_obligations_not_silently_overwritten(self):
        b1, _ = load_b1()
        for field in ("transfer_mechanism", "synchronization", "layout"):
            with self.subTest(field=field):
                changed = copy.deepcopy(b1)
                changed["T3_communication_union.json"]["requirements"][0][field] = "different-semantics"
                with self.assertRaisesRegex(ValueError, "review instead of overwriting"):
                    derive(changed)


if __name__ == "__main__":
    unittest.main()
