"""Contract gates must reject matched but inadmissible measurements."""
import unittest
import warnings

from calibration.e2_manifest import content_id
from cost_model.fusion_kernel_profile import KernelLatency, KeyedKernelProfile, key_id
from cost_model.measurement_labels import MeasurementContractError, measurement_contract_id
from cost_model.measured_profile import MeasuredHardwareProfile
from cost_model.srop_e0 import coupling_fixture, run_mode
from cost_model.tests.test_measured_profile import _hardware, _raw, _matmul, _group
from simulator import OpGraph


CONTRACT = {"executor": "test", "submission_included": True, "allocation_included": True,
    "granularity": "per_call", "unit": "latency",
    "environment": {"torch": "test", "hip": None, "triton": None, "device_model": "test"}}
CID = measurement_contract_id(CONTRACT)


class TestMeasurementLabels(unittest.TestCase):
    def test_contract_hash_uses_existing_content_id(self):
        self.assertEqual(CID, content_id("measurement-contract", CONTRACT))
        with self.assertRaises(ValueError):
            measurement_contract_id({"executor": "test"})

    def group_profile(self, admission="accepted", contract=CID, expected=CID, allow_unbound=False):
        graph, hardware, registry, profile = coupling_fixture()
        labels = {key: KernelLatency(entry.duration_s, "measured", "test-only", contract,
                                    admission=admission) for key, entry in profile.latencies.items()}
        guarded = KeyedKernelProfile(profile.keys, labels, profile.base,
            measurement_contract_id=expected, allow_unbound=allow_unbound)
        return graph, registry, profile, guarded

    def test_group_mismatch_raises_even_with_missing_key_fallback_enabled(self):
        graph, registry, old, profile = self.group_profile(expected="sha256:other")
        profile.strict = False
        with self.assertRaisesRegex(MeasurementContractError, "contract rejected"):
            run_mode(graph, profile, registry, "joint", method="exact")
        self.assertEqual(set(old.latencies), set(profile.latencies))

    def test_existing_search_rejects_legacy_unbound_labels(self):
        graph, registry, _, profile = self.group_profile("legacy_unbound", None)
        with self.assertRaisesRegex(MeasurementContractError, "legacy_unbound"):
            run_mode(graph, profile, registry, "joint", method="mcts", iterations=8)

    def test_matching_accepted_labels_allow_search_without_rekeying(self):
        graph, registry, old, profile = self.group_profile()
        result = run_mode(graph, profile, registry, "joint", method="exact")
        self.assertGreater(result["makespan_s"], 0)
        self.assertEqual(set(old.latencies), set(profile.latencies))

    def test_replaced_label_cannot_reuse_cached_accepted_contract(self):
        graph, registry, _, profile = self.group_profile()
        identifier, old = next(iter(profile.latencies.items()))
        profile.latencies[identifier] = KernelLatency(old.duration_s, "measured", "new-unbound-label")
        from cost_model.fusion_kernel_profile import candidate_structure
        from cost_model.legal_search import LegalFusionSearchSpace
        candidate = next(c for c in LegalFusionSearchSpace(graph, 2, registry, require_executable=False).candidates
            if any(key_id(profile.keys.describe(candidate_structure(graph, c, d))[0]) == identifier
                   for d in (0, 1)))
        for device in (0, 1):
            group = candidate_structure(graph, candidate, device)
            if key_id(profile.keys.describe(group)[0]) == identifier:
                with self.assertRaisesRegex(MeasurementContractError, "legacy_unbound"):
                    profile.estimate_group(graph, group)

    def test_conditional_and_unbound_diagnostics_are_explicit_low_confidence(self):
        for admission, contract in (("conditional_dev", CID), ("legacy_unbound", None), ("accepted", "other")):
            with self.subTest(admission=admission):
                graph, registry, _, profile = self.group_profile(admission, contract)
                with self.assertRaises(MeasurementContractError):
                    run_mode(graph, profile, registry, "joint", method="exact")
                profile.allow_unbound = True
                with warnings.catch_warnings(record=True) as emitted:
                    warnings.simplefilter("always")
                    result = run_mode(graph, profile, registry, "joint", method="exact")
                self.assertTrue(emitted)
                self.assertTrue(result["makespan_s"] > 0)
                group = next(c for c in profile.keys.graph.nodes if c == "silu")
                from cost_model.fusion_kernel_profile import candidate_structure
                from cost_model.legal_search import LegalFusionSearchSpace
                candidate = next(c for c in LegalFusionSearchSpace(graph, 2, registry, require_executable=False).candidates
                                 if c.node_ids == (group,))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cost = profile.estimate_group(graph, candidate_structure(graph, candidate, 0))
                self.assertLessEqual(cost.confidence, .25)
                self.assertTrue(cost.ood_reasons)

    def test_shape_exact_and_interpolation_use_same_gate(self):
        profile = MeasuredHardwareProfile(_hardware(), _raw(), measurement_contract_id=CID)
        for m in (256, 257):
            graph = OpGraph(name="gate")
            graph.add_node(_matmul("mm", m=m, n=1024, k=1024))
            with self.assertRaises(MeasurementContractError):
                profile.estimate_group(graph, _group(graph, "mm"))
            profile.allow_unbound = True
            with self.assertWarns(RuntimeWarning):
                cost = profile.estimate_group(graph, _group(graph, "mm"))
            self.assertLessEqual(cost.confidence, .25)
            self.assertTrue(cost.ood_reasons)
            profile.allow_unbound = False

    def test_shape_matching_contract_is_required(self):
        raw = _raw()
        for dev in raw["devices"]:
            for item in dev["matmul"]:
                item.update(measurement_contract_id=CID, admission="accepted")
        graph = OpGraph(name="matched")
        graph.add_node(_matmul("mm", m=256, n=1024, k=1024))
        with self.assertRaises(MeasurementContractError):
            MeasuredHardwareProfile(_hardware(), raw, measurement_contract_id="other").estimate_group(graph, _group(graph, "mm"))
        cost = MeasuredHardwareProfile(_hardware(), raw, measurement_contract_id=CID).estimate_group(graph, _group(graph, "mm"))
        self.assertEqual(cost.duration, 10e-6)
        self.assertEqual(cost.confidence, .95)
        self.assertAlmostEqual(cost.compute_time + cost.launch_time, cost.duration)


if __name__ == "__main__":
    unittest.main()
