"""KERV development-source substitute: shape, liveness, fallback and provenance."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from calibration.kerv_silu_mul import (
    DEFAULT_FLAGSCALE, build_capability, candidate_applicability, check_capability,
    development_registry,
)
from calibration.probe_kerv_silu_mul import checked_fused_call, load_kernel_module, run_probe
from cost_model.fusion_kernel_profile import GroupKernelKeys, candidate_structure, key_id
from cost_model.legal_search import LegalFusionSearchSpace
from cost_model.search_evaluation import build_evaluation_graph
from cost_model.srop_e0 import coupling_fixture, run_mode
from simulator import OpNode, TensorEdge, default_fusion_registry


class TestKervGraphGuards(unittest.TestCase):
    def setUp(self):
        self.graph, self.hardware, _, self.profile = coupling_fixture()
        self.manifest = {"authority": "development_surrogate_not_project_downstream_contract"}

    def test_shape_boundary_and_qwen_widths(self):
        for width, eligible in ((3072, True), (8192, True), (11008, False)):
            graph = copy.deepcopy(self.graph)
            for node in graph.nodes.values():
                node.extra["output_shape"] = [4, width]
            result = candidate_applicability(graph, ("silu", "mul"))
            self.assertEqual(result["eligible"], eligible, width)
        registry = development_registry(self.manifest)
        qwen = LegalFusionSearchSpace(build_evaluation_graph("layer"), 2, registry,
                                      require_executable=False)
        large = LegalFusionSearchSpace(build_evaluation_graph("model_3b"), 2, registry,
                                       require_executable=False)
        self.assertEqual(sum(len(c.node_ids) > 1 for c in qwen.candidates), 1)
        self.assertEqual(sum(len(c.node_ids) > 1 for c in large.candidates), 0)

    def test_inplace_gate_cannot_destroy_another_consumers_input(self):
        self.graph.add_node(OpNode("other_consumer", "add"))
        self.graph.add_edge(TensorEdge("gate", "other_consumer"))
        result = candidate_applicability(self.graph, ("silu", "mul"))
        self.assertFalse(result["eligible"])
        self.assertIn("gate_original_has_external_consumers", result["reasons"])
        space = LegalFusionSearchSpace(self.graph, 2, development_registry(self.manifest),
                                       require_executable=False)
        self.assertTrue(all(len(c.node_ids) == 1 for c in space.candidates))

    def test_exported_gate_is_not_overwritten(self):
        self.graph.extra["output_nodes"] = ["gate", "out"]
        self.assertIn("gate_original_is_exported_output",
                      candidate_applicability(self.graph, ("silu", "mul"))["reasons"])

    def test_unknown_layout_is_conditional_not_verified(self):
        for name in ("gate", "up"):
            del self.graph.nodes[name].extra["layout"]
        strict = candidate_applicability(self.graph, ("silu", "mul"))
        self.assertFalse(strict["eligible"])
        conditional = candidate_applicability(self.graph, ("silu", "mul"),
                                               layout_policy="require_contiguous_at_runtime")
        self.assertTrue(conditional["eligible"])
        self.assertEqual(len(conditional["requirements"]), 2)

    def test_noncontiguous_layout_cannot_be_assumed_away(self):
        self.graph.nodes["gate"].extra["strides"] = [16, 2]
        result = candidate_applicability(self.graph, ("silu", "mul"),
                                         layout_policy="require_contiguous_at_runtime")
        self.assertFalse(result["eligible"])
        self.assertIn("gate_is_not_contiguous", result["reasons"])

    def test_mixed_dtype_and_input_arity_are_rejected(self):
        self.graph.nodes["up"].extra["dtype"] = "fp32"
        self.assertFalse(candidate_applicability(self.graph, ("silu", "mul"))["eligible"])
        self.graph.add_edge(TensorEdge("gate", "mul"))
        self.assertIn("unsupported_input_arity",
                      candidate_applicability(self.graph, ("silu", "mul"))["reasons"])

    def test_backward_phase_and_column_major_layout_are_rejected(self):
        for node in self.graph.nodes.values():
            node.extra["phase"] = "backward"
        self.assertIn("forward_only_or_unknown_execution_phase",
                      candidate_applicability(self.graph, ("silu", "mul"))["reasons"])
        for node in self.graph.nodes.values():
            node.extra["phase"] = "forward"
        self.graph.nodes["gate"].extra["layout"]["order"] = "column_major"
        self.assertIn("gate_is_not_contiguous",
                      candidate_applicability(self.graph, ("silu", "mul"))["reasons"])

    def test_proxy_never_promotes_default_registry_or_deployability(self):
        registry = development_registry(self.manifest)
        self.assertFalse(default_fusion_registry().get("silu_mul").executable)
        self.assertFalse(registry.get("silu_mul").executable)
        self.assertNotIn("elementwise_chain", {p.pattern_id for p in registry.patterns()})
        row = run_mode(self.graph, self.profile, registry, "joint", method="exact")
        self.assertLess(row["makespan_s"], .011)
        self.assertFalse(row["artifact"]["extra"]["deployable"])

    def test_probe_rejects_cpu_and_oversized_shapes_before_measurement(self):
        with self.assertRaisesRegex(RuntimeError, "CPU fallback is forbidden"):
            run_probe("unused.json", device="cpu")
        with self.assertRaisesRegex(ValueError, "single-Kernel guard"):
            run_probe("unused.json", shape=(1, 11008))


@unittest.skipUnless((DEFAULT_FLAGSCALE / ".git").exists(), "local FlagScale source tree unavailable")
class TestPinnedKervSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = build_capability()

    def test_source_pinning_and_recomputed_but_changed_contract_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "capability.json"
            path.write_text(json.dumps(self.manifest), encoding="utf-8")
            self.assertEqual(check_capability(path), self.manifest)
            changed = copy.deepcopy(self.manifest)
            del changed["capability_id"]
            changed["static_contract"]["maximum_width"] = 11008
            changed["capability_id"] = key_id(changed)
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "reviewed contract"):
                check_capability(path)

    def test_binding_is_separate_from_synthetic_or_unbound_kernel(self):
        graph, hardware, registry, profile = coupling_fixture()
        candidate = next(c for c in LegalFusionSearchSpace(graph, 2, registry,
                         require_executable=False).candidates if c.pattern_id == "silu_mul")
        group = candidate_structure(graph, candidate, 0)
        bound = GroupKernelKeys(graph, hardware, {"silu_mul": self.manifest["binding"]},
                                profile.keys.runtime)
        self.assertNotEqual(key_id(bound.describe(group)[0]), key_id(profile.keys.describe(group)[0]))
        self.assertFalse(self.manifest["validation"]["GPU_validated"])

    def test_actual_cpu_native_alias_and_auto_dispatch_are_not_fusion_evidence(self):
        import torch
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "capability.json"
            path.write_text(json.dumps(self.manifest), encoding="utf-8")
            module, _ = load_kernel_module(DEFAULT_FLAGSCALE, path)
            old_backend = module._BACKEND
            old_enabled = module._ENABLED_OPERATORS
            try:
                module.configure_backend("auto")
                module.configure_enabled_operators(None)
                self.assertFalse(module._use_triton("kerv_silu_mul"))
                generator = torch.Generator().manual_seed(1234)
                gate = torch.randn(2, 8, generator=generator)
                up = torch.randn(2, 8, generator=generator)
                original_up = up.clone()
                expected = torch.nn.functional.silu(gate.clone()) * up
                result = module.kerv_silu_mul(gate, up)
                self.assertEqual(result.data_ptr(), gate.data_ptr())
                torch.testing.assert_close(result, expected)
                self.assertTrue(torch.equal(up, original_up))
                with self.assertRaisesRegex(RuntimeError, "CPU fallback is forbidden"):
                    checked_fused_call(module, gate, up)
            finally:
                module.configure_backend(old_backend)
                module.configure_enabled_operators(old_enabled)


if __name__ == "__main__":
    unittest.main()
