"""CPU contract tests; mocked dispatch never counts as GPU execution evidence."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from calibration.kerv_silu_mul import DEFAULT_BUNDLE, DEFAULT_FLAGSCALE
from calibration.validate_kerv_adapter import main, prepare, verify_device_trace
from cost_model.search_evaluation import build_evaluation_graph
from integration.flagscale import kerv_execution as execution
from simulator import enumerate_fusion_candidates, load_hardware, make_reference_plan
from simulator.placement_artifacts import PlacedOpGraph, load_placed_op_graph


ROOT = Path(__file__).resolve().parents[2]


def fake_module():
    module = SimpleNamespace(_BACKEND="auto", _ENABLED_OPERATORS=None,
                             _RECORD_PATH=None, _RECORD_ONCE=True, _RECORDED={"old"})
    module.configure_backend = lambda value: setattr(module, "_BACKEND", value)
    module.configure_enabled_operators = lambda value: setattr(module, "_ENABLED_OPERATORS", value)

    def recording(path, enabled):
        module._RECORD_PATH = path if enabled else None
        module._RECORDED.clear()

    module.configure_recording = recording
    return module


class TestKervExecution(unittest.TestCase):
    def setUp(self):
        self.module = fake_module()
        self.manifest = {"binding": {"id": "test-only", "version": "test-only"}}
        with patch.object(execution, "load_kernel_module", return_value=(self.module, self.manifest)):
            self.adapter = execution.KervSiluMulAdapter("test-only.json")
        graph = build_evaluation_graph("small")
        # Keep actual reference node IDs and FFN width, reduce only sequence.
        for node in graph.nodes.values():
            shape = list(node.extra["output_shape"])
            shape[1] = 1
            node.extra["output_shape"] = shape
        candidate = next(c for c in enumerate_fusion_candidates(graph, self.adapter.registry,
                          require_executable=True) if c.pattern_id == "silu_mul")
        hardware = load_hardware(ROOT / "data/bw1000_2card/bw1000_2card.calibrated.json")
        plan = make_reference_plan(graph, device=0, registry=self.adapter.registry)
        plan.groups = [g for g in plan.groups if g.node_ids[0] not in candidate.node_ids]
        # Both projections precede the fused group; down_proj follows it.
        plan.groups.insert(len(plan.groups) - 1, candidate.to_group(0))
        self.placed = PlacedOpGraph(graph, hardware, plan)
        self.index = next(i for i, g in enumerate(plan.groups) if len(g.node_ids) > 1)
        self.bound = self.adapter.bind(self.placed, self.index, {0: "cuda:0"})
        self.gate = torch.randn(1, 1, 3072, dtype=torch.bfloat16)
        self.up = torch.randn_like(self.gate)

    def values(self):
        return {self.bound.up_node: self.up, self.bound.gate_node: self.gate}

    def mock_fused(self, module, gate, up):
        self.assertEqual(module._BACKEND, "triton")
        self.assertEqual(module._ENABLED_OPERATORS, {"kerv_silu_mul"})
        gate.copy_(F.silu(gate) * up)
        return gate

    def test_mapping_inplace_alias_and_live_gate_invalidation(self):
        expected, original_up = F.silu(self.gate) * self.up, self.up.clone()
        values = self.values()
        with patch.object(execution, "_check_device"), patch.object(
                execution, "checked_fused_call", side_effect=self.mock_fused) as dispatch:
            call = self.bound(values)
        dispatch.assert_called_once()
        torch.testing.assert_close(call.output, expected)
        self.assertEqual(call.output.data_ptr(), self.gate.data_ptr())
        self.assertEqual(call.output_node, self.bound.node_ids[-1])
        self.assertNotIn(self.bound.gate_node, values)
        self.assertNotIn(self.bound.node_ids[0], values)
        self.assertIs(values[call.output_node], self.gate)
        self.assertTrue(torch.equal(original_up, self.up))
        self.assertIsNone(call.observed_jit_launches)
        self.assertEqual(self.module._BACKEND, "auto")
        self.assertIsNone(self.module._ENABLED_OPERATORS)
        self.assertEqual(self.module._RECORDED, {"old"})

    def test_cpu_cannot_enter_fused_dispatch(self):
        with patch.object(execution, "checked_fused_call") as dispatch:
            with self.assertRaisesRegex(RuntimeError, "CPU fallback is forbidden"):
                self.bound(self.values())
        dispatch.assert_not_called()

    def test_dispatch_failure_restores_backend_and_does_not_publish_output(self):
        values = self.values()
        with patch.object(execution, "_check_device"), patch.object(
                execution, "checked_launch_probe", side_effect=RuntimeError("no JIT launch")):
            with self.assertRaisesRegex(RuntimeError, "no JIT launch"):
                self.bound(values, observe_dispatch=True)
        self.assertIn(self.bound.gate_node, values)
        self.assertNotIn(self.bound.node_ids[-1], values)
        self.assertEqual(self.module._BACKEND, "auto")

    def test_storage_alias_to_up_or_another_live_tensor_is_rejected(self):
        for name in (self.bound.up_node, "other_live_node"):
            values = self.values()
            values[name] = self.gate.view_as(self.gate)
            with patch.object(execution, "_check_device"), patch.object(execution, "checked_fused_call") as dispatch:
                with self.assertRaisesRegex(ValueError, "storage aliases"):
                    self.bound(values)
                dispatch.assert_not_called()

    def test_runtime_metadata_and_autograd_are_checked_before_dispatch(self):
        for tensor in (self.gate.float(), self.gate[:, :, :8],
                       torch.randn(1, 1, 6144, dtype=self.gate.dtype)[:, :, ::2],
                       self.gate.clone().requires_grad_()):
            values = self.values()
            values[self.bound.gate_node] = tensor
            with patch.object(execution, "checked_fused_call") as dispatch:
                with self.assertRaises(ValueError):
                    self.bound(values)
                dispatch.assert_not_called()

    def test_external_gate_consumers_and_exported_intermediate_are_rejected(self):
        graph = self.placed.graph
        graph.extra["output_nodes"] = [self.bound.node_ids[0]]
        with self.assertRaisesRegex(ValueError, "exported SiLU"):
            self.adapter.bind(self.placed, self.index, {0: "cuda:0"})
        graph.extra.pop("output_nodes")
        from simulator import TensorEdge
        graph.add_edge(TensorEdge(self.bound.gate_node, "L000.mlp.down_proj"))
        with self.assertRaisesRegex(ValueError, "gate_original_has_external_consumers"):
            self.adapter.bind(self.placed, self.index, {0: "cuda:0"})

    def test_placement_and_explicit_implementation_cannot_change_after_binding(self):
        group = self.placed.plan.groups[self.index]
        group.device = 1
        with self.assertRaisesRegex(ValueError, "placement or implementation changed"):
            self.bound(self.values())
        group.device = 0
        group.extra["implementation"] = {"id": "other"}
        with self.assertRaisesRegex(ValueError, "verified capability"):
            self.adapter.bind(self.placed, self.index, {0: "cuda:0"})

    def test_missing_layers_and_wrong_runtime_mapping_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "explicit cuda:N"):
            self.adapter.bind(self.placed, self.index, {0: "cpu"})
        for node in self.bound.node_ids:
            self.placed.graph.nodes[node].extra.pop("layer")
        with self.assertRaisesRegex(ValueError, "same-layer metadata"):
            self.adapter.bind(self.placed, self.index, {0: "cuda:0"})

    def test_gate_and_up_resolution_does_not_depend_on_edge_list_order(self):
        self.placed.graph.edges.reverse()
        bound = self.adapter.bind(self.placed, self.index, {0: "cuda:0"})
        self.assertEqual(bound.gate_node, self.bound.gate_node)
        self.assertEqual(bound.up_node, self.bound.up_node)


class TestEvidenceGates(unittest.TestCase):
    def test_device_trace_requires_one_named_device_kernel(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            kernel = {"cat": "kernel", "name": "_silu_mul_kernel_0"}
            for events in ([], [{"cat": "cpu_op", "name": "kerv_silu_mul"}],
                           [kernel, kernel], [{"cat": "kernel", "name": "aten_mul"}]):
                path.write_text(json.dumps({"traceEvents": events}))
                with self.assertRaisesRegex(RuntimeError, "exactly one KERV"):
                    verify_device_trace(path)
            path.write_text(json.dumps({"traceEvents": [kernel]}))
            self.assertEqual(verify_device_trace(path)["device_kernel_count"], 1)

    def test_no_gpu_cli_records_failure_without_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with patch("torch.cuda.is_available", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "no CPU/eager fallback"):
                    main(["run", "--placed", "unused.json", "--out-dir", str(output)])
            failure = json.loads((output / "failure.json").read_text())
            self.assertFalse(failure["gpu_validation_passed"])
            self.assertFalse(failure["profile_accepted"])
            self.assertFalse((output / "result.json").exists())


@unittest.skipUnless((DEFAULT_FLAGSCALE / ".git").exists(), "pinned FlagScale checkout unavailable")
class TestPinnedAdapterPreparation(unittest.TestCase):
    def test_capability_opt_in_full_graph_and_artifact_binding(self):
        capability = DEFAULT_BUNDLE / "capability.json"
        registry, manifest = execution.load_kerv_registry(capability)
        from simulator import default_fusion_registry
        self.assertFalse(default_fusion_registry().get("silu_mul").executable)
        self.assertTrue(registry.get("silu_mul").executable)
        self.assertFalse(manifest["validation"]["GPU_validated"])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            index = prepare(output, capability, DEFAULT_FLAGSCALE,
                            ROOT / "data/bw1000_2card/bw1000_2card.calibrated.json")
            info = json.loads((output / "prepare.json").read_text())
            self.assertEqual((info["nodes"], info["executable_silu_mul_candidates"]), (591, 28))
            placed = load_placed_op_graph(output / "placed.json", registry=registry)
            adapter = execution.KervSiluMulAdapter(capability)
            bound = adapter.bind(placed, index, {0: "cuda:0"})
            self.assertEqual(bound.node_ids, ("L000.mlp.silu", "L000.mlp.gated_mul"))
            self.assertEqual(placed.plan.groups[index].extra["implementation"], manifest["binding"])
            self.assertFalse(placed.extra["deployable"])
        with tempfile.TemporaryDirectory() as directory:
            bad = copy.deepcopy(manifest)
            bad["static_contract"]["maximum_width"] = 11008
            path = Path(directory) / "capability.json"
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                execution.load_kerv_registry(path)


if __name__ == "__main__":
    unittest.main()
