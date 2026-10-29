from dataclasses import replace
import unittest

import torch

from simulator import PlacedOpGraph, TensorEdge, make_reference_plan
from execution.decoder_reference import ContinuousDecoderExecutor
from execution.gpu_holdout import CanonicalReferenceExecutor
from execution.verify_decoder_stack import (
    canonical_tensor, independent_decoder_forward, placed_stack, small_spec, verify,
)


class ContinuousDecoderTests(unittest.TestCase):
    def test_two_four_layer_formula_and_false_positive_controls(self):
        report = verify()
        self.assertEqual(len(report["cases"]), 32)
        self.assertTrue(report["all_pass"])
        self.assertFalse(report["gpu_copy_validated"])

    def test_first_layer_output_change_reaches_final_output(self):
        class ChangedFirstLayer(ContinuousDecoderExecutor):
            def _execute_node(self, node_id, inputs, device):
                result = super()._execute_node(node_id, inputs, device)
                return result + .1 if node_id == "L000.residual_add_2" else result

        with torch.no_grad():
            for layers in (2, 4):
                spec = small_spec(layers)
                initial = canonical_tensor((2, 5, 16), torch.float32, 1234, 11)
                expected = independent_decoder_forward(spec, initial, first_layer_delta=.1)
                changed, *_ = ChangedFirstLayer(placed_stack(spec), ("cpu", "cpu")).run_once(False)
                ordinary, *_ = ContinuousDecoderExecutor(placed_stack(spec), ("cpu", "cpu")).run_once(False)
                torch.testing.assert_close(changed, expected, rtol=1e-5, atol=1e-7)
                self.assertGreater((changed - ordinary).abs().max().item(), .05)

    def test_single_layer_remains_identical_to_legacy(self):
        with torch.no_grad():
            for dtype in ("fp32", "bf16"):
                placed = placed_stack(small_spec(1, dtype))
                old, *_ = CanonicalReferenceExecutor(placed, ("cpu", "cpu"), 1234, {}).run_once(False)
                new, *_ = ContinuousDecoderExecutor(placed, ("cpu", "cpu")).run_once(False)
                self.assertTrue(torch.equal(old, new))

    def test_projection_width_can_differ_from_residual_width(self):
        spec = replace(small_spec(), hidden_size=12)
        initial = canonical_tensor((2, 5, 12), torch.float32, 1234, 11)
        with torch.no_grad():
            actual, *_ = ContinuousDecoderExecutor(placed_stack(spec), ("cpu", "cpu")).run_once(False)
            expected = independent_decoder_forward(spec, initial)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)

    def test_connected_input_is_an_alias_and_checks_shape_dtype_device(self):
        executor = ContinuousDecoderExecutor(placed_stack(small_spec()), ("cpu", "cpu"))
        value = torch.ones(2, 5, 16)
        self.assertIs(executor._execute_node("L001.input", [value], torch.device("cpu")), value)
        for bad in (torch.ones(1, 5, 16), value.to(torch.float64)):
            with self.assertRaises(ValueError):
                executor._execute_node("L001.input", [bad], torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "consumer device"):
            executor._execute_node("L001.input", [value], torch.device("meta"))

    def test_disconnected_or_ambiguous_layer_input_is_rejected(self):
        for mode in ("disconnected", "ambiguous"):
            placed = placed_stack(small_spec())
            graph = placed.graph
            if mode == "disconnected":
                graph.edges = [edge for edge in graph.edges if edge.dst != "L001.input"]
            else:
                graph.add_edge(TensorEdge("L000.input", "L001.input"))
            placed = PlacedOpGraph(graph, placed.hardware, make_reference_plan(graph))
            with self.assertRaisesRegex(ValueError, "requires 1 predecessors"):
                ContinuousDecoderExecutor(placed, ("cpu", "cpu"))


if __name__ == "__main__":
    unittest.main()
