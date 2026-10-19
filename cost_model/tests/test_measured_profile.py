"""M3b measured full-shape and message-size profile tests."""

from __future__ import annotations

import unittest

from cost_model import MeasuredHardwareProfile, analyze_plan_structure
from simulator import FusionGroup, OpGraph, OpNode, Plan, TensorEdge, hardware_from_dict


SOURCE = "unit-test-source"


def _timing(value: float) -> dict:
    return {"median": value}


def _raw() -> dict:
    devices = []
    for index, scale in ((0, 1.0), (1, 1.1)):
        devices.append(
            {
                "index": index,
                "matmul": [
                    {
                        "shape": {"m": 256, "n": 1024, "k": 1024},
                        "duration_s": _timing(10e-6 * scale),
                    },
                    {
                        "shape": {"m": 256, "n": 3072, "k": 1024},
                        "duration_s": _timing(20e-6 * scale),
                    },
                    {
                        "shape": {"batch": 16, "m": 256, "n": 256, "k": 128},
                        "duration_s": _timing(30e-6 * scale),
                    },
                ],
                "reference_ops": [
                    {"name": "rmsnorm_reference", "duration_s": _timing(6e-6 * scale)},
                    {"name": "residual_add", "duration_s": _timing(5e-6 * scale)},
                    {"name": "silu_mul_eager_reference", "duration_s": _timing(7e-6 * scale)},
                ],
            }
        )
    points_01 = [
        {"size_bytes": 4096, "duration_s": _timing(8e-6)},
        {"size_bytes": 1024**2, "duration_s": _timing(20e-6)},
        {"size_bytes": 16 * 1024**2, "duration_s": _timing(100e-6)},
    ]
    points_10 = [
        {"size_bytes": 4096, "duration_s": _timing(9e-6)},
        {"size_bytes": 1024**2, "duration_s": _timing(22e-6)},
        {"size_bytes": 16 * 1024**2, "duration_s": _timing(110e-6)},
    ]
    return {
        "format": "fusion-placement-search.bw1000-calibration",
        "version": "1.0",
        "source_id": SOURCE,
        "dtype": "bf16",
        "model": {
            "seq_length": 256,
            "micro_batch_size": 1,
            "hidden_size": 1024,
            "ffn_hidden_size": 3072,
        },
        "devices": devices,
        "links": [
            {
                "src": 0,
                "dst": 1,
                "points": points_01,
                "fit": {"bandwidth_bytes_per_s": 200e9},
            },
            {
                "src": 1,
                "dst": 0,
                "points": points_10,
                "fit": {"bandwidth_bytes_per_s": 190e9},
            },
        ],
    }


def _hardware():
    return hardware_from_dict(
        {
            "name": "bw-test",
            "comm_bw": 20e9,
            "comm_latency": 5e-6,
            "launch_overhead": 2e-6,
            "link_matrix": [[None, 200e9], [190e9, None]],
            "devices": [
                {
                    "name": "BW",
                    "flops": 100e12,
                    "mem_bw": 1e12,
                    "max_mem": 64e9,
                    "calibrated": True,
                    "source": SOURCE,
                    "physical_index": index,
                }
                for index in (0, 1)
            ],
        }
    )


def _matmul(node_id: str, *, m: int, n: int, k: int, dtype: str = "bf16") -> OpNode:
    return OpNode(
        node_id,
        "matmul",
        flops=float(2 * m * n * k),
        out_bytes=float(m * n * 2),
        extra={
            "shape": {"m": m, "n": n, "k": k},
            "output_shape": (m, n),
            "dtype": dtype,
            "phase": "forward",
            "layer": 0,
        },
    )


def _group(graph: OpGraph, node_id: str, device: int = 0):
    plan = Plan(
        name="one",
        groups=[
            FusionGroup(
                [current],
                device if current == node_id else 0,
                {
                    "pattern_id": "unfused",
                    "backend": "reference",
                    "implementation_status": "reference",
                },
            )
            for current in graph.topological_order()
        ],
    )
    structure = analyze_plan_structure(graph, plan, 2)
    return next(group for group in structure.groups if group.node_ids == (node_id,))


class TestMeasuredHardwareProfile(unittest.TestCase):
    def setUp(self):
        self.profile = MeasuredHardwareProfile(_hardware(), _raw())

    def test_exact_full_shape_distinguishes_same_k(self):
        graph = OpGraph(name="same-k")
        graph.add_node(_matmul("small-n", m=256, n=1024, k=1024))
        graph.add_node(_matmul("large-n", m=256, n=3072, k=1024))
        small = self.profile.estimate_group(graph, _group(graph, "small-n"))
        large = self.profile.estimate_group(graph, _group(graph, "large-n"))
        self.assertEqual(small.duration, 10e-6)
        self.assertEqual(large.duration, 20e-6)
        self.assertEqual(small.source, "measured_exact_shape")
        self.assertEqual(large.source, "measured_exact_shape")
        self.assertEqual(small.ood_reasons, ())

    def test_batched_measurement_matches_flattened_graph_shape(self):
        graph = OpGraph(name="attention")
        graph.add_node(_matmul("qk", m=4096, n=256, k=128))
        cost = self.profile.estimate_group(graph, _group(graph, "qk"))
        self.assertEqual(cost.duration, 30e-6)
        self.assertEqual(cost.source, "measured_exact_shape")

    def test_dtype_mismatch_falls_back_and_is_ood(self):
        graph = OpGraph(name="dtype")
        graph.add_node(_matmul("fp32", m=256, n=1024, k=1024, dtype="fp32"))
        cost = self.profile.estimate_group(graph, _group(graph, "fp32"))
        self.assertEqual(cost.source, "analytic_fallback")
        self.assertTrue(cost.ood_reasons)
        self.assertLessEqual(cost.confidence, 0.25)

    def test_message_size_table_interpolates_and_preserves_direction(self):
        exact = self.profile.estimate_transfer(0, 1, 1024**2)
        reverse = self.profile.estimate_transfer(1, 0, 1024**2)
        between = self.profile.estimate_transfer(0, 1, 4 * 1024**2)
        self.assertEqual(exact.duration, 20e-6)
        self.assertEqual(reverse.duration, 22e-6)
        self.assertGreater(between.duration, 20e-6)
        self.assertLess(between.duration, 100e-6)
        self.assertEqual(exact.source, "measured_link_table")

    def test_rmsnorm_requires_matching_normalized_dimension(self):
        graph = OpGraph(name="rmsnorm")
        for node_id, tokens, inner in (("hidden", 256, 1024), ("head", 2048, 128)):
            graph.add_node(
                OpNode(
                    node_id,
                    "rmsnorm",
                    flops=1.0,
                    out_bytes=1.0,
                    extra={
                        "output_shape": (1, tokens, inner),
                        "dtype": "bf16",
                        "phase": "forward",
                        "layer": 0,
                    },
                )
            )
        hidden = self.profile.estimate_group(graph, _group(graph, "hidden"))
        head = self.profile.estimate_group(graph, _group(graph, "head"))
        self.assertEqual(hidden.source, "measured_exact_shape")
        self.assertEqual(head.source, "analytic_fallback")

    def test_eager_silu_mul_is_not_loaded_as_fused_kernel(self):
        graph = OpGraph(name="fused")
        for node_id, op_type in (("silu", "silu"), ("mul", "mul")):
            graph.add_node(
                OpNode(
                    node_id,
                    op_type,
                    flops=1.0,
                    out_bytes=1.0,
                    extra={
                        "output_shape": (256, 3072),
                        "dtype": "bf16",
                        "phase": "forward",
                        "layer": 0,
                    },
                )
            )
        graph.add_edge(TensorEdge("silu", "mul", 1.0))
        plan = Plan(
            name="fused",
            groups=[
                FusionGroup(
                    ["silu", "mul"],
                    0,
                    {
                        "pattern_id": "silu_mul",
                        "backend": "template",
                        "implementation_status": "planned",
                    },
                )
            ],
        )
        group = analyze_plan_structure(graph, plan, 2).groups[0]
        cost = self.profile.estimate_group(graph, group)
        self.assertEqual(cost.source, "analytic_fallback")
        self.assertIn("pattern:silu_mul:no_measured_fused_kernel", cost.ood_reasons)
        self.assertLessEqual(cost.confidence, 0.15)


if __name__ == "__main__":
    unittest.main()
