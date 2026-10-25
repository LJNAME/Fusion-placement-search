"""E0 counterfactual ranking-sensitivity tests."""

from __future__ import annotations

import unittest

from cost_model import (
    GroupCost,
    StructuralCostModel,
    TransferCost,
    analyze_plan_structure,
)
from cost_model.search_sensitivity import (
    CounterfactualHardwareProfile,
    SymmetricHardwareProfile,
    device_exchange_permutations,
    evaluate_counterfactual_sensitivity,
    permute_plan_devices,
    plan_id,
    shape_key_for_group,
)
from simulator import FusionGroup, OpGraph, OpNode, Plan


def _group(node_id: str, device: int) -> FusionGroup:
    return FusionGroup(
        [node_id],
        device,
        {
            "pattern_id": "unfused",
            "backend": "reference",
            "implementation_status": "reference",
        },
    )


class _FixedProfile:
    durations = {
        (0, "a"): 5.0e-3,
        (1, "a"): 8.0e-3,
        (0, "b"): 1.0e-3,
        (1, "b"): 5.2e-3,
        (0, "same"): 2.0e-3,
        (1, "same"): 3.0e-3,
    }

    @property
    def n_devs(self) -> int:
        return 2

    @property
    def overlap_send(self) -> bool:
        return True

    @property
    def interference(self) -> float:
        return 0.0

    def max_memory(self, device: int) -> float:
        return 1e12

    def estimate_group(self, graph, group) -> GroupCost:
        node = graph.nodes[group.node_ids[0]]
        duration = self.durations[(group.device, node.op_type)]
        return GroupCost(
            duration=duration,
            compute_time=duration,
            memory_time=0.0,
            launch_time=0.0,
            confidence=0.5,
            source="fixed",
        )

    def estimate_transfer(self, src_device, dst_device, nbytes) -> TransferCost:
        duration = 1.0e-3 if (src_device, dst_device) == (0, 1) else 3.0e-3
        return TransferCost(duration, 1.0, "fixed")

    def link_key(self, src_device, dst_device) -> tuple[int, int]:
        return tuple(sorted((int(src_device), int(dst_device))))


def _node(node_id: str, op_type: str) -> OpNode:
    return OpNode(
        node_id,
        op_type,
        flops=1.0,
        out_bytes=2.0,
        extra={"dtype": "bf16", "output_shape": (1, 1)},
    )


class TestCounterfactualSensitivity(unittest.TestCase):
    def test_detects_top1_flip_and_critical_path_switch(self):
        graph = OpGraph(name="ranking-flip")
        graph.add_node(_node("a", "a"))
        graph.add_node(_node("b", "b"))
        colocated = Plan([_group("a", 0), _group("b", 0)], name="colocated")
        split = Plan([_group("a", 0), _group("b", 1)], name="split")

        report = evaluate_counterfactual_sensitivity(
            graph,
            (colocated, split),
            _FixedProfile(),
            perturbations=(-0.20, 0.20),
            top_k=1,
        )

        by_label = {
            item["label"]: item for item in report["shape_sensitivities"]
        }
        b_on_device_1 = next(
            item
            for label, item in by_label.items()
            if label.startswith("device_class=physical:1") and "|b[bf16]" in label
        )
        a_on_device_0 = next(
            item
            for label, item in by_label.items()
            if label.startswith("device_class=physical:0") and "|a[bf16]" in label
        )
        self.assertTrue(b_on_device_1["top1_flip"])
        self.assertTrue(b_on_device_1["top_k_flip"])
        self.assertGreaterEqual(b_on_device_1["max_rank_shift"], 1)
        self.assertGreater(a_on_device_0["critical_path_switches"], 0)
        self.assertEqual(report["summary"]["ranking_signal"], "strong")

    def test_profile_scales_every_occurrence_of_one_shared_shape(self):
        graph = OpGraph(name="shared-shape")
        graph.add_node(_node("left", "same"))
        graph.add_node(_node("right", "same"))
        plan = Plan([_group("left", 0), _group("right", 0)])
        structure = analyze_plan_structure(graph, plan, 2)
        target = shape_key_for_group(graph, structure.groups[0])
        perturbed = CounterfactualHardwareProfile(
            _FixedProfile(), target, relative_delta=0.20
        )

        costs = [
            perturbed.estimate_group(graph, group).duration
            for group in structure.groups
        ]
        self.assertEqual(costs, [2.4e-3, 2.4e-3])
        self.assertEqual(
            shape_key_for_group(graph, structure.groups[0]),
            shape_key_for_group(graph, structure.groups[1]),
        )

    def test_symmetric_profile_collapses_device_swaps(self):
        graph = OpGraph(name="device-symmetry")
        graph.add_node(_node("a", "a"))
        graph.add_node(_node("b", "b"))
        colocated = Plan([_group("a", 0), _group("b", 0)], name="colocated")
        split = Plan([_group("a", 0), _group("b", 1)], name="split")
        classes = ("model:bw1000", "model:bw1000")
        exchange = device_exchange_permutations(classes)
        symmetric = SymmetricHardwareProfile(_FixedProfile(), classes)
        swapped_colocated = permute_plan_devices(colocated, (1, 0))
        swapped_split = permute_plan_devices(split, (1, 0))

        model = StructuralCostModel()
        self.assertAlmostEqual(
            model.predict(graph, colocated, symmetric).makespan,
            model.predict(graph, swapped_colocated, symmetric).makespan,
        )
        self.assertEqual(
            plan_id(colocated, exchange),
            plan_id(swapped_colocated, exchange),
        )
        report = evaluate_counterfactual_sensitivity(
            graph,
            (colocated, swapped_colocated, split, swapped_split),
            symmetric,
            perturbations=(-0.20, 0.20),
            top_k=1,
        )
        self.assertEqual(report["summary"]["candidate_plans"], 2)
        self.assertEqual(report["summary"]["shape_keys"], 2)
        self.assertTrue(report["configuration"]["symmetry_enabled"])
        self.assertEqual(
            report["configuration"]["max_exchange_score_delta_ms"], 0.0
        )
        self.assertEqual(symmetric.estimate_transfer(0, 1, 1.0).duration, 2.0e-3)
        self.assertEqual(symmetric.estimate_transfer(1, 0, 1.0).duration, 2.0e-3)

    def test_rejects_invalid_experiment_inputs(self):
        graph = OpGraph(name="invalid")
        graph.add_node(_node("a", "a"))
        plan = Plan([_group("a", 0)])
        with self.assertRaisesRegex(ValueError, "at least two"):
            evaluate_counterfactual_sensitivity(graph, (plan,), _FixedProfile())
        with self.assertRaisesRegex(ValueError, "non-zero"):
            evaluate_counterfactual_sensitivity(
                graph,
                (plan, Plan([_group("a", 1)])),
                _FixedProfile(),
                perturbations=(0.0,),
            )


if __name__ == "__main__":
    unittest.main()
