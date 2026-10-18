"""M3a common-structure and hardware-profile cost tests."""

from __future__ import annotations

import unittest

from cost_model import (
    CostEstimate,
    GroupCost,
    StructuralCostModel,
    TransferCost,
    analyze_plan_structure,
)
from simulator import (
    FusionGroup,
    OpGraph,
    OpNode,
    Plan,
    TensorEdge,
    hardware_from_dict,
)


def _node(node_id: str, flops: float = 1000.0, param_bytes: float = 16.0) -> OpNode:
    return OpNode(
        id=node_id,
        op_type="add",
        flops=flops,
        out_bytes=64.0,
        param_bytes=param_bytes,
        extra={
            "output_shape": (32,),
            "dtype": "bf16",
            "phase": "forward",
            "layer": 0,
        },
    )


def _graph() -> OpGraph:
    graph = OpGraph(name="three-op-chain")
    for node_id in ("a", "b", "c"):
        graph.add_node(_node(node_id))
    graph.add_edge(TensorEdge("a", "b", 64.0))
    graph.add_edge(TensorEdge("b", "c", 64.0))
    return graph


def _plan(devices=(0, 1, 1), workspace: float = 0.0) -> Plan:
    return Plan(
        name="placed-chain",
        extra={"search_space": "legal_fusion_v1"},
        groups=[
            FusionGroup(
                [node_id],
                devices[index],
                {
                    "pattern_id": "unfused",
                    "backend": "reference",
                    "implementation_status": "reference",
                    "workspace_bytes": workspace if node_id == "b" else 0.0,
                },
            )
            for index, node_id in enumerate(("a", "b", "c"))
        ],
    )


def _hardware(
    *,
    rate: float = 1e6,
    max_mem: float = 1e6,
    calibrated: bool = True,
):
    return hardware_from_dict(
        {
            "name": "test-hardware",
            "comm_bw": rate,
            "comm_latency": 1e-5,
            "link_bw": rate,
            "devices": [
                {
                    "name": "d0",
                    "flops": rate,
                    "mem_bw": rate,
                    "max_mem": max_mem,
                    "calibrated": calibrated,
                },
                {
                    "name": "d1",
                    "flops": rate,
                    "mem_bw": rate,
                    "max_mem": max_mem,
                    "calibrated": calibrated,
                },
            ],
        }
    )


class _FixedProfile:
    n_devs = 2
    overlap_send = True
    interference = 0.0

    def max_memory(self, device: int) -> float:
        return 1e9

    def estimate_group(self, graph, group) -> GroupCost:
        return GroupCost(1.0, 0.6, 0.4, 0.0, 1.0, "fixed")

    def estimate_transfer(self, src_device, dst_device, nbytes) -> TransferCost:
        return TransferCost(0.5, 1.0, "fixed")

    def link_key(self, src_device, dst_device):
        return (0, 0)


class TestPlanStructure(unittest.TestCase):
    def test_extracts_hardware_independent_group_dag(self):
        structure = analyze_plan_structure(_graph(), _plan(), n_devs=2)
        self.assertEqual([group.group_id for group in structure.groups], ["fg0000", "fg0001", "fg0002"])
        self.assertEqual(len(structure.dependencies), 2)
        self.assertEqual(structure.cross_device_bytes, 64.0)
        self.assertEqual(structure.groups[1].parameter_bytes, 16.0)
        self.assertEqual(structure.groups[1].input_bytes, 80.0)
        self.assertEqual(structure.groups[1].output_bytes, 64.0)

    def test_structure_does_not_take_hardware(self):
        first = analyze_plan_structure(_graph(), _plan(), n_devs=2)
        second = analyze_plan_structure(_graph(), _plan(), n_devs=2)
        self.assertEqual(first, second)


class TestStructuralCostModel(unittest.TestCase):
    def test_returns_physical_cost_estimate_and_critical_path(self):
        estimate = StructuralCostModel().predict(_graph(), _plan(), _FixedProfile())
        self.assertIsInstance(estimate, CostEstimate)
        self.assertEqual(estimate.makespan, 3.5)
        self.assertEqual(estimate.communication_time, 0.5)
        self.assertEqual(estimate.cross_device_bytes, 64.0)
        self.assertEqual(estimate.critical_path, ("fg0000", "fg0001", "fg0002"))
        self.assertEqual(estimate.confidence, 1.0)
        self.assertFalse(estimate.out_of_distribution)

    def test_hardware_profile_changes_latency_not_structure(self):
        model = StructuralCostModel()
        slow = model.predict(_graph(), _plan(), _hardware(rate=1e6))
        fast = model.predict(_graph(), _plan(), _hardware(rate=2e6))
        self.assertLess(fast.makespan, slow.makespan)
        self.assertEqual(fast.cross_device_bytes, slow.cross_device_bytes)
        self.assertEqual(fast.critical_path, slow.critical_path)

    def test_uncalibrated_profile_is_marked_out_of_distribution(self):
        estimate = StructuralCostModel().predict(
            _graph(),
            _plan(),
            _hardware(calibrated=False),
        )
        self.assertTrue(estimate.out_of_distribution)
        self.assertIn("device:0:uncalibrated", estimate.ood_reasons)
        self.assertIn("link:0-1:uncalibrated", estimate.ood_reasons)
        self.assertLessEqual(estimate.confidence, 0.2)

    def test_workspace_and_memory_overflow_are_reported(self):
        estimate = StructuralCostModel().predict(
            _graph(),
            _plan(workspace=512.0),
            _hardware(max_mem=200.0),
        )
        self.assertGreater(estimate.peak_memory[1], 200.0)
        self.assertEqual(estimate.details["overflow_devices"], [1])
        self.assertIn("device:1:memory_overflow", estimate.ood_reasons)

    def test_search_candidate_bridge_preserves_prediction_metadata(self):
        plan = _plan()
        estimate = StructuralCostModel().predict(_graph(), plan, _FixedProfile())
        candidate = estimate.as_search_candidate(1, plan)
        self.assertEqual(candidate.rank, 1)
        self.assertEqual(candidate.predicted_makespan, estimate.makespan)
        self.assertEqual(candidate.predicted_peak_memory, estimate.peak_memory)
        self.assertEqual(candidate.extra["cost_model"], "structural-v1")

    def test_illegal_plan_is_rejected_before_costing(self):
        plan = _plan()
        plan.groups[0].extra.pop("pattern_id")
        with self.assertRaisesRegex(ValueError, "missing extra.pattern_id"):
            StructuralCostModel().predict(_graph(), plan, _FixedProfile())

    def test_negative_workspace_is_rejected_before_scheduling(self):
        plan = _plan(workspace=-1.0)
        with self.assertRaisesRegex(ValueError, "workspace_bytes"):
            StructuralCostModel().predict(_graph(), plan, _FixedProfile())


if __name__ == "__main__":
    unittest.main()
