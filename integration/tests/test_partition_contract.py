"""Local output-consumer checks using existing frozen placed-graph artifacts."""
import json
from pathlib import Path
import unittest

from integration.partition_contract import (
    balanced_group_assignment, equal_layer_assignment, stage_metrics, validate_stage_assignment,
)
from simulator.placement_artifacts import load_placed_op_graph, placed_op_graph_artifact

ROOT = Path(__file__).resolve().parents[2]


def artifact(fused=True):
    directory = ROOT / "data/mcts_mainline_v1"
    entries = json.loads((directory / "best_plans.json").read_text()).values()
    entry = next(e for e in entries if e["scale"] == "small" and
                 any(len(g["node_ids"]) > 1 for g in e["plan"]["groups"]) == fused)
    return load_placed_op_graph(directory / entry["artifact"])


class TestPartitionContract(unittest.TestCase):
    def test_group_proposal_preserves_nodes_devices_and_input_artifact(self):
        placed = artifact(); before = placed_op_graph_artifact(placed)
        assignment = balanced_group_assignment(placed, 2)
        devices = {n: g.device for g in placed.plan.groups for n in g.node_ids}
        validate_stage_assignment(placed, assignment, 2, device_by_node=devices)
        metrics = stage_metrics(placed, assignment, 2)
        self.assertEqual(len(metrics["stage_group_work_s"]), 2)
        self.assertGreaterEqual(metrics["imbalance_ratio"], 0.)
        self.assertEqual(placed_op_graph_artifact(placed), before)

    def test_equal_layers_is_not_silently_replaced_by_operator_counts(self):
        placed = artifact()
        one = equal_layer_assignment(placed, 1)
        validate_stage_assignment(placed, one, 1)
        self.assertEqual(stage_metrics(placed, one, 1)["interstage_dependency_bytes"], 0.)
        with self.assertRaisesRegex(ValueError, "not_applicable"):
            equal_layer_assignment(placed, 2)

    def test_operator_cut_can_violate_a_fused_group(self):
        placed = artifact(); order = placed.graph.topological_order()
        assignment = {n: int(i >= len(order)//2) for i, n in enumerate(order)}
        with self.assertRaisesRegex(ValueError, "cuts fusion group"):
            validate_stage_assignment(placed, assignment, 2)
        with self.assertRaisesRegex(ValueError, "cuts fusion group"):
            stage_metrics(placed, assignment, 2)
        unfused = artifact(False)
        validate_stage_assignment(unfused, assignment, 2)

    def test_missing_or_extra_nodes_and_remapped_devices_are_rejected(self):
        placed = artifact(); assignment = balanced_group_assignment(placed, 2)
        missing = dict(assignment); missing.pop(next(iter(missing)))
        extra = dict(assignment, unknown=0)
        for proposal in (missing, extra):
            with self.assertRaisesRegex(ValueError, "exactly the original node IDs"):
                validate_stage_assignment(placed, proposal, 2)
        devices = {n: g.device for g in placed.plan.groups for n in g.node_ids}
        devices[next(iter(devices))] = 1
        with self.assertRaisesRegex(ValueError, "fixed device labels"):
            validate_stage_assignment(placed, assignment, 2, device_by_node=devices)

    def test_backward_stage_dependency_and_empty_stage_are_rejected(self):
        placed = artifact()
        assignment = {n: int(n == "L000.mlp.gate_proj") for n in placed.graph.nodes}
        with self.assertRaisesRegex(ValueError, "reverses an original dependency"):
            validate_stage_assignment(placed, assignment, 2)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            validate_stage_assignment(placed, {n: 0 for n in placed.graph.nodes}, 2)

    def test_planned_capability_cannot_pass_registry_executable_mode(self):
        placed = artifact()
        with self.assertRaisesRegex(ValueError, "executable"):
            validate_stage_assignment(placed, equal_layer_assignment(placed, 1), 1,
                                      require_registry_executable=True)
        unfused = artifact(False)
        validate_stage_assignment(unfused, equal_layer_assignment(unfused, 1), 1,
                                  require_registry_executable=True)
        self.assertFalse(unfused.extra["deployable"])


if __name__ == "__main__":
    unittest.main()
