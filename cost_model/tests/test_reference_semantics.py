import math
import unittest

from cost_model.hardware_profile import GroupCost, TransferCost
from cost_model.reference_semantics import HostSubmissionProfile, ReferenceEagerCostModel
from cost_model.structural import StructuralCostModel
from simulator import FusionGroup, OpGraph, OpNode, Plan, TensorEdge


class FixedProfile:
    n_devs = 2
    overlap_send = True
    interference = 0.0

    def max_memory(self, device):
        return 1e9

    def estimate_group(self, graph, group):
        duration = {"input": 99, "a": 10, "b": 4, "join": 1}[group.node_ids[0]]
        return GroupCost(duration, duration, 0, 0, 1, "fixture")

    def estimate_transfer(self, src, dst, nbytes):
        return TransferCost(1, 1, "fixture")

    def link_key(self, src, dst):
        return (src, dst)


def fixture(order=("input", "a", "b", "join")):
    graph = OpGraph(name="two-branches")
    for name, operator in (("input", "input"), ("a", "rmsnorm"), ("b", "matmul"), ("join", "add")):
        graph.add_node(OpNode(name, operator, 0, 4, 0, {"output_shape": [1], "dtype": "fp32"}))
    for src, dst in (("input", "a"), ("input", "b"), ("a", "join"), ("b", "join")):
        graph.add_edge(TensorEdge(src, dst, 4))
    plan = Plan([FusionGroup([name], 1 if name == "b" else 0,
        {"pattern_id": "unfused", "backend": "reference"}) for name in order])
    return graph, plan


class ReferenceSemanticsTests(unittest.TestCase):
    def test_copy_is_delayed_by_already_queued_source_work(self):
        graph, plan = fixture()
        model = ReferenceEagerCostModel()
        estimate = model.predict(graph, plan, FixedProfile())
        first, second = estimate.details["transfers"]
        self.assertEqual((first["start"], first["end"]), (10, 11))
        self.assertEqual((second["start"], second["end"]), (15, 16))
        self.assertEqual(estimate.makespan, 17)
        self.assertEqual(estimate.computation_time, 15)
        self.assertEqual(estimate.cross_device_bytes, 8)

    def test_early_submission_allows_parallel_branches_and_copy_waits_for_destination(self):
        graph, plan = fixture(("input", "b", "a", "join"))
        estimate = ReferenceEagerCostModel().predict(graph, plan, FixedProfile())
        groups = {row["node_id"]: row for row in estimate.details["groups"]}
        self.assertEqual((groups["a"]["start"], groups["b"]["start"]), (1, 1))
        self.assertEqual(estimate.details["transfers"][1]["start"], 11)
        self.assertEqual(estimate.makespan, 13)
        self.assertLess(estimate.makespan, estimate.computation_time + estimate.communication_time)

    def test_host_submission_can_remove_overlap_without_serializing_all_devices(self):
        graph, plan = fixture(("input", "b", "a", "join"))
        host = HostSubmissionProfile(operator_submit_s={"matmul": 6}, calibrated=True, source="fixture")
        estimate = ReferenceEagerCostModel(host).predict(graph, plan, FixedProfile())
        groups = {row["node_id"]: row for row in estimate.details["groups"]}
        self.assertEqual(groups["b"]["start"], 1)
        self.assertEqual(groups["a"]["start"], 6)
        self.assertEqual(estimate.makespan, 18)

    def test_cached_input_has_no_gpu_service_and_missing_host_cost_is_explicit(self):
        graph, plan = fixture()
        estimate = ReferenceEagerCostModel().predict(graph, plan, FixedProfile())
        self.assertEqual(estimate.details["groups"][0]["duration"], 0)
        self.assertIn("reference:host_submission_uncalibrated", estimate.ood_reasons)
        self.assertTrue(estimate.out_of_distribution)

    def test_full_tensor_bytes_override_partial_edge_metadata(self):
        graph, plan = fixture()
        graph.edges[1].bytes = 1

        class CheckInputProfile(FixedProfile):
            def estimate_group(inner, graph, group):
                if group.node_ids == ("b",):
                    self.assertEqual(group.input_bytes, 4)
                return super().estimate_group(graph, group)

        estimate = ReferenceEagerCostModel().predict(graph, plan, CheckInputProfile())
        self.assertEqual(estimate.details["transfers"][0]["bytes"], 4)
        self.assertEqual(estimate.cross_device_bytes, 8)

    def test_repeated_edges_issue_repeated_copies(self):
        graph, plan = fixture()
        graph.add_edge(TensorEdge("input", "b", 4))
        estimate = ReferenceEagerCostModel().predict(graph, plan, FixedProfile())
        self.assertEqual(len(estimate.details["transfers"]), 3)
        self.assertEqual(estimate.cross_device_bytes, 12)

    def test_original_dag_prediction_is_unchanged(self):
        graph, plan = fixture()
        before = StructuralCostModel().predict(graph, plan, FixedProfile())
        ReferenceEagerCostModel().predict(graph, plan, FixedProfile())
        after = StructuralCostModel().predict(graph, plan, FixedProfile())
        self.assertEqual(before.to_dict(), after.to_dict())

    def test_rejects_invalid_host_cost_and_non_singleton_groups(self):
        for value in (-1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                HostSubmissionProfile(copy_submit_s=value)
        graph, plan = fixture()
        plan.groups[0].node_ids.append("a")
        with self.assertRaisesRegex(ValueError, "singleton"):
            ReferenceEagerCostModel().predict(graph, plan, FixedProfile())


if __name__ == "__main__":
    unittest.main()
