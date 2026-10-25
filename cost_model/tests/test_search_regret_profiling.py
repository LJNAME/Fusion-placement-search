"""E1 search-regret-aware profiling replay tests."""

from __future__ import annotations

import math
import unittest

from cost_model import GroupCost, TransferCost
from cost_model.search_regret_profiling import (
    CalibrationReplayProfile,
    evaluate_search_regret_profiling,
)
from cost_model.search_sensitivity import shape_key_for_group
from cost_model.structure import analyze_plan_structure
from simulator import FusionGroup, OpGraph, OpNode, Plan


def _group(device: int) -> FusionGroup:
    return FusionGroup(
        ["op"],
        device,
        {
            "pattern_id": "unfused",
            "backend": "reference",
            "implementation_status": "reference",
        },
    )


class _ReplayTestProfile:
    def __init__(self, durations, source):
        self.durations = dict(durations)
        self.source = source

    @property
    def n_devs(self):
        return 2

    @property
    def overlap_send(self):
        return True

    @property
    def interference(self):
        return 0.0

    def max_memory(self, device):
        return 1e12

    def estimate_group(self, graph, group):
        duration = float(self.durations[group.device])
        return GroupCost(
            duration=duration,
            compute_time=duration,
            memory_time=0.0,
            launch_time=0.0,
            confidence=0.95 if "measured" in self.source else 0.25,
            source=self.source,
        )

    def estimate_transfer(self, src_device, dst_device, nbytes):
        return TransferCost(0.0, 1.0, "fixed_link")

    def link_key(self, src_device, dst_device):
        return tuple(sorted((int(src_device), int(dst_device))))


def _fixture():
    graph = OpGraph(name="e1-replay")
    graph.add_node(
        OpNode(
            "op",
            "rope",
            flops=1.0,
            out_bytes=2.0,
            extra={"dtype": "bf16", "output_shape": (1, 1)},
        )
    )
    plans = (Plan([_group(0)], name="device-0"), Plan([_group(1)], name="device-1"))
    oracle = _ReplayTestProfile(
        {0: 1.0e-3, 1: 4.0e-3}, "measured_exact_shape"
    )
    fallback = _ReplayTestProfile({0: 3.0e-3, 1: 2.0e-3}, "analytic_roofline")
    return graph, plans, oracle, fallback


class TestSearchRegretProfiling(unittest.TestCase):
    def test_replay_reveals_only_selected_shape(self):
        graph, plans, oracle, fallback = _fixture()
        structure = analyze_plan_structure(graph, plans[0], 2)
        key = shape_key_for_group(graph, structure.groups[0])
        replay = CalibrationReplayProfile(oracle, fallback, (key,))

        revealed = replay.estimate_group(graph, structure.groups[0])
        hidden_structure = analyze_plan_structure(graph, plans[1], 2)
        hidden = replay.estimate_group(graph, hidden_structure.groups[0])

        self.assertEqual(revealed.duration, 1.0e-3)
        self.assertIn("e1_revealed", revealed.source)
        self.assertEqual(hidden.duration, 2.0e-3)
        self.assertIn("e1_hidden", hidden.source)

    def test_budgeted_replay_reports_regret_and_recovers_oracle(self):
        graph, plans, oracle, fallback = _fixture()
        report = evaluate_search_regret_profiling(
            graph,
            plans,
            oracle,
            fallback,
            budgets=(0, 1, 2),
            random_seeds=(0, 1),
            top_k=1,
            impact_delta=0.2,
        )

        self.assertEqual(report["summary"]["profiled_shapes"], 2)
        self.assertTrue(report["summary"]["full_budget_recovers_oracle"])
        for policy in report["policies"].values():
            by_budget = {item["budget"]: item for item in policy["by_budget"]}
            self.assertAlmostEqual(by_budget[0]["top1_regret_ms_mean"], 3.0)
            self.assertAlmostEqual(by_budget[2]["top1_regret_ms_mean"], 0.0)
            self.assertEqual(by_budget[2]["top1_hit_rate"], 1.0)
            self.assertIsNotNone(policy["summary"]["first_zero_mean_regret_budget"])
            self.assertIsNotNone(
                policy["summary"]["first_full_mean_top_k_recall_budget"]
            )

        by_label = {item["device_class"]: item for item in report["profiled_shapes"]}
        self.assertAlmostEqual(
            by_label["physical:0"]["uncertainty_estimate"],
            abs(math.log(2.0 / 4.0)),
        )
        self.assertAlmostEqual(
            by_label["physical:1"]["uncertainty_estimate"],
            abs(math.log(3.0 / 1.0)),
        )


if __name__ == "__main__":
    unittest.main()
