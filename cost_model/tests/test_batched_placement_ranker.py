import copy
import json
from pathlib import Path
import unittest

from calibration.e2_records import load_jsonl_records
from cost_model.batched_placement_ranker import BatchedPlacementRanker
from cost_model.e2_audit import PreparedCandidateRanker
from cost_model.e2_srop_proxy import (WORKLOADS, ObjectCounterfactualProfile, _oracle_profile,
    build_workload_graph, object_map_for_workload, evaluate_fixed_candidate_srop)
from cost_model.placement_probes import graph_wide_candidates
from cost_model.tests.test_e2_audit import ForkProfile
from simulator import OpGraph, OpNode, TensorEdge, make_reference_plan
from simulator.hardware import load_hardware


class BatchedRankerTests(unittest.TestCase):
    def test_resource_recurrence_matches_for_directed_and_shared_links(self):
        graph = OpGraph()
        for key in ("a", "b", "c", "d"):
            graph.add_node(OpNode(key, "add", 1, 64, 16))
        for src, dst in (("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")):
            graph.add_edge(TensorEdge(src, dst, 64))
        plans = []
        base = make_reference_plan(graph)
        for bits in range(16):
            plan = copy.deepcopy(base)
            for index, group in enumerate(plan.groups):
                group.device = (bits >> index) & 1
            plans.append(plan)
        exchange = ((0, 1),)
        reference = PreparedCandidateRanker(graph, plans, 2, exchange)
        batched = BatchedPlacementRanker(graph, plans, 2, exchange)
        for slow in ("b", "c"):
            for overlap in (False, True):
                for interference in (0, 0.25):
                    for shared in (True, False):
                        profile = ForkProfile(slow, overlap, interference)
                        if not shared:
                            profile.link_key = lambda src, dst: (src, dst)
                        expected, actual = reference.rank(profile), batched.rank(profile)
                        self.assertEqual([(row.plan_id, row.makespan, row.confidence) for row in expected],
                                         [(row.plan_id, row.makespan, row.confidence) for row in actual])
                        profiles = [ForkProfile(key, overlap, interference) for key in ("b", "c")]
                        if not shared:
                            for item in profiles:
                                item.link_key = lambda src, dst: (src, dst)
                        # More than one chunk exercises scenario indexing as well as links.
                        profiles *= 18
                        reference_rows = [batched.rank(item) for item in profiles]
                        batch_rows = batched.rank_many(profiles)
                        self.assertEqual([[(r.plan_id, r.makespan, r.confidence) for r in rows] for rows in reference_rows],
                                         [[(r.plan_id, r.makespan, r.confidence) for r in rows] for rows in batch_rows])

    def test_changing_transfer_behavior_falls_back_without_changing_rankings(self):
        graph = OpGraph()
        for key in ("a", "b", "c", "d"):
            graph.add_node(OpNode(key, "add", 1, 64, 16))
        for src, dst in (("a", "b"), ("b", "c"), ("c", "d")):
            graph.add_edge(TensorEdge(src, dst, 64))
        plan = make_reference_plan(graph)
        for index, group in enumerate(plan.groups):
            group.device = index % 2
        ranker = BatchedPlacementRanker(graph, [plan], 2, ((0, 1),))
        for change in ("link", "overlap", "duration"):
            profiles = [ForkProfile("b", False, 0), ForkProfile("c", False, 0)]
            if change == "link":
                profiles[1].link_key = lambda src, dst: (src, dst)
            elif change == "overlap":
                profiles[1] = ForkProfile("c", True, 0)
            else:
                from dataclasses import replace
                original = profiles[1].estimate_transfer
                profiles[1].estimate_transfer = lambda *args: replace(original(*args), duration=123.0)
            expected = [ranker.rank(item) for item in profiles]
            actual = ranker.rank_many(profiles)
            self.assertEqual([[(r.plan_id, r.makespan, r.confidence) for r in rows] for rows in expected],
                             [[(r.plan_id, r.makespan, r.confidence) for r in rows] for rows in actual])
        self.assertEqual(ranker.rank_many([]), [])

    def test_complete_graph_and_counterfactual_rankings_match(self):
        root = Path(__file__).resolve().parents[2]
        records = load_jsonl_records(root / "data/bw1000_e2/profiling_dataset.jsonl")
        hardware = load_hardware(root / "data/bw1000_2card/bw1000_2card.calibrated.json")
        workload = WORKLOADS["model_3b_s256"]
        graph = build_workload_graph(workload, "model")
        mapping = object_map_for_workload(records, workload)
        profile = _oracle_profile(records, hardware, mapping)
        plans, _ = graph_wide_candidates(graph, [make_reference_plan(graph)], profile.device_classes)
        exchange = ((0, 1), (1, 0))
        reference = PreparedCandidateRanker(graph, plans, 2, exchange)
        batched = BatchedPlacementRanker(graph, plans, 2, exchange)
        for current in (profile, ObjectCounterfactualProfile(profile, mapping["self_attention.q_rope"], 0.2)):
            self.assertEqual([(row.plan_id, row.makespan) for row in reference.rank(current)],
                             [(row.plan_id, row.makespan) for row in batched.rank(current)])

    def test_complete_policy_trajectories_match(self):
        root = Path(__file__).resolve().parents[2]
        records = load_jsonl_records(root / "data/bw1000_e2/profiling_dataset.jsonl")
        hardware = load_hardware(root / "data/bw1000_2card/bw1000_2card.calibrated.json")
        contract = json.loads((root / "data/bw1000_e2/split_contract.json").read_text(encoding="utf-8"))
        workload = WORKLOADS["model_3b_s256"]
        graph = build_workload_graph(workload, "layer")
        plans, _ = graph_wide_candidates(graph, [make_reference_plan(graph)], ("model:bw", "model:bw"))
        reports = [evaluate_fixed_candidate_srop(graph, plans, records, contract, hardware, workload,
                    random_seeds=(0,), ranker_factory=factory)
                   for factory in (PreparedCandidateRanker, BatchedPlacementRanker)]
        for policy in reports[0]["policies"]:
            left = reports[0]["policies"][policy]["full_trajectories"]
            right = reports[1]["policies"][policy]["full_trajectories"]
            strip = lambda rows: [[{key: value for key, value in point.items() if "wall_time" not in key}
                                   for point in trajectory] for trajectory in rows]
            self.assertEqual(strip(left), strip(right))


if __name__ == "__main__":
    unittest.main()
