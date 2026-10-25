from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from calibration.e2_records import load_jsonl_records
from cost_model.e2_audit import PreparedCandidateRanker, ranking_metrics, selection_diagnostics
from cost_model.e2_srop_proxy import WORKLOADS, build_workload_graph, evaluate_fixed_candidate_srop
from cost_model.e2_task_suite import build_task_suite
from cost_model.hardware_profile import GroupCost, TransferCost
from cost_model.search_sensitivity import _RankedPlan
from cost_model.structural import StructuralCostModel
from simulator import OpGraph, OpNode, TensorEdge, make_reference_plan
from simulator.hardware import load_hardware


class ForkProfile:
    n_devs = 2
    device_classes = ("bw", "bw")

    def __init__(self, slow, overlap_send, interference):
        self.slow = slow
        self.overlap_send = overlap_send
        self.interference = interference

    def max_memory(self, device):
        return 1e9

    def estimate_group(self, graph, group):
        duration = 20.0 if group.node_ids[0] == self.slow else 1.0
        return GroupCost(duration, duration, 0.0, 0.0, 1.0, "test")

    def estimate_transfer(self, src, dst, nbytes):
        return TransferCost(0.1, 1.0, "test")

    def link_key(self, src, dst):
        return (0, 1)


class E2AuditTests(unittest.TestCase):
    def test_cached_scheduler_preserves_memory_transfers_and_critical_switches(self):
        graph = OpGraph(name="fork")
        for node_id in ("a", "b", "c", "d"):
            graph.add_node(OpNode(node_id, "add", 1.0, 64.0, 16.0))
        for src, dst in (("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")):
            graph.add_edge(TensorEdge(src, dst, 64.0))
        plan = make_reference_plan(graph)
        for group in plan.groups:
            group.device = int(group.node_ids[0] == "c")
        model = StructuralCostModel()
        prepared = model.prepare(graph, plan, 2)
        paths = []
        for slow in ("b", "c"):
            for overlap in (False, True):
                for interference in (0.0, 0.25):
                    profile = ForkProfile(slow, overlap, interference)
                    expected = model.predict(graph, plan, profile)
                    actual = model.predict_prepared(prepared, profile)
                    self.assertEqual(expected.to_dict(), actual.to_dict())
                    paths.append(actual.critical_path)
        self.assertGreater(len(set(paths)), 1)

    def test_ties_and_best_of_k_are_distinct_metrics(self):
        plan = make_reference_plan(OpGraph())
        oracle = [_RankedPlan(key, plan, cost, (), 1.0)
                  for key, cost in (("a", 1.0), ("b", 2.0), ("c", 2.0), ("d", 3.0))]
        predicted = [_RankedPlan(key, plan, float(index + 1), (), 1.0)
                     for index, key in enumerate(("a", "c", "b", "d"))]
        metrics = ranking_metrics(oracle, predicted, top_k=2)
        self.assertEqual(metrics["strict_top_k_recall"], 0.5)
        self.assertEqual(metrics["tie_aware_top_k_recall"], 2 / 3)
        self.assertEqual(metrics["best_of_k_regret_ms"], 0.0)

    def test_zero_scores_recommend_stopping_and_record_ties(self):
        audit = selection_diagnostics({"a": 0.0, "b": 0.0}, ("a", "b"))
        self.assertTrue(audit["stop_recommended"])
        self.assertEqual(audit["score_tie_fraction"], 1.0)
        self.assertFalse(selection_diagnostics({"a": 1e-30}, ("a",))["stop_recommended"])


class E2AuditIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[2]
        cls.records = load_jsonl_records(root / "data/bw1000_e2/profiling_dataset.jsonl")
        cls.contract = json.loads((root / "data/bw1000_e2/split_contract.json").read_text(encoding="utf-8"))
        cls.manifest = json.loads((root / "data/bw1000_e2/shape_manifest.json").read_text(encoding="utf-8"))
        cls.hardware = load_hardware(root / "data/bw1000_2card/bw1000_2card.calibrated.json")

    def test_cached_and_reference_select_identical_trajectories(self):
        workload = WORKLOADS["model_3b_s256"]
        graph = build_workload_graph(workload, "layer")
        base = make_reference_plan(graph)
        split = copy.deepcopy(base)
        for group in split.groups[len(split.groups)//2:]:
            group.device = 1
        common = (graph, [base, split], self.records, self.contract, self.hardware, workload)
        expected = evaluate_fixed_candidate_srop(*common, random_seeds=(0,), use_prepared=False)
        actual = evaluate_fixed_candidate_srop(*common, random_seeds=(0,), use_prepared=True)
        for policy in expected["policies"]:
            left = expected["policies"][policy]["full_trajectories"][0]
            right = actual["policies"][policy]["full_trajectories"][0]
            self.assertEqual(len(left), len(right))
            for lrow, rrow in zip(left, right):
                for key in lrow:
                    if "wall_time" not in key:
                        self.assertEqual(lrow[key], rrow[key], (policy, key))
        self.assertTrue(actual["excluded_non_graph_hidden_objects"])
        self.assertEqual(actual["measurement_pool_objects"], 10)

    def test_suite_uses_label_independent_nested_subsets(self):
        suite = build_task_suite(self.manifest, self.contract)
        changed = copy.deepcopy(self.manifest)
        for item in changed["objects"]:
            item["ignored_hidden_label"] = 999999
        self.assertEqual(suite, build_task_suite(changed, self.contract))
        self.assertEqual(len(suite["tasks"]), 84)
        groups = {}
        for task in suite["tasks"]:
            train = set(task["train_object_ids"])
            self.assertFalse(train & set(task["hidden_object_ids"]))
            if task["subset_seed"] is not None:
                key = task["split_id"], task["model_id"], task["scale"], task["subset_seed"]
                groups.setdefault(key, {})[task["initial_train_count"]] = train
        for subsets in groups.values():
            self.assertLess(subsets[16], subsets[32])


if __name__ == "__main__":
    unittest.main()
