from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from calibration.e2_records import load_jsonl_records
from cost_model.e2_srop_proxy import (
    WORKLOADS,
    build_workload_graph,
    evaluate_fixed_candidate_srop,
    object_map_for_workload,
)
from simulator import make_reference_plan
from simulator.hardware import load_hardware


class E2SropProxyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[2]
        cls.records = load_jsonl_records(
            root / "data" / "bw1000_e2" / "profiling_dataset.jsonl"
        )
        cls.contract = json.loads(
            (root / "data" / "bw1000_e2" / "split_contract.json").read_text(
                encoding="utf-8"
            )
        )
        cls.hardware = load_hardware(
            root / "data" / "bw1000_2card" / "bw1000_2card.calibrated.json"
        )
        cls.workload = WORKLOADS["model_3b_s256"]
        cls.graph = build_workload_graph(cls.workload, "layer")
        reference = make_reference_plan(cls.graph, device=0)
        split = copy.deepcopy(reference)
        split.name = "two-device-split"
        midpoint = len(split.groups) // 2
        for group in split.groups[midpoint:]:
            group.device = 1
        cls.plans = [reference, split]

    def test_workload_mapping_covers_every_timed_layer_operator(self) -> None:
        mapping = object_map_for_workload(self.records, self.workload)
        mapped = 0
        for node in self.graph.nodes.values():
            local_name = str(node.extra.get("local_name", node.id))
            if node.op_type != "input":
                self.assertIn(local_name, mapping)
                mapped += 1
        self.assertEqual(mapped, len(self.graph.nodes) - 1)

    def test_full_budget_recovers_oracle_for_every_policy(self) -> None:
        report = evaluate_fixed_candidate_srop(
            self.graph,
            self.plans,
            self.records,
            self.contract,
            self.hardware,
            self.workload,
            random_seeds=(0,),
        )
        for policy in report["policies"].values():
            full = policy["by_budget"][-1]
            self.assertAlmostEqual(full["top1_regret_ms_mean"], 0.0)
            self.assertAlmostEqual(full["top_k_recall_mean"], 1.0)

    def test_hidden_labels_do_not_change_first_selection(self) -> None:
        original = evaluate_fixed_candidate_srop(
            self.graph,
            self.plans,
            self.records,
            self.contract,
            self.hardware,
            self.workload,
            random_seeds=(0,),
        )
        hidden_ids = set(self.contract["splits"][0]["hidden_object_ids"])
        tampered = copy.deepcopy(self.records)
        for record in tampered:
            if record["object_id"] in hidden_ids:
                record["measurement"]["duration_s"] *= 100.0
                record["target"]["log_residual"] += 10.0
        changed = evaluate_fixed_candidate_srop(
            self.graph,
            self.plans,
            tampered,
            self.contract,
            self.hardware,
            self.workload,
            random_seeds=(0,),
        )
        for policy in self.contract["active_profiling_protocol"]["policies"]:
            original_first = original["policies"][policy]["runs"][0]["next_selection"]
            changed_first = changed["policies"][policy]["runs"][0]["next_selection"]
            self.assertEqual(original_first, changed_first)


if __name__ == "__main__":
    unittest.main()
