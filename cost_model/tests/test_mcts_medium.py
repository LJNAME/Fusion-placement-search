"""Exact references, boundary extraction and memory diagnostics stay distinct."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cost_model.hardware_profile import AnalyticHardwareProfile
from cost_model.search_evaluation import _modes, build_evaluation_graph
from simulator.dataset import graph_to_dict
from simulator.hardware import hardware_to_dict, load_hardware
from tools.evaluate_mcts_medium import (
    HARDWARE, SELECTIONS, extract_graph, final_requests, prepare, run, run_one, verify,
)


class TestMCTSMedium(unittest.TestCase):
    def test_extraction_preserves_internal_nodes_and_edges(self):
        full = build_evaluation_graph("layer")
        for name in SELECTIONS:
            graph = extract_graph(full, name)
            self.assertTrue(8 <= len(graph.nodes) <= 10)
            for node_id in graph.extra["selected_original_nodes"]:
                self.assertEqual(graph.nodes[node_id], full.nodes[node_id])
            for node_id in graph.extra["boundary_input_nodes"]:
                node = graph.nodes[node_id]
                self.assertEqual(node.op_type, "input")
                self.assertEqual(node.flops + node.param_bytes, 0.)
                self.assertEqual(node.out_bytes, full.nodes[node_id].out_bytes)
            for edge in graph.edges:
                self.assertIn(edge, full.edges)

    def test_capacity_variants_change_no_timing_constants(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "audit"
            plan = prepare(root)
            baseline = hardware_to_dict(load_hardware(HARDWARE))
            self.assertEqual(len(plan["tasks"]), 5)
            for task in plan["tasks"]:
                hw = json.loads((root / task["hardware"]).read_text())
                for i, device in enumerate(hw["devices"]):
                    device["max_mem"] = baseline["devices"][i]["max_mem"]
                self.assertEqual(hw, baseline)
            mlp = next(t for t in plan["tasks"] if t["task_id"] == "mlp_parameter_cap_90")
            self.assertLess(mlp["max_memory_bytes"][0], mlp["parameter_sum_bytes"])

    def test_overflow_does_not_silently_become_a_valid_result(self):
        graph = extract_graph(build_evaluation_graph("layer"), "mlp")
        hw = copy.deepcopy(load_hardware(HARDWARE))
        for device in hw.devices:
            device.max_mem = 1.
        mode = next(m for m in _modes(2) if m.name == "joint")
        result, _, _ = run_one(graph, AnalyticHardwareProfile(hw), mode, "mcts", 2, 0)
        self.assertFalse(result["metrics"]["selected_feasible"])
        self.assertIsNone(result["metrics"]["best_feasible_scored_cost_s"])
        self.assertGreater(result["metrics"]["selected_cost_s"], 0.)
        self.assertEqual(final_requests({"stages": [{"scorer_requests": 2}, {"scorer_requests": 3}]},
                                        [0, 1, 2, 3, 4]), [2, 3, 4])

    def test_protocol_verifier_and_missing_feasibility_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "audit"
            plan = prepare(root)
            plan["tasks"] = [t for t in plan["tasks"] if t["task_id"] in
                             ("mlp_actual_capacity", "mlp_parameter_cap_90")]
            plan.update(seeds=[0], budgets=[2], expected_exact_runs=8, expected_mcts_runs=8)
            (root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
            with patch("cost_model.search_evaluation.legal_beam_search", side_effect=AssertionError("Beam called")):
                report = run(root)
            self.assertEqual(report["mcts_runs"], 8)
            self.assertTrue(verify(root)["pass_checks"])
            refs = json.loads((root / "references.json").read_text())
            self.assertIsNone(refs["mlp_parameter_cap_90"]["fusion_only"]["cost_s"])
            self.assertIsNotNone(refs["mlp_parameter_cap_90"]["joint"]["cost_s"])
            with (root / "results.csv").open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            invalid = [r for r in rows if r["selected_feasible"] == "False"]
            self.assertTrue(invalid)
            self.assertTrue(all(r["selected_mode_reference_gap"] == "" for r in invalid))
            rows[0]["selected_mode_reference_gap"] = "123"
            with (root / "results.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, list(rows[0]), lineterminator="\n")
                writer.writeheader(); writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "reference gap mismatch"):
                verify(root)


if __name__ == "__main__":
    unittest.main()
