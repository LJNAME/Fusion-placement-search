import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from calibration.e2_dataset import make_dataset_view
from calibration.e2_records import load_jsonl_records
from cost_model.e2_profiling_cost import ProfilingCostPredictor, time_budget_summary
from cost_model.e2_srop_proxy import (_oracle_profile, _run_policy_trajectory,
    build_workload_graph, object_map_for_workload, WORKLOADS)
from cost_model.batched_placement_ranker import BatchedPlacementRanker
from cost_model.e2_diagnostics import measurement_pool
from simulator import make_reference_plan, default_fusion_registry
from simulator.hardware import load_hardware


def row(key, family, wall=None):
    result = {"object_id": key, "analytic": {"estimate_s": .001},
        "features": {"categorical": {"operator": family}}, "label_access": {"visible": wall is not None}}
    if wall is not None:
        result["measurement"] = {"profiling_wall_time_s": wall}
    return result


class ProfilingCostTests(unittest.TestCase):
    def test_family_global_and_empty_fallbacks(self):
        predictor = ProfilingCostPredictor().fit([row("a", "matmul", 1), row("b", "matmul", 3),
                                                  row("c", "rope", 9), row("d", "softmax")])
        self.assertEqual(predictor.predict(row("x", "matmul")), 2)
        self.assertEqual(predictor.predict(row("x", "rope")), 9)
        self.assertEqual(predictor.predict(row("x", "new_family")), 3)
        self.assertEqual(predictor.predict(row("x", "matmul"), "analytic_repetitions"), .36)
        self.assertEqual(ProfilingCostPredictor().fit([row("x", "new")]).predict(row("q", "new")), .36)

    def test_hidden_labels_rejected_and_duration_target_is_unused(self):
        hidden = row("a", "rope", 1)
        hidden["label_access"]["visible"] = False
        with self.assertRaisesRegex(ValueError, "visible"):
            ProfilingCostPredictor().fit([hidden])
        public = row("a", "rope", 1)
        first = ProfilingCostPredictor().fit([public])
        public["target"] = {"log_residual": 1e100}
        public["measurement"]["duration_s"] = 1e100
        second = ProfilingCostPredictor().fit([public])
        self.assertEqual(first.predict(row("x", "rope")), second.predict(row("x", "rope")))

    def test_invalid_labels_and_duplicate_objects_fail(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "positive"):
                ProfilingCostPredictor().fit([row("a", "rope", value)])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ProfilingCostPredictor().fit([row("a", "rope", 1), row("a", "rope", 2)])

    def test_time_auc_waits_for_observation_and_holds_stopped_decision(self):
        trace = [{"budget": 0, "profiling_wall_time_s": 0, "top1_regret_ms": 10},
                 {"budget": 1, "profiling_wall_time_s": 2, "top1_regret_ms": 0}]
        summary = time_budget_summary(trace, 10)
        self.assertEqual(summary["step_regret_time_auc_ms"], 2)
        self.assertEqual(next(p for p in summary["curve"] if p["cost_fraction"] == .1)["top1_regret_ms"], 10)
        self.assertEqual(next(p for p in summary["curve"] if p["cost_fraction"] == 1)["top1_regret_ms"], 0)
        trace[-1]["top1_regret_ms"] = 5
        self.assertEqual(time_budget_summary(trace, 10)["step_regret_time_auc_ms"], 6)
        with self.assertRaisesRegex(ValueError, "excludes"):
            time_budget_summary(trace, 1)

    def test_hidden_wall_costs_do_not_change_acquisition_before_their_reveal(self):
        root = Path(__file__).resolve().parents[2]
        records = load_jsonl_records(root / "data/bw1000_e2/profiling_dataset.jsonl")
        runtime = json.loads((root / "data/bw1000_e2/split_contract.json").read_text(encoding="utf-8"))
        runtime["active_profiling_protocol"]["profiling_cost_variant"] = "family_wall_median"
        split = runtime["splits"][0]
        hardware = load_hardware(root / "data/bw1000_2card/bw1000_2card.calibrated.json")
        workload = WORKLOADS["model_3b_s256"]
        graph = build_workload_graph(workload, "layer")
        mapping = object_map_for_workload(records, workload)
        pool = measurement_pool(graph, mapping, split["train_object_ids"])
        plans = [make_reference_plan(graph)]
        ranker = BatchedPlacementRanker(graph, plans, 2, ((0, 1), (1, 0)))
        oracle = ranker.rank(_oracle_profile(records, hardware, mapping))

        def run(dataset):
            with patch("cost_model.e2_refit_acquisition.expected_refit_losses",
                       side_effect=lambda profile, uncertainty, public, remaining, *args: {key: 1 for key in remaining}):
                return _run_policy_trajectory("srop_refit", graph, plans, dataset, runtime, hardware,
                    mapping, pool, oracle, split_id=split["split_id"], random_seed=None,
                    registry=default_fusion_registry(), exchange_permutations=((0, 1), (1, 0)), ranker=ranker)

        before = run(records)
        altered = copy.deepcopy(records)
        first_reveal = before[1]["selected_object_ids"][0]
        for item in altered:
            if item["object_id"] not in set(split["train_object_ids"]) | {first_reveal}:
                item["measurement"]["profiling_wall_time_s"] *= 1e6
        after = run(altered)
        for index in (0, 1):
            for field in ("next_selection", "next_selection_score", "next_cost_proxy_s"):
                self.assertEqual(before[index][field], after[index][field])
        # Both acquisition and the reported next cost use the freshly fitted public model.
        for point in before[:-1]:
            public = make_dataset_view(records, runtime, split_id=split["split_id"],
                mode="active_calibration", role="sampler", revealed_object_ids=point["selected_object_ids"])
            predictor = ProfilingCostPredictor().fit(public)
            selected = next(item for item in public if item["object_id"] == point["next_selection"])
            expected = predictor.predict(selected)
            self.assertEqual(point["next_cost_proxy_s"], expected)
            self.assertEqual(point["next_selection_score"], 1/expected)


if __name__ == "__main__":
    unittest.main()
