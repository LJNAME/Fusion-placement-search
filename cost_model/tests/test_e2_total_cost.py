import copy
import json
from pathlib import Path
import unittest

from calibration.e2_dataset import make_dataset_view
from calibration.e2_records import load_jsonl_records
from cost_model.batched_placement_ranker import BatchedPlacementRanker
from cost_model.e2_srop_proxy import _active_profile, _oracle_profile, _run_policy_trajectory, build_workload_graph, object_map_for_workload, WORKLOADS
from cost_model.e2_diagnostics import measurement_pool
from cost_model.e2_total_cost import decision_timeline, cost_summary, task_summary
from cost_model.e2_total_cost_evaluation import semantic_trace
from simulator import default_fusion_registry, make_reference_plan
from simulator.hardware import load_hardware


class TotalCostTests(unittest.TestCase):
    def test_new_decision_waits_for_refit_and_old_decision_survives_acquisition(self):
        trace = [dict(budget=0, decision_ready_algorithm_wall_time_s=1.0, algorithm_wall_time_s=3.0,
                      profiling_wall_time_s=0.0, top1_regret_ms=4.0),
                 dict(budget=1, decision_ready_algorithm_wall_time_s=4.0, algorithm_wall_time_s=5.0,
                      profiling_wall_time_s=2.0, top1_regret_ms=0.0)]
        timeline = decision_timeline(trace, 1.0, 10.0)
        # Initial plan at t=2; acquisition to t=4; profiling to t=6; refit to t=7.
        self.assertEqual([row["time_s"] for row in timeline["events"]], [0, 2, 7])
        self.assertEqual(timeline["observation_completion_times_s"], [6])
        curve = cost_summary(timeline)["curve"]
        at4 = next(row for row in curve if row["budget_s"] == 4)
        self.assertEqual(at4["step_auc_ms"], 7.0)  # (2*10 + 2*4)/4
        self.assertEqual(at4["regret_ms"], 4.0)
        at8 = next(row for row in curve if row["budget_s"] == 8)
        self.assertEqual(at8["step_auc_ms"], 5.0)  # (2*10 + 5*4)/8
        self.assertEqual(at8["completed_measurements"], 1)
        self.assertEqual(timeline["completion_s"], 8)

    def test_task_aggregation_keeps_policies_separate_and_zero_budget_controls(self):
        def run(policy, regret):
            return {"policy": policy, "preparation_s": .1, "trace": [dict(budget=0,
                decision_ready_algorithm_wall_time_s=.1, algorithm_wall_time_s=.2,
                profiling_wall_time_s=0, top1_regret_ms=regret, stop_recommended=True)]}
        result = task_summary([run("random", 0), run("random", 2), run("srop_refit", 5)], 10)
        self.assertEqual(result["forced"]["random"]["final_regret_ms"], 1)
        self.assertEqual(result["forced"]["srop_refit"]["final_regret_ms"], 5)
        self.assertEqual(result["forced"]["reference_only"]["completion_s"], 0)
        self.assertEqual(result["forced"]["reference_only"]["curve"][0]["regret_ms"], 10)
        # Identical initial decisions must not gain from timing noise in shared initialization.
        initial, sampler = run("initial_only", 0), run("srop_refit", 0)
        initial["repeat"] = sampler["repeat"] = 0
        sampler["trace"][0].update(decision_ready_algorithm_wall_time_s=.8, algorithm_wall_time_s=.9)
        aligned = task_summary([initial, sampler], 10)["forced"]
        self.assertEqual(aligned["initial_only"]["curve"], aligned["srop_refit"]["curve"])

    def test_recovery_time_does_not_treat_a_temporary_zero_as_success(self):
        timeline = dict(events=[dict(time_s=0, regret_ms=0, published_measurements=0),
            dict(time_s=1, regret_ms=5, published_measurements=0),
            dict(time_s=2, regret_ms=0, published_measurements=1)], observation_completion_times_s=[1.5],
            completion_s=3, final_regret_ms=0)
        self.assertEqual(cost_summary(timeline)["stable_zero_time_s"], 2)

    def test_batched_baselines_match_complete_scalar_trajectories_and_hidden_labels_stay_masked(self):
        root = Path(__file__).resolve().parents[2]
        records = load_jsonl_records(root/"data/bw1000_e2/profiling_dataset.jsonl")
        contract = json.loads((root/"data/bw1000_e2/split_contract.json").read_text())
        contract["active_profiling_protocol"]["uncertainty_variant"] = "distance_floor"
        workload = WORKLOADS["model_3b_s256"]
        split_id = workload.split_id
        hardware = load_hardware(root/"data/bw1000_2card/bw1000_2card.calibrated.json")
        graph = build_workload_graph(workload, "layer")
        mapping = object_map_for_workload(records, workload)
        train = next(row["train_object_ids"] for row in contract["splits"] if row["split_id"] == split_id)
        pool = measurement_pool(graph, mapping, train)
        reference = make_reference_plan(graph)
        split = copy.deepcopy(reference)
        for group in split.groups[len(split.groups)//2:]:
            group.device = 1
        plans = [reference, split]
        ranker = BatchedPlacementRanker(graph, plans, 2, ((0, 1), (1, 0)))
        oracle = ranker.rank(_oracle_profile(records, hardware, mapping))
        for policy in ("impact_only", "srop_proxy"):
            trajectories = []
            for batch in (False, True):
                runtime = copy.deepcopy(contract)
                runtime["active_profiling_protocol"]["counterfactual_batch_scenarios"] = batch
                trajectories.append(_run_policy_trajectory(policy, graph, plans, records, runtime, hardware,
                    mapping, pool, oracle, split_id=split_id, random_seed=None, registry=default_fusion_registry(),
                    exchange_permutations=((0, 1), (1, 0)), ranker=ranker, capture_decision_timing=True))
            self.assertEqual(semantic_trace(trajectories[0]), semantic_trace(trajectories[1]))
            altered = copy.deepcopy(records)
            for row in altered:
                if row["object_id"] not in train:
                    row["measurement"]["profiling_wall_time_s"] *= 1000
                    row["measurement"]["duration_s"] *= 100
                    row["target"]["log_residual"] += 10
            changed = _run_policy_trajectory(policy, graph, plans, altered, runtime, hardware, mapping,
                pool, oracle, split_id=split_id, random_seed=None, registry=default_fusion_registry(),
                exchange_permutations=((0, 1), (1, 0)), ranker=ranker, capture_decision_timing=True)
            for key in ("next_selection", "next_selection_score", "next_cost_proxy_s"):
                self.assertEqual(trajectories[1][0][key], changed[0][key])


if __name__ == "__main__":
    unittest.main()
