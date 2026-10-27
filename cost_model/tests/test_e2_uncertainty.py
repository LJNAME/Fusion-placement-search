import copy
import json
import math
from pathlib import Path
import unittest

from calibration.e2_dataset import make_dataset_view
from calibration.e2_records import load_jsonl_records
from cost_model.e2_residual import NearestShapePredictor
from cost_model.e2_uncertainty import TrainingDistanceUncertainty, interval_metrics
from cost_model.e2_srop_proxy import (_active_profile, _oracle_profile, _run_policy_trajectory, _expected_decision_regret_ms,
    object_map_for_workload, build_workload_graph, WORKLOADS)
from cost_model.batched_placement_ranker import BatchedPlacementRanker
from cost_model.e2_diagnostics import measurement_pool
from simulator import make_reference_plan, default_fusion_registry
from simulator.hardware import load_hardware


class UncertaintyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[2]
        cls.records = load_jsonl_records(root / "data/bw1000_e2/profiling_dataset.jsonl")
        cls.contract = json.loads((root / "data/bw1000_e2/split_contract.json").read_text(encoding="utf-8"))
        cls.protocol = cls.contract["prediction_protocol"]
        cls.split = cls.contract["splits"][0]
        cls.public = make_dataset_view(cls.records, cls.contract, split_id=cls.split["split_id"], mode="zero_shot", role="predictor")
        cls.visible = [row for row in cls.public if "target" in row]
        cls.hardware = load_hardware(root / "data/bw1000_2card/bw1000_2card.calibrated.json")

    def test_every_fold_excludes_object_and_its_features_from_normalization(self):
        model = TrainingDistanceUncertainty().fit(self.visible, self.protocol)
        for fold in model.folds:
            self.assertNotIn(fold["object_id"], fold["train_object_ids"])
            train = [row for row in self.visible if row["object_id"] != fold["object_id"]]
            expected = NearestShapePredictor().fit_mean(train, self.protocol)
            self.assertEqual(fold["encoder_mean"], expected.encoder.mean.tolist())
            self.assertEqual(fold["encoder_scale"], expected.encoder.scale.tolist())

    def test_hidden_label_changes_leave_calibration_identical(self):
        altered = copy.deepcopy(self.records)
        for row in altered:
            if row["object_id"] in self.split["hidden_object_ids"]:
                row["target"]["log_residual"] += 100
                row["measurement"]["duration_s"] *= 100
        public = make_dataset_view(altered, self.contract, split_id=self.split["split_id"], mode="zero_shot", role="predictor")
        first = TrainingDistanceUncertainty().fit(self.visible, self.protocol)
        second = TrainingDistanceUncertainty().fit([row for row in public if "target" in row], self.protocol)
        self.assertEqual(first.folds, second.folds)
        self.assertEqual([first.describe(row) for row in self.public], [second.describe(row) for row in public])

    def test_uncertainty_variants_do_not_change_mean_durations(self):
        mapping = object_map_for_workload(self.records, WORKLOADS["model_3b_s256"])
        profiles = []
        for variant in ("legacy", "strict_family", "distance_floor"):
            runtime = copy.deepcopy(self.contract)
            runtime["active_profiling_protocol"]["uncertainty_variant"] = variant
            profile, _ = _active_profile(self.records, runtime, self.hardware, mapping,
                split_id=self.split["split_id"], revealed_object_ids=())
            profiles.append(profile.durations_s)
        self.assertEqual(profiles[0], profiles[1])
        self.assertEqual(profiles[0], profiles[2])

    def test_distance_rule_preserves_family_floor(self):
        model = TrainingDistanceUncertainty().fit(self.visible, self.protocol)
        for row in self.public:
            values = model.describe(row)
            self.assertGreaterEqual(values["distance_floor"], values["strict_family"])

    def test_interval_score_penalizes_unnecessary_width(self):
        rows = [{"abs_log_error": 0.0, "q90": {"narrow": 0.1, "wide": 10.0}}]
        self.assertEqual(interval_metrics(rows, "narrow")["coverage"], 1.0)
        self.assertEqual(interval_metrics(rows, "wide")["coverage"], 1.0)
        self.assertLess(interval_metrics(rows, "narrow")["mean_interval_score_log"],
                        interval_metrics(rows, "wide")["mean_interval_score_log"])

    def test_unrevealed_calibration_labels_rejected(self):
        with self.assertRaisesRegex(ValueError, "visible"):
            TrainingDistanceUncertainty().fit(self.public, self.protocol)

    def test_actual_stopping_matches_prefix_of_full_replay(self):
        workload = WORKLOADS["model_3b_s256"]
        graph = build_workload_graph(workload, "layer")
        mapping = object_map_for_workload(self.records, workload)
        pool = measurement_pool(graph, mapping, self.split["train_object_ids"])
        reference = make_reference_plan(graph)
        split = copy.deepcopy(reference)
        for group in split.groups[len(split.groups)//2:]:
            group.device = 1
        plans = [reference, split]
        exchange = ((0, 1), (1, 0))
        ranker = BatchedPlacementRanker(graph, plans, 2, exchange)
        oracle = ranker.rank(_oracle_profile(self.records, self.hardware, mapping))
        common = ("srop_proxy", graph, plans, self.records, self.contract, self.hardware, mapping, pool, oracle)
        kwargs = dict(split_id=self.split["split_id"], random_seed=None, registry=default_fusion_registry(),
                      exchange_permutations=exchange, ranker=ranker)
        full = _run_policy_trajectory(*common, **kwargs)
        stopped = _run_policy_trajectory(*common, **kwargs, stop_on_recommendation=True)
        index = next((index for index, point in enumerate(full) if point["stop_recommended"]), len(full)-1)
        self.assertEqual(len(stopped), index+1)
        strip = lambda rows: [{key: value for key, value in row.items() if "wall_time" not in key} for row in rows]
        self.assertEqual(strip(stopped), strip(full[:index+1]))

    def test_wide_intervals_keep_small_positive_counterfactual_multipliers(self):
        workload = WORKLOADS["model_3b_s256"]
        graph = build_workload_graph(workload, "layer")
        mapping = object_map_for_workload(self.records, workload)
        profile, _ = _active_profile(self.records, self.contract, self.hardware, mapping,
            split_id=self.split["split_id"], revealed_object_ids=())
        reference = make_reference_plan(graph)
        other = copy.deepcopy(reference)
        for group in other.groups[len(other.groups)//2:]:
            group.device = 1
        plans = [reference, other]
        ranker = BatchedPlacementRanker(graph, plans, 2, ((0, 1), (1, 0)))
        score = _expected_decision_regret_ms(graph, plans, profile, mapping["self_attention.q_proj"], 120.0,
            ranker.rank(profile), standard_normal_quantiles=(-1.2815515655, 0.0, 1.2815515655),
            q90_normalizer=1.644853627, registry=default_fusion_registry(),
            exchange_permutations=((0, 1), (1, 0)), ranker=ranker)
        self.assertTrue(math.isfinite(score))


if __name__ == "__main__":
    unittest.main()
