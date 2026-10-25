from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from calibration.e2_records import load_jsonl_records
from cost_model.e2_residual import evaluate_zero_shot


class E2ResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[2]
        cls.records = load_jsonl_records(root / "data" / "bw1000_e2" / "profiling_dataset.jsonl")
        cls.contract = json.loads(
            (root / "data" / "bw1000_e2" / "split_contract.json").read_text(encoding="utf-8")
        )
        cls.split_id = cls.contract["splits"][0]["split_id"]
        cls.report = evaluate_zero_shot(cls.records, cls.contract, split_id=cls.split_id)

    def test_all_preregistered_baselines_run_on_hidden_objects(self) -> None:
        self.assertEqual(
            set(self.report["methods"]),
            {"analytic_only", "family_median", "nearest_shape", "shared_ridge"},
        )
        self.assertGreater(self.report["train_records"], 0)
        self.assertGreater(self.report["hidden_records"], 0)
        self.assertEqual(len(self.report["predictions"]), self.report["hidden_records"])
        for metrics in self.report["methods"].values():
            self.assertGreaterEqual(metrics["mape"], 0.0)
            self.assertGreaterEqual(metrics["interval_coverage"], 0.0)
            self.assertLessEqual(metrics["interval_coverage"], 1.0)

    def test_ridge_selection_stays_inside_frozen_protocol(self) -> None:
        alpha = self.report["methods"]["shared_ridge"]["selected_alpha"]
        self.assertIn(alpha, self.contract["prediction_protocol"]["ridge_alphas"])

    def test_hidden_label_changes_do_not_change_predictions(self) -> None:
        tampered = copy.deepcopy(self.records)
        hidden_ids = set(self.contract["splits"][0]["hidden_object_ids"])
        for record in tampered:
            if record["object_id"] in hidden_ids:
                record["measurement"]["duration_s"] *= 100.0
                record["target"]["log_residual"] += 10.0
        changed = evaluate_zero_shot(tampered, self.contract, split_id=self.split_id)
        original_predictions = {
            row["object_id"]: {
                method: row[method]["predicted_s"] for method in self.report["methods"]
            }
            for row in self.report["predictions"]
        }
        changed_predictions = {
            row["object_id"]: {
                method: row[method]["predicted_s"] for method in changed["methods"]
            }
            for row in changed["predictions"]
        }
        self.assertEqual(original_predictions, changed_predictions)


if __name__ == "__main__":
    unittest.main()
