"""Training-only uncertainty calibration without changing NN mean predictions.

These are empirical leave-one-key-out intervals, not conformal guarantees.
Each calibration fold fits normalization and neighbors on its training rows.
"""

from __future__ import annotations

import math
import statistics

import numpy as np

from .e2_residual import NearestShapePredictor, _operator, _quantile, _target


VARIANTS = ("legacy", "strict_family", "distance_floor")
PROTOCOL = {
    "version": "training-distance-v1",
    "folds": "leave-one-complete-object-out; refit normalization inside each fold",
    "scale": "1 + nearest_distance / median_positive_fold_distance",
    "distance_reference_fallback": 1.0,
    "distance_interval": "max(strict_family_q90, q90(abs_error / scale) * query_scale)",
    "tuning": "no held-out labels or failed-case errors; coverage and family support use original protocol",
    "distance_bins": [1.0, 2.0],
    "guarantee": "empirical intervals; no exchangeability or coverage guarantee under workload shift",
}


def _distance(predictor, record):
    index = predictor._nearest_index(record)
    vector = predictor.encoder.transform([record])[0]
    return float(np.linalg.norm(predictor.matrix[index] - vector)), predictor.records[index]["object_id"]


class TrainingDistanceUncertainty:
    def fit(self, visible_records, protocol):
        self.records = list(visible_records)
        if len(self.records) < 3 or len({row["object_id"] for row in self.records}) != len(self.records):
            raise ValueError("calibration needs at least three distinct visible measurement objects")
        if any("target" not in row or not row.get("label_access", {}).get("visible", False) for row in self.records):
            raise ValueError("calibration accepts visible public training labels only")
        self.predictor = NearestShapePredictor().fit_mean(self.records, protocol)
        self.coverage = float(protocol["uncertainty_interval_coverage"])
        self.minimum = int(protocol["family_uncertainty_min_samples"])
        self.folds = []
        for index, row in enumerate(self.records):
            train = self.records[:index] + self.records[index+1:]
            predictor = NearestShapePredictor().fit_mean(train, protocol)
            distance, source = _distance(predictor, row)
            self.folds.append({
                "object_id": row["object_id"], "family": _operator(row),
                "train_object_ids": [item["object_id"] for item in train],
                "neighbor_object_id": source, "distance": distance,
                "abs_log_error": abs(_target(row) - predictor._nearest(row)),
                "encoder_mean": predictor.encoder.mean.tolist(),
                "encoder_scale": predictor.encoder.scale.tolist(),
                "family_seen_in_fold": any(_operator(item) == _operator(row) for item in train),
            })
        self.by_family = {}
        self.global_parameters = self._parameters(self.folds)
        for family in sorted({_operator(row) for row in self.records}):
            group = [fold for fold in self.folds if fold["family"] == family]
            if len(group) >= self.minimum:
                self.by_family[family] = self._parameters(group)
        return self

    def _parameters(self, folds):
        distances = [fold["distance"] for fold in folds if fold["distance"] > 1e-12]
        reference = statistics.median(distances) if distances else 1.0
        family_q = max(_quantile([fold["abs_log_error"] for fold in folds], self.coverage), 1e-6)
        normalized = [fold["abs_log_error"] / (1.0 + fold["distance"] / reference) for fold in folds]
        coefficient = max(_quantile(normalized, self.coverage), 1e-6)
        return {"distance_reference": reference, "family_q90": family_q,
                "distance_coefficient": coefficient, "folds": len(folds)}

    def describe(self, record):
        family = _operator(record)
        parameters = self.by_family.get(family, self.global_parameters)
        distance, source = _distance(self.predictor, record)
        ratio = distance / parameters["distance_reference"]
        scaled = parameters["distance_coefficient"] * (1.0 + ratio)
        return {
            "strict_family": parameters["family_q90"],
            "distance_floor": max(parameters["family_q90"], scaled),
            "nearest_distance": distance, "distance_ratio": ratio,
            "distance_bin": "near" if ratio <= 1.0 else "mid" if ratio <= 2.0 else "far",
            "neighbor_object_id": source,
            "family_seen": any(_operator(row) == family for row in self.records),
            "family_calibration": family in self.by_family,
        }

    def q90(self, record, variant):
        if variant not in {"strict_family", "distance_floor"}:
            raise ValueError(f"unknown calibrated uncertainty: {variant}")
        return self.describe(record)[variant]


def interval_metrics(rows, variant):
    if not rows:
        return {"objects": 0}
    errors = [row["abs_log_error"] for row in rows]
    widths = [row["q90"][variant] for row in rows]
    # Central 90% interval score in log-duration space. Penalize both width
    # and misses so making all intervals arbitrarily wide cannot win silently.
    scores = [2.0 * width + 20.0 * max(0.0, error - width) for error, width in zip(errors, widths)]
    return {"objects": len(rows), "coverage": statistics.fmean(error <= width for error, width in zip(errors, widths)),
            "mean_q90_log": statistics.fmean(widths), "median_q90_log": statistics.median(widths),
            "max_q90_log": max(widths), "mean_interval_score_log": statistics.fmean(scores)}
