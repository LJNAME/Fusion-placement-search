"""Strict zero-shot residual baselines for E2 GPU performance prediction."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from calibration.e2_dataset import make_dataset_view
from calibration.e2_records import load_jsonl_records


def _target(record: Mapping[str, Any]) -> float:
    return float(record["target"]["log_residual"])


def _operator(record: Mapping[str, Any]) -> str:
    return str(record["features"]["categorical"]["operator"])


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability, method="linear"))


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 2:
        return 0.0
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    if float(np.std(left_ranks)) == 0.0 or float(np.std(right_ranks)) == 0.0:
        return 0.0
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


@dataclass(frozen=True)
class ResidualPrediction:
    mean: float
    q90: float


class _UncertaintyModel:
    def __init__(self, errors: Sequence[tuple[str, float]], *, min_family_samples: int, coverage: float):
        absolute = [abs(float(error)) for _, error in errors]
        self.global_q = max(_quantile(absolute, coverage), 1e-6)
        self.by_family: dict[str, float] = {}
        for family in sorted({family for family, _ in errors}):
            values = [abs(float(error)) for key, error in errors if key == family]
            if len(values) >= min_family_samples:
                self.by_family[family] = max(_quantile(values, coverage), 1e-6)

    def q90(self, record: Mapping[str, Any]) -> float:
        return self.by_family.get(_operator(record), self.global_q)


class AnalyticOnlyPredictor:
    def fit(self, records: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> "AnalyticOnlyPredictor":
        errors = [(_operator(record), _target(record)) for record in records]
        self.uncertainty = _UncertaintyModel(
            errors,
            min_family_samples=int(protocol["family_uncertainty_min_samples"]),
            coverage=float(protocol["uncertainty_interval_coverage"]),
        )
        return self

    def predict(self, record: Mapping[str, Any]) -> ResidualPrediction:
        return ResidualPrediction(0.0, self.uncertainty.q90(record))


class FamilyMedianPredictor:
    def _fit_medians(self, records: Sequence[Mapping[str, Any]]) -> tuple[float, dict[str, float]]:
        global_median = float(statistics.median(_target(record) for record in records))
        families = {
            family: float(statistics.median(_target(record) for record in records if _operator(record) == family))
            for family in sorted({_operator(record) for record in records})
        }
        return global_median, families

    def fit(self, records: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> "FamilyMedianPredictor":
        self.global_median, self.families = self._fit_medians(records)
        loo_errors: list[tuple[str, float]] = []
        for index, record in enumerate(records):
            remaining = list(records[:index]) + list(records[index + 1 :])
            global_median, families = self._fit_medians(remaining)
            prediction = families.get(_operator(record), global_median)
            loo_errors.append((_operator(record), _target(record) - prediction))
        self.uncertainty = _UncertaintyModel(
            loo_errors,
            min_family_samples=int(protocol["family_uncertainty_min_samples"]),
            coverage=float(protocol["uncertainty_interval_coverage"]),
        )
        return self

    def predict(self, record: Mapping[str, Any]) -> ResidualPrediction:
        mean = self.families.get(_operator(record), self.global_median)
        return ResidualPrediction(mean, self.uncertainty.q90(record))


class _NumericEncoder:
    def __init__(self, names: Sequence[str]):
        self.names = tuple(names)

    @staticmethod
    def _raw(record: Mapping[str, Any], name: str) -> float:
        value = float(record["features"]["numeric"][name])
        if value < 0.0:
            raise ValueError(f"E2 numeric feature must be non-negative: {name}")
        return math.log1p(value)

    def fit(self, records: Sequence[Mapping[str, Any]]) -> "_NumericEncoder":
        matrix = np.asarray(
            [[self._raw(record, name) for name in self.names] for record in records],
            dtype=np.float64,
        )
        self.mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        self.scale = np.where(scale > 1e-12, scale, 1.0)
        return self

    def transform(self, records: Sequence[Mapping[str, Any]]) -> np.ndarray:
        matrix = np.asarray(
            [[self._raw(record, name) for name in self.names] for record in records],
            dtype=np.float64,
        )
        return (matrix - self.mean) / self.scale


class NearestShapePredictor:
    def fit_mean(self, records: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> "NearestShapePredictor":
        """Fit the mean response only; no uncertainty is constructed here."""
        self.records = list(records)
        self.encoder = _NumericEncoder(protocol["numeric_features"]).fit(records)
        self.matrix = self.encoder.transform(records)
        return self

    def fit(self, records: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> "NearestShapePredictor":
        self.fit_mean(records, protocol)
        loo_errors: list[tuple[str, float]] = []
        for index, record in enumerate(records):
            prediction = self._nearest(record, exclude_index=index)
            loo_errors.append((_operator(record), _target(record) - prediction))
        self.uncertainty = _UncertaintyModel(
            loo_errors,
            min_family_samples=int(protocol["family_uncertainty_min_samples"]),
            coverage=float(protocol["uncertainty_interval_coverage"]),
        )
        return self

    def _nearest_index(self, record: Mapping[str, Any], *, exclude_index: int | None = None) -> int:
        vector = self.encoder.transform([record])[0]
        candidates = [
            index
            for index, candidate in enumerate(self.records)
            if index != exclude_index and _operator(candidate) == _operator(record)
        ]
        if not candidates:
            candidates = [index for index in range(len(self.records)) if index != exclude_index]
        if not candidates:
            raise ValueError("nearest-shape predictor requires at least two training records")
        return min(candidates, key=lambda item: (float(np.linalg.norm(self.matrix[item] - vector)), self.records[item]["object_id"]))

    def _nearest(self, record: Mapping[str, Any], *, exclude_index: int | None = None) -> float:
        return _target(self.records[self._nearest_index(record, exclude_index=exclude_index)])

    def predict(self, record: Mapping[str, Any]) -> ResidualPrediction:
        return ResidualPrediction(self._nearest(record), self.uncertainty.q90(record))


class _RidgeEncoder:
    def __init__(self, numeric: Sequence[str], categorical: Sequence[str]):
        self.numeric = _NumericEncoder(numeric)
        self.categorical_names = tuple(categorical)

    def fit(self, records: Sequence[Mapping[str, Any]]) -> "_RidgeEncoder":
        self.numeric.fit(records)
        self.categories = {
            name: tuple(sorted({str(record["features"]["categorical"][name]) for record in records}))
            for name in self.categorical_names
        }
        return self

    def transform(self, records: Sequence[Mapping[str, Any]]) -> np.ndarray:
        numeric = self.numeric.transform(records)
        rows: list[list[float]] = []
        for record in records:
            categories: list[float] = []
            for name in self.categorical_names:
                value = str(record["features"]["categorical"][name])
                categories.extend(1.0 if value == category else 0.0 for category in self.categories[name])
            rows.append(categories)
        categorical = np.asarray(rows, dtype=np.float64)
        intercept = np.ones((len(records), 1), dtype=np.float64)
        return np.concatenate((intercept, numeric, categorical), axis=1)


def _ridge_fit(matrix: np.ndarray, targets: np.ndarray, alpha: float) -> np.ndarray:
    penalty = np.eye(matrix.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    system = matrix.T @ matrix + penalty
    rhs = matrix.T @ targets
    return np.linalg.pinv(system) @ rhs


def _fold_assignments(records: Sequence[Mapping[str, Any]], folds: int) -> np.ndarray:
    keyed = sorted(
        range(len(records)),
        key=lambda index: hashlib.sha256(
            str(records[index]["object_id"]).encode("utf-8")
        ).hexdigest(),
    )
    assignments = np.empty(len(records), dtype=np.int64)
    for position, index in enumerate(keyed):
        assignments[index] = position % folds
    return assignments


class SharedRidgePredictor:
    def _cross_validated_predictions(
        self, records: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any], alpha: float
    ) -> np.ndarray:
        folds = min(int(protocol["ridge_cv_folds"]), len(records))
        assignments = _fold_assignments(records, folds)
        predictions = np.empty(len(records), dtype=np.float64)
        for fold in range(folds):
            validation = np.flatnonzero(assignments == fold)
            if len(validation) == 0:
                continue
            training = np.flatnonzero(assignments != fold)
            train_records = [records[index] for index in training]
            validation_records = [records[index] for index in validation]
            encoder = _RidgeEncoder(
                protocol["numeric_features"], protocol["categorical_features"]
            ).fit(train_records)
            coefficient = _ridge_fit(
                encoder.transform(train_records),
                np.asarray([_target(record) for record in train_records], dtype=np.float64),
                alpha,
            )
            predictions[validation] = encoder.transform(validation_records) @ coefficient
        return predictions

    def fit(self, records: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> "SharedRidgePredictor":
        targets = np.asarray([_target(record) for record in records], dtype=np.float64)
        scores: dict[float, float] = {}
        cached: dict[float, np.ndarray] = {}
        for alpha_value in protocol["ridge_alphas"]:
            alpha = float(alpha_value)
            predictions = self._cross_validated_predictions(records, protocol, alpha)
            cached[alpha] = predictions
            scores[alpha] = float(np.mean(np.square(targets - predictions)))
        self.alpha = min(scores, key=lambda value: (scores[value], value))
        self.cv_mse = scores[self.alpha]
        errors = [
            (_operator(record), float(target - prediction))
            for record, target, prediction in zip(records, targets, cached[self.alpha])
        ]
        self.uncertainty = _UncertaintyModel(
            errors,
            min_family_samples=int(protocol["family_uncertainty_min_samples"]),
            coverage=float(protocol["uncertainty_interval_coverage"]),
        )
        self.encoder = _RidgeEncoder(
            protocol["numeric_features"], protocol["categorical_features"]
        ).fit(records)
        self.coefficient = _ridge_fit(self.encoder.transform(records), targets, self.alpha)
        return self

    def predict(self, record: Mapping[str, Any]) -> ResidualPrediction:
        mean = float(self.encoder.transform([record])[0] @ self.coefficient)
        return ResidualPrediction(mean, self.uncertainty.q90(record))


def _metrics(
    records: Sequence[Mapping[str, Any]], predictions: Sequence[ResidualPrediction], coverage: float
) -> dict[str, float]:
    measured = [float(record["measurement"]["duration_s"]) for record in records]
    predicted = [
        float(record["analytic"]["estimate_s"]) * math.exp(prediction.mean)
        for record, prediction in zip(records, predictions)
    ]
    ape = [abs(actual - estimate) / actual for actual, estimate in zip(measured, predicted)]
    log_errors = [
        abs(_target(record) - prediction.mean)
        for record, prediction in zip(records, predictions)
    ]
    empirical_coverage = statistics.fmean(
        float(error <= prediction.q90) for error, prediction in zip(log_errors, predictions)
    )
    return {
        "mape": statistics.fmean(ape),
        "mdape": float(statistics.median(ape)),
        "p90_ape": _quantile(ape, 0.90),
        "spearman": _spearman(measured, predicted),
        "log_mae": statistics.fmean(log_errors),
        "interval_coverage": empirical_coverage,
        "coverage_calibration_error": abs(empirical_coverage - coverage),
    }


def evaluate_zero_shot(
    records: Sequence[Mapping[str, Any]], contract: Mapping[str, Any], *, split_id: str
) -> dict[str, Any]:
    protocol = contract["prediction_protocol"]
    predictor_view = make_dataset_view(
        records,
        contract,
        split_id=split_id,
        mode="zero_shot",
        role="predictor",
    )
    oracle_view = make_dataset_view(
        records,
        contract,
        split_id=split_id,
        mode="zero_shot",
        role="oracle_evaluator",
    )
    train = [record for record in predictor_view if "target" in record]
    split = next(item for item in contract["splits"] if item["split_id"] == split_id)
    hidden_ids = set(split["hidden_object_ids"])
    hidden = [record for record in oracle_view if record["object_id"] in hidden_ids]
    models = {
        "analytic_only": AnalyticOnlyPredictor(),
        "family_median": FamilyMedianPredictor(),
        "nearest_shape": NearestShapePredictor(),
        "shared_ridge": SharedRidgePredictor(),
    }
    method_reports: dict[str, Any] = {}
    prediction_table: list[dict[str, Any]] = [
        {
            "object_id": record["object_id"],
            "operator": _operator(record),
            "measured_s": float(record["measurement"]["duration_s"]),
        }
        for record in hidden
    ]
    for name in protocol["baselines"]:
        model = models[name].fit(train, protocol)
        predictions = [model.predict(record) for record in hidden]
        details: dict[str, Any] = _metrics(
            hidden, predictions, float(protocol["uncertainty_interval_coverage"])
        )
        details["by_operator"] = {}
        for operator in sorted({_operator(record) for record in hidden}):
            indices = [
                index for index, record in enumerate(hidden) if _operator(record) == operator
            ]
            details["by_operator"][operator] = {
                "records": len(indices),
                **_metrics(
                    [hidden[index] for index in indices],
                    [predictions[index] for index in indices],
                    float(protocol["uncertainty_interval_coverage"]),
                ),
            }
        if isinstance(model, SharedRidgePredictor):
            details.update({"selected_alpha": model.alpha, "train_cv_mse": model.cv_mse})
        method_reports[name] = details
        for row, record, prediction in zip(prediction_table, hidden, predictions):
            row[name] = {
                "predicted_s": float(record["analytic"]["estimate_s"]) * math.exp(prediction.mean),
                "residual_mean": prediction.mean,
                "residual_q90": prediction.q90,
            }
    return {
        "split_id": split_id,
        "mode": "zero_shot",
        "train_records": len(train),
        "hidden_records": len(hidden),
        "shared_seen_records": len(split["shared_seen_object_ids"]),
        "protocol": copy.deepcopy(protocol),
        "methods": method_reports,
        "predictions": prediction_table,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    records = load_jsonl_records(args.dataset)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    report = {
        "format": "fusion-placement-search.e2-zero-shot-evaluation",
        "version": "1.0",
        "dataset": str(args.dataset),
        "contract_id": contract["contract_id"],
        "splits": [
            evaluate_zero_shot(records, contract, split_id=split["split_id"])
            for split in contract["splits"]
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for split in report["splits"]:
        print(
            f"{split['split_id']}: train={split['train_records']} "
            f"hidden={split['hidden_records']} shared_seen={split['shared_seen_records']}"
        )
        for name, metrics in split["methods"].items():
            suffix = f" alpha={metrics['selected_alpha']:g}" if "selected_alpha" in metrics else ""
            print(
                f"  {name}: MAPE={metrics['mape']:.6f} "
                f"P90={metrics['p90_ape']:.6f} Spearman={metrics['spearman']:.6f} "
                f"coverage={metrics['interval_coverage']:.6f}{suffix}"
            )
    print(f"report: {args.output}")
    print("Fusion E2b strict zero-shot evaluation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
