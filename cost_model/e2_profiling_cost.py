"""Training-only estimates of the wall time of one profiling operation.

Wall labels include preparation, warmup, timing loops and synchronization.
They are distinct from the GPU duration target used by the residual predictor.
The family median is predeclared, not selected using evaluation labels.
"""

from __future__ import annotations

import math
import statistics


VARIANTS = ("analytic_repetitions", "global_wall_median", "family_wall_median")
PROTOCOL = {
    "variants": list(VARIANTS),
    "active_variant": "family_wall_median",
    "target": "measurement.profiling_wall_time_s",
    "family": "features.categorical.operator",
    "fallback": "unseen family -> visible global median; no wall labels -> analytic repetitions",
    "selection": "predeclared family median; no held-out model selection or hyperparameters",
    "scope": "same BW1000 environment and profiling recipe only; no cross-recipe transfer claim",
}


def analytic_cost_s(record, repetitions):
    return max(float(record["analytic"]["estimate_s"]) * repetitions, 1e-12)


class ProfilingCostPredictor:
    def fit(self, public_records, *, repetitions=360):
        if repetitions <= 0:
            raise ValueError("profiling repetitions must be positive")
        self.repetitions = repetitions
        values = []
        by_family = {}
        self.train_object_ids = []
        for row in public_records:
            measurement = row.get("measurement", {})
            if "profiling_wall_time_s" not in measurement:
                continue
            if not row.get("label_access", {}).get("visible", False):
                raise ValueError("cost fit requires an explicitly visible public measurement")
            value = float(measurement["profiling_wall_time_s"])
            if not math.isfinite(value) or value <= 0:
                raise ValueError("profiling wall time must be finite and positive")
            key = row["object_id"]
            if key in self.train_object_ids:
                raise ValueError("duplicate profiling measurement object")
            self.train_object_ids.append(key)
            values.append(value)
            family = str(row["features"]["categorical"]["operator"])
            by_family.setdefault(family, []).append(value)
        self.global_median_s = statistics.median(values) if values else None
        self.family_medians_s = {key: statistics.median(items) for key, items in by_family.items()}
        return self

    def predict(self, record, variant="family_wall_median"):
        if variant not in VARIANTS:
            raise ValueError(f"unknown profiling cost variant: {variant}")
        if variant == "analytic_repetitions" or self.global_median_s is None:
            return analytic_cost_s(record, self.repetitions)
        if variant == "global_wall_median":
            return self.global_median_s
        family = str(record["features"]["categorical"]["operator"])
        return self.family_medians_s.get(family, self.global_median_s)


def time_budget_summary(trajectory, horizon_s):
    """Non-preemptive profiling: an observation is usable only after completion.

    A common, order-independent horizon is the sum of all pool wall costs.
    Hold the current decision during measurement and after an early stop.
    CPU selection time is reported separately, not counted as free acceleration.
    """
    if not trajectory or not math.isfinite(horizon_s) or horizon_s <= 0:
        raise ValueError("a positive common horizon and nonempty trajectory are required")
    times = [float(point["profiling_wall_time_s"]) for point in trajectory]
    if times[0] != 0 or any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("profiling completion times must increase from zero")
    if times[-1] > horizon_s + 1e-9:
        raise ValueError("common horizon excludes measured costs")
    area = sum((right-left) * float(point["top1_regret_ms"])
               for left, right, point in zip(times, times[1:], trajectory))
    area += max(horizon_s-times[-1], 0) * float(trajectory[-1]["top1_regret_ms"])
    curve = []
    for fraction in (0, .01, .02, .05, .1, .25, .5, 1):
        available = [point for point in trajectory if point["profiling_wall_time_s"] <= fraction*horizon_s]
        point = available[-1]
        curve.append({"cost_fraction": fraction, "budget_s": fraction*horizon_s,
            "completed_measurements": point["budget"], "top1_regret_ms": point["top1_regret_ms"]})
    return {"horizon_s": horizon_s, "step_regret_time_auc_ms": area/horizon_s, "curve": curve}
