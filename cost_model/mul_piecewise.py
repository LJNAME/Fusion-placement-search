"""CPU-only two-segment Mul experiment; the batch-01 linear model stays the baseline.

Development data (batches 01 and 03, quality_passed) was used to discover the
jump, so comparisons on it are diagnostics, not independent tests. Predictions
for unmeasured holdout sizes are frozen here for a later collection.
"""
import argparse
import json
from pathlib import Path
import statistics

from cost_model.fusion_kernel_profile import key_id

HIDDEN = 3072
BATCHES = {"batch01": "data/stage_size_calibration_20261106_01/review/evaluation.json",
           "batch03": "data/stage_size_batch03_mul_20261106_01/review/evaluation.json"}
FIT01 = "data/stage_size_calibration_20261106_01/fit/fit.json"
EXCLUDED = {"batch02": "noisy batch (SiLU anchor +10.9%, 3/8 failed); documented, not used"}
BREAK_AFTER, BREAK_BEFORE = 448, 512
HOLDOUT_SIZES = (320, 416, 640, 896)


def ols(points):
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


def development_points(root):
    points = []
    for name, path in BATCHES.items():
        cases = json.loads((root / path).read_text(encoding="utf-8"))["cases"]
        for case, q in cases.items():
            if case.startswith("mul/S") and q["status"] == "quality_passed":
                s = int(case.split("S")[1])
                points.append({"batch": name, "S": s, "n": s * HIDDEN, "us": q["statistics"]["median_us"]})
    return sorted(points, key=lambda p: (p["S"], p["batch"]))


def fit_piecewise(points):
    low = [(p["n"], p["us"]) for p in points if p["S"] <= BREAK_AFTER]
    high = [(p["n"], p["us"]) for p in points if p["S"] >= BREAK_BEFORE]
    if len({x for x, _ in low}) < 2 or len({x for x, _ in high}) < 2:
        return None
    return {"low": ols(low), "high": ols(high)}


def predict_piecewise(model, s):
    if model is None or BREAK_AFTER < s < BREAK_BEFORE:
        return None  # breakpoint location inside (448, 512) is unknown
    a, b = model["low"] if s <= BREAK_AFTER else model["high"]
    return a + b * s * HIDDEN


def leave_one_size_out(points):
    rows = []
    for s in sorted({p["S"] for p in points}):
        train = [p for p in points if p["S"] != s]
        lin = ols([(p["n"], p["us"]) for p in train])
        pw = fit_piecewise(train)
        for p in (p for p in points if p["S"] == s):
            lp, pp = lin[0] + lin[1] * p["n"], predict_piecewise(pw, s)
            rows.append({"S": s, "batch": p["batch"], "measured_us": p["us"],
                         "linear_refit_error": (lp - p["us"]) / p["us"],
                         "piecewise_error": None if pp is None else (pp - p["us"]) / p["us"]})
    mape = lambda k: statistics.fmean(abs(r[k]) for r in rows if r[k] is not None)
    covered = [r for r in rows if r["piecewise_error"] is not None]
    return {"rows": rows, "linear_mape": mape("linear_refit_error"),
            "piecewise_mape": mape("piecewise_error") if covered else None,
            "linear_mape_on_piecewise_covered": statistics.fmean(abs(r["linear_refit_error"]) for r in covered),
            "piecewise_coverage": f"{len(covered)}/{len(rows)}"}


def build_report(root=Path(".")):
    points = development_points(root)
    fit01 = {e["stage"]: e for e in json.loads((root / FIT01).read_text(encoding="utf-8"))["entries"]}["mul"]
    baseline = lambda s: fit01["a_us_composite_for_prediction"] + fit01["b_us_per_element"] * s * HIDDEN
    pw = fit_piecewise(points)
    in_sample = [{**p, "baseline_linear_us": baseline(p["S"]), "piecewise_us": predict_piecewise(pw, p["S"])}
                 for p in points]
    predictions = {f"mul/S{s}": {"baseline_linear_us": baseline(s), "piecewise_us": predict_piecewise(pw, s)}
                   for s in HOLDOUT_SIZES}
    return {"scope": "CPU two-segment experiment; baseline linear model unchanged, no replacement",
        "development_data": {"batches": list(BATCHES), "excluded": EXCLUDED, "points": len(points),
                             "role": "used to discover the jump; NOT independent test data"},
        "piecewise": {"breakpoint": f"between S={BREAK_AFTER} and S={BREAK_BEFORE}; undefined inside",
                      "low_a_us_b_us_per_element": pw["low"], "high_a_us_b_us_per_element": pw["high"]},
        "in_sample": in_sample,
        "leave_one_size_out_diagnostic": leave_one_size_out(points),
        "frozen_holdout_predictions": predictions,
        "frozen_holdout_prediction_id": key_id(predictions),
        "holdout_status": "not collected; sizes never measured in any batch",
        "replacement_rule": "replace only if piecewise beats baseline on the new holdout; threshold not preset",
        "model_replaced": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_report()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"prediction_id": report["frozen_holdout_prediction_id"]}))


if __name__ == "__main__":
    main()
