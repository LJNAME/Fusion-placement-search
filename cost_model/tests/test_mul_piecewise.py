import json
from pathlib import Path

from cost_model import mul_piecewise as mp


def test_holdout_sizes_never_measured_and_baseline_kept():
    measured = set()
    for path in [*mp.BATCHES.values(), "data/stage_size_followup_mul_20261106_01/review/evaluation.json"]:
        measured |= {int(c.split("S")[1]) for c in json.loads(Path(path).read_text())["cases"] if c.startswith("mul/S")}
    assert not set(mp.HOLDOUT_SIZES) & measured
    report = mp.build_report()
    assert report["model_replaced"] is False
    assert report["development_data"]["role"].endswith("NOT independent test data")
    assert mp.predict_piecewise(mp.fit_piecewise(mp.development_points(Path("."))), 480) is None
