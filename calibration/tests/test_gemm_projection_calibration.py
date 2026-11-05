"""CPU checks for the P2 GEMM collector, Profile overlay and rescoring."""
import copy
import math

import pytest
import torch

from calibration import gemm_projection_calibration as gp
from calibration import stage_size_calibration as base
from calibration.tests.test_stage_size_calibration import metadata_fixture, tensor_obs
from cost_model.measurement_labels import MeasurementContractError

MEASURED = {"gate": 16.0, "up": 16.1, "down": 15.0}


@pytest.fixture(scope="module")
def frozen():
    return gp.plan()


def records(frozen, fail=()):
    rows = []
    for attempt, order in enumerate(frozen["order"]):
        for case_id in order:
            case = frozen["cases"][case_id]
            tick = len(rows) * 10 + 1
            us = MEASURED[case["projection"]] * (1 + .001 * (attempt - 3))
            if case_id in fail and attempt > 3:
                us *= 2
            rows.append({"case": case_id, "round": attempt, "started_monotonic_ns": tick,
                "finished_monotonic_ns": tick + 5,
                "timing": {"elapsed_ms_per_50_calls": us * 50 / 1000, "error": None,
                    "event_boundary": {"device": case["timing_device"], "start_device": case["timing_device"],
                                       "end_device": case["timing_device"], "stream": 7, "calls_per_replay": 50}},
                "checks": base.expected_checks(case)})
    return rows


def prepared(frozen):
    src = frozen["collector"]["source_sha256"][gp.SOURCES[0]]
    out = {}
    for c, case in frozen["cases"].items():
        dev = case["timing_device"]
        out[c] = {"recipe": copy.deepcopy(gp.RECIPE), "source_sha256": src, "tensors": {
            "x": tensor_obs(case["x_shape"], dev), "weight": tensor_obs(case["weight_shape"], dev),
            "output": tensor_obs(case["output_shape"], dev)}}
    return out


def write(directory, frozen, fail=()):
    rows = records(frozen, fail)
    _, runtime = metadata_fixture(base.plan())
    prep = prepared(frozen)
    status = gp.evaluate(frozen, rows)["status"]
    content = {"plan.json": frozen, "preparation.json": prep, "runtime.json": runtime,
        "identity.json": gp.identity(frozen, prep, runtime),
        "warmup.json": {c: base.expected_checks(k) for c, k in frozen["cases"].items()},
        "process.json": {"python_returncode": 0 if status == "calibration_candidates_ready" else 1,
                         "log_returncode": 0, "automatic_admission": False},
        "result.json": {"scope": gp.SCOPE, "complete": True, "records": rows, "automatic_admission": False}}
    content.update({f"attempt_{i:03d}.json": r for i, r in enumerate(rows)})
    for name, value in content.items():
        base.save(directory / name, value)


def test_plan_freezes_shapes_and_current_predictions(frozen):
    assert len(frozen["cases"]) == 6 and sum(len(o) for o in frozen["order"]) == 42
    down = frozen["cases"]["gemm/down/gpu1"]
    assert (down["m"], down["k"], down["n"], down["timing_device"]) == (256, 3072, 1024, "cuda:1")
    pred = frozen["frozen_predictions"]["gemm/gate/gpu0"]
    assert pred["predicted_us"] == pytest.approx(22.884743841557214) and pred["source"] == "analytic_ffn_gemm_proxy"
    assert frozen["prediction_error_threshold"] is None


def test_cpu_gemm_numerics_use_linear_weight_layout():
    torch.manual_seed(1)
    x = torch.randn(8, 16, dtype=torch.bfloat16)
    w = (torch.randn(4, 16) / math.sqrt(16)).to(torch.bfloat16)
    calls = [gp.GemmCall(x, w) for _ in range(2)]
    for c in calls:
        c()
    snaps = (x.view(torch.uint8).clone(), w.view(torch.uint8).clone())
    refs = (x @ w.t(), x.float() @ w.float().t())
    assert gp.check(torch, calls, x, w, snaps, refs)["outputs_checked"] == 2
    assert calls[0].observation["tensors"]["weight"]["shape"] == [4, 16]
    with pytest.raises(ValueError):
        gp.GemmCall(x, w.t().contiguous())


def test_review_reports_prediction_error_and_rejects_tampering(tmp_path, frozen):
    write(tmp_path, frozen)
    result = gp.review(tmp_path)
    err = result["prediction_error"]["gemm/down/gpu0"]
    assert err["measured_us"] == pytest.approx(15.0)
    assert err["signed_error"] == pytest.approx((err["predicted_us"] - 15.0) / 15.0)
    ident = base.read(tmp_path / "identity.json")
    ident["cases"]["gemm/up/gpu1"]["contract_id"] = "x"
    base.save(tmp_path / "identity.json", ident)
    with pytest.raises(MeasurementContractError):
        gp.review(tmp_path)


def test_identity_is_per_gpu(frozen):
    _, runtime = metadata_fixture(base.plan())
    ident = gp.identity(frozen, prepared(frozen), runtime)["cases"]
    assert ident["gemm/gate/gpu0"]["gpu_key_id"] != ident["gemm/gate/gpu1"]["gpu_key_id"]


def test_profile_overlay_and_rescoring(tmp_path, frozen):
    from cost_model import ffn_gemm_rescoring as rs
    raw, rev = tmp_path / "raw", tmp_path / "review"
    raw.mkdir(); rev.mkdir()
    write(raw, frozen, fail=("gemm/down/gpu1",))
    base.save(rev / "evaluation.json", gp.review(raw))
    search, full, native, measured, store, review, registry = rs.profiles(tmp_path)
    assert store.failed == ["gemm/down/gpu1"]
    from simulator import FusionGroup, Plan
    from cost_model.ffn_profile_search import COMPUTE, with_boundaries
    from cost_model.structural import StructuralCostModel
    device = {"L000.mlp.up_proj": 1, "L000.mlp.down_proj": 1}
    plan = with_boundaries(Plan(groups=[FusionGroup([n], device.get(n, 0), {"pattern_id": "unfused"}) for n in COMPUTE]))
    groups = {(g.node_ids[0], g.device): g for g in
              StructuralCostModel(registry=registry).prepare(full, plan, measured.n_devs).structure.groups}
    hit = measured.estimate_group(full, groups["L000.mlp.up_proj", 1])
    assert hit.source == "measured_gemm_projection" and hit.duration == pytest.approx(16.1e-6)
    miss = measured.estimate_group(full, groups["L000.mlp.down_proj", 1])
    assert miss.source == "analytic_ffn_gemm_proxy" and "gemm_profile:not_reviewed_or_quality_failed" in miss.ood_reasons
    silu = measured.estimate_group(full, groups["L000.mlp.silu", 0])
    assert silu.source == "measured_native_isolated_stage"
    report = rs.build_report(tmp_path)
    assert report["after"]["search_space"] == 48 and len(report["after"]["mcts"]) == 3
    assert report["gemm_quality_failed_left_as_proxy"] == ["gemm/down/gpu1"]
