"""CPU checks with synthetic receipts; no GPU timing evidence."""
import copy
import hashlib

import pytest
import torch

from calibration import stage_size_calibration as ssc
from cost_model import stage_linear_calibration as slc
from cost_model.measurement_labels import MeasurementContractError

TRUTH = {"silu": (1.5, 6e-6), "mul": (1.0, 3e-6), "chain": (1.8, 8.5e-6)}
COPY_US = {"copy/524288/0->1": 20., "copy/524288/1->0": 19., "copy/1572864/0->1": 30., "copy/1572864/1->0": 29.}


def truth_us(case):
    if case["kind"] == "copy":
        return COPY_US[f"copy/{case['bytes']}/{case['source']}->{case['destination']}"]
    a, b = TRUTH[case["stage"]]
    return a + b * case["elements"]


def fixture_records(frozen):
    rows = []
    for attempt, order in enumerate(frozen["order"]):
        for case_id in order:
            case = frozen["cases"][case_id]
            tick = len(rows) * 10 + 1
            # Round 3 equals the truth, so the median is exact.
            value = truth_us(case) * (1 + .001 * (attempt - 3))
            rows.append({"case": case_id, "round": attempt, "started_monotonic_ns": tick,
                "finished_monotonic_ns": tick + 5,
                "timing": {"elapsed_ms_per_50_calls": value * 50 / 1000, "error": None,
                    "event_boundary": {"device": case["timing_device"], "start_device": case["timing_device"],
                        "end_device": case["timing_device"], "stream": 7, "calls_per_replay": 50}},
                "checks": ssc.expected_checks(case)})
    return rows


def tensor_obs(case_shape, device):
    return {"shape": list(case_shape), "dtype": "bf16", "layout": "contiguous", "strides": ssc.strides(case_shape),
            "layout_provenance": {"kind": "runtime_tensor", "device": device,
                                  "observer": "Tensor.shape/dtype/stride/is_contiguous"}}


def metadata_fixture(frozen):
    prepared = {}
    source = frozen["collector"]["source_sha256"][ssc.SOURCES[0]]
    for case_id, case in frozen["cases"].items():
        if case["kind"] == "copy":
            prepared[case_id] = {"recipe": copy.deepcopy(ssc.RECIPES["copy"]), "source_sha256": source,
                "tensors": {"source": tensor_obs(case["shape"], f"cuda:{case['source']}"),
                            "destination": tensor_obs(case["shape"], f"cuda:{case['destination']}")}}
        else:
            roles = ("gate", "up", "left", "output") + (("mid",) if case["stage"] == "chain" else ())
            prepared[case_id] = {"recipe": copy.deepcopy(ssc.RECIPES[case["stage"]]), "source_sha256": source,
                "tensors": {r: tensor_obs(case["shape"], "cuda:0") for r in roles}}
    runtime = {"runtime": {"accelerator": "HIP", "hip": "fixture", "driver": "fixture",
        "grad_enabled": False, "autocast_enabled": False, "deterministic_algorithms": False,
        "torch_distribution": "fixture", "torch_git_revision": "fixture",
        "torch_extension_sha256": "fixture", "torch_build_sha256": "fixture"},
        "devices": {str(i): {"architecture": "gfx936:sramecc+:xnack-", "name": "fixture", "ordinal": i} for i in (0, 1)},
        "peer_capability": {"0->1": True, "1->0": True}, "provenance": {"uuid": {"0": "u0", "1": "u1"}}}
    return prepared, runtime


def write_fixture(directory, mutate=None):
    frozen = ssc.plan()
    rows = fixture_records(frozen)
    if mutate:
        mutate(rows)
    prep, runtime = metadata_fixture(frozen)
    status = ssc.evaluate(frozen, rows)["status"]
    content = {"plan.json": frozen, "preparation.json": prep, "runtime.json": runtime,
        "identity.json": ssc.identity(frozen, prep, runtime),
        "warmup.json": {c: ssc.expected_checks(case) for c, case in frozen["cases"].items()},
        "process.json": {"python_returncode": 0 if status == "calibration_candidates_ready" else 1,
                         "log_returncode": 0, "automatic_admission": False},
        "result.json": {"scope": ssc.SCOPE, "complete": True, "records": rows,
                        "automatic_admission": False, "known_instrumentation_guard_passed": True}}
    content.update({f"attempt_{i:03d}.json": r for i, r in enumerate(rows)})
    for name, value in content.items():
        ssc.save(directory / name, value)
    return frozen


def reviewed(tmp_path, mutate=None):
    frozen = write_fixture(tmp_path, mutate)
    return ssc.review(tmp_path), frozen


def test_plan_freezes_sizes_roles_copies_and_protocol():
    a = ssc.plan()
    assert a == ssc.plan()
    assert len(a["cases"]) == 19 and sum(len(o) for o in a["order"]) == 133
    roles = {c: v.get("role") for c, v in a["cases"].items() if v["kind"] == "stage"}
    assert {c for c, r in roles.items() if r == "holdout"} == {"silu/S128", "silu/S512", "mul/S128", "mul/S512"}
    assert all(roles[f"chain/S{s}"] == "chain_validation" for s in ssc.SIZES)
    copies = {(v["bytes"], v["source"], v["destination"]) for v in a["cases"].values() if v["kind"] == "copy"}
    assert copies == {(524288, 0, 1), (524288, 1, 0), (1572864, 0, 1), (1572864, 1, 0)}
    assert a["fit_protocol"]["holdout_error_threshold"] is None
    assert a["order"][1] == a["order"][0][::-1]
    assert not a["model_parameters_updated"] and not a["automatic_admission"]


@pytest.mark.parametrize("stage", ssc.STAGES)
def test_cpu_stage_numerics_and_independent_buffers(stage):
    torch.manual_seed(3)
    gate, up = [torch.randn(ssc.shape(4), dtype=torch.bfloat16) for _ in range(2)]
    left = torch.nn.functional.silu(gate)
    inputs = {"gate": gate, "up": up, "left": left}
    snaps = {k: v.view(torch.uint8).clone() for k, v in inputs.items()}
    calls = [ssc.StageCall(stage, gate, up, left) for _ in range(3)]
    for c in calls:
        c.invalidate(); c()
    refs = ssc.references(torch, stage, gate, up, left)
    assert ssc.check_stage(torch, calls, inputs, snaps, refs)["outputs_checked"] == 3
    if stage == "chain":
        assert set(calls[0].observation["tensors"]) == {"gate", "up", "left", "output", "mid"}
        calls[0].mid.zero_()
        with pytest.raises(AssertionError, match="intermediate"):
            ssc.check_stage(torch, calls, inputs, snaps, refs)


@pytest.mark.parametrize("bad", ["missing", "reorder", "overlap", "copy_device", "checks"])
def test_bad_receipts_rejected(bad):
    frozen = ssc.plan()
    rows = fixture_records(frozen)
    if bad == "missing": rows.pop()
    elif bad == "reorder": rows[:2] = rows[1::-1]
    elif bad == "overlap": rows[0]["finished_monotonic_ns"] = 10 ** 6
    elif bad == "copy_device":
        row = next(r for r in rows if r["case"] == "copy/524288/0->1")
        row["timing"]["event_boundary"]["device"] = "cuda:0"
    else: rows[0]["checks"]["outputs_independent"] = False
    with pytest.raises(MeasurementContractError):
        ssc.evaluate(frozen, rows)


def test_quality_failure_is_per_case_and_blocks_only_that_fit():
    frozen = ssc.plan()
    rows = fixture_records(frozen)
    for r in rows:
        if r["case"] == "mul/S64" and r["round"] > 3:
            r["timing"]["elapsed_ms_per_50_calls"] *= 2
    result = ssc.evaluate(frozen, rows)
    assert result["status"] == "partial_quality_failed" and result["quality_failed"] == ["mul/S64"]
    fitted = slc.fit({**result, "identity": ssc.identity(frozen, *metadata_fixture(frozen))}, frozen)
    assert fitted["stages"]["mul"]["status"] == "not_fitted"
    assert fitted["stages"]["silu"]["status"] == "fitted"
    assert [e["stage"] for e in fitted["entries"]] == ["silu"]


def test_identity_uuid_is_provenance_and_peer_is_required():
    frozen = ssc.plan()
    prep, runtime = metadata_fixture(frozen)
    original = ssc.identity(frozen, prep, runtime)
    runtime["provenance"]["uuid"] = {"0": "other", "1": "other"}
    assert ssc.identity(frozen, prep, runtime) == original
    runtime["runtime"]["torch_build_sha256"] = "changed"
    assert ssc.identity(frozen, prep, runtime) != original
    runtime["peer_capability"]["1->0"] = False
    with pytest.raises(MeasurementContractError, match="Peer"):
        ssc.identity(frozen, prep, runtime)


def test_review_recomputes_and_rejects_tampering(tmp_path):
    result, _ = reviewed(tmp_path)
    assert result["status"] == "calibration_candidates_ready"
    row = ssc.read(tmp_path / "attempt_005.json")
    row["timing"]["elapsed_ms_per_50_calls"] *= 1.5
    ssc.save(tmp_path / "attempt_005.json", row)
    with pytest.raises(MeasurementContractError, match="differ"):
        ssc.review(tmp_path)


def test_fit_recovers_line_and_never_reads_holdout_labels(tmp_path):
    result, frozen = reviewed(tmp_path)
    fitted = slc.fit(result, frozen)
    for stage in slc.FITTED_STAGES:
        line = fitted["stages"][stage]
        a, b = TRUTH[stage]
        assert line["status"] == "fitted" and line["b_identified"] and line["a_separately_identified"]
        for case, p in line["holdout_predictions"].items():
            assert p["predicted_us"] == pytest.approx(a + b * p["elements"], rel=1e-9)
    assert not any("S128" in c or "S512" in c or c.startswith("chain") or c.startswith("copy")
                   for c in fitted["labels_read"])
    poisoned = copy.deepcopy(result)
    for case in ("silu/S128", "mul/S512", "chain/S256", "copy/524288/0->1"):
        poisoned["cases"][case] = {"status": "quality_failed", "statistics": None, "samples_us": []}
    assert slc.fit(poisoned, frozen) == fitted


def test_parameter_lookup_is_per_gpu_and_contract(tmp_path):
    result, frozen = reviewed(tmp_path)
    fitted = slc.fit(result, frozen)
    store = slc.StageLinearCalibration(fitted)
    entry = fitted["entries"][0]
    args = (entry["gpu_key"], entry["contract_id"], entry["recipe_id"])
    inside = store.predict(entry["stage"], 384 * 3072, *args)
    assert inside["extrapolated"] is False and inside["validated_range"] is None
    assert store.predict(entry["stage"], 4096 * 3072, *args)["extrapolated"] is True
    other_gpu = copy.deepcopy(entry["gpu_key"]); other_gpu["device"]["ordinal"] = 1
    assert store.predict(entry["stage"], 384 * 3072, other_gpu, *args[1:]) is None
    assert store.predict(entry["stage"], 384 * 3072, args[0], "other-contract", args[2]) is None


def test_validate_reports_holdout_chain_and_rejects_changed_predictions(tmp_path):
    result, frozen = reviewed(tmp_path)
    fitted = slc.fit(result, frozen)
    report = slc.validate(fitted, result, frozen, 1.34e12)
    assert report["linear_fit"]["metrics"]["coverage"] == "4/4"
    assert report["linear_fit"]["metrics"]["mape"] == pytest.approx(0, abs=1e-9)
    assert report["linear_fit"]["ordering"]["misordered_pairs"] == []
    assert report["analytic_baseline_frozen_assumptions"]["refit"] is False
    row = next(r for r in report["chain_check"]["rows"] if r["S"] == 256)
    n = 256 * 3072
    assert row["delta_add_us"] == pytest.approx(sum(TRUTH[s][0] + TRUTH[s][1] * n for s in ("silu", "mul"))
                                                - (TRUTH["chain"][0] + TRUTH["chain"][1] * n))
    assert report["copies"]["copy/1572864/1->0"]["median_us"] == pytest.approx(29.)
    tampered = copy.deepcopy(fitted)
    tampered["stages"]["silu"]["holdout_predictions"]["silu/S128"]["predicted_us"] += 1
    with pytest.raises(ValueError, match="changed after fitting"):
        slc.validate(tampered, result, frozen, 1.34e12)


def test_ffn_search_check_runs_after_validation_and_records_sources(tmp_path):
    from cost_model import ffn_calibrated_search as search
    result, frozen = reviewed(tmp_path)
    validation = slc.validate(slc.fit(result, frozen), result, frozen, 1.34e12)
    report = search.build_report(validation)
    assert report["cost_sources_after"]["L000.mlp.silu@gpu0"] == "measured_stage_size_batch"
    assert report["cost_sources_after"]["L000.mlp.silu@gpu1"] == "analytic_ffn_stage_proxy"
    assert report["cost_sources_after"]["copy/524288/0->1"] == "measured_graph_copy_batch"
    assert report["best_unchanged_is_not_accuracy_evidence"] is True
    assert len(report["ranking_after"]) == 48
    empty = copy.deepcopy(validation)
    empty["linear_fit"]["metrics"]["coverage"] = "0/4"
    with pytest.raises(ValueError, match="before the search"):
        search.build_report(empty)


def test_no_gpu_failure_retains_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    with pytest.raises(RuntimeError, match="HIP GPU0 and GPU1"):
        ssc.main(["run", "--output", str(tmp_path)])
    assert (tmp_path / "plan.json").exists() and ssc.read(tmp_path / "failure.json")["traceback"]
