"""CPU checks for the GPU1 completion collector and overlay; synthetic receipts.

The LightOP Kernel Key replay needs the vendor runtime, so the fused case is
covered by a synthetic identity here and by the real review on the server.
"""
import copy

import pytest

from calibration import gpu1_cost_completion as g1
from calibration import native_stage_calibration as native
from calibration import stage_size_calibration as base
from calibration.tests.test_stage_size_calibration import metadata_fixture, tensor_obs
from cost_model.fusion_kernel_profile import key_id
from cost_model.measurement_labels import MeasurementContractError

US = {"stage/silu/gpu1": 7.0, "stage/mul/gpu1": 4.3, "fused/silu_mul/gpu1": 17.9,
      "gemm/down/gpu1": 27.3, "gemm/down/gpu0": 27.4}


@pytest.fixture(scope="module")
def frozen():
    return g1.plan()


def prepared(frozen):
    src = frozen["collector"]["source_sha256"]
    out = {}
    for c, case in frozen["cases"].items():
        dev = case["timing_device"]
        if case["kind"] == "stage":
            out[c] = {"recipe": copy.deepcopy(native.RECIPES[case["stage"]]), "schema": native.SCHEMAS[case["stage"]],
                      "source_sha256": src["calibration/native_stage_calibration.py"],
                      "tensors": {r: tensor_obs(case["shape"], dev) for r in ("gate", "up", "left", "output")}}
        elif case["kind"] == "gemm":
            out[c] = {"recipe": copy.deepcopy(g1.gp.RECIPE), "source_sha256": src["calibration/gemm_projection_calibration.py"],
                      "tensors": {"x": tensor_obs(case["x_shape"], dev), "weight": tensor_obs(case["weight_shape"], dev),
                                  "output": tensor_obs(case["output_shape"], dev)}}
        else:
            out[c] = {"one_call": {"synthetic": True}, "call_count": 50}
    return out


FUSED = {"kernel_key": {"synthetic": "gpu1"}, "identity_gaps": [], "measurement_contract_id": "sha256:contract",
         "group_index": 3}
FUSED["kernel_key_id"] = key_id(FUSED["kernel_key"])


def receipt(frozen, fail=()):
    rows = []
    for attempt, order in enumerate(frozen["order"]):
        for c in order:
            case = frozen["cases"][c]
            tick = len(rows) * 10 + 1
            us = US[c] * (1 + .001 * (attempt - 3)) * (2 if c in fail and attempt > 3 else 1)
            rows.append({"case": c, "round": attempt, "started_monotonic_ns": tick, "finished_monotonic_ns": tick + 5,
                "timing": {"elapsed_ms_per_50_calls": us * 50 / 1000, "error": None,
                    "event_boundary": {"device": case["timing_device"], "start_device": case["timing_device"],
                                       "end_device": case["timing_device"], "stream": 7, "calls_per_replay": 50}},
                "checks": base.expected_checks(case)})
    return rows


def test_plan_reuses_declared_recipes_and_freezes_proxies(frozen):
    assert len(frozen["cases"]) == 5 and sum(len(o) for o in frozen["order"]) == 35
    assert frozen["recipes"]["stage"] == native.RECIPES
    assert frozen["cases"]["gemm/down/gpu0"]["role"] == "same_batch_reference_not_a_new_profile"
    assert frozen["frozen_proxy_predictions"]["stage/silu/gpu1"]["source"] == "analytic_ffn_stage_proxy"
    assert frozen["frozen_proxy_predictions"]["fused/silu_mul/gpu1"]["source"] == "analytic_ffn_fused_proxy"
    assert frozen["known_gaps"]


def test_identity_rejects_wrong_device_and_incomplete_key(frozen):
    _, runtime = metadata_fixture(base.plan())
    ident = g1.identity(frozen, prepared(frozen), runtime, FUSED)
    assert ident["cases"]["fused/silu_mul/gpu1"]["case_identity"]["kernel_key_id"] == FUSED["kernel_key_id"]
    bad = prepared(frozen)
    bad["stage/mul/gpu1"]["tensors"]["output"]["layout_provenance"]["device"] = "cuda:0"
    with pytest.raises(MeasurementContractError):
        g1.identity(frozen, bad, runtime, FUSED)
    with pytest.raises(MeasurementContractError):
        g1.identity(frozen, prepared(frozen), runtime, {**FUSED, "identity_gaps": ["runtime.driver_version"]})


def test_overlay_serves_only_matching_gpu1_groups(tmp_path, frozen):
    from cost_model.ffn_profile_search import COMPUTE, with_boundaries
    from cost_model.ffn_gemm_rescoring import profiles
    from cost_model.gpu1_profile import MeasuredGpu1Profile
    from cost_model.structural import StructuralCostModel
    from simulator import FusionGroup, Plan
    _, runtime = metadata_fixture(base.plan())
    rows = receipt(frozen, fail=("gemm/down/gpu1",))
    ident = g1.identity(frozen, prepared(frozen), runtime, FUSED)
    evaluation = g1.evaluate(frozen, rows)
    raw = tmp_path / "raw"; raw.mkdir()
    base.save(raw / "identity.json", ident)
    evaluation.update(identity=ident, raw_sha256={"identity.json": base.digest(raw / "identity.json")})
    store = g1_store(raw, evaluation, tmp_path)
    assert store.failed == ["gemm/down/gpu1"] and "gemm/down/gpu0" not in {r["case_id"] for r in store.records.values()}
    _, full, native_profile, _, _, _, registry = profiles("data/gemm_projection_20261106_01")
    profile = MeasuredGpu1Profile(native_profile, store)
    plan = with_boundaries(Plan(groups=[FusionGroup([n], 1, {"pattern_id": "unfused"}) for n in COMPUTE]))
    groups = {g.node_ids[0]: g for g in StructuralCostModel(registry=registry).prepare(full, plan, 2).structure.groups}
    assert profile.estimate_group(full, groups["L000.mlp.silu"]).source == "measured_gpu1_isolated_stage"
    assert profile.estimate_group(full, groups["L000.mlp.gated_mul"]).duration == pytest.approx(4.3e-6)
    assert profile.estimate_group(full, groups["L000.mlp.down_proj"]).source == "analytic_ffn_gemm_proxy"


def g1_store(raw, evaluation, tmp_path):
    from cost_model.gpu1_profile import ReviewedGpu1Store
    base.save(tmp_path / "evaluation.json", evaluation)
    return ReviewedGpu1Store(raw, tmp_path / "evaluation.json", excluded_cases=("gemm/down/gpu0",))
