"""Stage boundaries, missing data and decision sensitivity; CPU only."""
import json
from pathlib import Path

import pytest

from cost_model.graph_analytic import Parameter, analyze, assume, parameters, recipes, sensitivity
from tools.analyze_graph_cost import ROOT, SCENARIOS, reviewed_complete_fused


def base():
    return parameters(1e12, "CPU test fixture, not device measurement")


def scenario(**updates):
    values = json.loads(SCENARIOS.read_text(encoding="utf-8"))["assumptions"]
    return assume(base(), {**values, **updates})


def test_unknown_is_not_zero_or_an_admissible_search_score():
    result = analyze(base())
    assert result["choice"] == "unknown"
    assert result["delta_fused_minus_native_us"] is None
    for arm in result["arms"].values():
        assert arm["duration_us"] is None
        assert "graph_stage_overhead_us" in arm["missing_parameters"]
        assert "silu_elements_per_s" in arm["missing_parameters"]
        assert "vector_flops_per_s" in arm["missing_parameters"]


def test_real_recipe_bytes_and_nonzero_nonlinear_work():
    cases = recipes()
    assert sum(s.read_bytes + s.write_bytes for s in cases["native"]) == 7.5 * 2**20
    assert sum(s.read_bytes + s.write_bytes for s in cases["fused"]) == 10.5 * 2**20
    assert cases["fused"][0].read_bytes == cases["fused"][0].write_bytes == 3 * 2**20
    assert cases["native"][0].silu_elements == 256 * 3072
    assert cases["fused"][1].silu_elements == 256 * 3072


@pytest.mark.parametrize("layout", ["unknown", "strided"])
def test_no_implicit_contiguous(layout):
    with pytest.raises(ValueError):
        analyze(base(), layout=layout)


@pytest.mark.parametrize("name,value", [("memory_bandwidth_Bps", 0),
    ("graph_stage_overhead_us", -1), ("coalescing_efficiency", 1.1),
    ("compute_utilization", 0), ("cache_hit_fraction", -0.1),
    ("silu_elements_per_s", float("nan")), ("lightop_compute_multiplier", 0)])
def test_bad_parameter_rejected(name, value):
    with pytest.raises(ValueError):
        scenario(**{name: value})


def test_parameter_cannot_silently_lose_unknown_status():
    with pytest.raises(ValueError):
        Parameter(None, "us", "execution", "missing", "Graph", status="measured")


def test_coalescing_changes_reads_not_writes():
    first = analyze(scenario())["arms"]["native"]["stages"][1]
    second = analyze(scenario(coalescing_efficiency=.5))["arms"]["native"]["stages"][1]
    assert second["dram_read_us"] == pytest.approx(first["dram_read_us"] * 2)
    assert second["write_us"] == first["write_us"]


def test_read_utilization_is_explicit():
    first = analyze(scenario())["arms"]["native"]["stages"][1]
    second = analyze(scenario(read_utilization=2))["arms"]["native"]["stages"][1]
    assert second["dram_read_us"] == pytest.approx(first["dram_read_us"] * 2)
    assert second["write_us"] == first["write_us"]


def test_cache_needs_rate_only_when_used_and_reduces_dram_reads():
    no_cache = analyze(scenario())
    assert no_cache["arms"]["native"]["duration_us"] is not None
    missing = analyze(scenario(cache_hit_fraction=.5))
    assert missing["arms"]["native"]["duration_us"] is None
    assert "cache_bandwidth_Bps" in missing["arms"]["native"]["missing_parameters"]
    cached = analyze(scenario(cache_hit_fraction=.5, cache_bandwidth_Bps=10e12))
    a = no_cache["arms"]["native"]["stages"][0]
    b = cached["arms"]["native"]["stages"][0]
    assert b["dram_read_us"] == pytest.approx(a["dram_read_us"] / 2)
    assert b["memory_us"] < a["memory_us"]


def test_implementation_rate_is_not_forced_equal():
    a = analyze(scenario())
    b = analyze(scenario(lightop_compute_multiplier=4))
    assert b["arms"]["fused"]["stages"][1]["compute_us"] == pytest.approx(
        a["arms"]["fused"]["stages"][1]["compute_us"] / 4)
    assert a["arms"]["native"] == b["arms"]["native"]


def test_shared_graph_overhead_cancels_only_for_equal_analytic_stage_counts():
    a = analyze(scenario(graph_stage_overhead_us=.1))
    b = analyze(scenario(graph_stage_overhead_us=4))
    assert a["delta_fused_minus_native_us"] == pytest.approx(b["delta_fused_minus_native_us"])
    assert b["arms"]["native"]["duration_us"] - a["arms"]["native"]["duration_us"] == pytest.approx(7.8)


def test_strict_complete_profile_replaces_whole_call_and_sign_flip_is_causal():
    exact = reviewed_complete_fused()
    assert exact["duration_us"] == pytest.approx(17.811180353164673)
    assert exact["additional_pack_us"] == exact["additional_launch_us"] == 0
    cfg = json.loads(SCENARIOS.read_text(encoding="utf-8"))
    assumptions = {**cfg["assumptions"], "silu_elements_per_s": 5e11}
    result = sensitivity(base(), assumptions, {"graph_stage_overhead_us": [.1, 20]},
                         complete_fused_us=exact["duration_us"])
    rows = result["joint_rows"]
    assert all(r["scores"]["exact_complete_fused"]["fused_us"] == exact["duration_us"] for r in rows)
    effects = result["effects"]["graph_stage_overhead_us"]["modes"]
    assert effects["analytic_both"]["max_delta_span_us"] == pytest.approx(0, abs=1e-10)
    witness = effects["exact_complete_fused"]["sign_flip_witness"]
    assert witness is not None
    assert {w["scores"]["exact_complete_fused"]["choice"] for w in witness} == {"native", "fused"}
    left, right = (dict(w["assumptions"]) for w in witness)
    left.pop("graph_stage_overhead_us")
    right.pop("graph_stage_overhead_us")
    assert left == right


def test_sensitive_implementation_can_change_analytic_choice():
    cfg = json.loads(SCENARIOS.read_text(encoding="utf-8"))
    assumptions = {**cfg["assumptions"], "silu_elements_per_s": 2e10}
    result = sensitivity(base(), assumptions, {"lightop_compute_multiplier": [.5, 4]})
    assert result["conditional_calibration_shortlist"]["analytic_both"] == ["lightop_compute_multiplier"]


def test_sensitivity_requires_complete_assumptions():
    with pytest.raises(ValueError):
        sensitivity(base(), {}, {"graph_stage_overhead_us": [.1, 1]})


def test_no_eager_or_matrix_rate_imported_as_execution_parameter():
    p = base()
    assert p["graph_stage_overhead_us"].value is None
    assert p["vector_flops_per_s"].value is None
    assert p["silu_elements_per_s"].value is None
    assert all(v.status != "measured" for v in scenario().values())
