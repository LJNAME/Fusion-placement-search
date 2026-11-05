"""Local search wiring with real reviewed costs and explicit result limits."""
import copy

import pytest

from cost_model import local_profile_search as local


def test_two_real_cost_choices_and_search_agree():
    result = local.build_report(seeds=(0, 1, 7))
    assert result["search_space_size"] == 2
    rows = {r["choice"]: r for r in result["candidates"]}
    assert rows["native"]["cost_us"] == pytest.approx(10.172800123691559)
    assert rows["fused"]["cost_us"] == pytest.approx(17.811180353164673)
    assert rows["fused"]["profile_identity_ids"] == [result["identity"]["fused_kernel_key_id"]]
    assert len(rows["native"]["profile_identity_ids"]) == 2
    assert [g["source"] for g in rows["native"]["estimate"]["details"]["groups"]].count(
        "measured_native_isolated_stage") == 2
    assert [g["source"] for g in rows["fused"]["estimate"]["details"]["groups"]].count(
        "measured_group_kernel") == 1
    assert all(r["best_matches_exact"] and r["cost_model_gap_us"] == 0 for r in result["mcts"])
    assert result["exact_choice"] == "native"
    assert result["allowed_devices"] == [0]
    assert result["independent_reference_regret_us"] is None
    assert not result["analytic_fallback_used"]
    assert not result["placement_benefit_tested"]
    assert not result["mcts_efficiency_or_superiority_tested"]


def test_missing_measured_stage_does_not_silently_use_legacy_analytic(monkeypatch):
    monkeypatch.setattr(local.ReviewedStageStore, "lookup", lambda *_: None)
    with pytest.raises(ValueError, match="Uncovered cost"):
        local.build_report(seeds=())


def test_different_runtime_does_not_mix_profile_sources(monkeypatch):
    original = local.load_reviewed_profile
    def mismatched(directory):
        profile = original(directory)
        profile.keys.runtime = copy.deepcopy(profile.keys.runtime)
        profile.keys.runtime["accelerator_version"] = "different"
        return profile
    monkeypatch.setattr(local, "load_reviewed_profile", mismatched)
    with pytest.raises(ValueError, match="runtime conditions"):
        local.build_case()


def test_report_replay_is_deterministic():
    assert local.build_report(seeds=(0,)) == local.build_report(seeds=(0,))
