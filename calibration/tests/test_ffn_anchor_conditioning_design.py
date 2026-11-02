"""CPU checks of timestamp interpretation and a design with no GPU entry."""
import copy
import pytest

from calibration.ffn_anchor_control import ROOT, read, pilot_plan
from calibration.ffn_anchor_conditioning_plan import proposal
from tools.analyze_ffn_anchor_position import describe


def rounds():
    records = []
    tick = 1000
    def sample(us):
        nonlocal tick
        tick += 10
        return {"batch_ms": us / 20, "timing_begin": tick, "validation_end": tick + 1}
    for i in range(7):
        records.append({"attempt": i, "pre": [sample(12), sample(10), sample(10)],
                        "target": sample(20), "post": [sample(10), sample(10), sample(10)]})
    return records


def test_preparation_reference_is_separate_without_removing_first_endpoint_values():
    rows, summary = describe(rounds(), {"anchor_warmup_end": 0})
    assert len(rows) == 14 and summary["pre"]["endpoints"] == 7
    assert rows[0]["first_gap_contains_target_preparation"]
    assert summary["pre"]["first_minus_next_two_us"]["n"] == 7
    assert summary["pre"]["first_host_gap_ms_excluding_different_preparation_reference"]["n"] == 6
    assert summary["pre"]["first_minus_next_two_us"]["median"] == 2
    assert all(not r["gpu_idle_or_clock_observed"] for r in rows)


def test_short_host_gap_extreme_remains_visible_and_does_not_change_quality():
    data = rounds()
    data[0]["post"][0]["batch_ms"] = 40 / 20
    original = copy.deepcopy(data)
    rows, summary = describe(data, {"anchor_warmup_end": 0})
    assert data == original and rows[1]["first_us"] == 40
    assert rows[1]["first_gap_reference"] == "same_round_target_validation_end"
    assert summary["post"]["first_minus_next_two_us"]["max"] == 30
    with pytest.raises(ValueError): describe(data[:-1], {"anchor_warmup_end": 0})
    with pytest.raises(ValueError): describe(data, {"anchor_warmup_end": 999999})


def test_design_is_single_intervention_with_replication_and_no_execution_entry():
    design = proposal()
    assert design == proposal()
    assert [c["condition"] for c in design["cells"]] == ["A", "B", "B", "A"]
    assert {c["row"] for c in design["cells"]} == {"down_proj@gpu0"}
    assert [c["conditioning_replays_per_endpoint"] for c in design["cells"]] == [0, 1, 1, 0]
    assert design["shared_recipe"]["measured_replays_per_endpoint"] == 3
    assert not design["gpu_entry_open"] and not design["gpu_collector_implemented"]
    assert not design["retroactive_quality_change"] and not design["formal_matrix_open"]
    # Existing GPU replay entry remains byte/source-identical to its v2 plan.
    assert pilot_plan() == read(ROOT / "data/ffn_anchor_control_v2/plan.json")
