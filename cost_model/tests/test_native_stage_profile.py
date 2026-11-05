"""Real reviewed costs, context misses and the existing scheduler interface."""
import copy
from dataclasses import replace

import pytest

from cost_model.hardware_profile import AnalyticHardwareProfile, GroupCost
from cost_model.native_stage_profile import NativeMeasuredFirstProfile, ReviewedStageStore
from cost_model.structural import StructuralCostModel
from cost_model.structure import analyze_plan_structure
from simulator import FusionGroup, OpGraph, OpNode, Plan, TensorEdge, load_hardware
from calibration.native_stage_calibration import ROOT


@pytest.fixture(scope="module")
def store():
    return ReviewedStageStore(ROOT / "data/native_stage_calibration_20261105_01/raw")


def case(store):
    records = {r["stage"]: r for r in store.records.values()}
    graph = OpGraph(name="observed-stage-boundaries")
    for name, op, tensor in (("gate", "input", "gate"), ("up", "input", "up"),
                             ("s", "silu", "output"), ("m", "mul", "output")):
        record = records["mul" if name == "m" else "silu"]
        implementation = record["identity"]["implementation"]
        observation = copy.deepcopy(implementation["tensors"][tensor])
        observation["output_shape"] = observation.pop("shape")
        if name in ("s", "m"):
            observation["implementation"] = {"id": implementation["recipe"]["id"],
                                             "version": implementation["source_sha256"]}
        graph.add_node(OpNode(name, op, 786432 if name in ("s", "m") else 0, 1572864, 0, observation))
    for a, b in (("gate", "s"), ("s", "m"), ("up", "m")):
        graph.add_edge(TensorEdge(a, b, 1572864))
    plan = Plan(groups=[FusionGroup([n], 0, {"pattern_id": "unfused"}) for n in ("gate", "up", "s", "m")])
    bindings = {n: {"identity": copy.deepcopy(records[s]["identity"]), "operands": operands}
                for n, s, operands in (("s", "silu", {"gate": "gate"}),
                                       ("m", "mul", {"left": "s", "up": "up"}))}
    base = AnalyticHardwareProfile(load_hardware(ROOT / "data/bw1000_2card/bw1000_2card.calibrated.json"))
    groups = analyze_plan_structure(graph, plan, base.n_devs, strict_fusion=False).groups
    return graph, plan, base, groups, bindings


def test_exact_reviewed_times_are_used_without_extra_launch_or_memory(store):
    graph, _, base, groups, bindings = case(store)
    profile = NativeMeasuredFirstProfile(graph, base, store, bindings)
    for index, expected_us in ((2, 6.537600159645081), (3, 3.6351999640464783)):
        cost = profile.estimate_group(graph, groups[index])
        assert cost.duration * 1e6 == pytest.approx(expected_us)
        assert cost.source == "measured_native_isolated_stage"
        assert cost.launch_time == cost.memory_time == 0
        assert "measurement_context:isolated_stage" in cost.ood_reasons


@pytest.mark.parametrize("bad", ["contract", "runtime", "shape", "layout", "input_layout", "implementation", "device", "left_origin", "aliased_operands", "unbound"])
def test_mismatch_uses_analytic_fallback_and_explains_miss(store, bad):
    graph, _, base, groups, bindings = case(store)
    group = groups[3]
    if bad == "contract": bindings["m"]["identity"]["contract"]["mode"] = "eager"
    elif bad == "runtime": bindings["m"]["identity"]["runtime"]["torch_build_sha256"] = "different"
    elif bad == "shape": graph.nodes["m"].extra["output_shape"] = [1, 128, 3072]
    elif bad == "layout": graph.nodes["m"].extra["layout"] = None
    elif bad == "input_layout": graph.nodes["up"].extra["strides"] = [1, 3072, 256]
    elif bad == "implementation": graph.nodes["m"].extra["implementation"]["id"] = "tensor.mul"
    elif bad == "device": group = replace(group, device=1)
    elif bad == "left_origin": graph.nodes["s"].op_type = "input"
    elif bad == "aliased_operands":
        bindings["m"]["operands"]["up"] = "s"
        graph.edges = [TensorEdge("s", "m", 1572864) if e.src == "up" else e for e in graph.edges]
    else: del bindings["m"]
    profile = NativeMeasuredFirstProfile(graph, base, store, bindings)
    cost = profile.estimate_group(graph, group)
    assert cost.duration == base.estimate_group(graph, group).duration
    assert cost.source == "analytic_roofline"
    assert any(r.startswith("stage_profile:") for r in cost.ood_reasons)


def test_fused_complete_group_is_delegated_without_stage_splitting(store):
    graph, _, base, groups, bindings = case(store)
    class CompleteCostBase:
        def estimate_group(self, *_):
            return GroupCost(17.8112e-6, 17.8112e-6, 0., 0., .95, "measured_complete_call")
    profile = NativeMeasuredFirstProfile(graph, CompleteCostBase(), store, bindings)
    group = replace(groups[3], node_ids=("s", "m"))
    assert profile.estimate_group(graph, group).source == "measured_complete_call"
    assert profile.estimate_group(graph, group).duration == 17.8112e-6


def test_existing_scheduler_consumes_sources_and_keeps_context_limitation(store):
    graph, plan, base, _, bindings = case(store)
    profile = NativeMeasuredFirstProfile(graph, base, store, bindings)
    estimate = StructuralCostModel(strict_fusion=False).predict(graph, plan, profile)
    assert estimate.makespan * 1e6 == pytest.approx(10.172800123691559)
    assert [g["source"] for g in estimate.details["groups"]][-2:] == ["measured_native_isolated_stage"] * 2
    assert "measurement_context:isolated_stage" in estimate.ood_reasons
    # This is a scheduler estimate from isolated profiles, not new GPU chain time.


def test_unknown_analytic_cost_propagates_instead_of_defaulting_to_zero(store):
    graph, _, _, groups, _ = case(store)
    class UnknownBase:
        def estimate_group(self, *_):
            raise ValueError("Graph parameters unresolved")
    profile = NativeMeasuredFirstProfile(graph, UnknownBase(), store, {})
    with pytest.raises(ValueError, match="unresolved"):
        profile.estimate_group(graph, groups[3])


def test_loading_measured_costs_does_not_run_analytic_plan(monkeypatch):
    from calibration import native_stage_calibration
    def forbidden():
        raise AssertionError("Analytic model planning is not needed for lookup")
    monkeypatch.setattr(native_stage_calibration, "plan", forbidden)
    loaded = ReviewedStageStore(ROOT / "data/native_stage_calibration_20261105_01/raw")
    assert len(loaded.records) == 2


def test_changed_raw_measurement_cannot_reuse_the_review_receipt(tmp_path):
    import shutil
    raw = tmp_path / "raw"
    shutil.copytree(ROOT / "data/native_stage_calibration_20261105_01/raw", raw)
    shutil.copytree(ROOT / "data/native_stage_calibration_20261105_01/review", tmp_path / "review")
    with (raw / "attempt_00.json").open("a") as f:
        f.write(" ")
    with pytest.raises(ValueError, match="receipt changed"):
        ReviewedStageStore(raw)
