"""Real graph contracts with mocked CPU dispatch; not GPU adapter evidence."""
import copy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from calibration.kerv_add_rmsnorm_probe import EPS, SHAPE, references, synthetic_inputs
from cost_model.search_evaluation import qwen25_3b_spec, qwen3_spec
from cost_model.structure import analyze_plan_structure
from integration.flagscale import kerv_add_rmsnorm as execution
from simulator import FusionGroup, Plan, TensorEdge, validate_fusion_plan
from simulator.fusion_rules import default_fusion_registry, enumerate_fusion_candidates
from simulator.hardware import load_hardware
from simulator.model_graphs import build_model_graph
from simulator.placement_artifacts import PlacedOpGraph


GROUP = ("L000.residual_add_1", "L000.pre_mlp_norm")
X, RESIDUAL = "L000.self_attention.out_proj", "L000.input"


@pytest.fixture(scope="module")
def graph():
    return build_model_graph(qwen25_3b_spec())


def placed_for(graph, adapter):
    groups = []
    for node in graph.topological_order():
        if node == GROUP[0]:
            continue
        groups.append(FusionGroup(list(GROUP), 0, {"pattern_id": "add_rmsnorm",
                      "implementation": copy.deepcopy(adapter.implementation)}) if node == GROUP[1]
                      else FusionGroup([node], 0, {"pattern_id": "unfused"}))
    hardware = load_hardware(Path(__file__).resolve().parents[2] / "data/bw1000_2card/bw1000_2card.calibrated.json")
    return PlacedOpGraph(graph, hardware, Plan(groups)), next(i for i, g in enumerate(groups) if len(g.node_ids) > 1)


@pytest.fixture
def bound(graph):
    adapter = execution.KervAddRmsnormAdapter()
    placed, index = placed_for(copy.deepcopy(graph), adapter)
    return adapter.bind(placed, index, {0: "cuda:0"})


def values():
    x, residual, weight = synthetic_inputs()
    return {X: x, RESIDUAL: residual}, weight


def fake_module():
    def entry(x, residual, normalized_shape, weight, eps):
        assert normalized_shape == (2048,) and eps == EPS
        summed, normalized, _ = references(x, residual, weight)
        x.copy_(summed)
        return normalized, x
    return SimpleNamespace(add_rms_norm_inference=entry)


def test_36_opt_in_candidates_without_changing_existing_registries(graph):
    registry = execution.adapter_registry()
    multi = lambda g, r: [c for c in enumerate_fusion_candidates(g, r, require_executable=True) if len(c.node_ids) > 1]
    candidates = multi(graph, registry)
    assert {c.node_ids for c in candidates} == {
        (f"L{i:03d}.residual_add_1", f"L{i:03d}.pre_mlp_norm") for i in range(36)}
    assert len(candidates) == 36 and {c.pattern_id for c in candidates} == {"add_rmsnorm"}
    assert multi(graph, default_fusion_registry()) == []
    assert multi(build_model_graph(qwen3_spec(1)), registry) == []
    from integration.lightop.add_rmsnorm import executable_registry
    assert multi(graph, executable_registry()) == []


def test_group_outputs_include_residual_branch(bound):
    placed = bound.placed
    validate_fusion_plan(placed.graph, placed.plan, 2, bound.adapter.registry, require_executable=True)
    structure = analyze_plan_structure(placed.graph, placed.plan, 2, registry=bound.adapter.registry)
    fused = next(g for g in structure.groups if g.node_ids == GROUP)
    assert fused.output_bytes == 2 * 1048576
    assert "L000.residual_add_2" in placed.graph.successors(GROUP[0])


def test_mapping_publishes_two_outputs_and_consumes_only_x(bound, monkeypatch):
    live, weight = values()
    x = live[X]
    residual_before, weight_before = live[RESIDUAL].clone(), weight.clone()
    expected_sum, expected_norm, _ = references(x, live[RESIDUAL], weight)
    module = fake_module()
    monkeypatch.setattr(execution, "_check_device", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "device", lambda *args: nullcontext())
    monkeypatch.setattr(execution, "load_module", lambda *args: module)
    call = bound(live, weight)
    assert X not in live and call.consumed_node == X
    assert live[GROUP[0]] is call.add_output and live[GROUP[1]] is call.normalized_output
    assert call.add_output.data_ptr() == x.data_ptr()
    assert torch.equal(call.add_output, expected_sum) and torch.equal(call.normalized_output, expected_norm)
    assert torch.equal(live[RESIDUAL], residual_before) and torch.equal(weight, weight_before)
    assert call.observations["inputs"]["x"]["layout"] == "contiguous"
    assert "layout" not in bound.placed.graph.nodes[X].extra  # runtime evidence is not a graph declaration
    assert call.observed_jit_launches is None


@pytest.mark.parametrize("bad", ["side_consumer", "exported_x", "shape", "missing_layer", "backward", "layout", "stride", "epsilon"])
def test_static_contract_rejections(graph, bad):
    g = copy.deepcopy(graph)
    if bad == "side_consumer":
        g.add_edge(TensorEdge(X, "L000.residual_add_2"))
    elif bad == "exported_x":
        g.extra["output_nodes"] = [X]
    elif bad == "shape":
        g.nodes[X].extra["output_shape"] = (1, 256, 1024)
    elif bad == "missing_layer":
        del g.nodes[X].extra["layer"]
    elif bad == "backward":
        g.nodes[X].extra["phase"] = "backward"
    elif bad == "layout":
        g.nodes[X].extra["layout"] = "strided"
    elif bad == "stride":
        g.nodes[X].extra["strides"] = (1, 2, 3)
    elif bad == "epsilon":
        g.nodes[GROUP[1]].extra["eps"] = 1e-5
    with pytest.raises(ValueError):
        execution.group_io(g, GROUP)


def test_edge_order_does_not_select_wrong_consumed_input(graph):
    g = copy.deepcopy(graph)
    g.edges.reverse()
    assert execution.group_io(g, GROUP) == (X, RESIDUAL)


def test_cpu_cannot_reach_dispatch(bound, monkeypatch):
    live, weight = values()
    monkeypatch.setattr(bound.adapter, "_invoke", lambda *args: pytest.fail("CPU reached dispatch"))
    with pytest.raises(RuntimeError, match="no CPU/native fallback"):
        bound(live, weight)
    assert X in live and not any(n in live for n in GROUP)


@pytest.mark.parametrize("alias", ["residual", "live_view", "weight"])
def test_inplace_storage_aliases_are_rejected(bound, monkeypatch, alias):
    live, weight = values()
    if alias == "residual":
        live[RESIDUAL] = live[X].view_as(live[X])
    elif alias == "live_view":
        live["unrelated_live_tensor"] = live[X].view(-1)[:8]
    else:
        weight = live[X].view(-1)[:2048]
    monkeypatch.setattr(execution, "_check_device", lambda *args: None)
    monkeypatch.setattr(bound.adapter, "_invoke", lambda *args: pytest.fail("alias reached dispatch"))
    with pytest.raises(ValueError, match="aliases"):
        bound(live, weight)
    assert X in live


@pytest.mark.parametrize("bad", ["dtype", "shape", "strided", "grad", "weight_shape", "weight_grad"])
def test_runtime_tensor_contract_is_checked_before_dispatch(bound, monkeypatch, bad):
    live, weight = values()
    if bad == "dtype":
        live[X] = live[X].float()
    elif bad == "shape":
        live[X] = live[X][:, :1]
    elif bad == "strided":
        live[X] = torch.empty(1, 256, 4096, dtype=torch.bfloat16)[..., ::2]
    elif bad == "grad":
        live[X].requires_grad_()
    elif bad == "weight_shape":
        weight = weight[:4]
    else:
        weight.requires_grad_()
    monkeypatch.setattr(bound.adapter, "_invoke", lambda *args: pytest.fail("invalid input reached dispatch"))
    with pytest.raises(ValueError):
        bound(live, weight)
    assert X in live


def test_dispatch_partial_failure_invalidates_consumed_value(bound, monkeypatch):
    live, weight = values()
    monkeypatch.setattr(execution, "_check_device", lambda *args: None)
    def failure(x, *args):
        x.zero_()
        raise RuntimeError("partial dispatch failure")
    monkeypatch.setattr(bound.adapter, "_invoke", failure)
    with pytest.raises(RuntimeError, match="partial dispatch"):
        bound(live, weight)
    assert X not in live and not any(n in live for n in GROUP)
    assert RESIDUAL in live


@pytest.mark.parametrize("bad", ["sum_alias", "norm_alias", "norm_shape"])
def test_bad_kernel_outputs_are_not_published(bound, monkeypatch, bad):
    live, weight = values()
    monkeypatch.setattr(execution, "_check_device", lambda *args: None)
    def wrong(x, residual, *args):
        if bad == "sum_alias":
            return torch.empty_like(x), x.clone()
        if bad == "norm_alias":
            return residual, x
        return torch.empty(1), x
    monkeypatch.setattr(bound.adapter, "_invoke", wrong)
    with pytest.raises(RuntimeError):
        bound(live, weight)
    assert X not in live and not any(n in live for n in GROUP)


@pytest.mark.parametrize("bad", ["implementation", "placement", "stale_output", "missing_input"])
def test_bound_state_cannot_silently_change(bound, monkeypatch, bad):
    live, weight = values()
    group = bound.placed.plan.groups[bound.index]
    if bad == "implementation":
        group.extra["implementation"]["kernel_source_sha256"] = "wrong"
    elif bad == "placement":
        group.device = 1
    elif bad == "stale_output":
        live[GROUP[0]] = torch.empty_like(live[X])
    else:
        del live[RESIDUAL]
    monkeypatch.setattr(bound.adapter, "_invoke", lambda *args: pytest.fail("changed state reached dispatch"))
    with pytest.raises(ValueError):
        bound(live, weight)


def test_explicit_binding_is_required(graph):
    adapter = execution.KervAddRmsnormAdapter()
    placed, index = placed_for(copy.deepcopy(graph), adapter)
    placed.plan.groups[index].extra.pop("implementation")
    with pytest.raises(ValueError, match="explicitly name"):
        adapter.bind(placed, index, {0: "cuda:0"})
