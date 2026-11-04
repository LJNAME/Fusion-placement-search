"""CPU fixtures exercise the adapter contract; never GPU or latency evidence."""
from contextlib import nullcontext
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from cost_model.search_evaluation import build_evaluation_graph
from integration.lightop import capabilities
from integration.lightop import silu_mul as execution
from simulator import enumerate_fusion_candidates, load_hardware, make_reference_plan, TensorEdge
from simulator.placement_artifacts import PlacedOpGraph, load_placed_op_graph, save_placed_op_graph


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fake_operator(output, packed):
    gate, up = packed.float().chunk(2, dim=-1)
    output.copy_((gate * torch.sigmoid(gate) * up).to(output.dtype))


@pytest.fixture
def case(monkeypatch):
    monkeypatch.setattr(execution, "check_installed", lambda: {"scope": "CPU test fixture only"})
    operator = Mock(side_effect=fake_operator)
    monkeypatch.setattr(execution, "resolve", lambda capability: operator if capability == execution.CAPABILITY else None)
    adapter = execution.LightopSiluMulAdapter()
    graph = build_evaluation_graph("small")
    candidate = next(c for c in enumerate_fusion_candidates(graph, adapter.registry, require_executable=True)
                     if c.pattern_id == "silu_mul")
    plan = make_reference_plan(graph, registry=adapter.registry)
    plan.groups = [g for g in plan.groups if g.node_ids[0] not in candidate.node_ids]
    fused = candidate.to_group(0)
    fused.extra["implementation"] = copy.deepcopy(adapter.implementation)
    plan.groups.insert(len(plan.groups) - 1, fused)
    hardware = load_hardware(ROOT / "data/bw1000_2card/bw1000_2card.calibrated.json")
    placed = PlacedOpGraph(graph, hardware, plan, extra={"deployable": False})
    index = next(i for i, group in enumerate(plan.groups) if len(group.node_ids) == 2)
    bound = adapter.bind(placed, index, {0: "cuda:0"})
    gate = torch.randn(execution.SHAPE, dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    values = {bound.gate_node: gate, bound.up_node: up}
    return adapter, placed, index, bound, values, operator


def prepare_cpu(monkeypatch, bound, values):
    # Only these tests bypass GPU guards. The production API has no CPU flag.
    monkeypatch.setattr(execution, "_check_device", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "device", lambda *_: nullcontext())
    return bound.prepare(values)


def test_packs_split_inputs_on_every_call_with_independent_output(case, monkeypatch):
    _, _, _, bound, values, operator = case
    gate, up = values[bound.gate_node], values[bound.up_node]
    snapshots = [x.view(torch.uint8).clone() for x in (gate, up)]
    call = prepare_cpu(monkeypatch, bound, values)
    operator.assert_not_called()
    for _ in range(3):
        output = call()
        expected = F.silu(gate.float()) * up.float()
        torch.testing.assert_close(output.float(), expected, rtol=1e-2, atol=1e-2)
        assert not torch._C._overlaps(output, gate) and not torch._C._overlaps(output, up)
        assert all(torch.equal(x.view(torch.uint8), old) for x, old in zip((gate, up), snapshots))
    assert operator.call_count == 3
    assert values.keys() == {bound.gate_node, bound.up_node}
    values[call.output_node] = call()
    assert values[bound.output_node] is call.output
    assert bound.node_ids[0] not in values
    assert values[bound.gate_node] is gate and values[bound.up_node] is up


def test_preallocated_timed_body_has_cat_before_kernel_and_no_allocation(case, monkeypatch):
    _, _, _, bound, values, operator = case
    call = prepare_cpu(monkeypatch, bound, values)
    original_cat, events = torch.cat, []

    def cat(tensors, *, dim, out):
        events.append("cat")
        assert out is call._packed and dim == -1
        return original_cat(tensors, dim=dim, out=out)

    def op(out, packed):
        events.append("kernel")
        assert out is call._output_2d and packed is call._packed
        fake_operator(out, packed)

    monkeypatch.setattr(torch, "cat", cat)
    operator.side_effect = op
    monkeypatch.setattr(torch, "empty", lambda *_, **__: pytest.fail("allocation entered the timed body"))
    pointer = call.output.data_ptr()
    call()
    # Updated inputs must be re-packed; a stale prepacked-only call would fail.
    values[bound.gate_node].fill_(2)
    call()
    torch.testing.assert_close(call.output.float(), F.silu(values[bound.gate_node].float()) * values[bound.up_node].float(),
                               rtol=1e-2, atol=1e-2)
    assert events == ["cat", "kernel", "cat", "kernel"]
    assert pointer == call.output.data_ptr()
    assert call.observation["explicit_buffer_bytes"] == 4718592
    assert call.observation["tensors"]["gate"]["layout_provenance"]["device"] == "cpu"
    assert not call.observation["gpu_execution_verified"] and not call.observation["profile_accepted"]


def test_distinct_live_outputs_need_distinct_prepared_calls(case, monkeypatch):
    _, _, _, bound, values, _ = case
    first = prepare_cpu(monkeypatch, bound, values)
    second = prepare_cpu(monkeypatch, bound, values)
    first()
    saved = first.output.clone()
    values[bound.up_node].zero_()
    second()
    assert torch.equal(first.output, saved) and torch.count_nonzero(second.output) == 0
    assert not torch._C._overlaps(first.output, second.output)
    assert not torch._C._overlaps(first._packed, second._packed)


def test_read_only_input_storage_aliases_are_safe(case, monkeypatch):
    _, _, _, bound, values, _ = case
    values[bound.up_node] = values[bound.gate_node]
    gate = values[bound.gate_node]
    old = gate.view(torch.uint8).clone()
    call = prepare_cpu(monkeypatch, bound, values)
    output = call()
    torch.testing.assert_close(output.float(), F.silu(gate.float()) * gate.float(), rtol=1e-2, atol=1e-2)
    assert torch.equal(gate.view(torch.uint8), old)
    assert call.observation["storage_overlap"] == {"gate_up": True, "output_gate": False, "output_up": False}


def test_cpu_cannot_reach_dispatch(case):
    _, _, _, bound, values, operator = case
    with pytest.raises(RuntimeError, match="no CPU/native fallback"):
        bound.prepare(values)
    operator.assert_not_called()


@pytest.mark.parametrize("bad", ["dtype", "shape", "strided", "grad", "non_tensor"])
def test_runtime_mismatch_has_no_implicit_conversion(case, monkeypatch, bad):
    _, _, _, bound, values, operator = case
    original = values[bound.gate_node]
    replacements = {
        "dtype": original.half(), "shape": original.squeeze(0),
        "strided": torch.empty(1, 256, 6144, dtype=original.dtype)[..., ::2],
        "grad": original.clone().requires_grad_(), "non_tensor": [1, 2, 3],
    }
    values[bound.gate_node] = replacements[bad]
    with pytest.raises((TypeError, ValueError)):
        prepare_cpu(monkeypatch, bound, values)
    operator.assert_not_called()


@pytest.mark.parametrize("field,value", [("strides", [1, 1, 1]), ("layout", "strided")])
def test_explicit_layout_declarations_are_not_overridden(case, monkeypatch, field, value):
    _, placed, _, bound, values, operator = case
    placed.graph.nodes[bound.gate_node].extra[field] = value
    with pytest.raises(ValueError, match="graph"):
        prepare_cpu(monkeypatch, bound, values)
    operator.assert_not_called()


def test_unknown_graph_layout_remains_unknown_despite_runtime_observation(case, monkeypatch):
    _, placed, _, bound, values, _ = case
    before = copy.deepcopy(placed.graph.nodes[bound.gate_node].extra)
    call = prepare_cpu(monkeypatch, bound, values)
    assert call.observation["tensors"]["gate"]["layout"] == "contiguous"
    assert placed.graph.nodes[bound.gate_node].extra == before
    assert "layout" not in before


def test_output_layout_declaration_must_also_match(case, monkeypatch):
    _, placed, _, bound, values, operator = case
    placed.graph.nodes[bound.output_node].extra["layout"] = "strided"
    with pytest.raises(ValueError, match="graph layout"):
        prepare_cpu(monkeypatch, bound, values)
    operator.assert_not_called()


def test_cross_device_inputs_are_not_relocated(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "CPU test fixture")
    gate = SimpleNamespace(is_cuda=True, device=torch.device("cuda:0"))
    up = SimpleNamespace(is_cuda=True, device=torch.device("cuda:1"))
    with pytest.raises(ValueError, match="already be on the placed runtime device"):
        execution._check_device(gate, up, torch.device("cuda:0"))


@pytest.mark.parametrize("bad", ["shape", "dtype", "layer", "phase", "export_silu", "silu_consumer"])
def test_graph_semantics_must_match_the_narrow_case(case, bad):
    adapter, placed, index, bound, _, operator = case
    node = placed.graph.nodes[bound.node_ids[0]]
    if bad == "shape":
        node.extra["output_shape"] = [1, 256, 4096]
    elif bad == "dtype":
        node.extra["dtype"] = "fp16"
    elif bad == "layer":
        placed.graph.nodes[bound.up_node].extra.pop("layer")
    elif bad == "phase":
        node.extra["phase"] = "backward"
    elif bad == "export_silu":
        placed.graph.extra["output_nodes"] = [node.id]
    else:
        placed.graph.add_edge(TensorEdge(node.id, "L000.mlp.down_proj"))
    with pytest.raises(ValueError):
        adapter.bind(placed, index, {0: "cuda:0"})
    operator.assert_not_called()


def test_gate_external_consumers_and_exports_are_allowed_without_mutation(case, monkeypatch):
    adapter, placed, index, bound, values, _ = case
    placed.graph.add_edge(TensorEdge(bound.gate_node, "L000.mlp.down_proj"))
    placed.graph.extra["output_nodes"] = [bound.gate_node]
    bound = adapter.bind(placed, index, {0: "cuda:0"})
    old = values[bound.gate_node].clone()
    prepare_cpu(monkeypatch, bound, values)()
    assert torch.equal(old, values[bound.gate_node])


@pytest.mark.parametrize("kind", ["missing_implementation", "other_implementation", "placement", "edges"])
def test_bound_identity_and_input_roles_cannot_silently_change(case, monkeypatch, kind):
    _, placed, index, bound, values, operator = case
    group = placed.plan.groups[index]
    if kind == "missing_implementation":
        group.extra.pop("implementation")
    elif kind == "other_implementation":
        group.extra["implementation"] = {"id": "kerv"}
    elif kind == "placement":
        group.device = 1
    else:
        for edge in placed.graph.in_edges(bound.node_ids[0]):
            edge.src = bound.up_node
    with pytest.raises(ValueError):
        prepare_cpu(monkeypatch, bound, values)
    operator.assert_not_called()


@pytest.mark.parametrize("device_map", [{}, {0: "cpu"}, {0: "cuda"}])
def test_runtime_device_mapping_must_be_explicit(case, device_map):
    adapter, placed, index, _, _, _ = case
    with pytest.raises(ValueError):
        adapter.bind(placed, index, device_map)


def test_exception_never_substitutes_native_or_publishes_output(case, monkeypatch):
    _, _, _, bound, values, operator = case
    call = prepare_cpu(monkeypatch, bound, values)
    old = {name: x.clone() for name, x in values.items()}
    operator.side_effect = RuntimeError("vendor dispatch failed")
    with pytest.raises(RuntimeError, match="vendor dispatch failed"):
        values[call.output_node] = call()
    assert set(values) == set(old)
    assert all(torch.equal(values[name], x) for name, x in old.items())


def test_package_identity_failure_prevents_resolve(monkeypatch):
    loader = Mock()
    monkeypatch.setattr(execution, "resolve", loader)
    def fail():
        raise ValueError("pinned implementation mismatch")
    monkeypatch.setattr(execution, "check_installed", fail)
    with pytest.raises(ValueError, match="pinned implementation mismatch"):
        execution.LightopSiluMulAdapter()
    loader.assert_not_called()


def test_edge_order_and_artifact_roundtrip_preserve_roles(case, tmp_path):
    adapter, placed, index, bound, _, _ = case
    placed.graph.edges.reverse()
    path = tmp_path / "placed.json"
    save_placed_op_graph(path, placed)
    recovered = load_placed_op_graph(path, registry=adapter.registry)
    rebound = adapter.bind(recovered, index, {0: "cuda:0"})
    assert (rebound.gate_node, rebound.up_node) == (bound.gate_node, bound.up_node)
    assert recovered.plan.groups[index].extra["implementation"] == adapter.implementation


def test_opt_in_rule_does_not_promote_default_registry_or_frozen_evidence():
    from simulator import default_fusion_registry
    from integration.lightop.catalog import catalog
    assert not default_fusion_registry().get("silu_mul").executable
    assert execution.adapter_registry().get("silu_mul").executable
    graph = build_evaluation_graph("model")
    matched = [c for c in enumerate_fusion_candidates(graph, execution.adapter_registry(), require_executable=True)
               if c.pattern_id == "silu_mul"]
    assert len(matched) == 28
    frozen = next(c for c in catalog()["capabilities"] if c["id"] == execution.CAPABILITY)
    assert not frozen["placed_graph_adapter"]
    live = [c for c in capabilities() if c["placed_graph_adapter"]]
    assert [c["id"] for c in live] == [execution.CAPABILITY]
    assert live[0]["project_adapter"]["validation"] == "cpu_contract_only_gpu_pending"
    assert not live[0]["project_adapter"]["gpu_execution_verified"]
    assert live[0]["project_adapter"]["accepted_profiles"] == 0
