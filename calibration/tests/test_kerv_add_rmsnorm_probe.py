"""CPU checks for source/liveness gates and evidence, not GPU accuracy."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from calibration import kerv_add_rmsnorm_probe as probe
from cost_model.search_evaluation import qwen25_3b_spec
from simulator.model_graphs import build_model_graph
from simulator.schema import TensorEdge


@pytest.fixture(scope="module")
def graph():
    return build_model_graph(qwen25_3b_spec())


def test_existing_graph_keeps_dual_output_and_unknown_layout(graph):
    result = probe.graph_contract(graph)
    assert result["eligible_graph_positions"] == 36
    assert all(row["add_outside_consumers"] == [f"L{i:03d}.residual_add_2"]
               and set(row["layouts"].values()) == {"unknown"}
               for i, row in enumerate(result["groups"]))
    assert not result["runtime_aliases_verified"]
    assert not result["runtime_producer_layouts_verified"]


@pytest.mark.parametrize("extra_use", ["consumer", "export"])
def test_overwrite_gate_rejects_live_attention_output(graph, extra_use):
    graph = copy.deepcopy(graph)
    if extra_use == "consumer":
        graph.add_edge(TensorEdge("L000.self_attention.out_proj", "L000.residual_add_2"))
    else:
        graph.extra["output_nodes"] = ["L000.self_attention.out_proj"]
    result = probe.graph_contract(graph)
    assert result["eligible_graph_positions"] == 35
    assert not result["groups"][0]["graph_overwrite_precondition"]


def test_source_mismatch_is_rejected_before_import(tmp_path):
    path = tmp_path / probe.SOURCE
    path.parent.mkdir(parents=True)
    path.write_text("raise RuntimeError('must not import')\n")
    with pytest.raises(ValueError, match="audited snapshot"):
        probe.load_module(tmp_path)


def test_references_use_bf16_add_and_separate_rounding_boundary():
    x, residual, weight = probe.synthetic_inputs()
    summed, handwritten, aten = probe.references(x, residual, weight)
    assert torch.equal(summed, x + residual)
    y = summed.float() * torch.rsqrt(summed.float().square().mean(-1, keepdim=True) + probe.EPS)
    assert torch.equal(handwritten, y.bfloat16() * weight)
    assert torch.equal(aten, torch.nn.functional.rms_norm(summed, (2048,), weight, eps=probe.EPS))


def test_dispatch_restores_entry_on_failure():
    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                raise RuntimeError("launch failed")
            return launch
    original = Kernel()
    module = SimpleNamespace(**{probe.KERNEL: original})

    def entry(*args):
        return getattr(module, probe.KERNEL)[(256,)](num_warps=4)
    module.add_rms_norm_inference = entry
    with pytest.raises(RuntimeError, match="launch failed"):
        probe.dispatch_once(module, None, None, None)
    assert getattr(module, probe.KERNEL) is original


def test_trace_requires_actual_target_kernel(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"traceEvents": [{"cat": "kernel", "name": "aten_native_fallback"}]}))
    with pytest.raises(RuntimeError, match="target AddRMS"):
        probe.verify_trace(path)
    path.write_text(json.dumps({"traceEvents": [{"cat": "kernel", "name": probe.KERNEL}]}))
    assert probe.verify_trace(path)["kernel_count"] == 1


def test_no_gpu_cannot_produce_success_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        probe.run(tmp_path, tmp_path, "cuda:0", {})
    assert not (tmp_path / "result.json").exists()


def test_adapter_probe_prepares_existing_first_group_without_gpu():
    from calibration.kerv_silu_mul import DEFAULT_FLAGSCALE
    bound, descriptor = probe.prepare_adapter(DEFAULT_FLAGSCALE, "cuda:0")
    assert bound.node_ids == ("L000.residual_add_1", "L000.pre_mlp_norm")
    assert bound.x_node == "L000.self_attention.out_proj"
    assert descriptor["implementation"] == bound.adapter.implementation
    assert descriptor["gpu_mapping_previously_verified"] is False
    assert bound.adapter._module is None  # no Triton/GPU load in CPU preparation


@pytest.mark.parametrize("consume", [True, False])
def test_adapter_probe_checks_publication_and_input_invalidation(consume):
    class Bound:
        x_node, residual_node, node_ids = "x", "residual", ("add", "norm")

        def __call__(self, values, weight, *, observe_dispatch):
            assert observe_dispatch
            x = values["x"]
            if consume:
                del values["x"]
            normalized = x.clone()
            values["add"], values["norm"] = x, normalized
            return SimpleNamespace(add_output=x, normalized_output=normalized, observed_jit_launches=1)

    if not consume:
        with pytest.raises(RuntimeError, match="live tensor mapping"):
            probe.invoke_case(None, torch.ones(1), torch.zeros(1), torch.ones(1), Bound())
    else:
        outputs, dispatch, mapping = probe.invoke_case(None, torch.ones(1), torch.zeros(1), torch.ones(1), Bound())
        assert len(outputs) == 2 and all(mapping.values())
        assert dispatch == [{"kernel": probe.KERNEL, "count": 1}]
