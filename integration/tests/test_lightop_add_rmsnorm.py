"""CPU checks for the Add+RMSNorm graph adapter; GPU numerics run separately."""
import copy

import pytest

from cost_model.hardware_profile import AnalyticHardwareProfile
from cost_model.search_evaluation import build_evaluation_graph
from cost_model.structure import analyze_plan_structure
from simulator import FusionGroup, Plan, validate_fusion_plan
from simulator.fusion_rules import FusionRuleError, default_fusion_registry, enumerate_fusion_candidates
from simulator.hardware import load_hardware

lightop = pytest.importorskip("integration.lightop.add_rmsnorm")


@pytest.fixture(scope="module")
def layer():
    return build_evaluation_graph("layer")


def test_candidates_are_intra_layer_add_then_norm(layer):
    reg = lightop.executable_registry()
    multi = [c for c in enumerate_fusion_candidates(layer, reg, require_executable=True) if len(c.node_ids) > 1]
    assert {(c.pattern_id, c.node_ids) for c in multi} == {
        ("add_rmsnorm", ("L000.residual_add_1", "L000.pre_mlp_norm")),
        ("silu_mul", ("L000.mlp.silu", "L000.mlp.gated_mul"))}
    model = build_evaluation_graph("model")
    counts = {}
    for c in enumerate_fusion_candidates(model, reg, require_executable=True):
        if len(c.node_ids) > 1:
            counts[c.pattern_id] = counts.get(c.pattern_id, 0) + 1
    assert counts == {"add_rmsnorm": 28, "silu_mul": 28}
    assert default_fusion_registry().get("silu_mul").implementation_status == "planned"  # default untouched


def test_group_with_exported_add_is_legal_and_has_two_outputs(layer):
    reg = lightop.executable_registry()
    group = ("L000.residual_add_1", "L000.pre_mlp_norm")
    order = [n for n in layer.topological_order() if n not in group[:-1]]
    plan = Plan(groups=[FusionGroup(list(group), 0, {"pattern_id": "add_rmsnorm"}) if n == group[-1]
                        else FusionGroup([n], 0, {"pattern_id": "unfused"}) for n in order])
    validate_fusion_plan(layer, plan, 2, registry=reg, require_executable=True)
    structure = analyze_plan_structure(layer, plan, 2, registry=reg)
    fused = next(g for g in structure.groups if g.pattern_id == "add_rmsnorm")
    assert fused.output_bytes == 2 * 524288  # add output (residual stream) + norm output


@pytest.mark.parametrize("bad", ["shape", "layer", "norm_input"])
def test_rejects_outside_evidenced_contract(layer, bad):
    graph = copy.deepcopy(layer)
    group = ("L000.residual_add_1", "L000.pre_mlp_norm")
    if bad == "shape":
        graph.nodes[group[0]].extra["output_shape"] = (1, 512, 1024)
    elif bad == "layer":
        graph.nodes[group[1]].extra["layer"] = 1
    else:
        from simulator.schema import TensorEdge
        graph.add_edge(TensorEdge("L000.input", group[1], 524288.0))
    with pytest.raises(FusionRuleError):
        lightop.group_io(graph, group) if bad != "layer" else lightop._LightopAddRmsnormPattern().validate(graph, group)
