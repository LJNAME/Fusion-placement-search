"""Cross-layer measured Profile reuse: matching by condition, never by node name."""
import copy

import pytest

pytest.importorskip("lightop")  # store replays the LightOP admission (vendor env)

from cost_model.hardware_profile import AnalyticHardwareProfile
from cost_model.layer_invariant_profile import LayerInvariantMeasuredProfile, LayerInvariantStore, contract_class
from cost_model.search_evaluation import build_evaluation_graph
from cost_model.structural import StructuralCostModel
from integration.lightop.add_rmsnorm import executable_registry
from simulator import FusionGroup, Plan
from simulator.hardware import load_hardware


@pytest.fixture(scope="module")
def setup():
    reg = executable_registry()
    hw = load_hardware("data/bw1000_2card/bw1000_2card.calibrated.json")
    graph = build_evaluation_graph("model")
    prof = LayerInvariantMeasuredProfile(AnalyticHardwareProfile(hw), LayerInvariantStore())
    return graph, prof, reg


def groups_for(graph, reg, layer, device, fuse=()):
    pairs = {f"L{layer:03d}.{a}": f"L{layer:03d}.{b}" for a, b in fuse}
    skip, groups = set(pairs), []
    for n in graph.topological_order():
        if n in skip:
            continue
        if n in pairs.values():
            first = next(k for k, v in pairs.items() if v == n)
            pattern = "silu_mul" if "silu" in first else "add_rmsnorm"
            groups.append(FusionGroup([first, n], device, {"pattern_id": pattern}))
        else:
            groups.append(FusionGroup([n], device if n.startswith(f"L{layer:03d}.") else 0, {"pattern_id": "unfused"}))
    plan = Plan(groups=groups)
    return {g.node_ids: g for g in StructuralCostModel(registry=reg).prepare(graph, plan, 2).structure.groups}


@pytest.mark.parametrize("device", [0, 1])
def test_same_conditions_in_different_layers_hit_the_same_record(setup, device):
    graph, prof, reg = setup
    costs = {}
    for layer in (0, 13, 27):
        g = groups_for(graph, reg, layer, device)
        for local in ("mlp.silu", "mlp.gated_mul", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            cost = prof.estimate_group(graph, g[(f"L{layer:03d}.{local}",)])
            assert cost.source == "measured_layer_invariant"
            costs.setdefault(local, set()).add(cost.duration)
    assert all(len(v) == 1 for v in costs.values())
    assert costs["mlp.gate_proj"] == costs["mlp.up_proj"]  # same m/n/k condition


def test_fused_and_add_rmsnorm_sources_are_explicit(setup):
    graph, prof, reg = setup
    g = groups_for(graph, reg, 5, 1, fuse=(("mlp.silu", "mlp.gated_mul"), ("residual_add_1", "pre_mlp_norm")))
    fused = prof.estimate_group(graph, g[("L005.mlp.silu", "L005.mlp.gated_mul")])
    assert fused.source == "measured_layer_invariant" and fused.duration == pytest.approx(18.9792e-6, rel=1e-4)
    norm = prof.estimate_group(graph, g[("L005.residual_add_1", "L005.pre_mlp_norm")])
    assert norm.source != "measured_layer_invariant" and "measured_profile_miss:add_rmsnorm_not_measured" in norm.ood_reasons
    attention = prof.estimate_group(graph, g[("L005.self_attention.q_proj",)])
    assert attention.source != "measured_layer_invariant"  # different m/n/k: no measurement


def test_condition_change_misses(setup):
    graph, prof, reg = setup
    changed = copy.deepcopy(graph)
    changed.nodes["L003.mlp.silu"].extra["strides"] = [1, 1, 1]
    g = groups_for(changed, reg, 3, 0)
    cost = prof.estimate_group(changed, g[("L003.mlp.silu",)])
    assert "measured_profile_miss:dtype_or_layout_outside_contract" in cost.ood_reasons
    with pytest.raises(ValueError):
        contract_class({"mode": "eager", "calls_per_replay": 50, "statistic": "median", "unit": "us/call"})
