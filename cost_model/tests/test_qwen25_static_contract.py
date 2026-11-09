"""Static declarations reuse real local costs, never assert model observations."""
import copy

import pytest

from cost_model.native_add_rmsnorm_profile import load_record
from cost_model.qwen25_add_rmsnorm import NativeChainCostModel,annotate,reference_plan,search_registry
from cost_model.qwen25_static_contract import StaticContractProfile,conditions,qualify_static
from cost_model.profile_hardware_class import hardware_class_audit
from cost_model.search_evaluation import qwen25_3b_spec
from simulator import FusionGroup
from simulator.model_graphs import build_model_graph
from simulator.placement_artifacts import PlacedOpGraph,graph_fingerprint
from tools.evaluate_qwen25_add_rmsnorm import make_case
from tools.run_qwen25_static_search import export_plan


@pytest.fixture(scope="module")
def case():
    graph=build_model_graph(qwen25_3b_spec());_,base,_=make_case(graph,conditional=False)
    native=load_record("data/native_add_rmsnorm_timing_01")
    profile=StaticContractProfile(graph,base.hardware,base.adapter,base.measured,native)
    registry=search_registry(profile)
    model=NativeChainCostModel(registry=registry,scheduling_policy="dependency_level_node_id_v1")
    return graph,profile,registry,model


def test_all_layers_match_conditions_without_model_observations(case):
    graph,p,_,_=case
    assert len(p.reuse)==len(p.native_reuse)==72 and not p.conditional
    assert not p.declaration["model_runtime_observed"]
    assert {d for _,d in p.reuse}=={0,1}
    assert len({q["condition_id"] for q in p.qualifications.values() if q["profile"] in
                ("static_contract_equivalent","hardware_class_transfer")})==1
    assert all("layout" not in graph.nodes[n].extra for pair,_ in p.reuse for n in pair)
    pair=("L000.residual_add_1","L000.pre_mlp_norm")
    assert p.qualifications[(pair,1)]["profile"]=="hardware_class_transfer"
    assert p.qualifications[(pair,1)]["measured_on"]==0
    assert p.qualifications[(pair,1)]["reuse_on"]==1
    assert p.qualifications[(pair,1)]["target_device_independently_measured"] is False
    assert p.qualifications[(("L035.residual_add_2","final_norm"),0)]["capability"]=="not_fusible"


def test_only_provenance_is_ignored_in_complete_condition_comparison():
    first={"shape":[1,256,2048],"strides":[524288,2048,1],"layout_provenance":{"kind":"runtime_tensor"}}
    second=copy.deepcopy(first);second["layout_provenance"]={"kind":"graph_declaration"}
    assert conditions(first)==conditions(second)
    second["strides"]=[1,256,2048]
    assert conditions(first)!=conditions(second)


def test_physical_identity_does_not_block_hardware_class_match(case):
    _,p,_,_=case
    key=copy.deepcopy(p.accepted["kernel_key"])
    moved=copy.deepcopy(key)
    moved["hardware"].update(device_index=1,physical_index=1)
    moved["hardware"]["source"]["execution_device"]="cuda:1"
    assert conditions(key)==conditions(moved)
    moved["hardware"]["source"]["architecture"]="gfx90a"
    assert conditions(key)!=conditions(moved)
    moved=copy.deepcopy(key);moved["runtime"]["driver_version"]="another"
    assert conditions(key)!=conditions(moved)


def test_hardware_fingerprints_are_equal_and_heterogeneous_cards_are_rejected(case):
    _,p,_,_=case
    evidence=p.declaration["hardware_class_audit"]
    assert evidence["software_stack_match"]
    assert evidence["devices"]["0"]["hardware_class_key"]==evidence["devices"]["1"]["hardware_class_key"]
    changed=copy.deepcopy(p.hardware);changed.devices[1].extra["class"]="different_gpu"
    report=hardware_class_audit(changed,p.accepted)
    assert report["devices"]["0"]["class_and_stack_match"]
    assert not report["devices"]["1"]["class_and_stack_match"]


def test_gpu1_cost_is_transferred_and_keeps_measured_origin(case):
    graph,p,registry,_=case
    pair=("L002.residual_add_1","L002.pre_mlp_norm")
    group=next(g for g in reference_plan(graph,registry,fused=True,device=1).groups if tuple(g.node_ids)==pair)
    cost=p.estimate_group(graph,group)
    assert cost.duration==p.accepted["latency"]["duration_s"]
    assert cost.source=="transferred_hardware_class"
    assert "hardware_class_transfer_not_target_measurement" in cost.ood_reasons
    assert p.native_reuse[(pair,1)].duration==p.native_reuse[(pair,0)].duration
    assert p.native_reuse[(pair,1)].source=="transferred_native_chain_hardware_class"
    assert p.accepted["kernel_key"]["hardware"]["device_index"]==0


@pytest.mark.parametrize("bad",["shape","dtype","strides","epsilon","alias","producer","missing","runtime","contract"])
def test_unproved_or_conflicting_position_is_not_priced_as_fused(case,bad):
    original,p,_,_=case;g=copy.deepcopy(original);declaration=copy.deepcopy(p.declaration)
    pair=("L001.residual_add_1","L001.pre_mlp_norm");x="L001.self_attention.out_proj"
    if bad=="shape": g.nodes[x].extra["output_shape"]=(1,256,1024)
    elif bad=="dtype": g.nodes[x].extra["dtype"]="fp32"
    elif bad=="strides": g.nodes[x].extra["strides"]=[1,1,1]
    elif bad=="epsilon":g.nodes[pair[1]].extra["eps"]=1e-5
    elif bad=="alias": g.nodes[x].extra["alias_of"]="L001.input"
    elif bad=="producer":g.nodes[x].op_type="view"
    elif bad=="missing":del g.nodes[x].extra["output_shape"]
    elif bad=="runtime":declaration["target_runtime"]["torch_build_sha256"]="different"
    else: declaration["fused_measurement_contract"]["calls_per_replay"]=100
    q,request=qualify_static(g,p.hardware,p.adapter,pair,0,declaration,p.accepted,p.native_record)
    assert request is None and q["profile"]!="static_contract_equivalent"


def test_measured_chain_and_fusion_costs_keep_original_records(case):
    graph,p,registry,model=case;before=graph_fingerprint(graph)
    native=reference_plan(graph,registry,fused=False);fused=reference_plan(graph,registry,fused=True)
    a=model.predict(graph,native,p);b=model.predict(graph,fused,p)
    delta=p.native_record["duration_s"]-p.accepted["latency"]["duration_s"]
    assert a.makespan-b.makespan==pytest.approx(36*delta)
    assert all(cost.launch_time==cost.memory_time==0 for cost in p.native_reuse.values())
    assert len(native.groups)==687 and len(fused.groups)==651 and graph_fingerprint(graph)==before


def test_native_fallback_export_does_not_assign_chain_timing_to_original_nodes(case):
    graph,p,registry,model=case
    plan=reference_plan(graph,registry,fused=True)
    for i,group in enumerate(plan.groups):
        if group.node_ids==["L000.residual_add_1","L000.pre_mlp_norm"]:
            plan.groups[i:i+1]=[FusionGroup([n],0,{"pattern_id":"unfused"}) for n in group.node_ids]
            break
    plan=annotate(plan,p)
    for group in plan.groups:
        if len(group.node_ids)>1:
            group.extra["measurement_ref"]={"measured_kernel_key_id":p.accepted["kernel_key_id"]}
    estimate=model.predict(graph,plan,p)
    placed=PlacedOpGraph(graph,p.hardware,plan,predicted_makespan=estimate.makespan,predicted_peak_memory=estimate.peak_memory)
    artifact=export_plan(placed,estimate,p,registry)
    assert artifact["summary"]["fused_groups"]==35
    assert artifact["summary"]["nodes"]==687
    assert len(artifact["native_chain_execution_units"])==1
    assert not any("predicted" in g for g in artifact["groups"])
