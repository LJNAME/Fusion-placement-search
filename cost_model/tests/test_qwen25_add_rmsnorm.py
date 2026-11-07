"""Real 3B graph audit; fake runtime metadata below is CPU policy testing only."""
import copy
import json

import pytest

from cost_model.hardware_profile import GroupCost
from cost_model.qwen25_add_rmsnorm import (
    PROFILE_DIRECTORY, GraphPlanningProfile, IntegrationScorer, activation_slice, audit, qualify_context, reference_plan, search_registry,
)
from cost_model.search_evaluation import qwen25_3b_spec
from simulator.fusion_rules import enumerate_fusion_candidates
from simulator.model_graphs import build_model_graph
from simulator.placement_artifacts import graph_fingerprint
from tools.evaluate_qwen25_add_rmsnorm import make_case, run_mode


@pytest.fixture(scope="module")
def case():
    graph=build_model_graph(qwen25_3b_spec())
    registry,profile,model=make_case(graph)
    return graph,registry,profile,model


def context(graph,profile,pair):
    identity=json.loads((PROFILE_DIRECTORY/"raw/identity.json").read_text())
    runtime=json.loads((PROFILE_DIRECTORY/"raw/runtime.json").read_text())
    return {"evidence_kind":"model_runtime","source_id":"CPU_UNIT_TEST_FIXTURE_NOT_REAL_MODEL_EVIDENCE",
            "graph_fingerprint":graph_fingerprint(graph),"node_ids":list(pair),"device_index":0,
            "observations":copy.deepcopy(identity["tensor_observations"]),"runtime":runtime["runtime"],
            "device":runtime["device"],"measurement_contract":identity["measurement_contract"],
            "implementation":profile.adapter.implementation,
            "storage_aliases":{"x_residual":False,"x_weight":False,"x_other_live":False}}


def test_audit_includes_final_norm_and_never_fills_unknown_layouts(case):
    graph,_,profile,_=case
    fingerprint=graph_fingerprint(graph);report=audit(graph,profile)
    assert report["direct_positions"]==37 and report["static_adapter_matches"]==36
    assert report["capability_counts"]=={"fusible":0,"not_fusible":1,"insufficient_info":36}
    assert report["rows"][-1]["nodes"]==["L035.residual_add_2","final_norm"]
    assert report["exact_equivalent_profile_uses"]==0 and not profile.reuse
    assert report["matching_measured_native_chains"]==0 and report["reliable_measured_pairs"]==0
    assert report["fused_query_sources"]=={"analytic_graph_proxy":72,"unavailable":2}
    assert graph_fingerprint(graph)==fingerprint


def test_key_equivalence_needs_explicit_layer_specific_evidence(case):
    graph,_,profile,_=case;pair=("L001.residual_add_1","L001.pre_mlp_norm")
    ctx=context(graph,profile,pair)
    result=qualify_context(graph,profile.hardware,profile.adapter,pair,0,ctx,profile.accepted)
    assert result["profile"]=="exact_equivalent"
    assert result["kernel_key_id"]==profile.accepted["kernel_key_id"]
    other=("L002.residual_add_1","L002.pre_mlp_norm")
    assert qualify_context(graph,profile.hardware,profile.adapter,other,0,ctx,profile.accepted)["capability"]=="insufficient_info"


@pytest.mark.parametrize("bad",["shape","layout","alias","implementation","contract","runtime","stale"])
def test_incompatible_context_never_reuses_layer0_measurement(case,bad):
    graph,_,profile,_=case;pair=("L001.residual_add_1","L001.pre_mlp_norm");ctx=context(graph,profile,pair)
    if bad=="shape":ctx["observations"]["x"]["shape"]=[1,256,1024]
    elif bad=="layout":ctx["observations"]["x"]["layout"]="unknown"
    elif bad=="alias":ctx["storage_aliases"]["x_residual"]=True
    elif bad=="implementation":ctx["implementation"]={"id":"another"}
    elif bad=="contract":ctx["measurement_contract"]["calls_per_replay"]=100
    elif bad=="runtime":ctx["runtime"]["torch_build_sha256"]="another-build"
    else:ctx["graph_fingerprint"]="stale"
    assert qualify_context(graph,profile.hardware,profile.adapter,pair,0,ctx,profile.accepted)["profile"]!="exact_equivalent"


def test_explicit_equivalence_gate_can_price_only_qualified_usage(case):
    graph,_,original,_=case;pair=("L001.residual_add_1","L001.pre_mlp_norm")
    ctx=context(graph,original,pair)
    profile=GraphPlanningProfile(graph,original.hardware,original.adapter,original.measured,{pair[-1]+"@gpu0":ctx})
    assert len(profile.reuse)==1 and (("L002.residual_add_1","L002.pre_mlp_norm"),0) not in profile.reuse
    selected,target=profile.reuse[(pair,0)]
    cost=selected.estimate_group(selected.keys.graph,target)
    assert cost.duration==pytest.approx(5.353559851646423e-6)


def test_known_alias_violation_is_rejected_before_cost_scoring(case):
    graph,registry,original,model=case;pair=("L001.residual_add_1","L001.pre_mlp_norm")
    ctx=context(graph,original,pair);ctx["storage_aliases"]["x_other_live"]=True
    profile=GraphPlanningProfile(graph,original.hardware,original.adapter,original.measured,{pair[-1]+"@gpu0":ctx})
    scorer=IntegrationScorer(graph,profile,model,64);scorer.start_run()
    scorer(reference_plan(graph,registry,fused=True))
    assert not scorer.last_feasible and scorer.run_model_evaluations==0 and scorer.rejected_requests==1


@pytest.mark.parametrize("fused_us,want_fused",[(2.,True),(20.,False)])
def test_existing_mcts_keeps_native_when_fused_is_slower(case,monkeypatch,fused_us,want_fused):
    # Costs below are test fixtures, never included as experiment measurements.
    graph,_,_,_=case;sliced,boundaries=activation_slice(graph)
    registry,profile,model=make_case(sliced,boundaries=boundaries)
    original=profile.estimate_group
    def costs(g,group):
        if len(group.node_ids)>1:
            return GroupCost(fused_us/1e6,fused_us/1e6,0.,0.,0.,"unit_test_fixture")
        return original(g,group)
    monkeypatch.setattr(profile,"estimate_group",costs)
    result,_,_=run_mode(sliced,registry,profile,model,"joint",budget=16)
    assert bool(result["fused_groups"]) is want_fused


def test_strict_mode_does_not_enable_unknown_physical_conditions(case):
    graph,registry,original,model=case
    profile=GraphPlanningProfile(graph,original.hardware,original.adapter,original.measured,conditional=False)
    assert profile.plan_blockers(reference_plan(graph,registry,fused=True))
    assert not profile.plan_blockers(reference_plan(graph,registry,fused=False))
    assert profile.hardware.launch_overhead==pytest.approx(1.5e-6)
    assert not [c for c in enumerate_fusion_candidates(graph,search_registry(profile),require_executable=True) if len(c.node_ids)>1]
    result,_,_=run_mode(graph,search_registry(profile),profile,model,"joint",budget=16)
    assert result["fused_groups"]==0 and result["cost_model_evaluations"]<=16


def test_missing_graph_shape_is_information_gap_not_proven_incompatibility(case):
    graph,_,profile,_=case;g=copy.deepcopy(graph);pair=("L001.residual_add_1","L001.pre_mlp_norm")
    del g.nodes[pair[-1]].extra["output_shape"]
    assert qualify_context(g,profile.hardware,profile.adapter,pair,0,None,profile.accepted)["capability"]=="insufficient_info"


def test_graph_declaration_is_not_mislabeled_as_observed_runtime_layout(case):
    graph,_,profile,_=case;pair=("L001.residual_add_1","L001.pre_mlp_norm");ctx=context(graph,profile,pair)
    ctx["observations"]["x"]["layout_provenance"]={"kind":"graph_declaration","source_id":"planning_only"}
    result=qualify_context(graph,profile.hardware,profile.adapter,pair,0,ctx,profile.accepted)
    assert result["capability"]=="insufficient_info" and result["profile"]!="exact_equivalent"


def test_same_gpu_model_never_substitutes_gpu0_profile_for_gpu1(case):
    graph,_,profile,_=case;pair=("L001.residual_add_1","L001.pre_mlp_norm");ctx=context(graph,profile,pair)
    ctx["device_index"]=1
    for observation in ctx["observations"].values():observation["layout_provenance"]["device"]="cuda:1"
    result=qualify_context(graph,profile.hardware,profile.adapter,pair,1,ctx,profile.accepted)
    assert result["profile"]=="missing" and "no_device_specific_accepted_profile" in result["reasons"]
