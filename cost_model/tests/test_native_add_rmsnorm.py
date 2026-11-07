"""Native chain accounting and contract gates; GPU timings below are fixtures."""
import copy
import json

import pytest
import torch

from calibration import native_add_rmsnorm_timing as timing
from calibration.kerv_add_rmsnorm_probe import prepare_adapter, synthetic_inputs, references
from calibration.kerv_silu_mul import DEFAULT_FLAGSCALE
from cost_model import native_add_rmsnorm_profile as profiles
from cost_model.hardware_profile import GroupCost
from cost_model.qwen25_add_rmsnorm import (
    PROFILE_DIRECTORY, activation_slice, audit, reference_plan,
)
from cost_model.search_evaluation import qwen25_3b_spec
from simulator.model_graphs import build_model_graph
from simulator.placement_artifacts import graph_fingerprint
from tools.evaluate_qwen25_add_rmsnorm import make_case, run_mode


def test_native_executes_whole_chain_and_preserves_all_inputs():
    x,residual,weight=synthetic_inputs()
    before=[t.clone() for t in (x,residual,weight)]
    summed,normalized=timing.native_call(x,residual,weight)
    expected=references(x,residual,weight)
    assert torch.equal(summed,expected[0]) and torch.equal(normalized,expected[2])
    assert all(torch.equal(a,b) for a,b in zip((x,residual,weight),before))
    assert len({t.untyped_storage().data_ptr() for t in (x,residual,weight,summed,normalized)})==5


def test_native_plan_reuses_existing_fused_without_changing_its_identity():
    before=(PROFILE_DIRECTORY/"review/profile.json").read_bytes()
    frozen=timing.plan(DEFAULT_FLAGSCALE)
    assert frozen["graph_contract"]["eligible_graph_positions"]==36
    assert frozen["graph_contract"]["runtime_producer_layouts_verified"] is False
    assert frozen["contract_template"]["calls_per_replay"]==50
    assert frozen["contract_template"]["rounds"]==7
    assert frozen["fused_remeasurement"] is False
    fused=json.loads(before)
    contract={**frozen["contract_template"],"environment":fused["measurement_contract"]["environment"]}
    assert timing.comparable_contract(contract)==timing.comparable_contract(fused["measurement_contract"])
    assert (PROFILE_DIRECTORY/"review/profile.json").read_bytes()==before


@pytest.mark.parametrize("field",["calls_per_replay","warmup_replays","cache_context","environment"])
def test_comparison_does_not_hide_changed_measurement_boundaries(field):
    contract=json.loads((PROFILE_DIRECTORY/"review/profile.json").read_text())["measurement_contract"]
    changed=copy.deepcopy(contract);changed[field]="different"
    assert timing.comparable_contract(changed)!=timing.comparable_contract(contract)


def rows():
    return [{"round":i,"checks":dict(timing.CHECKS),"timing":{"elapsed_ms_per_50_calls":.5,"error":None,
        "event_boundary":{"device":"cuda:0","start_device":"cuda:0","end_device":"cuda:0", "stream":4,"calls_per_replay":50}}}
        for i in range(7)]


def test_native_summary_never_replaces_failed_rounds():
    data=rows(); assert timing.summarize_native(data)["statistics"]["median_us"]==10.
    data[1]["timing"]["error"]="failed"
    assert timing.summarize_native(data)["status"]=="quality_failed" and len(data)==7
    with pytest.raises(ValueError):timing.summarize_native(data[:-1])
    data=rows();data[0]["checks"]["all_inputs_unchanged"]=False
    with pytest.raises(ValueError):timing.summarize_native(data)


def test_no_gpu_run_fails_without_creating_a_latency(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    output=tmp_path/"no_gpu"
    with pytest.raises(RuntimeError,match="HIP GPU0 required"):
        timing.main(["run","--flagscale-root",str(DEFAULT_FLAGSCALE),"--output",str(output)])
    assert (output/"failure.json").exists() and not (output/"result.json").exists()


@pytest.fixture(scope="module")
def case():
    graph,_=activation_slice(build_model_graph(qwen25_3b_spec()))
    _,base,_=make_case(graph,conditional=False)
    pair=("L000.residual_add_1","L000.pre_mlp_norm")
    identity=json.loads((PROFILE_DIRECTORY/"raw/identity.json").read_text())
    runtime=json.loads((PROFILE_DIRECTORY/"raw/runtime.json").read_text())
    context={"evidence_kind":"model_runtime","source_id":"CPU_TEST_FIXTURE_NOT_MODEL_EVIDENCE",
        "graph_fingerprint":graph_fingerprint(graph),"node_ids":list(pair),"device_index":0,
        "observations":identity["tensor_observations"],"runtime":runtime["runtime"],"device":runtime["device"],
        "measurement_contract":identity["measurement_contract"],"implementation":base.adapter.implementation,
        "storage_aliases":{"x_residual":False,"x_weight":False,"x_other_live":False}}
    bound,_=prepare_adapter(DEFAULT_FLAGSCALE,"cuda:0")
    implementation={"id":"torch.native.add_then_rmsnorm.dual_output.v1","version":"CPU_TEST_FIXTURE"}
    native=timing.native_identity(bound,context["observations"],context["runtime"],context["device"],implementation)
    native.update(implementation=implementation,runtime=runtime,measurement_contract=identity["measurement_contract"],duration_s=10e-6)
    return graph,base,pair,context,native


def test_native_chain_is_qualified_as_a_whole_not_singleton_costs(case):
    graph,base,pair,context,record=case
    assert profiles.qualified_native_cost(graph,base.hardware,base.adapter,pair,0,context,base.accepted,record).duration==10e-6
    assert profiles.qualified_native_cost(graph,base.hardware,base.adapter,pair,0,None,base.accepted,record) is None
    assert profiles.qualified_native_cost(graph,base.hardware,base.adapter,pair,1,context,base.accepted,record) is None
    changed=copy.deepcopy(record);changed["measurement_contract"]["warmup_replays"]=11
    assert profiles.qualified_native_cost(graph,base.hardware,base.adapter,pair,0,context,base.accepted,changed) is None


@pytest.mark.parametrize("native_us,fused_expected",[(3.,False),(12.,True)])
def test_strict_mcts_can_choose_native_or_kerv_from_measured_chain(case,native_us,fused_expected):
    graph,base,pair,context,record=case
    record=copy.deepcopy(record);record["duration_s"]=native_us/1e6
    boundaries=set(graph.nodes)-set(pair)
    registry,profile,model=make_case(graph,{pair[-1]+"@gpu0":context},boundaries=boundaries,
                                    conditional=False,native_record=record)
    unfused=reference_plan(graph,registry,fused=False)
    original=copy.deepcopy(unfused)
    estimate=model.predict(graph,unfused,profile)
    assert estimate.makespan==pytest.approx(native_us/1e6)
    assert estimate.details["native_chain_execution_units"]==[list(pair)]
    assert unfused==original and all(len(g.node_ids)==1 for g in unfused.groups)
    assert sum(g["source"]=="measured_native_add_rmsnorm_chain" for g in estimate.details["groups"])==1
    result,_,_=run_mode(graph,registry,profile,model,"joint",budget=16)
    assert bool(result["fused_groups"]) is fused_expected
    report=audit(graph,profile)
    assert report["matching_measured_native_chains"]==report["reliable_measured_pairs"]==1


def test_split_device_native_pair_does_not_use_chain_measurement(case):
    graph,_,pair,context,record=case
    registry,profile,model=make_case(graph,{pair[-1]+"@gpu0":context},conditional=False,native_record=record)
    plan=reference_plan(graph,registry)
    next(g for g in plan.groups if g.node_ids==[pair[-1]]).device=1
    estimate=model.predict(graph,plan,profile)
    assert "native_chain_execution_units" not in estimate.details
    assert all(g["source"]!="measured_native_add_rmsnorm_chain" for g in estimate.details["groups"])


def test_strict_search_waits_for_a_matching_native_chain(case):
    graph,base,pair,context,record=case
    record=copy.deepcopy(record);record["measurement_contract"]["warmup_replays"]=11
    registry,profile,_=make_case(graph,{pair[-1]+"@gpu0":context},conditional=False,native_record=record)
    assert not profile.native_reuse
    assert all(len(g.node_ids)==1 for g in reference_plan(graph,registry,fused=True).groups)
    assert profile.plan_blockers(reference_plan(graph,base.adapter.registry,fused=True))
    assert audit(graph,profile)["reliable_measured_pairs"]==0


def test_another_layer_requires_its_own_context_before_reusing_native_cost(case):
    _,base,_,context,record=case
    graph=build_model_graph(qwen25_3b_spec())
    pair=("L001.residual_add_1","L001.pre_mlp_norm")
    ctx=copy.deepcopy(context);ctx.update(graph_fingerprint=graph_fingerprint(graph),node_ids=list(pair))
    assert profiles.qualified_native_cost(graph,base.hardware,base.adapter,pair,0,ctx,base.accepted,record) is not None
    other=("L002.residual_add_1","L002.pre_mlp_norm")
    assert profiles.qualified_native_cost(graph,base.hardware,base.adapter,other,0,ctx,base.accepted,record) is None


@pytest.mark.parametrize("native_ms,preferred",[(.1,"Native"),(.5,"KERV")])
def test_offline_review_accepts_either_ranking_and_lookup_checks_receipts(tmp_path,monkeypatch,case,native_ms,preferred):
    # The entire directory is a CPU fixture, never saved as experiment data.
    _,_,_,context,_=case
    monkeypatch.setattr(profiles,"verify_sources",lambda frozen:None)
    frozen=timing.plan(DEFAULT_FLAGSCALE)
    runtime=json.loads((PROFILE_DIRECTORY/"raw/runtime.json").read_text())
    implementation=timing.binding()
    implementation["version"]=runtime["runtime"]["framework_version"]
    bound,_=prepare_adapter(DEFAULT_FLAGSCALE,"cuda:0")
    identity=timing.native_identity(bound,context["observations"],runtime["runtime"],runtime["device"],implementation)
    contract={**frozen["contract_template"],"environment":runtime["environment"]}
    identity.update(tensor_observations=context["observations"],implementation=implementation,
        measurement_contract=contract,measurement_contract_id=timing.measurement_contract_id(contract))
    data=rows()
    for i,row in enumerate(data):
        row["timing"]["elapsed_ms_per_50_calls"]=native_ms
        row.update(started_monotonic_ns=i*10+1,finished_monotonic_ns=i*10+2)
    result={"complete":True,"gpu_executed":True,"graph_capture_verified":True,"identity":identity,
        "records":data,"quality":timing.summarize_native(data),"fused_remeasurement":False,"automatic_admission":False}
    files={"plan.json":frozen,"identity.json":identity,"runtime.json":runtime,"result.json":result,"progress.json":{}}
    files.update({f"round_{i}.json":row for i,row in enumerate(data)})
    for name,value in files.items():timing.save(tmp_path/name,value)
    (tmp_path/"fatal_traceback.log").write_bytes(b"")
    record=profiles.review(tmp_path)
    assert record["admission"]=="accepted" and record["comparison"]["preferred"]==preferred
    assert profiles.load_record(tmp_path)["duration_s"]==pytest.approx(native_ms/1000/50)
    (tmp_path/"round_0.json").write_text("{}")
    with pytest.raises(profiles.MeasurementContractError,match="receipt changed"):
        profiles.load_record(tmp_path)
