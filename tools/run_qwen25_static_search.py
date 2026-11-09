"""One strict 3B joint search under an explicit static execution contract."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from calibration.kerv_add_rmsnorm_probe import save, sha
from cost_model.kerv_add_rmsnorm_profile import load_profile
from cost_model.native_add_rmsnorm_profile import load_record
from cost_model.qwen25_add_rmsnorm import (
    PROFILE_DIRECTORY,SCHEDULING_POLICY,NativeChainCostModel,audit,reference_plan,search_registry,
)
from cost_model.qwen25_static_contract import StaticContractProfile
from cost_model.search_evaluation import qwen25_3b_spec
from integration.flagscale.kerv_add_rmsnorm import KervAddRmsnormAdapter
from simulator.fused_placed_export import export_fused_placed_graph,save_fused_placed_graph,validate_fused_placed_graph
from simulator.hardware import load_hardware
from simulator.model_graphs import build_model_graph
from simulator.placement_artifacts import PlacedOpGraph,graph_fingerprint,load_placed_op_graph,save_placed_op_graph
from tools.evaluate_qwen25_add_rmsnorm import describe,run_mode

NATIVE_DIRECTORY=ROOT/"data/native_add_rmsnorm_timing_01"


def export_plan(placed,estimate,profile,registry):
    native_pairs=[tuple(p) for p in estimate.details.get("native_chain_execution_units",[])]
    # A Native chain has one measured duration. Do not attach its scheduling
    # group indices to two original nodes as if they were independent timings.
    artifact=export_fused_placed_graph(placed,None if native_pairs else estimate,registry)
    if native_pairs:
        node_group={n:f"fg{i:04d}" for i,g in enumerate(placed.plan.groups) for n in g.node_ids}
        starts={p[0] for p in native_pairs};ends={p[1]:p for p in native_pairs}
        units=[(ends[g.node_ids[0]],g.device) if len(g.node_ids)==1 and g.node_ids[0] in ends
               else (tuple(g.node_ids),g.device) for g in placed.plan.groups
               if not (len(g.node_ids)==1 and g.node_ids[0] in starts)]
        membership={n:i for i,(nodes,_) in enumerate(units) for n in nodes};levels=[]
        for i,(nodes,_) in enumerate(units):
            parents={membership[e.src] for e in placed.graph.edges if e.dst in nodes and membership[e.src]!=i}
            levels.append(max((levels[p]+1 for p in parents),default=0))
        order=sorted(range(len(units)),key=lambda i:(levels[i],tuple(sorted(units[i][0]))))
        per_device={str(d):[] for d in range(profile.n_devs)}
        for i in order:
            nodes,device=units[i]
            per_device[str(device)].extend(dict.fromkeys(node_group[n] for n in nodes))
        artifact["execution_order"]={"source":"cost-model execution units; Native chains consecutive", "per_device":per_device}
        artifact["native_chain_execution_units"]=copy.deepcopy(placed.plan.extra["native_chain_execution_units"])
    artifact["execution_contract"]=copy.deepcopy(profile.declaration)
    artifact["hardware_class_audit"]=copy.deepcopy(profile.declaration["hardware_class_audit"])
    artifact["deployable_scope"]="executable rules and memory under declared contract; downstream runtime not verified"
    for group,row in zip(placed.plan.groups,artifact["groups"]):
        if len(group.node_ids)>1:
            row["implementation"]=copy.deepcopy(group.extra["implementation"])
            row["measurement_ref"]=copy.deepcopy(group.extra["measurement_ref"])
            row["runtime_requirements"]=profile.qualifications[(tuple(group.node_ids),group.device)]["runtime_requirements"]
    validate_fused_placed_graph(artifact,placed.graph)
    return artifact


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    graph=build_model_graph(qwen25_3b_spec());fingerprint=graph_fingerprint(graph)
    adapter=KervAddRmsnormAdapter()
    hardware=load_hardware(ROOT/"data/bw1000_2card/bw1000_2card.calibrated.json")
    measured,_=load_profile(PROFILE_DIRECTORY)
    native=load_record(NATIVE_DIRECTORY)
    profile=StaticContractProfile(graph,hardware,adapter,measured,native)
    registry=search_registry(profile);model=NativeChainCostModel(registry=registry,scheduling_policy=SCHEDULING_POLICY)
    coverage=audit(graph,profile)
    coverage["static_contract_profile_uses"]=len(profile.reuse)
    coverage["exact_equivalent_profile_uses"]=0  # Evidence provenance differs; conditions are equal.
    coverage["model_runtime_observed"]=False
    coverage["available_benchmark_profiles"]=2
    coverage["same_device_profile_uses"]=sum(d==0 for _,d in profile.reuse)
    coverage["hardware_class_transferred_uses"]=sum(d!=0 for _,d in profile.reuse)
    coverage["native_chain_cost_uses"]=len(profile.native_reuse)
    coverage["matching_measured_native_chains"]=sum(d==0 for _,d in profile.native_reuse)
    for row in coverage["rows"]:
        for device in row["devices"]:
            device["cost_pair_available"]=device["native_chain"] is not None and device["fused"] is not None
            if device.get("reuse_kind")=="homogeneous_hardware_class_transfer":
                device["native_measured_same_contract"]=False
                device["reliable_measured_pair"]=False
    coverage["reliable_measured_pairs"]=sum(d["reliable_measured_pair"] for row in coverage["rows"] for d in row["devices"])
    save(output/"capability_cost_audit.json",coverage);save(output/"execution_contract.json",profile.declaration)
    save(output/"hardware_class_audit.json",profile.declaration["hardware_class_audit"])
    references={name:describe(graph,reference_plan(graph,registry,fused=fused),profile,model)
                for name,fused in (("native_gpu0",False),("fused_gpu0",True))}
    queries_before=profile.fused_query_counts.copy()
    result,plan,trace=run_mode(graph,registry,profile,model,"joint",budget=64)
    search_queries={str(d):profile.fused_query_counts[d]-queries_before[d] for d in range(profile.n_devs)}
    for unit in plan.extra.get("native_chain_execution_units",[]):
        unit.update(measured_on=0,reuse_on=unit["device"],
                    reuse_kind="same_device" if unit["device"]==0 else "homogeneous_hardware_class_transfer")
    selected=[]
    for group in plan.groups:
        if len(group.node_ids)==1: continue
        key=(tuple(group.node_ids),group.device);q=profile.qualifications[key]
        if key not in profile.reuse or key not in profile.native_reuse:
            raise ValueError("selected fusion has no strict measured condition pair")
        cost=profile.estimate_group(graph,group)
        assert cost.duration==profile.accepted["latency"]["duration_s"]
        group.extra["measurement_ref"]={k:q[k] for k in ("kernel_key_id","measured_kernel_key_id",
            "native_measured_key_id","measurement_contract_id","condition_id",
            "measured_on","reuse_on","reuse_kind","hardware_class_key")}
        selected.append({"nodes":list(group.node_ids),"device":group.device,
                         **group.extra["measurement_ref"],"cost_source":cost.source,"duration_us":cost.duration*1e6})
    estimate=model.predict(graph,plan,profile)
    placed=PlacedOpGraph(graph,profile.hardware,plan,predicted_makespan=estimate.makespan,
        predicted_peak_memory=estimate.peak_memory,extra={"scope":"static contract strict search, not full-model execution",
            "execution_contract_id":profile.assumption_id,"model_layouts_observed":False})
    save_placed_op_graph(output/"joint_placed.json",placed,registry=registry)
    restored=load_placed_op_graph(output/"joint_placed.json",registry=registry)
    replay=model.predict(restored.graph,restored.plan,profile)
    assert restored.candidate_id==placed.candidate_id and graph_fingerprint(restored.graph)==fingerprint
    assert abs(replay.makespan-estimate.makespan)<1e-15 and abs(result["predicted_ms"]/1e3-estimate.makespan)<1e-15
    artifact=export_plan(restored,replay,profile,registry)
    save_fused_placed_graph(output/"joint_fused_placed.json",artifact)
    report={"scope":"one strict joint MCTS, declared target contract and existing measured local costs",
        "model":"Qwen2.5-3B","nodes":len(graph.nodes),"edges":len(graph.edges),"graph_fingerprint":fingerprint,
        "qualification_basis":"0.6B-style explicit static execution declaration; not model runtime observation",
        "audit_summary":{k:v for k,v in coverage.items() if k!="rows"},"references":references,"run":result,
        "hardware_class_audit":profile.declaration["hardware_class_audit"],
        "fused_cost_queries_during_search":search_queries,
        "migration_is_target_device_measurement":False,
        "selected_fusion_queries":selected,"cost_sources":dict(Counter(g["source"] for g in estimate.details["groups"])),
        "consistency":{"graph_unchanged":graph_fingerprint(graph)==fingerprint,"export_roundtrip":True,
            "rescored_makespan_matches":True,"every_selected_fusion_has_measured_pair":True,
            "group_dag_validated":True,"profile_records_copied_or_readmitted":False},
        "execution_contract_id":profile.assumption_id,"export_summary":artifact["summary"],
        "new_gpu_work":False,"new_measurements":0,"new_profiles_admitted":0,"whole_model_executed":False,
        "model_layouts_observed":False,"mcts_runs":1,"search_algorithm_changed":False,
        "full_plan_measured":False,"independent_regret":None,
        "input_sha256":{str(p.relative_to(ROOT)):sha(p) for p in (PROFILE_DIRECTORY/"review/profile.json",
            NATIVE_DIRECTORY/"review/profile.json",ROOT/"cost_model/qwen25_static_contract.py",
            ROOT/"cost_model/profile_hardware_class.py",Path(__file__))}}
    save(output/"search_trace.json",trace);save(output/"report.json",report)
    print(json.dumps({"cost_uses":len(profile.reuse),"hardware_class_transfers":coverage["hardware_class_transferred_uses"],
        "fused_queries_per_gpu":search_queries,"fused_groups":len(selected),
        "predicted_ms":result["predicted_ms"],"score_calls":result["cost_model_evaluations"],
        "export_consistent":True},indent=2))
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args().output)
