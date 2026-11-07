"""CPU audit: three fixed placements and one faithfully relabeled MCTS run."""
import argparse
from collections import Counter
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from calibration.kerv_add_rmsnorm_probe import save,sha
from cost_model.fusion_kernel_profile import key_id
from cost_model.kerv_add_rmsnorm_profile import load_profile
from cost_model.native_add_rmsnorm_profile import load_record
from cost_model.qwen25_add_rmsnorm import PROFILE_DIRECTORY,SCHEDULING_POLICY,NativeChainCostModel,reference_plan,search_registry
from cost_model.qwen25_static_contract import StaticContractProfile
from cost_model.search_evaluation import qwen25_3b_spec
from integration.flagscale.kerv_add_rmsnorm import KervAddRmsnormAdapter
from simulator.hardware import load_hardware
from simulator.model_graphs import build_model_graph
from simulator.placement_artifacts import graph_fingerprint,plan_to_dict
from tools.evaluate_qwen25_add_rmsnorm import run_mode


def relabel_plan(plan,logical_to_original):
    result=copy.deepcopy(plan)
    for group in result.groups: group.device=logical_to_original[group.device]
    return result


class PermutedProfile:
    """A pure change of labels, including links, capacity and measured sources."""
    def __init__(self,base,logical_to_original=(1,0)):
        if sorted(logical_to_original)!=list(range(base.n_devs)):
            raise ValueError("device mapping must be a permutation")
        self.base=base;self.mapping=tuple(logical_to_original)
        self.hardware=copy.deepcopy(base.hardware)
        self.hardware.devices=[copy.deepcopy(base.hardware.devices[d]) for d in self.mapping]
        if base.hardware.link_matrix is not None:
            self.hardware.link_matrix=[[base.hardware.link_matrix[a][b] for b in self.mapping] for a in self.mapping]
        if "link_matrix_nominal" in self.hardware.extra:
            matrix=base.hardware.extra["link_matrix_nominal"]
            self.hardware.extra["link_matrix_nominal"]=[[matrix[a][b] for b in self.mapping] for a in self.mapping]
        self.hardware.extra["logical_to_original_device"]=list(self.mapping)
        self.qualifications={(pair,d):copy.deepcopy(q) for d,original in enumerate(self.mapping)
                             for (pair,physical),q in base.qualifications.items() if physical==original}
        self.reuse={(pair,d):item for d,original in enumerate(self.mapping)
                    for (pair,physical),item in base.reuse.items() if physical==original}
        self.native_reuse={(pair,d):item for d,original in enumerate(self.mapping)
                           for (pair,physical),item in base.native_reuse.items() if physical==original}
        self.declaration={**copy.deepcopy(base.declaration),"logical_to_original_device":list(self.mapping),
                          "scope":"CPU relabel audit; original measurement ordinals retain their provenance"}
        self.assumption_id=key_id(self.declaration)

    def __getattr__(self,name): return getattr(self.base,name)
    def max_memory(self,device): return self.base.max_memory(self.mapping[device])
    def estimate_group(self,graph,group):
        return self.base.estimate_group(graph,replace(group,device=self.mapping[group.device]))
    def estimate_transfer(self,src,dst,nbytes):
        return self.base.estimate_transfer(self.mapping[src],self.mapping[dst],nbytes)
    def link_key(self,src,dst): return self.base.link_key(self.mapping[src],self.mapping[dst])
    def plan_blockers(self,plan): return self.base.plan_blockers(relabel_plan(plan,self.mapping))


def three_plans(graph,registry):
    a=reference_plan(graph,registry,fused=True,device=0)
    b=copy.deepcopy(a)
    for group in b.groups:
        layer=graph.nodes[group.node_ids[0]].extra["layer"]
        group.device=0 if layer<18 else 1
    c=reference_plan(graph,registry,fused=True,device=1)
    return {"A_all_gpu0":a,"B_layers_0_17_gpu0_18_35_gpu1":b,"C_all_gpu1":c}


def overlap_statistics(groups):
    events=[]
    for group in groups:
        if group["end"]>group["start"]:
            events.extend(((group["start"],1),(group["end"],-1)))
    active=peak=0;previous=0.;union=overlap=0.
    for moment,delta in sorted(events):
        if active: union+=moment-previous
        if active>1: overlap+=moment-previous
        active+=delta;peak=max(peak,active);previous=moment
    return {"union_compute_busy_ms":union*1e3,"multi_device_compute_overlap_ms":overlap*1e3,
            "peak_simultaneous_compute_devices":peak}


def summarize_plan(graph,plan,profile,model):
    if profile.plan_blockers(plan): raise ValueError("audit plan violates the current capability/cost contract")
    estimate=model.predict(graph,plan,profile)
    groups=estimate.details["groups"]
    spans={}
    for d in range(profile.n_devs):
        active=[g for g in groups if g["device"]==d and g["end"]>g["start"]]
        spans[str(d)]={"operator_busy_ms":estimate.per_device_time[d]*1e3,
            "first_compute_ms":min((g["start"] for g in active),default=0.)*1e3,
            "last_compute_ms":max((g["end"] for g in active),default=0.)*1e3,
            "busy_over_makespan":estimate.per_device_time[d]/estimate.makespan if estimate.makespan else 0.,
            "peak_memory_bytes":estimate.peak_memory[d]}
    crossing=[{"producer":edge.src,"consumer":edge.dst,"source_device":src.device,
               "destination_device":dst.device,"bytes":edge.bytes}
              for edge in graph.edges
              for src in [next(g for g in plan.groups if edge.src in g.node_ids)]
              for dst in [next(g for g in plan.groups if edge.dst in g.node_ids)] if src.device!=dst.device]
    return {"predicted_ms":estimate.makespan*1e3,"operator_work_sum_ms":estimate.computation_time*1e3,
            "communication_work_sum_ms":estimate.communication_time*1e3,"cross_device_bytes":estimate.cross_device_bytes,
            "cross_device_edges":crossing,"per_device":spans,**overlap_statistics(groups),
            "nodes_per_device":dict(Counter(str(g.device) for g in plan.groups for _ in g.node_ids)),
            "fusions_per_device":dict(Counter(str(g.device) for g in plan.groups if len(g.node_ids)>1)),
            "cost_sources":dict(Counter(g["source"] for g in groups)),
            "memory_feasible":all(v<=profile.max_memory(d) for d,v in enumerate(estimate.peak_memory))}


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    graph=build_model_graph(qwen25_3b_spec())
    hardware=load_hardware(ROOT/"data/bw1000_2card/bw1000_2card.calibrated.json")
    measured,_=load_profile(PROFILE_DIRECTORY)
    profile=StaticContractProfile(graph,hardware,KervAddRmsnormAdapter(),measured,load_record(ROOT/"data/native_add_rmsnorm_timing_01"))
    registry=search_registry(profile);model=NativeChainCostModel(registry=registry,scheduling_policy=SCHEDULING_POLICY)
    plans=three_plans(graph,registry)
    scores={name:summarize_plan(graph,p,profile,model) for name,p in plans.items()}
    print(json.dumps({k:{f:v[f] for f in ("predicted_ms","operator_work_sum_ms","communication_work_sum_ms","multi_device_compute_overlap_ms")}
                      for k,v in scores.items()},indent=2),flush=True)
    swapped=PermutedProfile(profile)
    swapped_registry=search_registry(swapped)
    swapped_model=NativeChainCostModel(registry=swapped_registry,scheduling_policy=SCHEDULING_POLICY)
    relabel_checks={}
    for name,p in plans.items():
        changed=summarize_plan(graph,relabel_plan(p,(1,0)),swapped,swapped_model)
        error=abs(changed["predicted_ms"]-scores[name]["predicted_ms"])
        if error>1e-10: raise AssertionError("device relabel changed the score of the same physical plan")
        relabel_checks[name]={"same_physical_plan_score_ms":changed["predicted_ms"],"absolute_error_ms":error}
    result,plan,trace=run_mode(graph,swapped_registry,swapped,swapped_model,"joint",budget=64)
    result["logical_to_original_device"]=[1,0]
    original_plan=relabel_plan(plan,(1,0))
    original_score=summarize_plan(graph,original_plan,profile,model)
    assert abs(original_score["predicted_ms"]-result["predicted_ms"])<1e-10
    previous=ROOT/"data/qwen25_hardware_class_joint_v1/report.json"
    baseline=json.loads(previous.read_text(encoding="utf-8"))
    assert abs(baseline["run"]["predicted_ms"]-scores["A_all_gpu0"]["predicted_ms"])<1e-10
    d0,d1=profile.hardware.devices
    report={"scope":"three fixed placements and one swapped-label MCTS; CPU model audit only",
        "graph_fingerprint":graph_fingerprint(graph),"fusion_partition":"same 36 KERV AddRMS pairs in A/B/C",
        "boundaries":"A/C include embedding/final norm/head on the same card; B embedding on 0 and final norm/head on 1",
        "plans":scores,"logical_swap_invariance":relabel_checks,
        "swapped_search":result,"swapped_search_in_original_labels":original_score,
        "original_search":{"input_report_sha256":sha(previous),"run":baseline["run"]},
        "original_plan_level_symmetry":{
            "gpu0_flops":d0.flops,"gpu1_flops":d1.flops,"gpu0_mem_bw":d0.mem_bw,"gpu1_mem_bw":d1.mem_bw,
            "gpu0_flops_advantage_fraction":d0.flops/d1.flops-1,
            "gpu0_mem_bw_advantage_fraction":d0.mem_bw/d1.mem_bw-1,
            "fused_costs_shared":True,"all_operator_costs_identical":False,
            "overlap_send":profile.overlap_send,"interference":profile.interference,
            "comm_latency_us":profile.hardware.comm_latency*1e6},
        "objective":"makespan of one graph/microbatch; no multi-microbatch pipeline throughput model",
        "scheduler":"one compute resource per GPU; dependency-level ordering; independent branches may overlap",
        "known_search_order":"guided completion starts with logical 0 then retains the last device; ties retain earlier insertion",
        "new_mcts_runs":1,"new_gpu_work":False,"new_measurements":0,"search_algorithm_changed":False,
        "global_optimum_proven":False,"input_sha256":{str(p.relative_to(ROOT)):sha(p) for p in
            (Path(__file__),ROOT/"cost_model/legal_search.py",ROOT/"cost_model/structural.py",previous)}}
    save(output/"report.json",report);save(output/"swapped_trace.json",trace)
    save(output/"swapped_plan.json",plan_to_dict(plan))
    print(json.dumps({"swapped_search_ms":result["predicted_ms"],"logical_devices":result["nodes_per_device"],
                      "original_devices":original_score["nodes_per_device"],"actual_scores":result["cost_model_evaluations"]},indent=2))
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args().output)
