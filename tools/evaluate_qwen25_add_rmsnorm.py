"""Audit real 3B AddRMS locations and wire existing searches, CPU only."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from calibration.kerv_add_rmsnorm_probe import save, sha
from cost_model.exact_search import exact_legal_search
from cost_model.fusion_kernel_profile import candidate_structure
from cost_model.kerv_add_rmsnorm_profile import load_profile
from cost_model.qwen25_add_rmsnorm import (
    PROFILE_DIRECTORY, PROXY_CONFIG, SCHEDULING_POLICY, GraphPlanningProfile, IntegrationScorer,
    activation_slice, annotate, audit, reference_plan, search_registry,
)
from cost_model.search_evaluation import _Mode, _run_search, qwen25_3b_spec
from cost_model.structural import StructuralCostModel
from integration.flagscale.kerv_add_rmsnorm import KervAddRmsnormAdapter
from simulator.fusion_rules import enumerate_fusion_candidates
from simulator.hardware import load_hardware
from simulator.model_graphs import build_model_graph
from simulator.placement_artifacts import PlacedOpGraph, save_placed_op_graph
from tools.evaluate_search_quality import EvaluationBudgetReached, summarize


def make_case(graph, contexts=None, boundaries=(), *, conditional=True):
    adapter = KervAddRmsnormAdapter()
    measured, _ = load_profile(PROFILE_DIRECTORY)
    hardware = load_hardware(ROOT/"data/bw1000_2card/bw1000_2card.calibrated.json")
    profile = GraphPlanningProfile(graph,hardware,adapter,measured,contexts,boundaries=boundaries,conditional=conditional)
    registry=search_registry(profile)
    model = StructuralCostModel(registry=registry,scheduling_policy=SCHEDULING_POLICY)
    return registry,profile,model


def describe(graph, plan, profile, model):
    if profile.plan_blockers(plan):
        raise ValueError("returned plan violates capability/residency constraints")
    estimate = model.predict(graph,plan,profile)
    return {"predicted_ms":estimate.makespan*1e3,
            "fused_groups":sum(len(g.node_ids)>1 for g in plan.groups),
            "fused_nodes":[g.node_ids for g in plan.groups if len(g.node_ids)>1],
            "nodes_per_device":dict(Counter(str(g.device) for g in plan.groups for _ in g.node_ids)),
            "sources":dict(Counter(g["source"] for g in estimate.details["groups"])),
            "feasible":all(v<=profile.max_memory(i) for i,v in enumerate(estimate.peak_memory))}


def run_mode(graph,registry,profile,model,mode,budget=64):
    scorer = IntegrationScorer(graph,profile,model,budget)
    completed = True
    try:
        _run_search("mcts",graph,scorer,_Mode(mode,registry,True,(0,1)),seed=0,beam=8,rollouts=3,
                    mcts_iterations=budget//4,top_k=1,unfused_guided_rollout=True)
    except EvaluationBudgetReached:
        completed = False
    if scorer.best_plan is None:
        raise ValueError("search did not produce a feasible plan")
    result = {"mode":mode,"seed":0,"evaluation_budget":budget,**scorer.run_metrics(),
              "rejected_capability_or_residency_requests":scorer.rejected_requests,
              "search_returned_normally":completed,**summarize(scorer.trace),
              **describe(graph,scorer.best_plan,profile,model)}
    return result,annotate(scorer.best_plan,profile),scorer.trace


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--contexts",type=Path,help="optional per-group runtime layout/contract evidence, never inferred")
    parser.add_argument("--conditional-planning",action="store_true",help="explicitly allow unverified fusion preconditions for CPU integration tests")
    args=parser.parse_args(argv);args.output.mkdir(parents=True,exist_ok=False)
    graph=build_model_graph(qwen25_3b_spec())
    contexts=json.loads(args.contexts.read_text(encoding="utf-8")) if args.contexts else None
    registry,profile,model=make_case(graph,contexts,conditional=args.conditional_planning)
    coverage=audit(graph,profile);save(args.output/"capability_cost_audit.json",coverage)
    singletons=[c for c in enumerate_fusion_candidates(graph,registry,require_executable=True) if len(c.node_ids)==1]
    singleton_coverage={}
    for device in range(2):
        singleton_coverage[str(device)] = dict(Counter(profile.estimate_group(graph,candidate_structure(graph,c,device)).source for c in singletons))
    references={}
    for name,fused in (("no_fusion",False),("all_static_matches_conditional",True)):
        ref=reference_plan(graph,profile.adapter.registry if fused else registry,fused=fused)
        blockers=profile.plan_blockers(ref)
        references[name] = ({"feasible":False,"predicted_ms":None,"capability_blockers":blockers} if blockers
                            else describe(graph,ref,profile,model))
    runs=[];traces=[]
    for mode in ("joint","sequential","placement_then_fusion"):
        result,placed_plan,trace=run_mode(graph,registry,profile,model,mode)
        runs.append(result);traces.append({"mode":mode,"trace":trace})
        if mode=="joint":
            placed=PlacedOpGraph(graph,profile.hardware,placed_plan,predicted_makespan=result["predicted_ms"]/1e3,
                                extra={"scope":"conditional planning integration; not deployed or measured",
                                       "source_model_layout_verified":False,"independent_regret":None})
            save_placed_op_graph(args.output/"joint_placed.json",placed,registry=registry)
    sliced,boundaries=activation_slice(graph)
    small_registry,small_profile,small_model=make_case(sliced,boundaries=boundaries,conditional=args.conditional_planning)
    exact=exact_legal_search(sliced,lambda p:small_model.predict(sliced,p,small_profile),2,
        registry=small_registry,allowed_devices=(0,1),top_k=1,is_feasible=lambda p:not small_profile.plan_blockers(p))
    small_run,_,_=run_mode(sliced,small_registry,small_profile,small_model,"joint")
    small_exact=describe(sliced,exact.plans[0],small_profile,small_model)
    report={"scope":"conditional 3B Graph planning integration, not GPU performance validation",
            "model":"Qwen2.5-3B","nodes":len(graph.nodes),"edges":len(graph.edges),
            "audit_summary":{k:v for k,v in coverage.items() if k!="rows"},
            "singleton_cost_sources":singleton_coverage,"planning_declaration":profile.declaration,
            "assumption_id":profile.assumption_id,"scheduling_policy":SCHEDULING_POLICY,
            "references":references,"runs":runs,
            "small_real_graph_slice":{"nodes":list(sliced.nodes),"resident_boundaries":sorted(boundaries),
              "cost_basis":"separate analytic-only slice; optional full-graph runtime contexts not applied",
              "exact_domain":exact.search_space_size,"exact_feasible_plans":exact.feasible_plans,
              "exact":small_exact,"mcts":small_run,
              "model_gap":small_run["predicted_ms"]/small_exact["predicted_ms"]-1},
            "inputs_sha256":{str(p.relative_to(ROOT)):sha(p) for p in (
                ROOT/"data/bw1000_2card/bw1000_2card.calibrated.json",PROXY_CONFIG,
                PROFILE_DIRECTORY/"review/profile.json",ROOT/"cost_model/qwen25_add_rmsnorm.py",Path(__file__))},
            "runtime_contexts_provided":len(contexts or {}),
            "conditional_fusion_actions_enabled":args.conditional_planning,
            "runtime_contexts_sha256":sha(args.contexts) if args.contexts else None,"search_algorithm_changed":False,
            "default_search_configuration_changed":False,"new_gpu_execution":False,
            "new_profiles_admitted":0,"independent_reference_regret":None,
            "reliable_native_fused_performance_comparison":False}
    save(args.output/"search_trace.json",traces);save(args.output/"report.json",report)
    print(json.dumps({"positions":coverage["direct_positions"],"capabilities":coverage["capability_counts"],
                     "profile_uses":len(profile.reuse),"runs":[{k:r[k] for k in ("mode","predicted_ms","fused_groups","cost_model_evaluations")} for r in runs]},ensure_ascii=False))


if __name__=="__main__":
    main()
