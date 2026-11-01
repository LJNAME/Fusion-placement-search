"""Fixed-profile MCTS audit: small Exact, observed candidates, layer budget/seeds."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics as st
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from calibration.probe_kerv_graph import save
from cost_model.hardware_profile import AnalyticHardwareProfile
from cost_model.structural import StructuralCostModel
from cost_model.search_evaluation import (
    _CachedScorer,_modes,_partition_key,_plan_key,_run_search,build_evaluation_graph,
)
from simulator import make_reference_plan,validate_fusion_plan
from simulator.dataset import plan_to_dict
from simulator.hardware import load_hardware
from simulator.placement_artifacts import PlacedOpGraph,graph_fingerprint,save_placed_op_graph

SOURCES=("tools/evaluate_mcts_mainline.py","cost_model/search_evaluation.py","cost_model/legal_search.py",
    "cost_model/exact_search.py","cost_model/hardware_profile.py","cost_model/structural.py",
    "cost_model/structure.py","simulator/fusion_rules.py","simulator/model_graphs/dense_layer.py",
    "simulator/shape_model.py","simulator/hardware.py")
ROW_FIELDS=("run_id","row_id","plan_signature","partition_signature","device_assignment",
    "fused_groups","used_devices","cross_device_bytes","makespan_s","memory_feasible")


def digest(value):
    return "sha256:"+hashlib.sha256(json.dumps(value,separators=(",",":")).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gap(value,reference):
    return (value-reference)/reference


class ObservedScorer(_CachedScorer):
    """Observe the existing scorer without changing any value or search policy."""
    def __init__(self,graph,profile,*,capacity=None):
        super().__init__(graph,profile,capacity=capacity)
        self.observing=True
        self.observed={}
        self.requests=[]
        self.node_order=tuple(graph.topological_order())

    def estimate(self,plan):
        result=super().estimate(plan)
        key=_plan_key(plan)
        if self.observing and key not in self.observed:
            devices={n:g.device for g in plan.groups for n in g.node_ids}
            self.observed[key]=dict(row_id=len(self.observed),plan_signature=digest(key),
                partition_signature=digest(_partition_key(plan)),
                device_assignment="".join(str(devices[n]) for n in self.node_order),
                fused_groups=sum(len(g.node_ids)>1 for g in plan.groups),
                used_devices=",".join(map(str,sorted(set(devices.values())))),
                cross_device_bytes=result.cross_device_bytes,makespan_s=result.makespan,
                memory_feasible=self.is_feasible(result))
        return result

    def __call__(self,plan):
        value=super().__call__(plan)
        self.requests.append(self.observed[_plan_key(plan)]["row_id"])
        return value


def observation_summary(rows,requests):
    by_id={r["row_id"]:r for r in rows}
    unique=[by_id[i] for i in sorted(set(requests))]
    if not unique:
        raise ValueError("no scored candidate")
    best=min(r["makespan_s"] for r in unique)
    return dict(score_requests=len(requests),unique_scored_plans=len(unique),
        unique_fusion_partitions=len({r["partition_signature"] for r in unique}),
        unique_node_device_assignments=len({r["device_assignment"] for r in unique}),
        device_sets=dict(Counter(r["used_devices"] for r in unique)),
        fused_candidates=sum(r["fused_groups"]>0 for r in unique),
        unfused_candidates=sum(r["fused_groups"]==0 for r in unique),
        cross_device_candidates=sum(r["cross_device_bytes"]>0 for r in unique),
        memory_infeasible_candidates=sum(not r["memory_feasible"] for r in unique),
        first_scored_cost_s=by_id[requests[0]]["makespan_s"],best_scored_cost_s=best,
        first_best_request=next(i+1 for i,k in enumerate(requests) if by_id[k]["makespan_s"]==best),
        improved_after_first_scored=best<by_id[requests[0]]["makespan_s"])


def one_run(profile,scale,mode,method,iterations,seed):
    graph=build_evaluation_graph(scale)
    scorer=ObservedScorer(graph,profile)  # Fresh cache for each run.
    plans,elapsed,budget=_run_search(method,graph,scorer,mode,seed=seed,beam=8,rollouts=3,
                                   mcts_iterations=iterations,top_k=4)
    scorer.observing=False
    rows=list(scorer.observed.values())
    summary=observation_summary(rows,scorer.requests)
    if len(rows)!=scorer.calls or len(scorer.requests)!=budget["scorer_requests"]:
        raise ValueError("observer/scorer counters differ")
    offset=0
    stages=[]
    for stage in budget["stages"]:
        count=stage["scorer_requests"]
        observed=observation_summary(rows,scorer.requests[offset:offset+count])
        stages.append({"stage":stage["stage"],**observed})
        if mode.name=="sequential" and stage["stage"]=="placement" and observed["unique_fusion_partitions"]!=1:
            raise ValueError("placement stage evaluated more than the frozen fusion partition")
        offset+=count
    for plan in plans:
        validate_fusion_plan(graph,plan,profile.n_devs,registry=mode.registry,require_executable=mode.require_executable)
        estimate=scorer.estimate(plan)
        if estimate.details["overflow_devices"]:
            raise ValueError("selected an infeasible best plan")
        if mode.name=="sequential":
            actual=[dict(pattern_id=p,node_ids=list(ns)) for p,ns in _partition_key(plan)]
            if actual!=budget["frozen_partition"]:
                raise ValueError("selected plan changed frozen partition")
    winner=plans[0]
    best=scorer.estimate(winner).makespan
    if best!=summary["best_scored_cost_s"]:
        raise ValueError("returned best differs from minimum scored candidate")
    result=dict(scale=scale,mode=mode.name,method=method,iterations=iterations if method=="mcts" else None,
        seed=seed,graph_hash=graph_fingerprint(graph),best_cost_s=best,best_plan_signature=digest(_plan_key(winner)),
        best_used_devices=sorted({g.device for g in winner.groups}),
        observed_search_wall_s=elapsed,budget=budget,observations=summary,stage_observations=stages)
    return result,winner,rows,scorer.requests


def run(output,*,seeds=(0,7,23),budgets=(16,64,256),reference_iterations=1024,small_iterations=16,
        hardware=ROOT/"data/bw1000_2card/bw1000_2card.calibrated.json"):
    if min((*budgets,small_iterations))<2 or reference_iterations<=max(budgets):
        raise ValueError("iterations must support both Sequential stages; reference must exceed grid")
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    source_hashes={name:file_sha(ROOT/name) for name in SOURCES}
    hardware_hash=file_sha(hardware)
    profile=AnalyticHardwareProfile(load_hardware(hardware))
    modes=_modes(profile.n_devs)
    plan=dict(scope="exposed_fixed_analytic_profile_development; no GPU or SROP",
        seeds=list(seeds),budgets=list(budgets),reference_iterations=reference_iterations,
        small_iterations=small_iterations,modes=[m.name for m in modes],rollouts=3,fusion_probability=.7,top_k=4,
        guided_rollout=True,value_mix=0.,cache="fresh per run; reuse only within a run",
        phase_order=["small_exact","small_mcts","layer_high_reference","layer_grid"],
        observation_scope="scored complete candidates including guided/random completions; not tree-state visitation",
        wall_scope="search including observer overhead; postchecks and file I/O excluded",
        source_sha256=source_hashes,hardware={"path":str(Path(hardware).resolve()),"sha256":hardware_hash},
        no_algorithm_or_profile_changes=True,statistical_significance_claim=False)
    save(output/"plan.json",plan)
    raw_runs=[];winners={};exact_refs={};exact_joint_rows={};small_grid=[];layer_refs=[];layer_grid=[]
    with gzip.open(output/"evaluated_candidates.csv.gz","wt",encoding="utf-8",newline="") as rf, \
         gzip.open(output/"score_requests.csv.gz","wt",encoding="utf-8",newline="") as qf, \
         (output/"runs.jsonl").open("w",encoding="utf-8",newline="\n") as runs_file:
        rw=csv.DictWriter(rf,fieldnames=ROW_FIELDS,lineterminator="\n");rw.writeheader()
        qw=csv.writer(qf,lineterminator="\n");qw.writerow(("run_id","request","row_id"))
        def perform(scale,mode,method,iterations,seed,phase):
            result,winner,rows,requests=one_run(profile,scale,mode,method,iterations,seed)
            result.update(run_id=len(raw_runs),phase=phase)
            sig=result["best_plan_signature"]
            if sig not in winners:
                graph=build_evaluation_graph(scale)
                estimate=StructuralCostModel().predict(graph,winner,profile)
                artifact=Path("best_plans")/(sig.removeprefix("sha256:")+".json")
                save_placed_op_graph(output/artifact,PlacedOpGraph(graph,profile.hardware,winner,
                    predicted_makespan=estimate.makespan,predicted_peak_memory=estimate.peak_memory,
                    extra={"deployable":False,"scope":plan["scope"]}))
                winners[sig]=dict(scale=scale,plan=plan_to_dict(winner),artifact=artifact.as_posix())
            for r in rows:rw.writerow({"run_id":result["run_id"],**r})
            for i,row_id in enumerate(requests):qw.writerow((result["run_id"],i+1,row_id))
            runs_file.write(json.dumps(result,allow_nan=False)+"\n");runs_file.flush()
            raw_runs.append(result)
            print(f'{phase} {scale}/{mode.name}/{method} n={iterations} seed={seed} '
                  f'best={result["best_cost_s"]*1e3:.6f}ms unique={len(rows)}',flush=True)
            return result,rows
        for mode in modes:
            r,rows=perform("small",mode,"exact",small_iterations,seeds[0],"small_exact")
            exact_refs[mode.name]=r
            if mode.name=="joint":exact_joint_rows={x["plan_signature"]:x["makespan_s"] for x in rows}
        save(output/"small_exact_references.json",exact_refs)
        for seed in seeds:
            for mode in modes:
                r,rows=perform("small",mode,"mcts",small_iterations,seed,"small_mcts")
                if any(x["plan_signature"] not in exact_joint_rows or
                       x["makespan_s"]!=exact_joint_rows[x["plan_signature"]] for x in rows):
                    raise ValueError("small scored candidate differs from the exhaustive joint table")
                small_grid.append(r)
        for seed in seeds:
            for mode in modes:
                r,_=perform("layer",mode,"mcts",reference_iterations,seed,"layer_high_reference")
                layer_refs.append(r)
        # Freeze this reference before running the layer budget grid.
        ref=min(layer_refs,key=lambda r:(r["best_cost_s"],r["best_plan_signature"]))
        frozen=dict(kind="best_discovered_not_optimality_oracle",cost_s=ref["best_cost_s"],
            plan_signature=ref["best_plan_signature"],run_id=ref["run_id"],iterations=reference_iterations,
            seeds=list(seeds),all_reference_run_ids=[r["run_id"] for r in layer_refs])
        save(output/"layer_frozen_reference.json",frozen)
        for iterations in budgets:
            for seed in seeds:
                for mode in modes:
                    r,_=perform("layer",mode,"mcts",iterations,seed,"layer_grid")
                    layer_grid.append(r)
    discovered=min((*layer_refs,*layer_grid),key=lambda r:(r["best_cost_s"],r["best_plan_signature"]))
    save(output/"best_plans.json",winners)
    discovered_ref=dict(kind="post_hoc_best_discovered_not_oracle",cost_s=discovered["best_cost_s"],
        plan_signature=discovered["best_plan_signature"],run_id=discovered["run_id"])
    save(output/"layer_discovered_reference.json",discovered_ref)
    results=[]
    for r in (*small_grid,*layer_refs,*layer_grid):
        small=r["scale"]=="small"
        reference=exact_refs["joint"]["best_cost_s"] if small else frozen["cost_s"]
        restricted=exact_refs[r["mode"]]["best_cost_s"] if small else None
        results.append(dict(run_id=r["run_id"],scale=r["scale"],phase=r["phase"],mode=r["mode"],
            iterations=r["iterations"],seed=r["seed"],best_cost_s=r["best_cost_s"],
            joint_exact_gap=gap(r["best_cost_s"],reference) if small else None,
            restricted_optimality_gap=gap(r["best_cost_s"],restricted) if small and r["mode"]!="sequential" else None,
            two_stage_exact_reference_gap=gap(r["best_cost_s"],restricted) if small and r["mode"]=="sequential" else None,
            frozen_discovered_reference_gap=gap(r["best_cost_s"],reference) if not small else None,
            posthoc_discovered_reference_gap=gap(r["best_cost_s"],discovered_ref["cost_s"]) if not small else None,
            observed_search_wall_s=r["observed_search_wall_s"],cost_model_evaluations=r["budget"]["cost_model_evaluations"],
            score_requests=r["observations"]["score_requests"],
            unique_scored_plans=r["observations"]["unique_scored_plans"],
            unique_fusion_partitions=r["observations"]["unique_fusion_partitions"],
            unique_node_device_assignments=r["observations"]["unique_node_device_assignments"],
            cross_device_candidates=r["observations"]["cross_device_candidates"],
            first_best_request=r["observations"]["first_best_request"],best_plan_signature=r["best_plan_signature"]))
    with (output/"results.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]),lineterminator="\n");writer.writeheader();writer.writerows(results)
    summary=[]
    for iterations in budgets:
        for mode in modes:
            rr=[r for r in layer_grid if r["iterations"]==iterations and r["mode"]==mode.name]
            summary.append(dict(iterations=iterations,mode=mode.name,
                min_s=min(r["best_cost_s"] for r in rr),median_s=st.median(r["best_cost_s"] for r in rr),
                max_s=max(r["best_cost_s"] for r in rr),
                unique_best_plans=len({r["best_plan_signature"] for r in rr}),
                min_unique_scored=min(r["observations"]["unique_scored_plans"] for r in rr),
                max_unique_scored=max(r["observations"]["unique_scored_plans"] for r in rr)))
    if any(file_sha(ROOT/name)!=h for name,h in source_hashes.items()) or file_sha(hardware)!=hardware_hash:
        raise ValueError("source or hardware changed during experiment")
    report=dict(complete=True,exact_runs=len(exact_refs),mcts_runs=len(small_grid)+len(layer_refs)+len(layer_grid),
        small_joint_space=len(exact_joint_rows),small_all_restricted_references_matched=all(
            r["best_cost_s"]==exact_refs[r["mode"]]["best_cost_s"] for r in small_grid),
        layer_frozen_reference=frozen,layer_posthoc_reference=discovered_ref,layer_budget_summary=summary,
        source_and_hardware_unchanged=True,algorithm_changed=False,beam_executed=False,
        gpu_measurements=False,statistical_significance_claim=False)
    save(output/"report.json",report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir",type=Path,required=True)
    parser.add_argument("--seeds",default="0,7,23")
    parser.add_argument("--budgets",default="16,64,256")
    parser.add_argument("--reference-iterations",type=int,default=1024)
    parser.add_argument("--small-iterations",type=int,default=16)
    args=parser.parse_args()
    run(args.out_dir,seeds=tuple(map(int,args.seeds.split(","))),budgets=tuple(map(int,args.budgets.split(","))),
        reference_iterations=args.reference_iterations,small_iterations=args.small_iterations)


if __name__=="__main__":
    main()
