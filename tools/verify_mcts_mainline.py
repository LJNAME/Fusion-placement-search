"""Offline checks for MCTS score traces and exact/discovered reference roles."""
import argparse
from collections import Counter,defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from cost_model.exact_search import count_exact_search_space
from cost_model.legal_search import LegalFusionSearchSpace
from cost_model.search_evaluation import build_evaluation_graph
from simulator.placement_artifacts import load_placed_op_graph


def require(value,message):
    if not value:
        raise ValueError(message)


def verify(directory,*,check_source=True):
    directory=Path(directory)
    read=lambda name:json.loads((directory/name).read_text(encoding="utf-8"))
    plan=read("plan.json");report=read("report.json")
    runs=[json.loads(s) for s in (directory/"runs.jsonl").read_text(encoding="utf-8").splitlines()]
    require(report["complete"] and not report["algorithm_changed"] and not report["beam_executed"],"scope changed")
    require(len(runs)==4+4*len(plan["seeds"])*(2+len(plan["budgets"])),"wrong run count")
    require([r["run_id"] for r in runs]==list(range(len(runs))),"run IDs changed")
    observed=defaultdict(dict);queries=defaultdict(list)
    with gzip.open(directory/"evaluated_candidates.csv.gz","rt",encoding="utf-8",newline="") as f:
        for row in csv.DictReader(f):
            run_id=int(row.pop("run_id"));row_id=int(row["row_id"])
            require(row_id not in observed[run_id],"duplicate observation")
            row["makespan_s"]=float(row["makespan_s"])
            row["cross_device_bytes"]=float(row["cross_device_bytes"])
            row["fused_groups"]=int(row["fused_groups"])
            observed[run_id][row_id]=row
    with gzip.open(directory/"score_requests.csv.gz","rt",encoding="utf-8",newline="") as f:
        for row in csv.DictReader(f):
            run_id=int(row["run_id"])
            require(int(row["request"])==len(queries[run_id])+1,"request order changed")
            queries[run_id].append(int(row["row_id"]))
    for run in runs:
        rows=observed[run["run_id"]];qq=queries[run["run_id"]]
        require(set(qq)==set(rows),"missing or unrequested candidate")
        ob=run["observations"]
        require(len(rows)==ob["unique_scored_plans"]==run["budget"]["cost_model_evaluations"],"cache counts disagree")
        require(len(qq)==ob["score_requests"]==run["budget"]["scorer_requests"],"request counts disagree")
        require(len({r["partition_signature"] for r in rows.values()})==ob["unique_fusion_partitions"],"partition count changed")
        require(len({r["device_assignment"] for r in rows.values()})==ob["unique_node_device_assignments"],"device count changed")
        require(dict(Counter(r["used_devices"] for r in rows.values()))==ob["device_sets"],"device sets changed")
        require(sum(r["cross_device_bytes"]>0 for r in rows.values())==ob["cross_device_candidates"],"cross-device count changed")
        best=min(r["makespan_s"] for r in rows.values())
        require(best==run["best_cost_s"]==ob["best_scored_cost_s"],"best cost changed")
        require(ob["first_best_request"]==next(i+1 for i,k in enumerate(qq) if rows[k]["makespan_s"]==best),"first best changed")
        offset=0
        for stage,summary in zip(run["budget"]["stages"],run["stage_observations"]):
            ids=qq[offset:offset+stage["scorer_requests"]]
            partitions={rows[k]["partition_signature"] for k in ids}
            require(len(partitions)==summary["unique_fusion_partitions"],"stage partition count changed")
            if run["mode"]=="sequential" and stage["stage"]=="placement":
                require(len(partitions)==1,"placement changed partition")
            offset+=stage["scorer_requests"]
    exact=read("small_exact_references.json")
    exact_joint=next(r for r in runs if r["phase"]=="small_exact" and r["mode"]=="joint")
    table={r["plan_signature"]:r["makespan_s"] for r in observed[exact_joint["run_id"]].values()}
    require(len(table)==48==report["small_joint_space"],"small Exact space changed")
    for r in runs:
        if r["phase"]=="small_mcts":
            require(all(x["plan_signature"] in table and table[x["plan_signature"]]==x["makespan_s"]
                        for x in observed[r["run_id"]].values()),"small score differs from Exact table")
    refs=[r for r in runs if r["phase"]=="layer_high_reference"]
    grid=[r for r in runs if r["phase"]=="layer_grid"]
    frozen=read("layer_frozen_reference.json");discovered=read("layer_discovered_reference.json")
    require(frozen["all_reference_run_ids"]==[r["run_id"] for r in refs],"frozen reference used grid data")
    require(max(r["run_id"] for r in refs)<min(r["run_id"] for r in grid),"reference was not frozen before grid")
    require(frozen["cost_s"]==min(r["best_cost_s"] for r in refs),"wrong frozen reference")
    require(discovered["cost_s"]==min(r["best_cost_s"] for r in (*refs,*grid)),"wrong posthoc reference")
    with (directory/"results.csv").open(encoding="utf-8",newline="") as f:
        results=list(csv.DictReader(f))
    require(len(results)==report["mcts_runs"],"missing result row")
    for row in results:
        r=runs[int(row["run_id"])];cost=r["best_cost_s"]
        expected={}
        if r["scale"]=="small":
            expected["joint_exact_gap"]=(cost-exact["joint"]["best_cost_s"])/exact["joint"]["best_cost_s"]
            field="two_stage_exact_reference_gap" if r["mode"]=="sequential" else "restricted_optimality_gap"
            expected[field]=(cost-exact[r["mode"]]["best_cost_s"])/exact[r["mode"]]["best_cost_s"]
            if r["mode"]=="sequential":require(row["restricted_optimality_gap"]=="","sequential mislabeled as global optimum")
        else:
            expected["frozen_discovered_reference_gap"]=(cost-frozen["cost_s"])/frozen["cost_s"]
            expected["posthoc_discovered_reference_gap"]=(cost-discovered["cost_s"])/discovered["cost_s"]
        require(all(math.isclose(float(row[k]),v,abs_tol=1e-12) for k,v in expected.items()),"reference gap mismatch")
    for signature,entry in read("best_plans.json").items():
        placed=load_placed_op_graph(directory/entry["artifact"])
        key=tuple((g.extra["pattern_id"],tuple(g.node_ids),int(g.device)) for g in placed.plan.groups)
        actual="sha256:"+hashlib.sha256(json.dumps(key,separators=(",",":")).encode()).hexdigest()
        require(actual==signature and placed.extra["deployable"] is False,"invalid best artifact")
    if check_source:
        require(all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest for name,digest in plan["source_sha256"].items()),"source changed")
        require(hashlib.sha256(Path(plan["hardware"]["path"]).read_bytes()).hexdigest()==plan["hardware"]["sha256"],"hardware changed")
    space=LegalFusionSearchSpace(build_evaluation_graph("layer"),2,require_executable=False)
    count=count_exact_search_space(space)
    joint_ids={x["plan_signature"] for r in (*refs,*grid) if r["mode"]=="joint" for x in observed[r["run_id"]].values()}
    return dict(pass_checks=True,exact_runs=report["exact_runs"],mcts_runs=report["mcts_runs"],
        scored_unique_rows=sum(map(len,observed.values())),score_requests=sum(map(len,queries.values())),
        small_exact_joint_space=48,layer_joint_legal_space_count=count,
        layer_joint_scored_union=len(joint_ids),layer_joint_scored_fraction=len(joint_ids)/count,
        first_best_at_first_request=sum(r["observations"]["first_best_request"]==1 for r in runs if r["method"]=="mcts"),
        reference_roles_verified=True,tree_state_coverage_claim=False,source_checked=check_source)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    args=parser.parse_args()
    result=verify(args.directory)
    (args.directory/"verification.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
