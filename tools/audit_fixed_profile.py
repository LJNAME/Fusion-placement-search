"""Read-only coverage and billing audit for the fixed mainline profile."""
import argparse
from collections import Counter
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from calibration.probe_kerv_graph import save
from cost_model.fusion_kernel_profile import GroupKernelKeys, candidate_structure, key_id
from cost_model.measurement_labels import MeasurementContractError
from cost_model.measured_profile import MeasuredHardwareProfile
from cost_model.search_evaluation import build_evaluation_graph
from cost_model.structural import StructuralCostModel
from simulator.dataset import plan_from_dict
from simulator.fusion_rules import default_fusion_registry, enumerate_fusion_candidates
from simulator.placement_artifacts import graph_fingerprint


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_save(path, rows):
    with path.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v) if isinstance(v,(list,dict,tuple)) else v for k,v in row.items()})


def audit_scale(scale, diagnostic, strict):
    graph=build_evaluation_graph(scale)
    candidates=enumerate_fusion_candidates(graph,default_fusion_registry(),require_executable=False)
    keys=GroupKernelKeys(graph,diagnostic.hardware)
    rows=[]
    for candidate in candidates:
        for device in range(diagnostic.n_devs):
            group=candidate_structure(graph,candidate,device)
            cost=diagnostic.estimate_group(graph,group)
            description,gaps=keys.describe(group)
            try:
                admitted=strict.estimate_group(graph,group)
                outcome="allowed_"+admitted.source
                reason=None
            except MeasurementContractError as error:
                outcome="blocked_by_label_contract"
                reason=str(error)
            rows.append(dict(scale=scale,node_ids=list(group.node_ids),device=device,
                pattern=group.pattern_id,implementation_status=group.implementation_status,
                candidate_executable=candidate.executable,source=cost.source,duration_s=cost.duration,
                compute_time_s=cost.compute_time,memory_time_s=cost.memory_time,launch_time_s=cost.launch_time,
                confidence=cost.confidence,ood_reasons=list(cost.ood_reasons),strict_query=outcome,strict_error=reason,
                unbound_template_hash=key_id(description) if description is not None else None,
                identity_gaps=list(gaps),identity_exempt_nonzero_cost=description is None and cost.duration>0))
    summary=dict(scale=scale,nodes=len(graph.nodes),graph_hash=graph_fingerprint(graph),
        local_candidates=len(candidates),multi_node_candidates=sum(len(c.node_ids)>1 for c in candidates),
        executable_candidates=sum(c.executable for c in candidates),candidate_device_usages=len(rows),
        cost_sources=dict(Counter(r["source"] for r in rows)),strict_queries=dict(Counter(r["strict_query"] for r in rows)),
        distinct_unbound_templates=len({r["unbound_template_hash"] for r in rows if r["unbound_template_hash"]}),
        complete_template_usages=sum(r["unbound_template_hash"] is not None and not r["identity_gaps"] for r in rows),
        identity_gaps_by_usage=dict(Counter(g for r in rows for g in r["identity_gaps"])),
        identity_exempt_nonzero_cost_usages=sum(r["identity_exempt_nonzero_cost"] for r in rows))
    links=[]
    for nbytes in sorted({float(e.bytes) for e in graph.edges if e.bytes and e.bytes>0}):
        for src,dst in ((0,1),(1,0)):
            cost=strict.estimate_transfer(src,dst,nbytes)
            links.append(dict(scale=scale,src=src,dst=dst,nbytes=nbytes,duration_s=cost.duration,
                source=cost.source,confidence=cost.confidence,ood_reasons=list(cost.ood_reasons),
                kernel_label_contract_gate_applied=False))
    summary["link_size_direction_queries"]=len(links)
    summary["link_extrapolated_queries"]=sum(bool(r["ood_reasons"]) for r in links)
    return summary,rows,links


def audit_selected(baseline,profile):
    result=json.loads(Path(baseline).read_text(encoding="utf-8"))
    groups,transfers,plans=[],[],[]
    for scale in result["scales"]:
        graph=build_evaluation_graph(scale["scale"])
        if graph_fingerprint(graph)!=scale["graph_hash"]:
            raise ValueError("baseline graph changed")
        for search in scale["searches"]:
            candidate=search["candidates"][0]
            plan=plan_from_dict(candidate["plan"])
            estimate=StructuralCostModel().predict(graph,plan,profile)
            if not math.isclose(estimate.makespan,candidate["makespan_s"],rel_tol=1e-12):
                raise ValueError("fixed-profile baseline prediction changed")
            prefix=dict(scale=scale["scale"],mode=search["mode"],method=search["method"])
            groups.extend({**prefix,**row} for row in estimate.details["groups"])
            transfers.extend({**prefix,**row} for row in estimate.details["transfers"])
            plans.append({**prefix,"makespan_s":estimate.makespan,
                "group_sources":dict(Counter(g["source"] for g in estimate.details["groups"])),
                "transfer_sources":dict(Counter(t["source"] for t in estimate.details["transfers"])),
                "computation_time_s":estimate.computation_time,"communication_time_s":estimate.communication_time,
                "timeline_group_duration_sum_s":sum(g["duration"] for g in estimate.details["groups"]),
                "confidence":estimate.confidence,"out_of_distribution":estimate.out_of_distribution})
    return plans,groups,transfers


def run(hardware,raw,baseline,output):
    inputs={"hardware":Path(hardware),"raw":Path(raw),"baseline":Path(baseline)}
    sources=("tools/audit_fixed_profile.py","cost_model/hardware_profile.py","cost_model/measured_profile.py",
        "cost_model/structural.py","cost_model/structure.py","cost_model/measurement_labels.py",
        "cost_model/fusion_kernel_profile.py","cost_model/search_evaluation.py","simulator/fusion_rules.py")
    before={name:sha(path) for name,path in inputs.items()}
    source_hashes={name:sha(ROOT/name) for name in sources}
    diagnostic=MeasuredHardwareProfile.from_files(hardware,raw,allow_unbound=True)
    strict=MeasuredHardwareProfile.from_files(hardware,raw)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    summaries,rows,links=[],[],[]
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always",RuntimeWarning)
        for scale in ("small","layer","model","model_3b"):
            summary,rr,ll=audit_scale(scale,diagnostic,strict)
            summaries.append(summary);rows.extend(rr);links.extend(ll)
        plans,groups,transfers=audit_selected(baseline,diagnostic)
    after={name:sha(path) for name,path in inputs.items()}
    if before!=after or any(sha(ROOT/name)!=digest for name,digest in source_hashes.items()):
        raise ValueError("audit inputs or source changed")
    hw=diagnostic.hardware
    result=dict(scope="read_only_local_candidate_coverage_and_baseline_billing_not_full_search_exhaustion",
        inputs={name:{"path":str(path.resolve()),"sha256":before[name]} for name,path in inputs.items()},
        source_sha256=source_hashes,inputs_unchanged=True,
        kernel_records=len(diagnostic._kernels),record_admissions=dict(Counter(m.label.admission for m in diagnostic._kernels)),
        expected_kernel_contract_id=None,
        hardware_parameters=dict(launch_us=hw.launch_overhead*1e6,fusion_us=hw.fusion_overhead*1e6,
            comm_latency_us=hw.comm_latency*1e6,interference=hw.interference,overlap_send=hw.overlap_send),
        link_tables=[dict(src=t.src_device,dst=t.dst_device,points=len(t.points),
            min_bytes=t.points[0].size_bytes,max_bytes=t.points[-1].size_bytes,
            kernel_label_contract_gate_applied=False) for t in diagnostic._links.values()],
        scales=summaries,selected_plans=plans,
        selected_group_sources=dict(Counter(g["source"] for g in groups)),
        selected_transfer_sources=dict(Counter(t["source"] for t in transfers)),
        diagnostic_warning_counts=dict(Counter(str(w.message) for w in recorded)),
        limitations=["shape hits do not bind canonical group identity or runtime/layout",
            "strict group fallback availability is not measured coverage or deployability",
            "links are not admitted by the Kernel measurement-contract gate",
            "host cost is not a separate scheduler component; unbound labels may contain harness overhead",
            "embedding is identity-exempt in GroupKernelKeys but receives nonzero analytic cost here"],
        automatic_model_writeback=False,new_gpu_measurements=False)
    save(output/"report.json",result)
    csv_save(output/"candidate_groups.csv",rows)
    csv_save(output/"link_queries.csv",links)
    csv_save(output/"selected_groups.csv",groups)
    if transfers:
        csv_save(output/"selected_transfers.csv",transfers)
    print(json.dumps({"record_admissions":result["record_admissions"],"scales":summaries,
        "selected_group_sources":result["selected_group_sources"],"selected_transfer_sources":result["selected_transfer_sources"]},indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware",type=Path,default=ROOT/"data/bw1000_2card/bw1000_2card.calibrated.json")
    parser.add_argument("--raw",type=Path,default=ROOT/"data/bw1000_2card/raw_measurements.json")
    parser.add_argument("--baseline",type=Path,default=ROOT/"data/mainline_four_mode_v1/run1/search.json")
    parser.add_argument("--out-dir",type=Path,required=True)
    args=parser.parse_args()
    run(args.hardware,args.raw,args.baseline,args.out_dir)


if __name__=="__main__":
    main()
