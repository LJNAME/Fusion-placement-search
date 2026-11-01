"""Frozen-task iteration controls; reuse existing 16-iteration results and Exact tables."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibration.probe_kerv_graph import save
from cost_model.hardware_profile import AnalyticHardwareProfile
from cost_model.search_evaluation import _modes
from simulator.dataset import graph_from_dict
from simulator.hardware import hardware_from_dict
from tools.evaluate_mcts_medium import CANDIDATE_FIELDS, check_inputs, run_one
from tools.evaluate_mcts_mainline import digest, file_sha


def evaluate(source, output):
    source, output = Path(source), Path(output)
    base_plan = json.loads((source / "plan.json").read_text(encoding="utf-8"))
    check_inputs(source, base_plan)
    task = next(t for t in base_plan["tasks"] if t["task_id"] == "mlp_parameter_cap_90")
    old_runs = [json.loads(s) for s in (source / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
    exact = next(r for r in old_runs if r["task_id"] == task["task_id"] and
                 r["method"] == "exact" and r["mode"] == "joint")
    table = {}
    with gzip.open(source / "candidates.csv.gz", "rt", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["run_id"]) == exact["run_id"]:
                table[row["plan_signature"]] = dict(cost=float(row["makespan_s"]),
                    feasible=row["memory_feasible"] == "True", partition=row["partition_signature"])
    references = json.loads((source / "references.json").read_text(encoding="utf-8"))[task["task_id"]]
    joint_ref = references["joint"]["cost_s"]
    settings = (("joint", 8), ("joint", 32), ("sequential", 32))
    inputs = {name: file_sha(source / name) for name in (
        "plan.json", "runs.jsonl", "references.json", "candidates.csv.gz", task["graph"], task["hardware"])}
    sources = dict(base_plan["source_sha256"])
    sources["tools/evaluate_mcts_budget_controls.py"] = file_sha(Path(__file__))
    output.mkdir(parents=True, exist_ok=False)
    save(output / "plan.json", dict(source=str(source.resolve()), task_id=task["task_id"],
        input_sha256=inputs, source_sha256=sources, seeds=[0, 7, 23], new_settings=settings,
        reused_settings=[("placement_only", 16), ("fusion_only", 16), ("sequential", 16), ("joint", 16)],
        total_iteration_pairs=[("joint", 32, "sequential", 32), ("joint", 16, "sequential", 16)],
        per_stage_iteration_pairs=[("joint", 8, "sequential", 16), ("joint", 16, "sequential", 32)],
        controls="same graph, explicit capacity, timing profile, seeds, rollouts and algorithms",
        limits="iterations are not equal model evaluations, wall time or identical tree actions; 3 exposed development seeds",
        no_commitment_causal_claim=True))
    graph = graph_from_dict(json.loads((source / task["graph"]).read_text(encoding="utf-8")))
    profile = AnalyticHardwareProfile(hardware_from_dict(json.loads((source / task["hardware"]).read_text(encoding="utf-8"))))
    runs = []
    for r in old_runs:
        if r["task_id"] == task["task_id"] and r["method"] == "mcts" and r["iterations"] == 16:
            runs.append(dict(r, origin="reused_frozen_v2", baseline_run_id=r["run_id"], control_id=len(runs)))
    with gzip.open(output / "new_candidates.csv.gz", "wt", encoding="utf-8", newline="") as cf, \
         gzip.open(output / "new_requests.csv.gz", "wt", encoding="utf-8", newline="") as qf:
        cw = csv.DictWriter(cf, CANDIDATE_FIELDS, lineterminator="\n"); cw.writeheader()
        qw = csv.writer(qf, lineterminator="\n"); qw.writerow(("control_id", "request", "row_id"))
        for mode_name, budget in settings:
            mode = next(m for m in _modes(2) if m.name == mode_name)
            for seed in (0, 7, 23):
                r, scorer, _ = run_one(graph, profile, mode, "mcts", budget, seed,
                                      capacity=tuple(task["max_memory_bytes"]))
                r.update(control_id=len(runs), task_id=task["task_id"], origin="new_run")
                for row in scorer.observed.values():
                    expected = table[row["plan_signature"]]
                    if (row["makespan_s"], row["memory_feasible"]) != (expected["cost"], expected["feasible"]):
                        raise ValueError("new candidate differs from frozen Joint Exact")
                    cw.writerow(dict(run_id=r["control_id"], **{k: json.dumps(v) if k == "peak_memory_bytes" else v
                        for k, v in row.items()}))
                for i, row_id in enumerate(scorer.requests):
                    qw.writerow((r["control_id"], i+1, row_id))
                runs.append(r)
                print(mode_name, budget, seed, r["metrics"]["selected_cost_s"], flush=True)
    results = []
    for r in runs:
        mm = r["metrics"]
        stages = r["budget"]["stages"]
        split = "+".join(str(s["iterations"]) for s in stages)
        if sum(s["iterations"] for s in stages) != r["iterations"]:
            raise ValueError("iteration split mismatch")
        if r["mode"] == "sequential":
            partition = tuple((x["pattern_id"], tuple(x["node_ids"])) for x in r["budget"]["frozen_partition"])
            signature = digest(partition)
            costs = [x["cost"] for x in table.values() if x["partition"] == signature and x["feasible"]]
            reference = min(costs) if costs else None
            kind = "exact_for_this_frozen_partition"
        else:
            signature = None
            reference = references[r["mode"]]["cost_s"]
            kind = "exact_joint_full_space" if r["mode"] == "joint" else "exact_mode_restricted_space"
        cost = mm["selected_cost_s"]
        if cost is not None and (not mm["selected_feasible"] or cost != mm["best_feasible_scored_cost_s"]):
            raise ValueError("invalid feasible selection")
        results.append(dict(control_id=r["control_id"], origin=r["origin"], mode=r["mode"],
            iterations=r["iterations"], budget_split=split, seed=r["seed"],
            selected_feasible=mm["selected_feasible"], reason=r["reason"], selected_cost_s=cost,
            selected_signature=r["selected_signature"], reference_kind=kind, reference_cost_s=reference,
            mode_reference_gap=(cost-reference)/reference if cost is not None and reference is not None else None,
            joint_exact_gap=(cost-joint_ref)/joint_ref if cost is not None else None,
            frozen_partition_signature=signature,
            frozen_partition_reference_gap=(reference-joint_ref)/joint_ref if r["mode"] == "sequential" and reference is not None else None,
            scorer_requests=mm["score_requests"], cost_model_evaluations=mm["unique_scored_plans"],
            stage_score_requests="+".join(str(s["scorer_requests"]) for s in stages),
            stage_model_evaluations="+".join(str(s["cost_model_evaluations"]) for s in stages),
            observed_search_wall_s=r["observed_search_wall_s"]))
    with (output / "results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, list(results[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(results)
    with (output / "runs.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in runs:
            f.write(json.dumps(r, allow_nan=False)+"\n")
    check_inputs(source, base_plan)
    if any(file_sha(source / name) != value for name, value in inputs.items()) or \
       any(file_sha(ROOT / name) != value for name, value in sources.items()):
        raise ValueError("source or frozen inputs changed")
    report = dict(complete=True, new_mcts_runs=9, reused_frozen_mcts_runs=12,
        no_new_exact_evaluations=True, joint_reference_s=joint_ref,
        sequential_partition_signatures=sorted({r["frozen_partition_signature"] for r in results if r["mode"] == "sequential"}),
        max_frozen_partition_reference_gap=max(r["frozen_partition_reference_gap"] for r in results if r["mode"] == "sequential"),
        all_new_candidates_match_frozen_exact=True, source_and_inputs_unchanged=True,
        algorithm_changed=False, profile_changed=False, gpu=False, beam=False, srop=False,
        scope="exposed task diagnostic; does not isolate a causal commitment effect")
    save(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
