"""Screen every equal-capacity interval from a frozen complete plan table."""
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
from simulator.dataset import graph_from_dict
from tools.evaluate_mcts_mainline import digest, file_sha
from tools.evaluate_mcts_medium import check_inputs


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def audit(source, controls, output):
    source, controls, output = Path(source), Path(controls), Path(output)
    plan = json.loads((source / "plan.json").read_text(encoding="utf-8"))
    check_inputs(source, plan)
    runs = [json.loads(s) for s in (source / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
    exact = [r for r in runs if r["method"] == "exact" and r["mode"] == "joint" and r["task_id"].startswith("mlp_")]
    run_tasks = {r["run_id"]: r["task_id"] for r in exact}
    tables = {r["task_id"]: {} for r in exact}
    files = ("plan.json", "runs.jsonl", "candidates.csv.gz", "inputs/mlp_actual_capacity.graph.json")
    hashes = {name: file_sha(source / name) for name in files}
    control_hash = file_sha(controls / "results.csv")
    output.mkdir(parents=True, exist_ok=False)
    save(output / "plan.json", dict(source=str(source.resolve()), controls=str(controls.resolve()),
        input_sha256=hashes, control_results_sha256=control_hash, tool_sha256=file_sha(Path(__file__)),
        scope="fixed 8-node MLP, fixed timing model and candidate set, equal per-device positive capacity",
        evaluation_axis="actual cost_model_evaluations; iterations retained only as configuration",
        no_new_plan_evaluations=True, no_seed_or_iteration_changes=True))
    graph = graph_from_dict(json.loads((source / files[-1]).read_text(encoding="utf-8")))
    order = graph.topological_order()
    si, mi = order.index("L000.mlp.silu"), order.index("L000.mlp.gated_mul")
    with gzip.open(source / "candidates.csv.gz", "rt", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            run_id = int(row["run_id"])
            if run_id not in run_tasks:
                continue
            peaks = tuple(json.loads(row["peak_memory_bytes"]))
            assignment = row["device_assignment"]
            if len(peaks) != 2 or len(assignment) != len(order) or not set(assignment) <= set("01"):
                raise ValueError("audit requires the frozen two-device encoding")
            tables[run_tasks[run_id]][row["plan_signature"]] = dict(
                signature=row["plan_signature"], partition=row["partition_signature"],
                cost=float(row["makespan_s"]), peaks=peaks, required_capacity=max(peaks),
                fused=int(row["fused_groups"]) > 0, cut=assignment[si] != assignment[mi],
                assignment=assignment)
    table = tables["mlp_actual_capacity"]
    if len(table) != 384:
        raise ValueError("incomplete fixed Joint space")
    for task, other in tables.items():
        if set(other) != set(table) or any((other[k]["cost"], other[k]["peaks"]) != (x["cost"], x["peaks"])
                                           for k, x in table.items()):
            raise ValueError("capacity-only timing/peak invariance failed: " + task)
    seq = [r for r in runs if r["method"] == "exact" and r["mode"] == "sequential" and r["task_id"].startswith("mlp_")]
    partitions = {digest(tuple((x["pattern_id"], tuple(x["node_ids"])) for x in r["budget"]["frozen_partition"])) for r in seq}
    if len(partitions) != 1:
        raise ValueError("Sequential first-stage partition changed with capacity")
    fixed = next(iter(partitions))
    fixed_rows = [x for x in table.values() if x["partition"] == fixed]
    if len(fixed_rows) != 128:
        raise ValueError("unexpected frozen-partition space")
    levels = sorted({x["required_capacity"] for x in table.values()})
    intervals = []
    for index, cap in enumerate([0., *levels]):
        valid = [x for x in table.values() if x["required_capacity"] <= cap]
        fixed_valid = [x for x in fixed_rows if x["required_capacity"] <= cap]
        best = min((x["cost"] for x in valid), default=None)
        frozen_best = min((x["cost"] for x in fixed_valid), default=None)
        cut_count = sum(x["cut"] for x in valid)
        intervals.append(dict(lower_capacity_bytes=cap, lower_inclusive=cap > 0,
            upper_capacity_bytes=levels[index] if index < len(levels) else None, upper_inclusive=False,
            feasible_plans=len(valid), frozen_partition_feasible_plans=len(fixed_valid),
            fused_feasible_plans=sum(x["fused"] for x in valid), cut_feasible_plans=cut_count,
            all_feasible_plans_cut=bool(valid) and cut_count == len(valid),
            joint_min_s=best, frozen_partition_min_s=frozen_best,
            joint_strictly_better=best is not None and (frozen_best is None or best < frozen_best)))
    # A finite certificate stronger than checking arbitrary capacity samples:
    # each Joint plan has a frozen-partition plan with no greater required
    # common capacity and no greater modeled cost.
    certificate = []
    for x in table.values():
        candidates = [q for q in fixed_rows if q["required_capacity"] <= x["required_capacity"] and q["cost"] <= x["cost"]]
        witness = min(candidates, key=lambda q: (q["cost"], q["required_capacity"], q["signature"])) if candidates else None
        certificate.append(dict(joint_signature=x["signature"], joint_required_capacity=x["required_capacity"],
            joint_cost_s=x["cost"], dominating_frozen_signature=witness["signature"] if witness else None,
            dominating_required_capacity=witness["required_capacity"] if witness else None,
            dominating_cost_s=witness["cost"] if witness else None))
    write_csv(output / "equal_capacity_intervals.csv", intervals)
    write_csv(output / "dominance_certificate.csv", certificate)
    witnesses = {}
    for name, predicate in (("any", lambda x: True), ("fused", lambda x: x["fused"]), ("cut_silu_mul", lambda x: x["cut"])):
        x = min((x for x in table.values() if predicate(x)), key=lambda x: (x["required_capacity"], x["cost"], x["signature"]))
        witnesses[name] = dict(x, device_by_node=dict(zip(order, map(int, x["assignment"]))))
    save(output / "witnesses.json", witnesses)
    with (controls / "results.csv").open(encoding="utf-8") as f:
        controls_rows = list(csv.DictReader(f))
    quality = [dict(cost_model_evaluations=int(r["cost_model_evaluations"]),
        joint_exact_gap_percent=float(r["joint_exact_gap"])*100 if r["joint_exact_gap"] else None,
        mode=r["mode"], seed=int(r["seed"]), selected_feasible=r["selected_feasible"], reason=r["reason"],
        observed_search_wall_s=float(r["observed_search_wall_s"]), scorer_requests=int(r["scorer_requests"]),
        configuration_iterations=int(r["iterations"]), configuration_budget_split=r["budget_split"],
        reference_kind=r["reference_kind"], control_id=int(r["control_id"])) for r in controls_rows]
    quality.sort(key=lambda x: (x["seed"], x["cost_model_evaluations"], x["mode"]))
    write_csv(output / "quality_vs_evaluations.csv", quality)
    check_inputs(source, plan)
    if any(file_sha(source / name) != h for name, h in hashes.items()) or file_sha(controls / "results.csv") != control_hash:
        raise ValueError("frozen source data changed")
    report = dict(complete=True, joint_plans=384, frozen_partition_plans=128,
        capacity_breakpoints=len(levels), capacity_intervals_including_empty=len(intervals),
        min_feasible_capacity_bytes=levels[0], min_fused_capacity_bytes=witnesses["fused"]["required_capacity"],
        min_cut_capacity_bytes=witnesses["cut_silu_mul"]["required_capacity"],
        forced_cut_intervals=sum(x["all_feasible_plans_cut"] for x in intervals),
        joint_advantage_intervals=sum(x["joint_strictly_better"] for x in intervals),
        dominance_witnesses=sum(bool(x["dominating_frozen_signature"]) for x in certificate),
        cross_capacity_tables_checked=len(tables), quality_points=len(quality),
        source_and_inputs_unchanged=True, new_plan_evaluations=0, new_mcts_runs=0,
        scope="fixed graph/profile/candidates; all equal per-device capacities, not asymmetric capacities or arbitrary graphs",
        no_general_homogeneous_separability_theorem=True)
    save(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("controls", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.source, args.controls, args.output), indent=2))


if __name__ == "__main__":
    main()
