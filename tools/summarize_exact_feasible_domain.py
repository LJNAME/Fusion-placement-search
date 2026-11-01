"""Summarize frozen Exact feasible costs without evaluating any new plan."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(source, output):
    source, output = Path(source), Path(output)
    names = ("plan.json", "runs.jsonl", "references.json", "candidates.csv.gz", "requests.csv.gz")
    hashes = {name: sha(source / name) for name in names}
    plan = json.loads((source / "plan.json").read_text(encoding="utf-8"))
    tasks = {t["task_id"]: t for t in plan["tasks"]}
    runs = [json.loads(s) for s in (source / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
    exact = {r["run_id"]: r for r in runs if r["method"] == "exact"}
    refs = json.loads((source / "references.json").read_text(encoding="utf-8"))
    rows, requests = {i: {} for i in exact}, {i: [] for i in exact}
    with gzip.open(source / "candidates.csv.gz", "rt", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            run_id = int(row["run_id"])
            if run_id in exact:
                peaks = json.loads(row["peak_memory_bytes"])
                capacity = tasks[exact[run_id]["task_id"]]["max_memory_bytes"]
                feasible = all(p <= c for p, c in zip(peaks, capacity))
                if len(peaks) != len(capacity) or feasible != (row["memory_feasible"] == "True"):
                    raise ValueError("stored feasibility disagrees with explicit capacity")
                rows[run_id][int(row["row_id"])] = (float(row["makespan_s"]), feasible)
    with gzip.open(source / "requests.csv.gz", "rt", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            run_id = int(row["run_id"])
            if run_id in exact:
                requests[run_id].append(int(row["row_id"]))
    result = []
    for run_id, run in exact.items():
        offset = sum(s["scorer_requests"] for s in run["budget"]["stages"][:-1])
        ids = set(requests[run_id][offset:])
        expected = run["budget"]["stages"][-1]["search_space_size"]
        if len(ids) != expected:
            raise ValueError("incomplete Exact final-stage table")
        costs = [rows[run_id][i][0] for i in ids if rows[run_id][i][1]]
        reference = refs[run["task_id"]][run["mode"]]
        if len(costs) != reference["feasible_count"] or (min(costs) if costs else None) != reference["cost_s"]:
            raise ValueError("reference count or minimum changed")
        low, middle, high = (min(costs), statistics.median(costs), max(costs)) if costs else (None, None, None)
        width = (high - low) / low if costs else None
        classification = ("no_feasible_plan" if width is None else "width_below_0.1_percent" if width < .001
                          else "width_above_1_percent" if width > .01 else "intermediate_width")
        result.append(dict(task_id=run["task_id"], mode=run["mode"], exact_run_id=run_id,
            scope="frozen_partition_final_stage" if run["mode"] == "sequential" else "mode_space",
            total_plans=expected, feasible_plans=len(costs),
            min_us=low*1e6 if costs else None, median_us=middle*1e6 if costs else None,
            max_us=high*1e6 if costs else None, relative_width=width,
            width_percent=width*100 if width is not None else None,
            within_0_1_percent=sum(c <= low*1.001 for c in costs) if costs else 0,
            within_1_percent=sum(c <= low*1.01 for c in costs) if costs else 0,
            distinct_exact_float_costs=len(set(costs)), classification=classification))
    if any(sha(source / name) != value for name, value in hashes.items()):
        raise ValueError("source data changed while reading")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "all_modes.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, list(result[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(result)
    joint = [r for r in result if r["mode"] == "joint"]
    report = dict(source=str(source.resolve()), source_sha256=hashes, joint_tasks=joint,
        exact_modes_checked=len(result), thresholds=dict(degenerate_below=.001, dispersed_above=.01),
        zero_feasible_width=None, new_plan_evaluations=0, capacity_sweep_performed=False,
        algorithm_changed=False, profile_changed=False,
        interpretation="global feasible cost width; not evidence of search difficulty or fusion-placement interaction")
    (output / "report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8", newline="\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
