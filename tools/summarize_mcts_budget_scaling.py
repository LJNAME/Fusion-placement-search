"""Verify all scaling prefixes, preserve null infeasible costs, write tables."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cost_model.measured_fusion_calls import sha
from tools.evaluate_mcts_budget_scaling import PARENT, prefix_metrics, read
from tools.evaluate_stage_search import save


FIELDS = ("budget", "seed", "found_feasible", "first_feasible_evaluation", "distinct_feasible_plans",
          "best_feasible_ms", "expanded_decision_nodes", "max_decision_depth", "max_scored_assigned_nodes", "wall_s")


def run(directory):
    directory = Path(directory)
    config = read(directory / "plan.json")
    for name, expected in config["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("frozen source changed: " + name)
    if sha(PARENT / "plan.json") != config["fixed_parent_sha256"]:
        raise ValueError("fixed parent changed")
    rows, evaluations = [], 0
    for seed in config["seeds"]:
        folder = directory / f"seed_{seed}"
        result = read(folder / "result.json")
        trace = [json.loads(line) for line in (folder / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
        if len(trace) != max(config["budgets"]) or len({r["candidate_id"] for r in trace}) != len(trace):
            raise AssertionError("distinct-plan budget accounting error")
        if result["actual_unique_evaluations"] != len(trace) or not result["independent_64_prefix_checked"]:
            raise AssertionError("incomplete trajectory or missing independent control")
        best, previous_time, feasible_ids = None, 0., set()
        for i, entry in enumerate(trace, 1):
            if entry["evaluation"] != i or entry["elapsed_s"] < previous_time:
                raise AssertionError("nonchronological trajectory")
            previous_time = entry["elapsed_s"]
            feasible = all(m <= b for m, b in zip(entry["peak_memory_bytes"], config["capacity"]))
            if feasible != entry["feasible"]:
                raise AssertionError("fixed memory feasibility differs")
            if feasible:
                feasible_ids.add(entry["candidate_id"])
                best = entry["makespan_ms"] if best is None else min(best, entry["makespan_ms"])
            if entry["best_ms"] != best:
                raise AssertionError("feasible incumbent lost")
        previous_nodes = 0
        for budget in config["budgets"]:
            point = read(folder / f"budget_{budget}.json")
            expected = prefix_metrics(trace, budget, point["tree_stats"])
            if any(point[k] != expected[k] for k in expected):
                raise AssertionError("prefix metric differs from its raw observations")
            if point["expanded_decision_nodes"] != trace[budget - 1]["expanded_decision_nodes"]:
                raise AssertionError("decision nodes snapshot came from a later budget")
            if point["expanded_decision_nodes"] < previous_nodes or point["tree_stats"]["unary_expansions"]:
                raise AssertionError("tree counter inconsistent")
            previous_nodes = point["expanded_decision_nodes"]
            rows.append(point)
        evaluations += len(trace)
    summary = []
    for budget in config["budgets"]:
        points = sorted((r for r in rows if r["budget"] == budget), key=lambda r: config["seeds"].index(r["seed"]))
        feasible = [r for r in points if r["found_feasible"]]
        summary.append({"budget": budget, "seed_order": config["seeds"], "feasible_runs": len(feasible),
            "first_feasible_evaluations": [r["first_feasible_evaluation"] for r in points],
            "distinct_feasible_plans": [r["distinct_feasible_plans"] for r in points],
            "best_feasible_ms": [r["best_feasible_ms"] for r in points],
            "expanded_decision_nodes": [r["expanded_decision_nodes"] for r in points],
            "max_decision_depths": [r["max_decision_depth"] for r in points],
            "max_scored_assigned_nodes": [r["max_scored_assigned_nodes"] for r in points],
            "wall_s": [r["wall_s"] for r in points],
            "median_wall_s": statistics.median(r["wall_s"] for r in points)})
    save(directory / "summary.json", summary)
    with (directory / "all_runs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(sorted(rows, key=lambda r: (r["budget"], r["seed"])))
    save(directory / "verification.json", {"passed": True, "budget_seed_observations": len(rows),
        "continuous_trajectories": len(config["seeds"]), "actual_full_model_evaluations": evaluations,
        "independent_64_prefix_controls": len(config["seeds"]), "raw_prefix_metrics_and_memory_checked": True,
        "tree_metric": config["decision_node_metric"], "source_hashes_match": True,
        "extra_validation_cost_model_calls": 0, "new_gpu_measurements": 0,
        "nested_observations_are_not_independent_runs": True, "verifier_sha256": sha(Path(__file__)),
        "artifact_sha256": {p.relative_to(directory).as_posix(): sha(p) for p in sorted(directory.rglob('*'))
                            if p.is_file() and p.name != "verification.json"}})
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    run(parser.parse_args().directory)
