"""Read-only search diagnosis; no additional latency-model evaluations."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cost_model.fusion_library import load_library
from cost_model.legal_search import LegalFusionSearchSpace
from simulator.dataset import graph_from_dict
from tools.evaluate_stage_search import save


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def forced_actions(graph, registry):
    """A named path diagnostic, not a statistic over every possible path."""
    space = LegalFusionSearchSpace(graph, 2, registry, placement_policy="hybrid", tie_shared_params=True)
    state = space.initial_state()
    rows = []
    while not space.is_terminal(state):
        actions = space.actions(state)
        rows.append({"anchor": space.anchor(state), "options": len(actions)})
        choice = next(a for a in actions if len(a.candidate.node_ids) == 1 and a.device == 0)
        state = space.apply(state, choice)
    return {"path": "all Native / GPU0", "actions": len(rows),
            "forced_single_option_actions": sum(r["options"] == 1 for r in rows),
            "choice_actions": sum(r["options"] > 1 for r in rows), "first20": rows[:20]}


def run(directory):
    directory = Path(directory)
    registry = load_library().registry()
    result = {"scope": "fixed-64 measured-fusion refresh; CPU predicted makespan only",
              "additional_cost_model_calls": 0, "models": {}}
    csv_rows = []
    for name in ("qwen3_06b", "qwen25_3b"):
        folder = directory / name
        report = read(folder / "report.json")
        if report["status"] != "complete":
            raise ValueError("unfinished search")
        graph = graph_from_dict(read(folder / "graph.json"))
        audit = read(folder / "fusion_cost_audit.json")
        old = read(ROOT / "data/full_joint_models_v1" / name / "report.json")
        model = {"nodes": len(graph.nodes), "action_path": forced_actions(graph, registry),
                 "registered_fusions": audit["registered"], "matched_fusions": report["fusion_candidates"],
                 "new_measured_candidate_device_pairs": dict(Counter(r["pattern_id"] for r in audit["candidate_costs"] if r["measurement"])),
                 "joint_qk_structural_groups": audit["joint_qk_structural_groups"],
                 "joint_qk_matched_groups": audit["joint_qk_matched_groups"], "tasks": {}}
        for cap, task in report["tasks"].items():
            stats = {}
            for method in report["scope"]["methods"]:
                runs = [r for r in task["runs"] if r["method"] == method]
                for row in runs:
                    if row["budget_used"] != 64 or len(row["trace"]) != 64:
                        raise AssertionError("budget changed")
                trees = [r for row in runs for r in row["trace"] if "tree_assigned_nodes" in r]
                bests = [r["best"]["makespan_ms"] for r in runs if r["best"]]
                before = old["tasks"][cap]["summary"][method]["median_ms"]
                data = {"feasible_runs": len(bests), "median_ms": statistics.median(bests) if bests else None,
                        "old_median_ms": before, "first_best_evaluations": [r["first_final_best_evaluation"] for r in runs],
                        "initial_best_ms": [r["initial_best_ms"] for r in runs],
                        "feasible_evaluations": [r["feasible_evaluations"] for r in runs],
                        "median_search_s": statistics.median(r["elapsed_s"] for r in runs),
                        "max_tree_assigned_nodes": max((r["tree_assigned_nodes"] for r in trees), default=None),
                        "max_tree_choices": max((r["tree_choices"] for r in trees), default=None),
                        "best_fusion_patterns": [r["best"]["fused_patterns"] if r["best"] else None for r in runs],
                        "measured_calls_in_best": [r["best"]["measured_call_count"] if r["best"] else None for r in runs],
                        "measured_group_evaluations": sum(r["measured_call_count"] for row in runs for r in row["trace"])}
                if trees:
                    data["minimum_rollout_fraction"] = 1 - data["max_tree_assigned_nodes"] / len(graph.nodes)
                for old_run, new_run in zip([r for r in old["tasks"][cap]["runs"] if r["method"] == method], runs):
                    # This quantifies identical proposal prefixes, independent of changed scores.
                    new_run_ids = [r["candidate_id"] for r in new_run["trace"]]
                    old_ids = [r["candidate_id"] for r in old_run["trace"]]
                    data.setdefault("same_candidate_positions_as_before", []).append(sum(a == b for a, b in zip(new_run_ids, old_ids)))
                    data.setdefault("selected_plan_changed", []).append(
                        (old_run["best"] or {}).get("candidate_id") != (new_run["best"] or {}).get("candidate_id"))
                stats[method] = data
                csv_rows.append((name, cap, method, len(bests), before, data["median_ms"], data["median_search_s"],
                                 data["max_tree_assigned_nodes"], data["measured_group_evaluations"]))
            model["tasks"][cap] = stats
        result["models"][name] = model
    save(directory / "diagnosis.json", result)
    import csv
    with (directory / "before_after.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("model", "capacity", "method", "feasible_runs", "before_ms", "after_ms", "search_s",
                         "max_tree_assigned_nodes", "measured_group_evaluations"))
        writer.writerows(csv_rows)
    print(json.dumps({name: {"nodes": m["nodes"], "action_path": {k: v for k, v in m["action_path"].items() if k != "first20"},
                                  "tasks": m["tasks"]} for name, m in result["models"].items()}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    run(parser.parse_args().directory)
