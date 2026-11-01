"""Verify the current small/layer four-mode development baseline and replay."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulator.dataset import plan_from_dict
from simulator.placement_artifacts import load_placed_op_graph, plan_fingerprint
from simulator.fusion_rules import default_fusion_registry


def without_elapsed(value):
    if isinstance(value, dict):
        return {k: without_elapsed(v) for k,v in value.items() if k != "elapsed_s"}
    if isinstance(value, list):
        return [without_elapsed(v) for v in value]
    return value


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(first, second):
    paths = (Path(first), Path(second))
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    require(without_elapsed(reports[0]) == without_elapsed(reports[1]), "replay differs outside elapsed_s")
    checked_candidates, checked_artifacts = 0, 0
    summaries = []
    for path, report in zip(paths, reports):
        require(report["version"] in ("1.2", "1.3"), "expected four-mode report version 1.2 or 1.3")
        for scale in report["scales"]:
            config = scale["configuration"]
            require(all(k in config for k in ("seed", "beam", "rollouts", "mcts_iterations", "top_k")),
                    "missing replay configuration")
            names = {s["mode"] for s in scale["searches"]}
            require(names == {"placement_only", "fusion_only", "sequential", "joint"}, "missing search mode")
            for search in scale["searches"]:
                frozen = search["budget"].get("frozen_partition")
                if search["mode"] == "sequential":
                    require(frozen is not None, "missing frozen partition")
                    require([s["stage"] for s in search["budget"]["stages"]] == ["fusion", "placement"],
                            "missing sequential stage")
                    if search["method"] == "mcts":
                        stages = search["budget"]["stages"]
                        require([s["iterations"] for s in stages] == config["sequential_iteration_split"],
                                "sequential iteration split mismatch")
                        require(sum(s["iterations"] for s in stages) == config["mcts_iterations"], "total budget mismatch")
                for candidate in search["candidates"]:
                    plan = plan_from_dict(candidate["plan"])
                    nodes = [n for g in plan.groups for n in g.node_ids]
                    require(len(nodes) == len(set(nodes)) == scale["nodes"], "duplicate or missing node")
                    require(candidate["deployable"] is False, "development result marked deployable")
                    if report["version"] == "1.3":
                        require(candidate["feasible"] and all(p <= c for p, c in
                            zip(candidate["peak_memory_bytes"], candidate["capacity_bytes"])),
                            "infeasible output candidate")
                    if frozen is not None:
                        actual = [dict(pattern_id=g.extra["pattern_id"],node_ids=list(g.node_ids)) for g in plan.groups]
                        require(actual == frozen, "placement changed frozen fusion partition")
                    checked_candidates += 1
                if not search["candidates"]:
                    expected = "no_feasible_in_space" if search["method"] == "exact" else "evaluated_no_feasible"
                    require(report["version"] == "1.3" and search["plan"] is None and
                            search["best_placed_op_graph"] is None and search["reason"] == expected,
                            "invalid no-plan outcome")
                    continue
                best = search["candidates"][0]
                actual_placeholders=sum(default_fusion_registry().get(g["extra"]["pattern_id"]).implementation_status
                                        == "planned" for g in best["plan"]["groups"])
                require(best["placeholder_groups"] == actual_placeholders, "placeholder annotation mismatch")
                if search["mode"] == "placement_only":
                    require(actual_placeholders == 0, "placement-only selected a fusion placeholder")
                artifact_path = path.parent / search["best_placed_op_graph"]
                placed = load_placed_op_graph(artifact_path)
                require(all(set(n for g in c["plan"]["groups"] for n in g["node_ids"]) == set(placed.graph.nodes)
                            for c in search["candidates"]), "candidate contains a different node set")
                require(set(nodes for g in placed.plan.groups for nodes in g.node_ids) == set(placed.graph.nodes),
                        "artifact does not cover original graph")
                require(len(placed.graph.nodes) == scale["nodes"] and placed.graph_hash == scale["graph_hash"],
                        "artifact graph differs from report")
                require(placed.candidate_id == plan_fingerprint(plan_from_dict(best["plan"])), "artifact plan mismatch")
                require(placed.predicted_makespan == best["makespan_s"], "artifact prediction mismatch")
                require(placed.extra["deployable"] is False, "artifact marked deployable")
                checked_artifacts += 1
                if path == paths[0]:
                    other = json.loads((paths[1].parent / search["best_placed_op_graph"]).read_text(encoding="utf-8"))
                    require(json.loads(artifact_path.read_text(encoding="utf-8")) == other, "artifact replay differs")
            if path == paths[0]:
                summaries.append(dict(scale=scale["scale"],nodes=scale["nodes"],searches=len(scale["searches"]),
                    cost_model_evaluations=scale["cost_model_evaluations"],fusion_observations=[
                        dict(mode=s["mode"],method=s["method"],fused_groups=(
                            s["candidates"][0]["fused_groups"] if s["candidates"] else None))
                        for s in scale["searches"]]))
    return {"pass": True, "replay_equal_except": ["elapsed_s"], "reports": list(map(str,paths)),
        "checked_candidates_in_both_runs": checked_candidates, "checked_artifacts_in_both_runs": checked_artifacts,
        "scales": summaries, "scope": "CPU diagnostic baseline; no GPU performance or downstream deployment claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.first, args.second)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
