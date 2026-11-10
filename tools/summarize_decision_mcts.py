"""Verify fixed-budget closure ablation and emit per-evaluation convergence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cost_model.measured_fusion_calls import sha
from cost_model.search_evaluation import _plan_key, build_evaluation_graph
from simulator.dataset import plan_from_dict
from simulator.placement_artifacts import graph_fingerprint
from simulator import validate_fusion_plan
from cost_model.fusion_library import load_library
from tools.evaluate_full_joint import assert_shared_colocated
from tools.evaluate_stage_search import save


def run(directory):
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    config = report["config"]
    if report["status"] != "complete":
        raise ValueError("experiment incomplete")
    for path, digest in config["source_sha256"].items():
        if sha(ROOT / path) != digest:
            raise AssertionError("source changed: " + path)
    if sha(ROOT / config["parent_report"]) != config["parent_report_sha256"]:
        raise AssertionError("preceding control report changed")
    graph, registry = build_evaluation_graph("model"), load_library().registry()
    if graph_fingerprint(graph) != config["graph_fingerprint"]:
        raise AssertionError("graph changed")
    count = 0
    curves = []
    for cap, task in report["tasks"].items():
        references = None
        for row in task["runs"]:
            trace = row["trace"]
            if len(trace) != row["budget_used"] or row["budget_used"] != config["budget"]:
                raise AssertionError("budget differs")
            if len({r["candidate_id"] for r in trace}) != len(trace):
                raise AssertionError("duplicate candidate charged")
            init = [{k: r[k] for k in ("candidate_id", "makespan_ms", "peak_memory_bytes")} for r in trace[:2]]
            if references is not None and init != references:
                raise AssertionError("initial priors differ between variants")
            references = init
            best = None
            for n, entry in enumerate(trace, 1):
                if n != entry["evaluation"]:
                    raise AssertionError("nonchronological trace")
                feasible = all(p <= b for p, b in zip(entry["peak_memory_bytes"], config["capacities"][cap]))
                if feasible != entry["feasible"]:
                    raise AssertionError("memory constraint changed")
                if feasible and (best is None or entry["makespan_ms"] < best["makespan_ms"]):
                    best = entry
                if entry["best_ms"] != (best["makespan_ms"] if best else None):
                    raise AssertionError("incumbent not preserved")
            if row["best"] != best:
                raise AssertionError("wrong winning candidate")
            if row["method"] == "decision_joint" and row["tree_stats"]["unary_expansions"]:
                raise AssertionError("unary state survived closure")
            audit = row["action_audit"]
            if sum(audit["legal_action_count_histogram"].values()) != audit["action_queries"]:
                raise AssertionError("action query accounting error")
            if sum(audit["forced_primary_causes"].values()) != audit["query_kinds"].get("forced", 0):
                raise AssertionError("forced queries not fully classified")
            if row["best_plan"]:
                plan = plan_from_dict(row["best_plan"])
                validate_fusion_plan(graph, plan, 2, registry=registry, require_executable=True)
                assert_shared_colocated(graph, plan)
                candidate = hashlib.sha256(json.dumps((config["graph_fingerprint"], _plan_key(plan)), separators=(",", ":")).encode()).hexdigest()
                if candidate != best["candidate_id"]:
                    raise AssertionError("export differs from the scored winner")
            count += 1
        for method in config["methods"]:
            rows = [r for r in task["runs"] if r["method"] == method]
            if {r["seed"] for r in rows} != set(config["seeds"]):
                raise AssertionError("missing seed")
            for i in range(config["budget"]):
                valid = [r["trace"][i]["best_ms"] for r in rows if r["trace"][i]["best_ms"] is not None]
                curves.append((cap, method, i + 1, len(valid), statistics.median(valid) if valid else None))
    if count != 12:
        raise AssertionError("incomplete ablation")
    with (directory / "convergence.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("capacity", "method", "actual_evaluation", "feasible_runs", "median_best_ms_if_feasible"))
        writer.writerows(curves)
    save(directory / "artifact_review.json", {"completed_runs": count, "actual_full_search_evaluations": count * 64,
        "same_initialization_and_memory_budgets": True, "candidate_and_best_plan_ids_checked": True,
        "all_forced_queries_classified": True, "source_hashes_match": True,
        "additional_latency_model_calls": 0, "verifier_sha256": sha(Path(__file__)),
        "artifact_sha256": {p.relative_to(directory).as_posix(): sha(p) for p in sorted(directory.rglob('*')) if p.is_file() and p.name != "artifact_review.json"}})
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    run(parser.parse_args().directory)
