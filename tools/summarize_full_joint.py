"""Verify completed full-graph artifacts without additional cost-model calls."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cost_model.fusion_library import load_library
from cost_model.search_evaluation import _plan_key
from simulator import validate_fusion_plan
from simulator.dataset import graph_from_dict, plan_from_dict
from simulator.placement_artifacts import graph_fingerprint
from tools.evaluate_full_joint import assert_shared_colocated
from tools.evaluate_stage_search import save


def run(directory):
    directory = Path(directory)
    registry = load_library().registry("full")
    summary = {"scope": "full forward makespan predictions; reference gaps are signed, never Exact gaps",
               "models": {}, "direct_full_model_gpu_execution": False}
    artifacts, count, evaluations, budgets = {}, 0, 0, set()
    for name in ("qwen3_06b", "qwen25_3b"):
        folder = directory / name
        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        if report["status"] != "complete":
            raise ValueError(f"experiment incomplete: {name}")
        config = json.loads((folder / "plan.json").read_text(encoding="utf-8"))
        if report["scope"] != config:
            raise AssertionError("report and frozen configuration differ")
        for path, digest in config["source_sha256"].items():
            if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
                raise AssertionError(f"source changed during experiment: {path}")
        graph = graph_from_dict(json.loads((folder / "graph.json").read_text(encoding="utf-8")))
        if graph_fingerprint(graph) != report["graph_fingerprint"]:
            raise AssertionError("graph changed")
        summary["models"][name] = {k: report[k] for k in ("nodes", "edges", "fusion_candidates", "branch_regions")}
        summary["models"][name]["tasks"] = {}
        for cap, task in report["tasks"].items():
            seen_runs = set()
            for row in task["runs"]:
                run_id = (row["method"], row["seed"])
                if run_id in seen_runs:
                    raise AssertionError("duplicate run")
                seen_runs.add(run_id)
                trace = row["trace"]
                if not (len(trace) == row["budget_used"] == row["unique_candidates"] == config["budget"]):
                    raise AssertionError("unequal real evaluation budget")
                if len({r["candidate_id"] for r in trace}) != len(trace):
                    raise AssertionError("duplicate candidate charged twice")
                best = None
                for i, entry in enumerate(trace, 1):
                    if entry["evaluation"] != i or not math.isfinite(entry["makespan_ms"]):
                        raise AssertionError("invalid evaluation trace")
                    feasible = all(p <= c for p, c in zip(entry["peak_memory_bytes"], task["capacity_bytes"]))
                    if entry["feasible"] != feasible:
                        raise AssertionError("memory decision differs from capacity")
                    if feasible and (best is None or entry["makespan_ms"] < best["makespan_ms"]):
                        best = entry
                    if entry["best_ms"] != (best["makespan_ms"] if best else None):
                        raise AssertionError("best feasible incumbent was lost")
                if row["best"] != best:
                    raise AssertionError("selected plan does not match best scored candidate")
                if best is not None:
                    path = folder / row["plan_path"]
                    plan = plan_from_dict(json.loads(path.read_text(encoding="utf-8")))
                    validate_fusion_plan(graph, plan, 2, registry=registry, require_executable=True)
                    assert_shared_colocated(graph, plan)
                    key = hashlib.sha256(json.dumps((report["graph_fingerprint"], _plan_key(plan)), separators=(",", ":")).encode()).hexdigest()
                    if key != best["candidate_id"] or key != path.stem:
                        raise AssertionError("exported winning plan differs from scored plan")
                    ref = task["best_known_reference"]
                    if ref:
                        gap = (best["makespan_ms"] / ref["makespan_ms"] - 1) * 100
                        if abs(gap - row["known_reference_gap_percent"]) > 1e-10:
                            raise AssertionError("reference gap mismatch")
                count += 1
                evaluations += row["budget_used"]
                budgets.add(row["budget_used"])
            expected = {(m, s) for m in config["methods"] for s in config["seeds"]}
            if seen_runs != expected:
                raise AssertionError("missing run")
            summary["models"][name]["tasks"][cap] = {
                "capacity_bytes": task["capacity_bytes"], "best_known_reference": task["best_known_reference"],
                "summary": task["summary"]}
        for path in sorted(folder.rglob("*.json")):
            artifacts[path.relative_to(directory).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    save(directory / "summary.json", summary)
    with (directory / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(("model", "capacity", "method", "feasible_runs", "runs", "median_ms", "median_known_reference_gap_percent", "median_cross_device_edges"))
        for name, model in summary["models"].items():
            for cap, task in model["tasks"].items():
                for method, result in task["summary"].items():
                    writer.writerow((name, cap, method, result["feasible_runs"], result["runs"], result["median_ms"], result["median_reference_gap_percent"], result["median_cross_device_edges"]))
    save(directory / "verification.json", {"completed_runs": count,
        "actual_search_evaluations": evaluations, "budgets_per_run": sorted(budgets),
        "all_model_source_hashes_match": True, "all_candidate_ids_unique_per_run": True,
        "feasible_incumbents_and_saved_plan_ids_verified": True,
        "all_saved_plans_executable_under_manifest_and_tied_weight_constraint": True,
        "additional_model_calls_for_verification": 0,
        "scope": "manifest/static memory feasibility, not GPU deployment or training correctness",
        "new_gpu_measurements": 0,
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "artifact_sha256": artifacts})
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    run(parser.parse_args().directory)
