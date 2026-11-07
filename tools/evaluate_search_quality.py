"""One fixed evaluation budget on the existing measured-cost full-model case.

The result is the best feasible complete plan observed within the budget.
Only the experiment scorer is bounded; the search algorithms are unchanged.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cost_model.search_evaluation import _CachedScorer, _Mode, _run_search
from simulator import validate_fusion_plan
from tools.evaluate_executable_search import REFERENCES, describe, reference_plan, setup

MODES = ("joint", "sequential", "placement_then_fusion")
SEEDS = (0, 7, 23)


class EvaluationBudgetReached(Exception):
    pass


class BudgetScorer(_CachedScorer):
    def __init__(self, graph, profile, model, budget):
        super().__init__(graph, profile)
        self.model, self.budget = model, budget
        # The existing full-model experiment already disables this cache.
        self.cache_enabled = False
        self.trace, self.best_plan, self.best_s = [], None, float("inf")

    def start_run(self):
        super().start_run()
        self.trace, self.best_plan, self.best_s = [], None, float("inf")
        self.started = time.perf_counter()

    def __call__(self, plan):
        if self.run_model_evaluations >= self.budget:
            raise EvaluationBudgetReached
        value = super().__call__(plan)
        if self.last_feasible and self.last_makespan < self.best_s:
            self.best_s, self.best_plan = self.last_makespan, copy.deepcopy(plan)
        self.trace.append({"evaluation": self.run_model_evaluations,
            "candidate_ms": self.last_makespan * 1e3, "feasible": self.last_feasible,
            "best_ms": self.best_s * 1e3 if self.best_plan is not None else None,
            "elapsed_s": time.perf_counter() - self.started})
        return value


def summarize(trace):
    feasible = [r for r in trace if r["feasible"]]
    initial = [r for r in trace[:4] if r["feasible"]]
    if not feasible or not initial:
        raise ValueError("No feasible plan in the initial completion block")
    best = min(feasible, key=lambda r: r["candidate_ms"])
    initial_ms = min(r["candidate_ms"] for r in initial)
    return {"first_score_ms": trace[0]["candidate_ms"],
        "initial_best_ms": initial_ms, "initial_evaluations": min(4, len(trace)),
        "final_best_ms": best["candidate_ms"], "first_final_best_evaluation": best["evaluation"],
        "improvement_after_initial_percent": 100 * (initial_ms - best["candidate_ms"]) / initial_ms}


def run_one(graph, base, registry, mode, seed, budget):
    scorer = BudgetScorer(graph, base.profile, base.model, budget)
    # Four scores per non-terminal leaf (three original + one unfused).
    # Staged search may request extra incumbent rescoring, also bounded here.
    iterations = budget // 4
    started, completed = time.perf_counter(), True
    try:
        _run_search("mcts", graph, scorer, _Mode(mode, registry, True, (0, 1)),
            seed=seed, beam=8, rollouts=3, mcts_iterations=iterations, top_k=1,
            unfused_guided_rollout=True)
    except EvaluationBudgetReached:
        completed = False
    elapsed = time.perf_counter() - started
    if scorer.best_plan is None:
        raise ValueError("No feasible complete plan observed")
    validate_fusion_plan(graph, scorer.best_plan, base.profile.n_devs,
                         registry=registry, require_executable=True)
    result = {"mode": mode, "seed": seed, "evaluation_budget": budget,
        **scorer.run_metrics(), **summarize(scorer.trace), "elapsed_s": elapsed,
        "search_returned_normally": completed,
        "fused_groups": sum(len(g.node_ids) > 1 for g in scorer.best_plan.groups),
        "devices": sorted({g.device for g in scorer.best_plan.groups})}
    return result, scorer.trace


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    registry, hardware, graph, base = setup()
    historical_path = ROOT / "data/executable_search_model_measured_v3/report.json"
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    references = {}
    started = time.perf_counter()
    for name, patterns in REFERENCES.items():
        references[name] = describe(base, reference_plan(graph, registry, patterns))
        if abs(references[name]["makespan_ms"] - historical["references"][name]["makespan_ms"]) > 1e-10:
            raise ValueError("Frozen reference costs changed: " + name)
    baseline_elapsed = time.perf_counter() - started
    best_reference = min(r["makespan_ms"] for r in references.values())
    runs = []
    with (args.output / "convergence.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, ["mode", "seed", "evaluation", "candidate_ms", "feasible", "best_ms", "elapsed_s"])
        writer.writeheader()
        for mode in MODES:
            for seed in SEEDS:
                row, trace = run_one(graph, base, registry, mode, seed, 64)
                row["gap_to_best_known_reference"] = row["final_best_ms"] / best_reference - 1
                runs.append(row)
                writer.writerows({"mode": mode, "seed": seed, **point} for point in trace)
                stream.flush()
                (args.output / "runs.json").write_text(json.dumps(runs, indent=2) + "\n", encoding="utf-8")
                print(json.dumps(row), flush=True)
    report = {"scope": "CPU search quality; fixed full-model graph and existing costs",
        "evaluation_budget": 64, "initial_block": "first four completed scores, not a tree-free initialization",
        "result_kind": "best_feasible_observed_within_budget",
        "budget_semantics": "actual cost-model evaluations, including staged incumbent rescoring; cache disabled",
        "stopping": "stop before the first score exceeding the cap; record the incumbent even if search unwinds",
        "reference_baseline": {"evaluations": len(REFERENCES), "elapsed_s": baseline_elapsed,
                               "best_ms": best_reference, "plans": references},
        "graph_nodes": len(graph.nodes), "store_records": len(base.profile.store.records),
        "historical_report_sha256": hashlib.sha256(historical_path.read_bytes()).hexdigest(),
        "runs": runs, "new_gpu_execution": False, "cost_model_formula_changed": False,
        "global_optimum_proven": False, "end_to_end_measured": False}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, list(runs[0]))
        writer.writeheader()
        writer.writerows(runs)


if __name__ == "__main__":
    main()
