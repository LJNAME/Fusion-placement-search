"""Exact-referenced medium subgraphs; observe existing MCTS without changing it."""
import argparse
import copy
import csv
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibration.probe_kerv_graph import save
from cost_model.exact_search import count_exact_search_space
from cost_model.hardware_profile import AnalyticHardwareProfile
from cost_model.legal_search import LegalFusionSearchSpace
from cost_model.search_evaluation import (
    INFEASIBLE_PENALTY_S, _modes, _partition_key, _plan_key, _run_search, build_evaluation_graph,
)
from simulator import validate_fusion_plan
from simulator.dataset import graph_from_dict, graph_to_dict, plan_to_dict
from simulator.hardware import hardware_from_dict, hardware_to_dict, load_hardware
from simulator.placement_artifacts import PlacedOpGraph, load_placed_op_graph, save_placed_op_graph
from simulator.schema import OpGraph
from tools.evaluate_mcts_mainline import ObservedScorer, SOURCES, digest, file_sha

HARDWARE = ROOT / "data/bw1000_2card/bw1000_2card.calibrated.json"
SELECTIONS = {
    "serial_tail": (
        "self_attention.qk_matmul", "self_attention.softmax", "self_attention.av_matmul",
        "self_attention.out_proj", "residual_add_1", "pre_mlp_norm",
    ),
    "attention_fork": (
        "self_attention.q_proj", "self_attention.k_proj", "self_attention.q_norm",
        "self_attention.k_norm", "self_attention.q_rope", "self_attention.k_rope",
        "self_attention.qk_matmul", "self_attention.softmax",
    ),
    "mlp": ("pre_mlp_norm", "mlp.gate_proj", "mlp.up_proj", "mlp.silu",
            "mlp.gated_mul", "mlp.down_proj", "residual_add_2"),
}
CANDIDATE_FIELDS = (
    "run_id", "row_id", "plan_signature", "partition_signature", "device_assignment",
    "fused_groups", "used_devices", "cross_device_bytes", "makespan_s", "memory_feasible",
    "peak_memory_bytes",
)


def extract_graph(full, name):
    selected = {"L000." + n for n in SELECTIONS[name]}
    boundary = {e.src for e in full.edges if e.dst in selected and e.src not in selected}
    graph = OpGraph(name="qwen3-medium-" + name)
    for node_id in full.topological_order():
        if node_id not in selected | boundary:
            continue
        node = copy.deepcopy(full.nodes[node_id])
        if node_id in boundary:
            node.op_type = "input"
            node.flops = node.param_bytes = 0.
            node.extra.update(boundary_source=node_id, boundary_stub=True)
        graph.add_node(node)
    for edge in full.edges:
        if edge.dst in selected and edge.src in selected | boundary:
            graph.add_edge(copy.deepcopy(edge))
    graph.extra.update(
        selected_original_nodes=sorted(selected), boundary_input_nodes=sorted(boundary),
        external_consumers=[copy.deepcopy(e.__dict__) for e in full.edges
                            if e.src in selected and e.dst not in selected],
        scope="forward subgraph; boundary tensors initially available at freely searched locations",
        boundary_output_lifetime="terminal outputs retained to subgraph end; no external schedule",
    )
    graph.topological_order()
    return graph


def prepare(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "inputs").mkdir()
    full = build_evaluation_graph("layer")
    hardware = load_hardware(HARDWARE)
    tasks = []
    for name in SELECTIONS:
        graph = extract_graph(full, name)
        variants = [("actual_capacity", None)]
        if name == "mlp":
            variants += [("parameter_cap_75", .75), ("parameter_cap_90", .90)]
        for variant, fraction in variants:
            hw = copy.deepcopy(hardware)
            if fraction is not None:
                cap = fraction * sum(n.param_bytes for n in graph.nodes.values())
                for device in hw.devices:
                    device.max_mem = cap
            task_id = name + "_" + variant
            graph_path = "inputs/" + task_id + ".graph.json"
            hardware_path = "inputs/" + task_id + ".hardware.json"
            save(output / graph_path, graph_to_dict(graph))
            save(output / hardware_path, hardware_to_dict(hw))
            counts = {}
            for mode in _modes(hw.n_devs):
                devices = (0,) if mode.name == "sequential" else mode.allowed_devices
                space = LegalFusionSearchSpace(graph, hw.n_devs, mode.registry,
                    require_executable=mode.require_executable, allowed_devices=devices)
                counts[mode.name] = count_exact_search_space(space)
            if not 8 <= len(graph.nodes) <= 10 or counts["joint"] > 10000:
                raise ValueError("task exceeds preregistered size limit")
            tasks.append(dict(task_id=task_id, family=name, nodes=len(graph.nodes),
                graph=graph_path, hardware=hardware_path,
                graph_sha256=file_sha(output / graph_path), hardware_sha256=file_sha(output / hardware_path),
                capacity_kind="actual_snapshot" if fraction is None else "synthetic_mechanism_only",
                capacity_rule=None if fraction is None else dict(fraction_of_parameter_sum=fraction),
                max_memory_bytes=[d.max_mem for d in hw.devices],
                parameter_sum_bytes=sum(n.param_bytes for n in graph.nodes.values()),
                structural_counts=counts,
                sequential_count_scope="fusion stage only; placement count depends on frozen partition"))
    sources = tuple(dict.fromkeys((*SOURCES, "tools/evaluate_mcts_medium.py", "simulator/schema.py",
                                  "simulator/dataset.py", "simulator/placement_artifacts.py")))
    plan = dict(tasks=tasks, seeds=[0, 7, 23], budgets=[16, 64, 256], rollouts=3, top_k=4,
        expected_exact_runs=20, expected_mcts_runs=180,
        phase_order=["all_exact", "freeze_references", "all_mcts"],
        source_sha256={p: file_sha(ROOT / p) for p in sources},
        hardware_source=dict(path=str(HARDWARE), sha256=file_sha(HARDWARE)),
        policy="all predefined tasks retained, including infeasible/no-improvement cases",
        scorer="finite makespan + penalty for over-capacity; strict feasible final selection",
        capacity_source="explicit task capacity tuple, not profile.max_memory",
        infeasible_penalty_s=INFEASIBLE_PENALTY_S,
        reference="feasible minimum of Exact scored candidates; Sequential final stage only, not global optimum",
        initialization="first scoring request in final stage; Sequential fusion stage excluded",
        observation="complete scored candidates, not tree node visits",
        scope="analytic development model, no GPU/SROP/Beam; no downstream execution claim",
        algorithm_changed="capacity penalty and final admission only; tree/rollout parameters unchanged",
        profile_timing_changed=False,
        capacity_changes="only two explicitly synthetic MLP variants; no timing constant changes")
    baseline = ROOT / "data/mcts_medium_v1"
    if (baseline / "plan.json").exists():
        plan["regression_baseline"] = dict(source_commit="ee2c9bf", path=str(baseline),
            sha256={name: file_sha(baseline / name) for name in (
                "plan.json", "runs.jsonl", "references.json", "candidates.csv.gz", "requests.csv.gz")})
    save(output / "plan.json", plan)
    return plan


class MediumScorer(ObservedScorer):
    def __init__(self, graph, profile, *, capacity=None):
        super().__init__(graph, profile, capacity=capacity)
        self.plans = {}

    def estimate(self, plan):
        result = super().estimate(plan)
        if self.observing:
            key = _plan_key(plan)
            if key not in self.plans:
                self.plans[key] = copy.deepcopy(plan)
                self.observed[key]["peak_memory_bytes"] = list(result.peak_memory)
        return result


def final_requests(budget, requests):
    offset = sum(s["scorer_requests"] for s in budget["stages"][:-1])
    return requests[offset:]


def metrics(rows, requests, eligible, selected_signature):
    by_id = {int(r["row_id"]): r for r in rows}
    selected = next((r for r in rows if r["plan_signature"] == selected_signature), None)
    feasible = [by_id[i] for i in sorted(set(eligible)) if by_id[i]["memory_feasible"]]
    best = min(feasible, key=lambda r: (r["makespan_s"], r["plan_signature"])) if feasible else None
    first = by_id[eligible[0]]
    improved = None
    if best and first["memory_feasible"]:
        improved = first["makespan_s"] - best["makespan_s"]
    return dict(score_requests=len(requests), unique_scored_plans=len(rows),
        eligible_unique_plans=len(set(eligible)), eligible_feasible_plans=len(feasible),
        selected_cost_s=selected["makespan_s"] if selected else None,
        selected_feasible=selected["memory_feasible"] if selected else False,
        selected_peak_memory_bytes=selected["peak_memory_bytes"] if selected else None,
        first_final_stage_cost_s=first["makespan_s"], first_final_stage_feasible=first["memory_feasible"],
        best_feasible_scored_cost_s=best["makespan_s"] if best else None,
        best_feasible_scored_signature=best["plan_signature"] if best else None,
        first_best_feasible_request=next((i+1 for i, k in enumerate(eligible)
            if by_id[k]["memory_feasible"] and best and by_id[k]["makespan_s"] == best["makespan_s"]), None),
        feasible_improvement_over_initialization_s=improved,
        feasibility_found_after_first=bool(best and not first["memory_feasible"]),
        unique_fusion_partitions=len({by_id[i]["partition_signature"] for i in set(eligible)}),
        unique_device_assignments=len({by_id[i]["device_assignment"] for i in set(eligible)}))


def run_one(graph, profile, mode, method, iterations, seed, *, capacity=None):
    capacity = tuple(capacity) if capacity is not None else tuple(d.max_mem for d in profile.hardware.devices)
    scorer = MediumScorer(graph, profile, capacity=capacity)
    plans, elapsed, budget = _run_search(method, graph, scorer, mode, seed=seed,
        beam=8, rollouts=3, mcts_iterations=iterations, top_k=4)
    rows = list(scorer.observed.values())
    requests = list(scorer.requests)
    eligible = final_requests(budget, requests)
    if len(rows) != scorer.calls or len(requests) != budget["scorer_requests"]:
        raise ValueError("observer counters changed")
    for candidate in plans:
        validate_fusion_plan(graph, candidate, profile.n_devs, registry=mode.registry,
                             require_executable=mode.require_executable)
    if mode.name == "sequential":
        expected = budget["frozen_partition"]
        if any([dict(pattern_id=p, node_ids=list(ns)) for p, ns in _partition_key(candidate)] != expected
               for candidate in plans):
            raise ValueError("returned Sequential plan changed partition")
        if len({rows[i]["partition_signature"] for i in set(eligible)}) != 1:
            raise ValueError("Sequential placement changed partition")
    selected_signature = digest(_plan_key(plans[0])) if plans else None
    result = dict(mode=mode.name, method=method, seed=seed,
        iterations=iterations if method == "mcts" else None,
        observed_search_wall_s=elapsed, budget=budget, selected_signature=selected_signature,
        selected_plan=plan_to_dict(plans[0]) if plans else None, reason=budget["reason"],
        metrics=metrics(rows, requests, eligible, selected_signature))
    return result, scorer, eligible


def check_inputs(output, plan):
    for path, expected in plan["source_sha256"].items():
        if file_sha(ROOT / path) != expected:
            raise ValueError("source changed: " + path)
    if file_sha(plan["hardware_source"]["path"]) != plan["hardware_source"]["sha256"]:
        raise ValueError("base hardware changed")
    for task in plan["tasks"]:
        for field in ("graph", "hardware"):
            if file_sha(output / task[field]) != task[field + "_sha256"]:
                raise ValueError("frozen input changed")
    baseline = plan.get("regression_baseline")
    if baseline and any(file_sha(Path(baseline["path"]) / name) != value
                        for name, value in baseline["sha256"].items()):
        raise ValueError("regression baseline changed")


def gap(cost, reference):
    return None if cost is None or reference is None else (cost - reference) / reference


def run(output):
    output = Path(output)
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    check_inputs(output, plan)
    if (output / "runs.jsonl").exists():
        raise ValueError("run already exists; prepare a new directory")
    runs, references, exact_tables, artifacts = [], {}, {}, {}
    loaded = {t["task_id"]: (graph_from_dict(json.loads((output / t["graph"]).read_text())),
        AnalyticHardwareProfile(hardware_from_dict(json.loads((output / t["hardware"]).read_text()))))
        for t in plan["tasks"]}
    with gzip.open(output / "candidates.csv.gz", "wt", encoding="utf-8", newline="") as cf, \
         gzip.open(output / "requests.csv.gz", "wt", encoding="utf-8", newline="") as qf, \
         (output / "runs.jsonl").open("w", encoding="utf-8", newline="\n") as rf:
        cw = csv.DictWriter(cf, CANDIDATE_FIELDS, lineterminator="\n"); cw.writeheader()
        qw = csv.writer(qf, lineterminator="\n"); qw.writerow(("run_id", "request", "row_id"))

        def perform(task, mode, method, iterations, seed):
            task_id = task["task_id"]
            graph, profile = loaded[task_id]
            result, scorer, eligible = run_one(graph, profile, mode, method, iterations, seed,
                                             capacity=tuple(task["max_memory_bytes"]))
            result.update(run_id=len(runs), task_id=task_id)
            rows = list(scorer.observed.values())
            if method == "mcts":
                table = exact_tables[task_id]
                if any(r["plan_signature"] not in table or
                       table[r["plan_signature"]] != (r["makespan_s"], r["memory_feasible"])
                       for r in rows):
                    raise ValueError("MCTS score differs from exhaustive Joint table")
            elif mode.name == "joint":
                exact_tables[task_id] = {r["plan_signature"]: (r["makespan_s"], r["memory_feasible"]) for r in rows}
            # Export feasible Exact minima and feasible actual MCTS winners only.
            sig = result["metrics"]["best_feasible_scored_signature"] if method == "exact" else (
                result["selected_signature"] if result["metrics"]["selected_feasible"] else None)
            if sig and task_id + ":" + sig not in artifacts:
                winner = next(scorer.plans[k] for k, r in scorer.observed.items() if r["plan_signature"] == sig)
                scorer.observing = False
                estimate = scorer.estimate(winner)
                artifact = "best_plans/" + task_id + "-" + sig.removeprefix("sha256:") + ".json"
                save_placed_op_graph(output / artifact, PlacedOpGraph(graph, profile.hardware, winner,
                    predicted_makespan=estimate.makespan, predicted_peak_memory=estimate.peak_memory,
                    extra=dict(deployable=False, scope="medium fixed analytic development", task_id=task_id)))
                artifacts[task_id + ":" + sig] = artifact
            for row in rows:
                cw.writerow(dict(run_id=result["run_id"], **{k: json.dumps(v) if k == "peak_memory_bytes" else v
                    for k, v in row.items()}))
            for i, row_id in enumerate(scorer.requests):
                qw.writerow((result["run_id"], i+1, row_id))
            rf.write(json.dumps(result, allow_nan=False) + "\n"); rf.flush()
            runs.append(result)
            mm = result["metrics"]
            print(f'{method} {task_id}/{mode.name} n={iterations} seed={seed} '
                  f'selected_feasible={mm["selected_feasible"]} best_feasible={mm["best_feasible_scored_cost_s"]}', flush=True)
            return result

        for task in plan["tasks"]:
            references[task["task_id"]] = {}
            for mode in _modes(2):
                r = perform(task, mode, "exact", 16, 0)
                references[task["task_id"]][mode.name] = dict(run_id=r["run_id"],
                    kind="two_stage_exact_feasible_reference" if mode.name == "sequential" else "restricted_feasible_exact",
                    cost_s=r["metrics"]["best_feasible_scored_cost_s"],
                    feasible_count=r["metrics"]["eligible_feasible_plans"])
        save(output / "references.json", references)
        for task in plan["tasks"]:
            for budget in plan["budgets"]:
                for seed in plan["seeds"]:
                    for mode in _modes(2):
                        perform(task, mode, "mcts", budget, seed)
    results = []
    for r in runs:
        if r["method"] != "mcts":
            continue
        mm = r["metrics"]; ref = references[r["task_id"]][r["mode"]]["cost_s"]
        joint = references[r["task_id"]]["joint"]["cost_s"]
        selected = mm["selected_cost_s"] if mm["selected_feasible"] else None
        results.append(dict(run_id=r["run_id"], task_id=r["task_id"], mode=r["mode"],
            iterations=r["iterations"], seed=r["seed"], selected_feasible=mm["selected_feasible"],
            reason=r["reason"],
            selected_cost_s=mm["selected_cost_s"],
            selected_mode_reference_gap=gap(selected, ref), selected_joint_exact_gap=gap(selected, joint),
            best_feasible_scored_cost_s=mm["best_feasible_scored_cost_s"],
            diagnostic_scored_mode_reference_gap=gap(mm["best_feasible_scored_cost_s"], ref),
            first_final_stage_feasible=mm["first_final_stage_feasible"],
            feasible_improvement_over_initialization_s=mm["feasible_improvement_over_initialization_s"],
            feasibility_found_after_first=mm["feasibility_found_after_first"],
            first_best_feasible_request=mm["first_best_feasible_request"],
            eligible_feasible_plans=mm["eligible_feasible_plans"],
            eligible_unique_plans=mm["eligible_unique_plans"], score_requests=mm["score_requests"],
            cost_model_evaluations=mm["unique_scored_plans"], observed_search_wall_s=r["observed_search_wall_s"]))
    with (output / "results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, list(results[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(results)
    save(output / "artifacts.json", artifacts)
    check_inputs(output, plan)
    report = dict(complete=True, exact_runs=sum(r["method"] == "exact" for r in runs),
        mcts_runs=len(results), selected_feasible_runs=sum(r["selected_feasible"] for r in results),
        infeasible_selected_despite_feasible_scored=sum(not r["selected_feasible"] and
            r["best_feasible_scored_cost_s"] is not None for r in results),
        scored_table_matches_exact=True, references_frozen_before_mcts=True,
        capacity_admission_fixed=True, profile_timing_changed=False, gpu=False, beam=False, srop=False)
    save(output / "report.json", report)
    return report


def verify_regression(plan, runs, observed, queries):
    if plan["expected_mcts_runs"] != 180 or "regression_baseline" not in plan:
        return dict(performed=False)
    baseline = Path(plan["regression_baseline"]["path"])
    old_plan = json.loads((baseline / "plan.json").read_text())
    old_tasks = {t["task_id"]: t for t in old_plan["tasks"]}
    for task in plan["tasks"]:
        if any(task[k] != old_tasks[task["task_id"]][k] for k in ("graph_sha256", "hardware_sha256")):
            raise ValueError("regression task input changed")
    old_runs = [json.loads(s) for s in (baseline / "runs.jsonl").read_text().splitlines()]
    key = lambda r: (r["task_id"], r["method"], r["mode"], r["iterations"], r["seed"])
    old_by_key = {key(r): r for r in old_runs}
    old_rows, old_queries = {}, {}
    with gzip.open(baseline / "candidates.csv.gz", "rt", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            old_rows[(int(row["run_id"]), int(row["row_id"]))] = row["plan_signature"]
    with gzip.open(baseline / "requests.csv.gz", "rt", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            run_id = int(row["run_id"])
            old_queries.setdefault(run_id, []).append(old_rows[(run_id, int(row["row_id"]))])
    checked = 0
    for r in runs:
        if r["method"] != "mcts" or not r["task_id"].endswith("actual_capacity"):
            continue
        old = old_by_key[key(r)]
        if r["selected_signature"] != old["selected_signature"] or \
           r["metrics"]["selected_cost_s"] != old["metrics"]["selected_cost_s"]:
            raise ValueError("original-capacity winner changed")
        by_id = {x["row_id"]: x["plan_signature"] for x in observed[r["run_id"]]}
        if [by_id[i] for i in queries[r["run_id"]]] != old_queries[old["run_id"]]:
            raise ValueError("original-capacity score request trace changed")
        checked += 1
    if checked != 108:
        raise ValueError("missing original-capacity regression run")
    return dict(performed=True, matched_original_capacity_runs=checked,
                winner_and_cost_identical=True, ordered_score_requests_identical=True,
                frozen_input_files_identical=2*len(plan["tasks"]))


def verify(output):
    output = Path(output)
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    check_inputs(output, plan)
    runs = [json.loads(s) for s in (output / "runs.jsonl").read_text().splitlines()]
    report = json.loads((output / "report.json").read_text())
    if len(runs) != plan["expected_exact_runs"] + plan["expected_mcts_runs"]:
        raise ValueError("wrong run count")
    if not all(r["method"] == "exact" for r in runs[:plan["expected_exact_runs"]]):
        raise ValueError("Exact references not run first")
    rows = {r["run_id"]: [] for r in runs}; requests = {r["run_id"]: [] for r in runs}
    with gzip.open(output / "candidates.csv.gz", "rt", encoding="utf-8") as f:
        for x in csv.DictReader(f):
            row = dict(x)
            for field in ("row_id", "fused_groups"):
                row[field] = int(row[field])
            for field in ("makespan_s", "cross_device_bytes"):
                row[field] = float(row[field])
            row["memory_feasible"] = row["memory_feasible"] == "True"
            row["peak_memory_bytes"] = json.loads(row["peak_memory_bytes"])
            rows[int(row.pop("run_id"))].append(row)
    with gzip.open(output / "requests.csv.gz", "rt", encoding="utf-8") as f:
        for x in csv.DictReader(f):
            qq = requests[int(x["run_id"])]
            if int(x["request"]) != len(qq)+1:
                raise ValueError("request order changed")
            qq.append(int(x["row_id"]))
    tables = {}
    tasks = {t["task_id"]: t for t in plan["tasks"]}
    for r in runs:
        rr, qq = rows[r["run_id"]], requests[r["run_id"]]
        if r["method"] == "exact" and r["mode"] == "joint":
            tables[r["task_id"]] = {x["plan_signature"]: (x["makespan_s"], x["memory_feasible"]) for x in rr}
        recomputed = metrics(rr, qq, final_requests(r["budget"], qq), r["selected_signature"])
        if recomputed != r["metrics"]:
            raise ValueError("observation metrics mismatch")
        for x in rr:
            feasible = all(p <= c for p, c in zip(x["peak_memory_bytes"], tasks[r["task_id"]]["max_memory_bytes"]))
            if feasible != x["memory_feasible"]:
                raise ValueError("explicit capacity mismatch")
        mm = r["metrics"]
        if r["selected_signature"] is None:
            expected = "no_feasible_in_space" if r["method"] == "exact" else "evaluated_no_feasible"
            if r["selected_plan"] is not None or r["reason"] != expected or mm["eligible_feasible_plans"]:
                raise ValueError("invalid no-plan reason")
        elif not mm["selected_feasible"] or mm["selected_cost_s"] != mm["best_feasible_scored_cost_s"] or r["reason"]:
            raise ValueError("did not return best scored feasible candidate")
        if len(rr) != r["budget"]["cost_model_evaluations"] or len(qq) != r["budget"]["scorer_requests"]:
            raise ValueError("budget mismatch")
    refs = json.loads((output / "references.json").read_text())
    for task in plan["tasks"]:
        for mode, ref in refs[task["task_id"]].items():
            r = runs[ref["run_id"]]
            if ref["cost_s"] != r["metrics"]["best_feasible_scored_cost_s"]:
                raise ValueError("reference cost mismatch")
    for r in runs:
        if r["method"] == "mcts" and any(tables[r["task_id"]].get(x["plan_signature"]) !=
            (x["makespan_s"], x["memory_feasible"]) for x in rows[r["run_id"]]):
            raise ValueError("MCTS candidate differs from Joint Exact")
    with (output / "results.csv").open(encoding="utf-8") as f:
        results = list(csv.DictReader(f))
    if len(results) != plan["expected_mcts_runs"]:
        raise ValueError("missing MCTS result")
    for row in results:
        r = runs[int(row["run_id"])]; mm = r["metrics"]
        selected = mm["selected_cost_s"] if mm["selected_feasible"] else None
        ref = refs[r["task_id"]][r["mode"]]["cost_s"]
        checks = dict(selected_mode_reference_gap=gap(selected, ref),
            selected_joint_exact_gap=gap(selected, refs[r["task_id"]]["joint"]["cost_s"]),
            diagnostic_scored_mode_reference_gap=gap(mm["best_feasible_scored_cost_s"], ref))
        for field, value in checks.items():
            if (row[field] != "" if value is None else abs(float(row[field])-value) > 1e-12):
                raise ValueError("reference gap mismatch")
    artifacts = json.loads((output / "artifacts.json").read_text())
    for signature, path in artifacts.items():
        placed = load_placed_op_graph(output / path)
        expected = signature.split(":", 1)[1]
        if digest(_plan_key(placed.plan)) != expected or placed.extra["deployable"] is not False:
            raise ValueError("invalid output artifact")
    if sum(r["metrics"]["selected_feasible"] for r in runs if r["method"] == "mcts") != report["selected_feasible_runs"]:
        raise ValueError("selected feasibility count mismatch")
    result = dict(pass_checks=True, runs=len(runs), exact_runs=report["exact_runs"],
        mcts_runs=report["mcts_runs"], scored_unique_rows=sum(map(len, rows.values())),
        score_requests=sum(map(len, requests.values())), artifacts=len(artifacts),
        memory_feasibility_kept_separate=True, source_and_inputs_unchanged=True,
        original_capacity_regression=verify_regression(plan, runs, rows, requests))
    save(output / "verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "verify"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = {"prepare": prepare, "run": run, "verify": verify}[args.phase](args.directory)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
