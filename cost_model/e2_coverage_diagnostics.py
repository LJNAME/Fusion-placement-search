"""Expand the frozen E2 pools structurally, then repeat the same diagnostics."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

from calibration.e2_manifest import content_id
from calibration.e2_records import load_jsonl_records
from simulator.dataset import plan_from_dict, plan_to_dict
from simulator.hardware import hardware_to_dict, load_hardware

from .e2_diagnostics import (
    FORMAT, VERSION, _read_checkpoint, _write_checkpoint, _write_json,
    build_workload_graph, evaluate_one_step, evaluate_zero_budget, summarize,
    workload_for_task,
)
from .e2_task_suite import build_task_suite
from .placement_probes import COVERAGE_VERSION, graph_wide_candidates


_CONTEXT = None


def _initialize(context):
    global _CONTEXT
    _CONTEXT = context


def _run_task(task, phase):
    manifest, contract, records, hardware, base_root, checkpoint_root = _CONTEXT
    path = checkpoint_root / (task["task_id"].split(":")[1] + ".json.gz")
    if phase == "one-step":
        result = _read_checkpoint(path)
        plans = [plan_from_dict(plan) for plan in result["candidate_plans"]]
        result["one_step"] = evaluate_one_step(task, plans, records, manifest, contract, hardware)
        return result
    stored = _read_checkpoint(base_root / (task["task_id"].split(":")[1] + ".json.gz"))
    if stored is None:
        raise ValueError("base candidate checkpoint missing; replay the original frozen diagnostics first")
    started = time.perf_counter()
    graph = build_workload_graph(workload_for_task(task, manifest), task["scale"])
    classes = tuple(f"model:{device.name.strip().lower()}" for device in hardware.devices)
    plans, metadata = graph_wide_candidates(graph,
        [plan_from_dict(plan) for plan in stored["candidate_plans"]], classes)
    return {
        "task_id": task["task_id"], "candidate_plans": [plan_to_dict(plan) for plan in plans],
        "candidate_generation_wall_time_s": time.perf_counter() - started,
        "candidate_pool": metadata,
        "zero": evaluate_zero_budget(task, plans, records, manifest, contract, hardware, graph=graph),
    }


def comparison_summary(results, base_report):
    old = {item["task_id"]: item["zero"] for item in base_report["tasks"]}
    zeros = [result["zero"] for result in results]
    return {
        "base_experiment_id": base_report["experiment_id"],
        "all_base_candidates_preserved": all(set(old[item["task_id"]]["candidate_ids"])
                                               <= set(item["candidate_ids"]) for item in zeros),
        "oracle_reference_improved_configs": sum(item["oracle_best_ms"] < old[item["task_id"]]["oracle_best_ms"] - 1e-9
                                                  for item in zeros),
        "new_positive_initial_regret_configs": sum(item["positive_initial_regret"] and not old[item["task_id"]]["positive_initial_regret"]
                                                    for item in zeros),
        "candidate_count_min": min((item["candidate_plans"] for item in zeros), default=0),
        "candidate_count_max": max((item["candidate_plans"] for item in zeros), default=0),
    }


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    directory = root / "data/bw1000_e2"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-report", type=Path, default=directory / "e2c_frozen_diagnostics.json")
    parser.add_argument("--checkpoint-dir", type=Path, default=directory / "diagnostics_checkpoints")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("zero", "one-step", "all"), default="all")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("workers and limit must be positive")
    base = json.loads(args.base_report.read_text(encoding="utf-8"))
    manifest = json.loads((directory / "shape_manifest.json").read_text(encoding="utf-8"))
    contract = json.loads((directory / "split_contract.json").read_text(encoding="utf-8"))
    suite = json.loads((directory / "e2c_task_suite.json").read_text(encoding="utf-8"))
    records = load_jsonl_records(directory / "profiling_dataset.jsonl")
    hardware = load_hardware(root / "data/bw1000_2card/bw1000_2card.calibrated.json")
    if suite != build_task_suite(manifest, contract):
        raise ValueError("frozen suite differs from its label-independent construction")
    if (base["configuration"]["suite_id"] != suite["suite_id"]
            or base["configuration"]["dataset_id"] != content_id("diagnostics-data", records)
            or base["configuration"]["hardware"] != hardware_to_dict(hardware)
            or {item["task_id"] for item in base["tasks"]} != {task["task_id"] for task in suite["tasks"]}
            or base["execution_status"] != "complete"):
        raise ValueError("base report must contain the same complete frozen tasks, dataset and hardware")
    source_names = ("e2_coverage_diagnostics.py", "placement_probes.py", "e2_diagnostics.py",
                    "e2_srop_proxy.py", "e2_audit.py", "e2_task_suite.py", "e2_residual.py",
                    "structure.py", "structural.py", "search_sensitivity.py")
    config = dict(base["configuration"])
    config.update({
        "source_sha256": hashlib.sha256(b"".join((Path(__file__).parent / name).read_bytes() for name in source_names)).hexdigest(),
        "candidate_coverage": COVERAGE_VERSION,
        "base_experiment_id": base["experiment_id"],
        "base_report_id": content_id("frozen-base-report", base),
        "candidate_scope": "original predicted+oracle union plus label-independent structural probes; development diagnostic only",
    })
    experiment_id = content_id("e2-coverage-experiment", config)
    checkpoint_root = args.checkpoint_dir / experiment_id.split(":")[1][:16]
    base_root = args.checkpoint_dir / base["experiment_id"].split(":")[1][:16]
    tasks = suite["tasks"] if args.limit is None else suite["tasks"][:args.limit]
    results = {}
    base_zeros = {item["task_id"]: item["zero"] for item in base["tasks"]}
    for task in tasks:
        filename = task["task_id"].split(":")[1] + ".json.gz"
        original = _read_checkpoint(base_root / filename)
        if original is None or original["zero"]["candidate_ids"] != base_zeros[task["task_id"]]["candidate_ids"]:
            raise ValueError("base checkpoint missing or mismatched; replay the original diagnostics first")
        stored = _read_checkpoint(checkpoint_root / filename)
        if stored is not None:
            results[task["task_id"]] = stored

    def save_report():
        published = [{key: value for key, value in results[task["task_id"]].items() if key != "candidate_plans"}
                     for task in tasks if task["task_id"] in results]
        summary = summarize(published, len(suite["tasks"]))
        complete = summary["completed_zero_configs"] == len(tasks) and (
            args.phase == "zero" or summary["completed_one_step_configs"] == len(tasks))
        _write_json(args.output, {"format": FORMAT, "version": VERSION,
            "experiment_id": experiment_id, "configuration": config,
            "scope": "all_frozen_configs" if len(tasks) == len(suite["tasks"]) else "partial_smoke_test",
            "execution_status": "complete" if complete else "running", "requested_phase": args.phase,
            "summary": summary, "comparison": comparison_summary(published, base), "tasks": published})

    context = (manifest, contract, records, hardware, base_root, checkpoint_root)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_initialize, initargs=(context,)) as executor:
        for phase in ("zero", "one-step") if args.phase != "zero" else ("zero",):
            pending = [task for task in tasks if task["task_id"] not in results or
                       (phase == "one-step" and "one_step" not in results[task["task_id"]])]
            futures = {executor.submit(_run_task, task, phase): task for task in pending}
            for future in as_completed(futures):
                task = futures[future]
                result = future.result()
                results[task["task_id"]] = result
                _write_checkpoint(checkpoint_root / (task["task_id"].split(":")[1] + ".json.gz"), result)
                save_report()
                if phase == "zero":
                    detail = f"candidates={result['zero']['candidate_plans']} regret_ms={result['zero']['top1_regret_ms']:.9f}"
                else:
                    detail = f"improving={result['one_step']['improving_objects']} best_gain_ms={result['one_step']['best_single_step_gain_ms']:.9f}"
                print(f"{phase} {task['model_id']} {task['scale']} train={task['initial_train_count']} {detail}", flush=True)
            save_report()
    print(json.dumps(summarize(list(results.values()), len(suite["tasks"])), indent=2), flush=True)
    print(f"report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
