"""Seal the small A3 development experiment before new plan measurements."""

import argparse
import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import random

from calibration.e2_manifest import content_id
from calibration.e2_records import load_jsonl_records
from calibration.gpu_holdout import ROOT, SOURCES, INPUTS, file_hashes
from calibration.reference_validation import check as check_reference, DEFAULT_BUNDLE as REFERENCE
from cost_model.decoder_candidate_coverage import structured_candidates, expand_directions, branch_groups
from cost_model.placement_probes import graph_wide_candidates
from cost_model.e2_residual import NearestShapePredictor
from cost_model.e2_srop_proxy import DatasetLatencyProfile, E2Workload, object_map_for_workload
from cost_model.reference_semantics import HostSubmissionProfile, ReferenceEagerCostModel
from cost_model.search_evaluation import qwen3_spec, qwen25_3b_spec
from cost_model.search_sensitivity import plan_id, permute_plan_devices
from execution.audit_gpu_holdout import write_json
from simulator import make_reference_plan, validate_fusion_plan
from simulator.dataset import graph_to_dict, graph_from_dict, plan_to_dict, plan_from_dict
from simulator.hardware import hardware_from_dict
from simulator.model_graphs.decoder_stack import build_decoder_stack


DEFAULT_BUNDLE = ROOT / "data/a3_decoder_competition_v1"


def select_structures(graph, pool):
    available = {plan_id(p): p for p in pool}
    selected, reasons = {}, {}
    reference = make_reference_plan(graph)

    def add(nodes, reason):
        plan = copy.deepcopy(reference)
        for group in plan.groups:
            group.device = int(group.node_ids[0] in nodes)
        for direction in ((0, 1), (1, 0)):
            physical = permute_plan_devices(plan, direction)
            key = plan_id(physical)
            if key not in available:
                raise ValueError("A3 structural witness is absent from A2 pool")
            selected[key] = available[key]
            reasons.setdefault(key, []).append(reason)

    add(set(), "uniform-control")
    branches = branch_groups(graph)
    anchors = sorted({min(branches), max(branches)})
    for layer in anchors:
        for family, masks in (("qkv", ("100", "110", "111")), ("gate_up", ("01", "10", "11"))):
            chains = branches[layer][family]
            for mask in masks:
                nodes = {key for bit, chain in zip(mask, chains) if bit == "1" for key in chain}
                add(nodes, f"local:{layer}:{family}:{mask}")
    for layer in sorted(branches)[1:]:
        add({key for key, n in graph.nodes.items() if n.extra["layer"] >= layer}, f"layer-boundary:{layer}")
    return selected, reasons


def build_schedule(workloads, protocol):
    slots = []
    for index, task in enumerate(workloads):
        anchors = task["uniform_plan_ids"]
        body = sorted(set(task["plans"]) - set(anchors))
        for repeat in range(protocol["repeats"]):
            before = anchors if repeat % 2 == 0 else list(reversed(anchors))
            order = body.copy()
            random.Random(protocol["order_seed"] + 100 * index + repeat).shuffle(order)
            entries = [("before", key) for key in before] + [("candidate", key) for key in order] + [
                ("after", key) for key in reversed(before)]
            for position, (phase, key) in enumerate(entries):
                slots.append({"slot_id": f"{index}:{repeat}:{position}", "task_id": task["task_id"],
                              "repeat": repeat, "phase": phase, "plan_id": key})
    return slots


def build_freeze():
    parent, _ = check_reference(REFERENCE)
    hardware = hardware_from_dict(parent["hardware"])
    train = [{**r, "label_access": {"visible": True}} for r in load_jsonl_records(ROOT / INPUTS[0])]
    contract = json.loads((ROOT / INPUTS[2]).read_text())
    predictor = NearestShapePredictor().fit_mean(train, contract["prediction_protocol"])
    by_id = {r["object_id"]: r for r in train}
    host = HostSubmissionProfile(**parent["host_profile"])
    workloads = []
    for family, spec in (("qwen3-0.6b", qwen3_spec(2)), ("qwen2.5-3b", qwen25_3b_spec())):
        for layers in (2, 4):
            graph = build_decoder_stack(replace(spec, num_layers=layers))
            base, _ = graph_wide_candidates(graph, [make_reference_plan(graph)], ("bw", "bw"))
            pool = structured_candidates(graph, expand_directions(base))
            plans, reasons = select_structures(graph, pool)
            mapping = object_map_for_workload(train, E2Workload("A3", "exposed-development",
                f"{family}-s256-b1", family, 256))
            keys = {mapping[n.extra["local_name"]] for n in graph.nodes.values() if n.op_type != "input"}
            durations = {key: by_id[key]["analytic"]["estimate_s"] * math.exp(predictor._nearest(by_id[key])) for key in keys}
            profile = DatasetLatencyProfile(hardware, mapping, durations, {key: "original112_nearest" for key in keys})
            scores = {key: ReferenceEagerCostModel(host).predict(graph, p, profile).makespan for key, p in plans.items()}
            ties = sorted(key for key, value in scores.items() if value <= min(scores.values()) + 1e-12)
            uniform = [plan_id(make_reference_plan(graph, device=device)) for device in (0, 1)]
            workloads.append({"task_id": f"{family}-decoder{layers}-s256-b1", "family": family, "layers": layers,
                "graph": graph_to_dict(graph), "plans": {key: plan_to_dict(p) for key, p in sorted(plans.items())},
                "selection_reasons": reasons, "a2_pool_physical_candidates": len(pool),
                "uniform_plan_ids": uniform, "initial_scores_s": scores, "initial_predicted_ties": ties,
                "initial_selected_id": ties[0], "kernel_durations_s": durations,
                "kernel_scope": "All operator Keys already occur in the original112 training records; not unseen-shape validation."})
    protocol = {"devices": [0, 1], "dtype": "bf16", "seed": 1234, "order_seed": 20261029,
        "warmup": 5, "iterations": 20, "repeats": 3, "rtol": .01, "atol": .001,
        "primary_epsilon": .01, "epsilon_values": [.001, .005, .01], "anchor_drift_limit": .10,
        "noise_relative_floor": .001, "noise_mad_multiplier": 3.,
        "timing": "run_once(False) synchronized forward wall; no per-group probes, constructor/cache creation outside timed forward",
        "aggregation": "median of three window medians; uniform control within-window is median(before,after) block medians",
        "drift_policy": "Keep every sample/window in primary results; flag >10% before-after anchor drift; no rerun-to-pass or deletion. Repeat-consistent evidence requires all windows pass drift.",
        "failure_policy": "Persist failed slot and abort; never exclude failed candidate or overwrite its slot. Incomplete runs have no regret conclusion.",
        "lifecycle": "Rebuild executor each block; reuse canonical CPU templates within a task; no empty_cache between blocks",
        "runtime": {"torch": "2.9.0", "hip": "6.3.26093", "device_name_contains": "BW"},
        "resume": "Skip already recorded slots on identical environment; a window spanning collection processes is flagged and cannot establish repeat-consistent evidence.",
        "selection": "A2 members only: both uniform directions; local QKV100/110/111 and gate-up01/10/11 on first/last layers; all whole-layer boundaries; expand both physical directions",
        "scope": "Development plan-competition diagnosis, not independent SROP benefit or training throughput. No new kernel profiling/refit/re-search."}
    sources = tuple(dict.fromkeys((*SOURCES, *parent["source_sha256"],
        "calibration/decoder_competition.py", "execution/decoder_competition.py", "execution/decoder_reference.py",
        "simulator/model_graphs/decoder_stack.py", "cost_model/decoder_candidate_coverage.py")))
    inputs = (*INPUTS, "data/reference_validation_v1/freeze.json", "data/a2_decoder_candidate_coverage.v1.json")
    payload = {"format": "fusion-placement-search.a3-decoder-competition-freeze", "version": "1.0",
        "parent_freeze_id": parent["freeze_id"], "source_sha256": file_hashes(sources), "input_sha256": file_hashes(inputs),
        "hardware": parent["hardware"], "host_profile": parent["host_profile"], "training_labels": len(train),
        "protocol": protocol, "workloads": workloads, "schedule": build_schedule(workloads, protocol)}
    return {**payload, "freeze_id": content_id("a3-decoder-competition", payload)}


def check(bundle=DEFAULT_BUNDLE):
    frozen = json.loads((bundle / "freeze.json").read_text(encoding="utf-8"))
    payload = {k: v for k, v in frozen.items() if k != "freeze_id"}
    if frozen["freeze_id"] != content_id("a3-decoder-competition", payload):
        raise ValueError("A3 freeze content hash mismatch")
    for key in ("source_sha256", "input_sha256"):
        if file_hashes(frozen[key]) != frozen[key]:
            raise ValueError(f"A3 {key} differs from sealed files")
    if frozen["schedule"] != build_schedule(frozen["workloads"], frozen["protocol"]):
        raise ValueError("A3 schedule differs from sealed recipe")
    for task in frozen["workloads"]:
        graph = graph_from_dict(task["graph"])
        for key, raw in task["plans"].items():
            plan = plan_from_dict(raw)
            if plan_id(plan) != key:
                raise ValueError("A3 plan ID mismatch")
            validate_fusion_plan(graph, plan, 2, require_executable=True)
        scores = task["initial_scores_s"]
        if set(scores) != set(task["plans"]):
            raise ValueError("initial scores differ from selected candidate set")
        ties = sorted(k for k, value in scores.items() if value <= min(scores.values()) + 1e-12)
        if task["initial_predicted_ties"] != ties or task["initial_selected_id"] != min(ties):
            raise ValueError("initial physical tie choice changed")
        if task["uniform_plan_ids"] != [plan_id(make_reference_plan(graph, device=d)) for d in (0, 1)]:
            raise ValueError("uniform control IDs changed")
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "check"))
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args()
    if args.action == "prepare":
        frozen = build_freeze()
        args.bundle.mkdir(parents=True, exist_ok=False)
        write_json(args.bundle / "freeze.json", frozen)
    frozen = check(args.bundle)
    print("freeze_id:", frozen["freeze_id"])
    print("tasks:", len(frozen["workloads"]), "physical_candidates:", sum(len(t["plans"]) for t in frozen["workloads"]),
          "blocks:", len(frozen["schedule"]), "formal_forwards:", len(frozen["schedule"]) * frozen["protocol"]["iterations"])


if __name__ == "__main__":
    main()
