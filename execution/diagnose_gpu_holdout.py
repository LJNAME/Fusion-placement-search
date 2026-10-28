"""Collect three-plan GPU baselines and separate annotated execution traces."""

import argparse
from collections import Counter
from dataclasses import asdict
import gc
import json
import math
from pathlib import Path
import platform
import socket
import time

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from calibration.e2_records import append_jsonl_record
from calibration.gpu_holdout import DEFAULT_BUNDLE, load_freeze
from calibration.profile_e2_bw1000 import _validate_runtime
from cost_model.e2_srop_proxy import DatasetLatencyProfile, E2Workload, object_map_for_workload
from cost_model.structural import StructuralCostModel
from execution.audit_gpu_holdout import timing_statistics, write_json
from execution.gpu_holdout import CanonicalReferenceExecutor, compare_outputs, measure_candidate
from simulator import PlacedOpGraph
from simulator.dataset import graph_from_dict, plan_from_dict
from simulator.hardware import hardware_from_dict


DEFAULT_AUDIT = DEFAULT_BUNDLE / "results_received_20261028/execution_log_audit.json"


class CopyRecorder(TorchDispatchMode):
    """Observe actual to-copy calls during a diagnostic forward, without waits."""

    def __init__(self):
        super().__init__()
        self.rows = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is not torch.ops.aten._to_copy.default:
            return func(*args, **kwargs)
        source = args[0]
        target = torch.device(kwargs.get("device") or source.device)
        label = f"fusion_copy::{source.device}->{target}"
        with torch.profiler.record_function(label):
            started = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - started
        self.rows.append({"source": str(source.device), "destination": str(result.device),
            "source_dtype": str(source.dtype), "destination_dtype": str(result.dtype),
            "shape": list(source.shape), "source_stride": list(source.stride()),
            "source_payload_bytes": source.numel() * source.element_size(),
            "destination_payload_bytes": result.numel() * result.element_size(),
            "cross_gpu": source.device.type == result.device.type == "cuda" and source.device != result.device,
            "cpu_call_ms": elapsed * 1000})
        return result


class NodeTraceExecutor(CanonicalReferenceExecutor):
    """Use the existing executor and annotate nodes; leave dispatch order intact."""

    def __init__(self, *args, **kwargs):
        self.node_calls = []
        super().__init__(*args, **kwargs)

    def _execute_node(self, node_id, inputs, device):
        with torch.profiler.record_function(f"fusion_node::{node_id}::{device}"):
            started = time.perf_counter()
            result = super()._execute_node(node_id, inputs, device)
            elapsed = time.perf_counter() - started
        self.node_calls.append({"node_id": node_id, "device": str(device),
            "input_shapes": [list(value.shape) for value in inputs],
            "input_strides": [list(value.stride()) for value in inputs],
            "output_shape": list(result.shape), "output_stride": list(result.stride()),
            "cpu_scope_ms": elapsed * 1000})
        return result


def expected_transfers(placed):
    """List copies issued by the reference predecessor loop, including repeats."""
    graph, plan = placed.graph, placed.plan
    devices = {node: group.device for group in plan.groups for node in group.node_ids}
    rows = []
    for group in plan.groups:
        for destination in group.node_ids:
            for source in graph.predecessors(destination):
                if devices[source] == group.device:
                    continue
                node = graph.nodes[source]
                shape = node.extra["output_shape"]
                element_size = {"bf16": 2, "fp16": 2, "fp32": 4}[node.extra["dtype"]]
                rows.append({"source_node": source, "destination_node": destination,
                    "source": f"cuda:{devices[source]}", "destination": f"cuda:{group.device}",
                    "payload_bytes": math.prod(shape) * element_size,
                    "shape": shape, "logical_output_bytes": node.out_bytes})
    return rows


def transfer_signature(rows, *, actual):
    key = "source_payload_bytes" if actual else "payload_bytes"
    return Counter((row["source"], row["destination"], row[key]) for row in rows)


def prediction_breakdown(freeze, selection):
    workload = next(row for row in freeze["workloads"] if row["model_id"] == selection["model_id"])
    graph = graph_from_dict(workload["graph"])
    hardware = hardware_from_dict(freeze["hardware"])
    mapping = object_map_for_workload(freeze["query_features"], E2Workload(workload["model_id"],
        "post-hoc-diagnosis", workload["model_id"], workload["model_family"], workload["sequence_length"]))
    profile = DatasetLatencyProfile(hardware, mapping,
        {key: row["nearest_s"] for key, row in freeze["zero_shot_predictions"].items()},
        {key: "frozen_zero_shot" for key in freeze["zero_shot_predictions"]})
    sealed = {row["plan_id"]: row["makespan_s"] for row in workload["predicted_ranking"]}
    model = StructuralCostModel()
    rows = []
    for item in selection["plans"]:
        plan = plan_from_dict(workload["candidate_plans"][item["plan_id"]])
        prepared = model.prepare(graph, plan, 2)
        estimate = model.predict_prepared(prepared, profile)
        if not math.isclose(estimate.makespan, sealed[item["plan_id"]], rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("structural prediction no longer reproduces the sealed score")
        details = asdict(estimate)
        for group, structure in zip(details["details"]["groups"], prepared.structure.groups):
            group["node_ids"] = list(structure.node_ids)
        rows.append({**item, "frozen_prediction": details,
            "reference_loop_expected_transfers": expected_transfers(PlacedOpGraph(graph, hardware, plan))})
    return {"freeze_id": freeze["freeze_id"], "model_id": selection["model_id"], "plans": rows,
        "scope": "Original frozen costs and scheduler, reproduced without new labels; selected workloads/plans are post-hoc diagnostics. Scheduled intervals are predictions, not observed GPU activity."}


def capture_trace(placed, devices, parameters, templates, reference, path):
    executor = NodeTraceExecutor(placed, devices, parameters["seed"], templates)
    with torch.no_grad():
        for _ in range(parameters["warmup"]):
            output, _, _, _ = executor.run_once(False)
            del output
        executor.node_calls.clear()
        copies = CopyRecorder()
        supported = torch.profiler.supported_activities()
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.profiler.ProfilerActivity.CUDA in supported:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=True) as profiler:
            with copies, torch.profiler.record_function("fusion_diagnostic_forward"):
                output, _, _, traced_total = executor.run_once(False)
        accuracy = compare_outputs(output, reference, rtol=parameters["rtol"], atol=parameters["atol"])
        del output
    profiler.export_chrome_trace(str(path))
    events = profiler.events()
    gpu_events = [event for event in events if getattr(event.device_type, "name", "") in ("CUDA", "HIP")]
    important = ("fusion_", "matmul", "mm", "repeat_interleave", "permute", "clone", "contiguous",
        "copy", "synchroniz", "Synchroniz", "StreamWait", "EventRecord")
    operators = [{"name": row.key, "calls": row.count,
        "cpu_total_us": row.cpu_time_total, "self_cpu_total_us": row.self_cpu_time_total,
        "device_total_us": row.device_time_total}
        for row in profiler.key_averages() if any(part in row.key for part in important)]
    actual = [row for row in copies.rows if row["cross_gpu"]]
    expected = expected_transfers(placed)
    return {"trace_file": path.name, "gpu_activity_events": len(gpu_events),
        "gpu_timeline_available": bool(gpu_events), "correctness": accuracy,
        "traced_forward_ms_diagnostic_only": traced_total * 1000,
        "cross_gpu_copy_calls": len(actual),
        "cross_gpu_payload_bytes": sum(row["source_payload_bytes"] for row in actual),
        "copy_signature_matches_reference_loop": transfer_signature(actual, actual=True) == transfer_signature(expected, actual=False),
        "expected_transfers": expected, "observed_to_copy_calls": copies.rows,
        "node_cpu_scopes": executor.node_calls, "profiler_operator_totals": operators,
        "interpretation": "Traced timings include profiler/dispatch overhead. CPU scopes can include enqueue and blocking waits; nested CPU/device totals overlap and must not be summed. Use the separate uninstrumented baseline for performance. Use GPU timeline events, when available, to assess overlap."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    freeze, manifest = load_freeze(args.bundle)
    _validate_runtime(manifest, (0, 1))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if audit["freeze_id"] != freeze["freeze_id"] or not all(audit["checks"].values()):
        raise ValueError("diagnosis requires the matching successful raw-log audit")
    selection = audit["representatives"]
    workload = next(row for row in freeze["workloads"] if row["model_id"] == selection["model_id"])
    chosen = selection["plans"]
    if len(chosen) != 3 or len({row["plan_id"] for row in chosen}) != 3 or any(row["plan_id"] not in workload["execution_plan_ids"] for row in chosen):
        raise ValueError("expected three distinct previously measured candidates")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError("use a new diagnostic output directory; do not overwrite prior runs")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    predictions = prediction_breakdown(freeze, selection)
    write_json(args.out_dir / "prediction_breakdown.json", predictions)
    hardware = hardware_from_dict(freeze["hardware"])
    graph = graph_from_dict(workload["graph"])
    placed = {row["role"]: PlacedOpGraph(graph, hardware, plan_from_dict(workload["candidate_plans"][row["plan_id"]])) for row in chosen}
    parameters = freeze["protocol"]["execution"]
    devices, templates = ("cuda:0", "cuda:1"), {}
    environment = {"host": socket.gethostname(), "platform": platform.platform(),
        "torch": torch.__version__, "hip": torch.version.hip,
        "devices": [{"index": index, "name": torch.cuda.get_device_properties(index).name,
            "total_memory": torch.cuda.get_device_properties(index).total_memory} for index in (0, 1)],
        "peer_access": {f"{a}->{b}": torch.cuda.can_device_access_peer(a, b) for a, b in ((0, 1), (1, 0))}}
    write_json(args.out_dir / "diagnostic_environment.json", environment)
    with torch.no_grad():
        reference_executor = CanonicalReferenceExecutor(placed["single_gpu_reference"], devices, parameters["seed"], templates)
        output, _, _, _ = reference_executor.run_once(False)
        reference = output.float().cpu()
        del output, reference_executor
    gc.collect()
    records = []
    for repeat in range(3):
        # Three cyclic orders give each role one early, middle and late block.
        order = chosen[repeat:] + chosen[:repeat]
        for item in order:
            row = {"role": item["role"], "plan_id": item["plan_id"], "repeat": repeat}
            row.update(measure_candidate(placed[item["role"]], devices, parameters, templates, reference))
            append_jsonl_record(args.out_dir / "baseline_samples.jsonl", row)
            records.append(row)
            print(f"baseline repeat={repeat} role={item['role']} ms={row['median_s']*1000:.6f}", flush=True)
            if not row["correctness"]["pass"]:
                raise RuntimeError("diagnostic baseline output mismatch; keep recorded samples")
            gc.collect()
    baselines = [{"role": item["role"], "plan_id": item["plan_id"], "old_measured_ms": item["old_measured_ms"],
        "pooled": timing_statistics([sample for row in records if row["role"] == item["role"] for sample in row["samples_s"]]),
        "rounds": [{"repeat": row["repeat"], **timing_statistics(row["samples_s"])} for row in records if row["role"] == item["role"]]}
        for item in chosen]
    report = {"format": "fusion-placement-search.gpu-three-plan-diagnosis", "freeze_id": freeze["freeze_id"],
        "model_id": selection["model_id"], "selection": selection, "environment": environment,
        "scope": "Post-hoc development diagnosis, no model or scheduler modification and no independent validation claim.",
        "baseline_protocol": {"repeats": 3, "warmup": parameters["warmup"], "iterations": parameters["iterations"], "order": "cyclic rotation"},
        "predictions": predictions, "baselines": baselines, "traces": [], "status": "baselines_complete"}
    write_json(args.out_dir / "diagnostic_report.json", report)
    for item in chosen:
        role = item["role"]
        try:
            trace = capture_trace(placed[role], devices, parameters, templates, reference, args.out_dir / f"{role}.trace.json")
            trace.update(role=role, plan_id=item["plan_id"])
        except Exception as exc:
            trace = {"role": role, "plan_id": item["plan_id"], "error_type": type(exc).__name__, "error": str(exc), "gpu_timeline_available": False}
        report["traces"].append(trace)
        write_json(args.out_dir / "diagnostic_report.json", report)
        print(f"trace role={role} gpu_timeline_available={trace['gpu_timeline_available']}", flush=True)
    report["status"] = "complete" if all(row["gpu_timeline_available"] for row in report["traces"]) else "baselines_complete_traces_partial"
    write_json(args.out_dir / "diagnostic_report.json", report)
    if any(not row.get("correctness", {}).get("pass", False) for row in report["traces"]):
        raise SystemExit("a diagnostic trace failed or produced incorrect output; baselines were retained")
    print(f"report: {args.out_dir / 'diagnostic_report.json'}; status={report['status']}")


if __name__ == "__main__":
    main()
