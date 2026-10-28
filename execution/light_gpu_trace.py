"""Trace critical branches without dispatch interception or tensor metadata."""

from collections import defaultdict
import json
import statistics

import torch

from execution.gpu_holdout import CanonicalReferenceExecutor, compare_outputs


WATCHED_NODES = frozenset({"L000.mlp.gate_proj", "L000.mlp.up_proj", "L000.mlp.silu",
    "L000.self_attention.k_rope", "L000.self_attention.q_proj"})
TIMING_DEVIATION_TARGET = .15
BASELINE_SHIFT_TARGET = .10


class BranchTraceExecutor(CanonicalReferenceExecutor):
    def _execute_node(self, node_id, inputs, device):
        if node_id not in WATCHED_NODES:
            return super()._execute_node(node_id, inputs, device)
        with torch.profiler.record_function(f"fusion_node::{node_id}::{device}"):
            return super()._execute_node(node_id, inputs, device)


def merge_intervals(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def overlap_ms(first, second):
    return sum(max(0.0, min(b, d) - max(a, c))
        for a, b in merge_intervals(first) for c, d in merge_intervals(second))


def analyze_trace_frames(events, prefix="fusion_light_forward::"):
    """Associate physical GPU intervals by CPU external IDs, never flow arrows."""
    cpu_events = [event for event in events if event.get("ph") == "X" and event.get("cat") == "cpu_op"]
    scopes = [event for event in events if event.get("ph") == "X" and event.get("cat") == "user_annotation"]
    roots = sorted((event for event in scopes if event["name"].startswith(prefix)), key=lambda event: event["ts"])
    nodes = [event for event in scopes if event["name"].startswith("fusion_node::")]
    by_external = defaultdict(list)
    for event in cpu_events:
        external = event.get("args", {}).get("External id")
        if external is not None:
            by_external[external].append(event)

    def contains(outer, inner):
        return (outer["pid"] == inner["pid"] and outer["tid"] == inner["tid"]
            and outer["ts"] <= inner["ts"] and outer["ts"] + outer["dur"] >= inner["ts"] + inner["dur"] - 1)

    assigned = [[] for _ in roots]
    unassigned = []
    for index, event in enumerate(events):
        if event.get("ph") != "X" or event.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        anchors = by_external[event.get("args", {}).get("External id")]
        root_indices = [] if len(anchors) != 1 else [i for i, root in enumerate(roots) if contains(root, anchors[0])]
        if len(root_indices) != 1:
            unassigned.append(index)
            continue
        frame = root_indices[0]
        parent = [node for node in nodes if contains(roots[frame], node) and contains(node, anchors[0])]
        start = (event["ts"] - roots[frame]["ts"]) / 1000
        assigned[frame].append({"raw_event_index": index, "kind": event["cat"],
            "device": event["args"]["device"], "stream": event["args"].get("stream"),
            "node_id": parent[0]["name"].split("::")[1] if len(parent) == 1 else None,
            "start_ms": start, "end_ms": start + event["dur"] / 1000,
            "duration_ms": event["dur"] / 1000, "bytes": event["args"].get("bytes")})
    frames = []
    for root, activity in zip(roots, assigned):
        kernels = [event for event in activity if event["kind"] == "kernel"]
        copies = [event for event in activity if event["kind"] == "gpu_memcpy"]
        intervals = {device: [(event["start_ms"], event["end_ms"]) for event in kernels if event["device"] == device] for device in (0, 1)}
        node_summary = {}
        for node_id in sorted({event["node_id"] for event in kernels if event["node_id"] is not None}):
            selected = [event for event in kernels if event["node_id"] == node_id]
            node_summary[node_id] = {"kernels": len(selected), "devices": sorted({event["device"] for event in selected}),
                "start_ms": min(event["start_ms"] for event in selected), "end_ms": max(event["end_ms"] for event in selected),
                "active_intervals_ms": merge_intervals([(event["start_ms"], event["end_ms"]) for event in selected])}
        branch_overlap = {}
        for label, first, second in (("gate_up", "L000.mlp.gate_proj", "L000.mlp.up_proj"),
                ("k_rope_q_proj", "L000.self_attention.k_rope", "L000.self_attention.q_proj")):
            pair = [node_summary.get(first), node_summary.get(second)]
            available = all(item is not None for item in pair)
            branch_overlap[label] = {"available": available,
                "cross_device": available and pair[0]["devices"] != pair[1]["devices"],
                "overlap_ms": overlap_ms(pair[0]["active_intervals_ms"], pair[1]["active_intervals_ms"]) if available else None}
        frames.append({"label": root["name"], "physical_kernels": len(kernels), "gpu_memcpy_events": len(copies),
            "gpu_memcpy_bytes": sum(event["bytes"] or 0 for event in copies),
            "gpu_memcpy_duration_sum_ms": sum(event["duration_ms"] for event in copies),
            "gpu0_gpu1_kernel_overlap_ms": overlap_ms(intervals[0], intervals[1]) if kernels else None,
            "branch_overlap": branch_overlap, "node_gpu_intervals": node_summary, "gpu_events": activity})
    return {"frames": frames, "unassigned_gpu_event_indices": unassigned,
        "gpu_event_attribution_complete": bool(roots) and not unassigned,
        "method": "Only physical kernel/memcpy/memset ph=X events; unique CPU External id and scope containment. ac2g flow arrows and gpu_user_annotation spans are excluded. Missing branch events yield unavailable, not zero overlap."}


def capture_light_trace(placed, devices, parameters, templates, reference, path):
    executor = BranchTraceExecutor(placed, devices, parameters["seed"], templates)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.profiler.ProfilerActivity.CUDA in torch.profiler.supported_activities():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    options = dict(activities=activities, record_shapes=False, profile_memory=False, with_stack=False, with_modules=False)
    with torch.no_grad():
        for _ in range(parameters["warmup"]):
            output, _, _, _ = executor.run_once(False)
            del output
        # Initialize the profiler and branch scopes in a discarded session.
        with torch.profiler.profile(**options):
            output, _, _, _ = executor.run_once(False)
            del output
        timings = []
        with torch.profiler.profile(**options) as profiler:
            for iteration in range(3):
                with torch.profiler.record_function(f"fusion_light_forward::{iteration}"):
                    output, _, _, elapsed = executor.run_once(False)
                timings.append(elapsed)
                # Keep transfers for correctness outside the profiling session.
                if iteration == 2:
                    final_output = output
                else:
                    del output
        accuracy = compare_outputs(final_output, reference, rtol=parameters["rtol"], atol=parameters["atol"])
        del final_output, output
    profiler.export_chrome_trace(str(path))
    parsed = analyze_trace_frames(json.loads(path.read_text(encoding="utf-8"))["traceEvents"])
    return {"trace_file": path.name, "mode": "light", "correctness": accuracy,
        "gpu_timeline_available": len(parsed["frames"]) == 3 and all(frame["physical_kernels"] > 0 for frame in parsed["frames"]),
        "traced_samples_s": timings, "traced_median_ms_diagnostic_only": statistics.median(timings) * 1000,
        "collection": {"dispatch_interception": False, "record_shapes": False, "profile_memory": False,
            "with_stack": False, "with_modules": False, "per_node_metadata": False,
            "profiler_warmup_forwards": 1, "captured_forwards": 3, "watched_nodes": sorted(WATCHED_NODES)},
        **parsed}


def assess_trace_timing(trace, before_ms, after_ms):
    samples = trace.get("traced_samples_s")
    if not samples:
        return {"available": False, "timing_target_pass": False}
    median_ms = statistics.median(samples) * 1000
    before_ratio, after_ratio = median_ms / before_ms, median_ms / after_ms
    shift = after_ms / before_ms - 1
    return {"available": True, "trace_over_before_ratio": before_ratio, "trace_over_after_ratio": after_ratio,
        "after_over_before_change_fraction": shift,
        "trace_deviation_target": TIMING_DEVIATION_TARGET, "baseline_shift_target": BASELINE_SHIFT_TARGET,
        "timing_target_pass": all(abs(ratio - 1) <= TIMING_DEVIATION_TARGET for ratio in (before_ratio, after_ratio)) and abs(shift) <= BASELINE_SHIFT_TARGET,
        "interpretation": "Predeclared engineering check of profiling perturbation, not proof of causal fidelity or independent validation. Keep every trace even when the target fails."}
