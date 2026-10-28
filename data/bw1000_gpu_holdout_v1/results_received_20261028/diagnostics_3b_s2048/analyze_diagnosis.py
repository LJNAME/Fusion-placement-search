"""Audit received diagnostic data; read raw GPU intervals without flow arrows."""

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from calibration.e2_manifest import content_id
from calibration.e2_records import load_jsonl_records
from calibration.gpu_holdout import load_freeze
from execution.audit_gpu_holdout import timing_statistics, write_json
from execution.diagnose_gpu_holdout import prediction_breakdown


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def overlap_ms(first, second):
    return sum(max(0.0, min(b, d) - max(a, c))
        for a, b in union(first) for c, d in union(second))


def trace_analysis(path):
    trace = json.loads(path.read_text(encoding="utf-8"))
    events = trace["traceEvents"]
    scopes = [event for event in events if event.get("cat") == "user_annotation" and event.get("ph") == "X"]
    root = next(event for event in scopes if event["name"] == "fusion_diagnostic_forward")
    nodes = [event for event in scopes if event["name"].startswith("fusion_node::")]
    cpu_by_external = defaultdict(list)
    for event in events:
        if event.get("cat") == "cpu_op" and event.get("ph") == "X":
            cpu_by_external[event.get("args", {}).get("External id")].append(event)
    gpu = []
    for index, event in enumerate(events):
        if event.get("ph") != "X" or event.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        anchors = cpu_by_external[event.get("args", {}).get("External id")]
        parent = None
        if len(anchors) == 1:
            anchor = anchors[0]
            parents = [node for node in nodes if node["pid"] == anchor["pid"] and node["tid"] == anchor["tid"]
                and node["ts"] <= anchor["ts"] and node["ts"] + node["dur"] >= anchor["ts"] + anchor["dur"] - 1]
            if len(parents) == 1:
                parent = parents[0]["name"].split("::")[1]
        gpu.append({"raw_event_index": index, "kind": event["cat"], "node_id": parent,
            "device": event["args"]["device"], "stream": event["args"].get("stream"),
            "start_ms": (event["ts"] - root["ts"]) / 1000,
            "end_ms": (event["ts"] + event["dur"] - root["ts"]) / 1000,
            "duration_ms": event["dur"] / 1000, "bytes": event["args"].get("bytes")})
    kernels = [event for event in gpu if event["kind"] == "kernel"]
    copies = [event for event in gpu if event["kind"] == "gpu_memcpy"]
    intervals = {device: [(event["start_ms"], event["end_ms"]) for event in kernels if event["device"] == device] for device in (0, 1)}
    node_summary = {}
    for name in sorted({event["node_id"] for event in kernels if event["node_id"] is not None}):
        selected = [event for event in kernels if event["node_id"] == name]
        node_summary[name] = {"kernels": len(selected), "device": sorted({event["device"] for event in selected}),
            "start_ms": min(event["start_ms"] for event in selected),
            "end_ms": max(event["end_ms"] for event in selected),
            "gpu_active_union_ms": sum(b - a for a, b in union([(event["start_ms"], event["end_ms"]) for event in selected]))}
    runtime = defaultdict(list)
    for event in events:
        if event.get("cat") == "cuda_runtime" and event.get("ph") == "X":
            runtime[event["name"]].append(event["dur"] / 1000)
    return {"trace_file": path.name, "physical_kernels": len(kernels), "gpu_memcpy_events": len(copies),
        "kernels_with_unique_cpu_node_association": sum(event["node_id"] is not None for event in kernels),
        "kernel_busy_union_ms_by_device": {str(device): sum(b - a for a, b in union(values)) for device, values in intervals.items()},
        "gpu0_gpu1_kernel_overlap_ms": overlap_ms(intervals[0], intervals[1]),
        "gpu_memcpy_duration_sum_ms": sum(event["duration_ms"] for event in copies),
        "gpu_memcpy_bytes": sum(event["bytes"] for event in copies),
        "runtime_calls": {name: {"calls": len(values), "cpu_duration_sum_ms": sum(values)} for name, values in runtime.items()},
        "node_gpu_intervals": node_summary, "gpu_events": gpu,
        "method": "GPU intervals use ph=X kernel/memcpy/memset only. ac2g flows and gpu_user_annotation ranges are excluded. CPU-node attribution uses unique External id plus CPU scope containment; no flow arrows are required."}


def main():
    base = Path(__file__).resolve().parent
    freeze, _ = load_freeze(base.parent.parent)
    report = json.loads((base / "diagnostic_report.json").read_text())
    environment = json.loads((base / "diagnostic_environment.json").read_text())
    predictions = json.loads((base / "prediction_breakdown.json").read_text())
    records = load_jsonl_records(base / "baseline_samples.jsonl")
    assert report["freeze_id"] == freeze["freeze_id"]
    assert environment == report["environment"]
    assert content_id("gpu-three-plan-diagnostic-environment", environment) == report["environment_id"]
    regenerated = json.loads(json.dumps(prediction_breakdown(freeze, report["selection"])))
    assert predictions == report["predictions"] == regenerated
    chosen = {row["role"]: row for row in report["selection"]["plans"]}
    expected_slots = {(role, repeat) for role in chosen for repeat in range(3)}
    assert len(records) == 9 and {(row["role"], row["repeat"]) for row in records} == expected_slots
    assert len({row["record_id"] for row in records}) == 9
    for row in records:
        assert row["record_id"] == content_id("gpu-three-plan-diagnostic-baseline", {key: value for key, value in row.items() if key != "record_id"})
        assert row["environment_id"] == report["environment_id"] and row["freeze_id"] == freeze["freeze_id"]
        assert row["plan_id"] == chosen[row["role"]]["plan_id"] and row["model_id"] == report["model_id"]
        assert len(row["samples_s"]) == 20 and all(math.isfinite(value) and value > 0 for value in row["samples_s"])
        assert row["median_s"] == statistics.median(row["samples_s"]) and row["correctness"]["pass"]
    ref = next(row["pooled"]["median_ms"] for row in report["baselines"] if row["role"] == "single_gpu_reference")
    baselines = []
    for row in report["baselines"]:
        source = [entry for entry in records if entry["role"] == row["role"]]
        pooled = timing_statistics([value for entry in source for value in entry["samples_s"]])
        rounds = [{"repeat": entry["repeat"], **timing_statistics(entry["samples_s"])} for entry in source]
        assert pooled == row["pooled"] and rounds == row["rounds"]
        baselines.append({**row, "relative_to_single_gpu_pct": (pooled["median_ms"] / ref - 1) * 100})
    trace_rows = []
    prediction_by_role = {row["role"]: row["frozen_prediction"] for row in predictions["plans"]}
    for row in report["traces"]:
        assert row["correctness"]["pass"] and row["copy_signature_matches_reference_loop"]
        trace = trace_analysis(base / row["trace_file"])
        copies = [entry for entry in row["observed_to_copy_calls"] if entry["cross_gpu"]]
        dma = [entry for entry in trace["gpu_events"] if entry["kind"] == "gpu_memcpy"]
        assert Counter(entry["source_payload_bytes"] for entry in copies) == Counter(entry["bytes"] for entry in dma)
        assert len(copies) == row["cross_gpu_copy_calls"] and trace["gpu_memcpy_bytes"] == row["cross_gpu_payload_bytes"]
        nodes = trace["node_gpu_intervals"]
        predicted = prediction_by_role[row["role"]]
        groups = {group["node_ids"][0]: group for group in predicted["details"]["groups"]}
        target, branch = (("L000.mlp.up_proj", "L000.mlp.gate_proj") if row["role"] != "intermediate" else
            ("L000.self_attention.k_rope", "L000.self_attention.q_proj"))
        actual_target = nodes[target]
        actual_branch = nodes[branch]
        predicted_overlap = overlap_ms([(groups[target]["start"] * 1000, groups[target]["end"] * 1000)],
            [(groups[branch]["start"] * 1000, groups[branch]["end"] * 1000)])
        trace.update(role=row["role"], frozen_makespan_ms=predicted["makespan"] * 1000,
            frozen_transfer_duration_sum_ms=predicted["communication_time"] * 1000,
            traced_forward_ms_diagnostic_only=row["traced_forward_ms_diagnostic_only"],
            traced_to_uninstrumented_median_ratio=row["traced_forward_ms_diagnostic_only"] /
                next(entry["pooled"]["median_ms"] for entry in baselines if entry["role"] == row["role"]),
            target_node=target, branch_node=branch, predicted_target_branch_overlap_ms=predicted_overlap,
            target_gpu_start_ms=actual_target["start_ms"], branch_gpu_end_ms=actual_branch["end_ms"],
            target_starts_after_branch_in_trace=actual_target["start_ms"] >= actual_branch["end_ms"])
        trace_rows.append(trace)
    output = {"freeze_id": freeze["freeze_id"], "checks_pass": True, "raw_records": len(records),
        "baseline_samples": sum(len(row["samples_s"]) for row in records),
        "baseline_order_same_in_all_rounds": all(sorted(chosen, key=lambda role: next(entry["median_s"] for entry in records if entry["role"] == role and entry["repeat"] == repeat)) ==
            ["single_gpu_reference", "intermediate", "predicted_top1"] for repeat in range(3)),
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(base.iterdir()) if path.name in
            {"diagnostic_report.json", "diagnostic_environment.json", "prediction_breakdown.json", "baseline_samples.jsonl", *(row["trace_file"] for row in report["traces"]) }},
        "baselines": baselines, "traces": trace_rows,
        "limitations": ["One already-exposed workload and three measured candidates; not an independent validation or SROP result.",
            "Traces are instrumented: forward times are 1.5x-3.7x the uninstrumented medians. Observed absence of overlap cannot by itself prove zero overlap in uninstrumented runs.",
            "ROCtracer duplicate-flow warning was reported by the user. Flow arrows and GPU annotation spans are excluded; no guarantee about tracer completeness or causal association follows from status=complete.",
            "CPU scopes, nested profiler totals and GPU activity durations are not additive. Do not attribute baseline differences by subtracting instrumented timings."]}
    write_json(base / "analysis.json", output)
    print(f"checks_pass=True records={len(records)} samples={output['baseline_samples']} stable_role_order={output['baseline_order_same_in_all_rounds']}")
    for row in baselines:
        print(row["role"], row["pooled"]["median_ms"], row["relative_to_single_gpu_pct"])
    for row in trace_rows:
        print(row["role"], "kernels", row["physical_kernels"], "copy_ms", row["gpu_memcpy_duration_sum_ms"],
            "gpu_overlap_ms", row["gpu0_gpu1_kernel_overlap_ms"], "expected_branch_overlap_ms", row["predicted_target_branch_overlap_ms"],
            "trace_ratio", row["traced_to_uninstrumented_median_ratio"])


if __name__ == "__main__":
    main()
