"""CPU-frozen anchor pilot definition; this module has no GPU collector."""
import math
import statistics
import copy
import hashlib
from pathlib import Path

from cost_model.fusion_kernel_profile import key_id

PROTOCOL = {
    "version": "ffn-anchor-control.v1", "scope": "pilot_only_not_profile_admission",
    "device": "logical GPU0; observe fresh UUID/build, require same device for all four cells",
    "cells": [
        {"id": "down_r1", "row": "down_proj@gpu0", "endpoint_replays": 1},
        {"id": "up_r3", "row": "up_proj@gpu0", "endpoint_replays": 3},
        {"id": "up_r1", "row": "up_proj@gpu0", "endpoint_replays": 1},
        {"id": "down_r3", "row": "down_proj@gpu0", "endpoint_replays": 3}],
    "only_changed_factor": "anchor endpoint repeat count and its prespecified aggregation",
    "fixed": {"attempts": 7, "calls_per_replay": 50, "anchor_operator": "F.silu",
              "anchor_shape": [1,256,3072], "dtype": "bf16", "target_warmup": 10,
              "anchor_warmup": 10, "anchor_warmup_position": "before target case preparation, same as v3",
              "path_trace_position": "after every target and anchor timing in the worker",
              "new_process_per_cell": True, "worker_retries": 0, "extra_rounds": 0,
              "parent_gpu_state": "CPU-only parent; no retained smoke runtime",
              "unused_device_work": "no GPU1 workload in a pilot cell",
              "restore_and_validation": "outside Event interval; preserve v3 order and buffers"},
    "endpoint": {"aggregation": "median of all repeats, no trimming",
                 "dispersion": "inclusive IQR/median for r3, null/not-estimable for r1",
                 "r3_dispersion_limit": .10},
    "round": {"drift": "abs(post_median-pre_median)/pre_median", "drift_limit": .10,
              "invalid": "any endpoint numerical/dispersion failure or drift above limit invalidates round",
              "cell": "all seven attempts must be valid; target inclusive IQR/median <= 0.10",
              "threshold_status": "preregistered engineering pilot thresholds, not validated statistical guarantees"},
    "required_timestamps_ns": ["reset_begin", "timing_begin", "event_wait_end", "all_device_sync_end", "validation_end"],
    "timestamp_clock": "time.perf_counter_ns, raw per repeat and target; no disk I/O inside Event interval",
    "required_context": ["UUID/build", "stream/device", "case preparation begin/end", "anchor warmup end"],
    "telemetry": {"fields_if_available": ["clock_mhz", "utilization_percent", "temperature_c"],
                  "missing": "null plus source/error; never substitute normal or zero",
                  "missing_blocks_environment_causal_verdict": True},
    "followup_gate": "all four pipelines/identities/numerical/path/timestamps complete; both r3 cells pass every preregistered quality check; r1 quality failures retained; no retry or selecting another protocol",
    "followup": "same allocation may continue to a separately frozen seven-item GPU0 matrix; not implemented/open here",
    "v3_v4_profile_mixing": False, "retroactive_admission": False,
    "gpu_collector_ready": False, "gpu_entry_open": False,
}


def frozen_protocol():
    root = Path(__file__).resolve().parents[1]
    payload = copy.deepcopy(PROTOCOL)
    payload["workloads"] = {"down_proj@gpu0":{"input":[1,256,3072],"weight":[3072,1024],"output":[1,256,1024]},
                            "up_proj@gpu0":{"input":[1,256,1024],"weight":[1024,3072],"output":[1,256,3072]}}
    payload["fixture_recipe"] = "data/ffn_measurement_b3/recipe.json; named fixtures/seeds unchanged"
    payload["source_sha256"] = {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
                               ("calibration/ffn_anchor_control_protocol.py","calibration/ffn_b4_runtime.py","data/ffn_measurement_b3/recipe.json")}
    return {**payload, "protocol_id": key_id(payload)}


def endpoint_summary(samples, repeats):
    if repeats not in (1,3) or len(samples) != repeats:
        raise ValueError("endpoint requires exactly the frozen repeat count")
    values = []
    for sample in samples:
        if set(PROTOCOL["required_timestamps_ns"]) - sample.keys():
            raise ValueError("missing endpoint timestamps")
        stamps = [sample[k] for k in PROTOCOL["required_timestamps_ns"]]
        if any(type(t) is not int or t < 0 for t in stamps) or stamps != sorted(stamps):
            raise ValueError("missing or unordered endpoint timestamps")
        if sample["numeric_ok"] is not True or sample["calls"] != 50:
            raise ValueError("endpoint numerical/count failure")
        value = sample["batch_ms"]*1000/50
        if not math.isfinite(value) or value <= 0:
            raise ValueError("invalid anchor timing")
        values.append(value)
    median = statistics.median(values)
    q = statistics.quantiles(values,n=4,method="inclusive") if repeats == 3 else None
    dispersion = (q[2]-q[0])/median if q else None
    return {"median_us": median, "iqr_over_median": dispersion,
            "endpoint_valid": dispersion is None or dispersion <= .10}
