"""Collect host submission costs on old training workloads, not held-out tasks."""

import argparse
from collections import Counter
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import sys

import torch

from calibration.e2_manifest import content_id
from calibration.e2_records import append_jsonl_record
from calibration.gpu_holdout import DEFAULT_BUNDLE, ROOT, load_freeze
from calibration.profile_e2_bw1000 import _validate_runtime
from cost_model.reference_semantics import HostSubmissionProfile
from execution.audit_gpu_holdout import write_json
from execution.diagnose_gpu_holdout import expected_transfers
from execution.gpu_holdout import CanonicalReferenceExecutor, compare_outputs, measure_candidate
from execution.host_submission import SubmissionProbeExecutor, summarize_blocks, training_cases
from simulator import PlacedOpGraph, make_reference_plan
from simulator.dataset import graph_from_dict, plan_from_dict
from simulator.hardware import hardware_from_dict


def measure_probe(placed, devices, parameters, templates, reference, *, mode, environment_id):
    executor = SubmissionProbeExecutor(placed, devices, parameters["seed"], templates,
        device_windows=mode == "window_probe", environment_id=environment_id)
    with torch.no_grad():
        output, _ = executor.run_probe()
        first = compare_outputs(output, reference, rtol=parameters["rtol"], atol=parameters["atol"])
        del output
        for _ in range(parameters["warmup"]):
            output, _ = executor.run_probe()
            del output
        samples = []
        for iteration in range(parameters["iterations"]):
            output, sample = executor.run_probe()
            samples.append(sample)
            if iteration + 1 < parameters["iterations"]:
                del output
        last = compare_outputs(output, reference, rtol=parameters["rtol"], atol=parameters["atol"])
        del output
    observed = [row for row in executor.catalog.values() if row["kind"] == "copy"]
    expected = expected_transfers(placed)
    if Counter((row["source"], row["destination"], row["payload_bytes"]) for row in observed) != Counter(
            (row["source"], row["destination"], row["payload_bytes"]) for row in expected):
        raise ValueError("probe copy catalog differs from the reference predecessor loop")
    return {"samples_s": [row["total_s"] for row in samples], "probe_samples": samples,
        "catalog": executor.catalog, "correctness": {"pass": first["pass"] and last["pass"], "first": first, "last": last}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    freeze, manifest = load_freeze(args.bundle)
    _validate_runtime(manifest, (0, 1))
    cases = training_cases(freeze, manifest)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError("use an empty output directory; keep earlier runs for comparison")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    environment = {"host": socket.gethostname(), "platform": platform.platform(), "python": sys.version,
        "processor": platform.processor(), "logical_cpus": os.cpu_count(),
        "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
        "torch": torch.__version__, "hip": torch.version.hip,
        "devices": [{"index": index, "name": torch.cuda.get_device_properties(index).name,
            "total_memory": torch.cuda.get_device_properties(index).total_memory} for index in (0, 1)],
        "peer_access": {f"{a}->{b}": bool(torch.cuda.can_device_access_peer(a, b)) for a, b in ((0, 1), (1, 0))}}
    environment_id = content_id("reference-host-environment", environment)
    parameters = {key: freeze["protocol"]["execution"][key] for key in ("seed", "rtol", "atol", "warmup", "iterations")}
    sources = ("calibration/calibrate_reference_host.py", "execution/host_submission.py", "execution/reference.py",
        "execution/gpu_holdout.py", "cost_model/reference_semantics.py", "simulator/model_graphs/dense_layer.py")
    protocol = {"format": "fusion-placement-search.reference-host-calibration", "freeze_id": freeze["freeze_id"],
        "training_sequences": [256, 1024], "cases": cases, "parameters": parameters,
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
        "scope": "Training-side host dispatch calibration only; no seq768/1280 or new holdout labels. GPU windows are context-dependent body candidates, not pure kernel sums or link service measurements."}
    protocol_id = content_id("reference-host-calibration-protocol", protocol)
    write_json(args.out_dir / "protocol.json", {"protocol_id": protocol_id, **protocol})
    write_json(args.out_dir / "environment.json", {"environment_id": environment_id, **environment})
    devices, hardware, templates = ("cuda:0", "cuda:1"), hardware_from_dict(freeze["hardware"]), {}
    references, records = {}, []
    for index, case in enumerate(cases):
        try:
            graph = graph_from_dict(case["graph"])
            if case["model_id"] not in references:
                executor = CanonicalReferenceExecutor(PlacedOpGraph(graph, hardware, make_reference_plan(graph, device=0)), devices,
                    parameters["seed"], templates)
                with torch.no_grad():
                    output, _, _, _ = executor.run_once(False)
                    references[case["model_id"]] = output.float().cpu()
                    del output, executor
                gc.collect()
            reference = references[case["model_id"]]
            placed = PlacedOpGraph(graph, hardware, plan_from_dict(case["plan"]))
            middle = ("cpu_probe", "window_probe") if index % 2 == 0 else ("window_probe", "cpu_probe")
            for mode in ("baseline_before", *middle, "baseline_after"):
                row = {"freeze_id": freeze["freeze_id"], "environment_id": environment_id, "protocol_id": protocol_id,
                    "case_id": case["case_id"], "model_id": case["model_id"], "plan_name": case["plan_name"], "mode": mode}
                result = (measure_candidate(placed, devices, parameters, templates, reference) if mode.startswith("baseline") else
                    measure_probe(placed, devices, parameters, templates, reference, mode=mode, environment_id=environment_id))
                row.update(result)
                row["record_id"] = content_id("reference-host-calibration-block", row)
                append_jsonl_record(args.out_dir / "blocks.jsonl", row)
                records.append(row)
                print(f"case={index+1}/{len(cases)} {case['model_id']} {case['plan_name']} mode={mode} correctness={result['correctness']['pass']}", flush=True)
                if not result["correctness"]["pass"]:
                    raise RuntimeError("calibration output mismatch; keep the raw block")
                gc.collect()
        except Exception as exc:
            write_json(args.out_dir / "collection_status.json", {"status": "incomplete", "case_id": case["case_id"],
                "error_type": type(exc).__name__, "error": str(exc)})
            raise
    summary = summarize_blocks(records)
    data_id = content_id("reference-host-calibration-data", records)
    host = HostSubmissionProfile(**summary["host_parameters"], source=f"reference-host-calibration:{data_id}")
    write_json(args.out_dir / "host_submission_profile.json", asdict(host))
    body_catalog = {item["body_id"]: item["body_key"] for row in records if row["mode"] == "window_probe"
        for item in row["catalog"].values() if item["kind"] == "node"}
    write_json(args.out_dir / "body_window_candidates.json", {"body_catalog": body_catalog, "windows": summary["body_window_candidates"],
        "status": "diagnostic_candidates_only", "environment_id": environment_id,
        "limitations": "Body implementation/layout identity differs from old torch.mm/bmm keys. Windows may depend on queue state and are kept per case. Copy windows can contain stream waits and must not replace link parameters."})
    report = {"status": "complete" if host.calibrated else "complete_cpu_probe_perturbed", "protocol_id": protocol_id,
        "environment_id": environment_id, "data_id": data_id, "records": len(records),
        "measured_forwards": sum(len(row["samples_s"]) for row in records), **summary,
        "host_profile": asdict(host), "scope": protocol["scope"],
        "limitations": ["Host calibrated=true describes this measured host/runtime, not compatibility with old GPU-run hosts.",
            "A start-to-start dispatch step includes executor bookkeeping; API CPU time and GPU stream windows are distinct and not summed.",
            "Idle sync cost excludes pending GPU work; final busy synchronization is never a host parameter.",
            "Body windows and full-graph timings are training-side data, not independent validation or SROP benefit."]}
    write_json(args.out_dir / "calibration_report.json", report)
    write_json(args.out_dir / "collection_status.json", {"status": report["status"], "records": len(records)})
    print(f"report: {args.out_dir / 'calibration_report.json'}; host_calibrated={host.calibrated}")


if __name__ == "__main__":
    main()
