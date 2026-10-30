"""Freeze, collect and independently verify development group-Key microbenchmarks.

GPU operations are confined to collection; check/verify need no active device.
It never executes an operator graph, updates a predictor, or grants a downstream
fusion capability.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import socket
import statistics
import time

from cost_model.fusion_kernel_profile import canonical, key_id
from .kerv_silu_mul import DEFAULT_BUNDLE, DEFAULT_FLAGSCALE, ROOT, check_capability


DEFAULT_DRAFT = ROOT / "data/group_key_remeasurement_draft_v1/manifest.json"
FORMAT = "fusion-placement-search.group-key-collection"
COLLECTOR_SOURCES = (
    "calibration/group_key_remeasurement.py", "calibration/probe_kerv_silu_mul.py",
    "calibration/kerv_silu_mul.py", "cost_model/fusion_kernel_profile.py",
)


def sealed(payload, field):
    return {**payload, field: key_id(payload)}


def check_seal(value, field):
    if value.get(field) != key_id({k: v for k, v in value.items() if k != field}):
        raise ValueError(f"{field} content hash mismatch")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_draft(path=DEFAULT_DRAFT, *, root=ROOT):
    draft = json.loads(Path(path).read_text(encoding="utf-8"))
    check_seal(draft, "draft_id")
    if draft.get("status") != "draft_not_collection_ready":
        raise ValueError("expected an unbound development draft")
    entries = draft["entries"]
    if len(entries) != draft["slot_count"] or not entries:
        raise ValueError("draft slot count mismatch")
    if len({e["slot_id"] for e in entries}) != len(entries):
        raise ValueError("duplicate draft slot")
    for entry in entries:
        if entry["specification_id"] != key_id(entry["kernel_key_template"]):
            raise ValueError("draft specification hash mismatch")
        if any(entry.get(k) is not None for k in ("kernel_id", "kernel_key", "latency_s")):
            raise ValueError("draft unexpectedly contains bound identities or labels")
        if entry["label_role"] != "development_calibration":
            raise ValueError("remeasurement is development data")
        operation_contract(entry["kernel_key_template"])
    for group in ("source_sha256", "input_sha256"):
        for name, expected in draft[group].items():
            if file_hash(Path(root) / name) != expected:
                raise ValueError(f"draft source/input changed: {name}")
    return draft


def strides(shape):
    result, stride = [], 1
    for size in reversed(shape):
        result.append(stride)
        stride *= size
    return list(reversed(result))


def operation_contract(key):
    """Specify physical inputs explicitly, including graph-implicit MM weights."""
    ops = key["operators"]
    dtype = ops[0]["output"]["dtype"]
    if dtype != "bf16" or any(o["output"]["dtype"] != dtype for o in ops):
        raise ValueError("this development collector supports BF16 only")
    output = list(ops[-1]["output"]["shape"])
    names = [o["operator"] for o in ops]
    if names == ["matmul"] and key["implementation"]["id"] == "torch.mm.reference":
        shape = ops[0]["attributes"]["shape"]
        m, n, k = (int(shape[f]) for f in ("m", "n", "k"))
        if math.prod(output) != m * n:
            raise ValueError("logical MM output disagrees with physical dimensions")
        inputs = [[m, k], [k, n]]
        kind = "mm"
    elif names == ["silu"] and key["implementation"]["id"] == "torch.silu.reference":
        inputs, kind = [output], "silu"
    elif names == ["mul"] and key["implementation"]["id"] == "torch.mul.reference":
        inputs, kind = [output, output], "mul"
    elif names == ["silu", "mul"] and key["implementation"]["id"] == "flagos_embodied.kerv_silu_mul.triton":
        inputs, kind = [output, output], "kerv_silu_mul"
        if output[-1] > 8192 or key["pattern_id"] != "silu_mul":
            raise ValueError("fused shape/pattern is outside the pinned KERV contract")
    else:
        raise ValueError(f"unsupported primitive/group: {names}")
    if any(not s or any(type(v) is not int or v <= 0 for v in s) for s in inputs + [output]):
        raise ValueError("invalid physical tensor shape")
    if any(o["attributes"].get("phase") != "forward" for o in ops):
        raise ValueError("collector supports forward kernels only")
    return {
        "kind": kind, "dtype": dtype,
        "inputs": [{"shape": s, "strides": strides(s), "layout": "contiguous_row_major"}
                   for s in inputs],
        "output": {"shape": output, "strides": strides(output)},
        "mm_output": "contiguous [M,N] with zero-copy logical output view",
        "input_initialization": "CPU Torch seeded normal BF16; Mul left=BF16 SiLU(gate)",
        "input_seed": 1234, "requires_grad": False,
        "restore_mutating_gate": kind == "kerv_silu_mul",
    }


def block_schedule(entries, samples, seed):
    """Interleave devices, reverse implementation order on alternate rounds."""
    by_name = {}
    for e in entries:
        name = e["slot_id"].rsplit("@gpu", 1)[0]
        by_name.setdefault(name, []).append(e)
    names = sorted(by_name)
    random.Random(seed).shuffle(names)
    schedule = []
    for sample in range(samples):
        for name in (names if sample % 2 == 0 else reversed(names)):
            for e in sorted(by_name[name], key=lambda e: e["device_index"], reverse=bool(sample % 2)):
                schedule.append({"block_index": len(schedule), "sample": sample,
                                 "slot_id": e["slot_id"]})
    return schedule


def bind_manifest(draft, environment, *, mode="smoke", order_seed=0, root=ROOT,
                  collector_sources=None):
    if mode not in {"smoke", "formal"}:
        raise ValueError("unknown collection mode")
    check_seal(environment, "environment_id")
    devices = {d["device_index"]: d for d in environment["devices"]}
    if set(devices) != {e["device_index"] for e in draft["entries"]}:
        raise ValueError("runtime devices do not cover the draft")
    runtime = environment["runtime"]
    if runtime.get("accelerator") != "hip" or any(not runtime.get(f) for f in (
            "framework_version", "accelerator_version", "triton_version", "torch_build_config_sha256")):
        raise ValueError("unbound HIP/Torch/Triton/compiler build identity")
    sources = (dict(collector_sources) if collector_sources is not None else
               {n: file_hash(Path(root) / n) for n in COLLECTOR_SOURCES})
    if set(sources) != set(COLLECTOR_SOURCES):
        raise ValueError("incomplete collector source identity")
    recipe = copy.deepcopy(draft["measurement_recipe"][
        "smoke" if mode == "smoke" else "formal_development_proposal"])
    recipe.update(mode=mode, order_seed=order_seed, input_seed=1234,
                  timing="one event pair per invocation; restore outside events; GPU synchronization per block",
                  context="resident warm inputs; Torch output allocation included; no graph execution",
                  numeric={"mm": {"rtol": .02, "atol": .05},
                           "elementwise": {"rtol": .01, "atol": .001}},
                  quality="retain every sample; report median, IQR, drift; no outlier trimming")
    entries = []
    for original in draft["entries"]:
        entry = {k: copy.deepcopy(original[k]) for k in (
            "slot_id", "specification_id", "candidate_nodes", "device_index", "pattern_id", "label_role")}
        key = copy.deepcopy(original["kernel_key_template"])
        contract = operation_contract(key)
        key["runtime"] = copy.deepcopy(runtime)
        key["hardware"] = copy.deepcopy(devices[entry["device_index"]])
        key["hardware"]["source"] = environment["environment_id"]
        for tensor in [*key["external_inputs"], *(o["output"] for o in key["operators"])]:
            tensor["strides"] = strides(tensor["shape"])
        if contract["kind"] != "kerv_silu_mul":
            key["implementation"]["version"] = "sha256:" + sources[COLLECTOR_SOURCES[0]]
            key["implementation"]["source_contract"] = "calibration.group_key_remeasurement.prepare_operation"
        key["implementation"]["measurement_contract"] = contract
        key["implementation"]["collector_source_sha256"] = sources
        entry.update(kernel_key=key, kernel_id=key_id(key))
        entries.append(entry)
    payload = {
        "format": FORMAT, "version": "1.0", "scope": "development_isolated_kernels",
        "draft_id": draft["draft_id"], "draft": draft,
        "capability_id": draft["capability_id"], "environment": environment,
        "collector_source_sha256": sources, "recipe": recipe, "entries": entries,
        "schedule": block_schedule(entries, recipe["samples"], order_seed),
        "downstream_deployable": False, "automatic_profile_writeback": False,
    }
    return sealed(payload, "manifest_id")


def capture_environment(torch, triton, device_indices):
    visibility = {n: os.environ.get(n) for n in (
        "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")}
    devices = []
    for index in sorted(device_indices):
        p = torch.cuda.get_device_properties(index)
        devices.append({
            "device_index": index, "name": p.name, "class": p.name,
            "physical_index": None,
            "uuid": str(p.uuid) if getattr(p, "uuid", None) is not None else None,
            "pci_bus_id": str(p.pci_bus_id) if getattr(p, "pci_bus_id", None) is not None else None,
            "total_memory": int(p.total_memory),
            "multiprocessors": int(p.multi_processor_count),
            "architecture": str(getattr(p, "gcnArchName", "unknown")),
            "locator": {"host": socket.gethostname(), "torch_logical_index": index,
                        "visibility": visibility},
            "mapping_scope": "Torch enumeration; physical ordinal not inferred from logical index",
        })
    runtime = {"framework": "pytorch", "framework_version": str(torch.__version__),
               "framework_git_version": str(torch.version.git_version),
               "accelerator": "hip", "accelerator_version": torch.version.hip,
               "triton_version": str(triton.__version__),
               "torch_build_config_sha256": hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
               "python_version": platform.python_version(),
               "float32_matmul_precision": torch.get_float32_matmul_precision(),
               "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
               "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    return sealed({"host": socket.gethostname(), "platform": platform.platform(),
                   "runtime": runtime, "torch_build_config": torch.__config__.show(),
                   "devices": devices}, "environment_id")


def prepare_operation(entry, torch, module, *, device=None):
    """Create one isolated operation; CPU use is limited to unit verification."""
    import torch.nn.functional as F
    contract = entry["kernel_key"]["implementation"]["measurement_contract"]
    kind = contract["kind"]
    target = device if device is not None else f"cuda:{entry['device_index']}"
    generator = torch.Generator(device="cpu").manual_seed(contract["input_seed"])
    cpu = [torch.randn(t["shape"], generator=generator, dtype=torch.bfloat16)
           for t in contract["inputs"]]
    if kind == "mul":
        cpu[0] = F.silu(cpu[0])
    inputs = [v.to(target) for v in cpu]
    for value, declared in zip(inputs, contract["inputs"]):
        if list(value.stride()) != declared["strides"] or not value.is_contiguous():
            raise ValueError("actual measurement input layout differs from frozen Key")
    original = inputs[0].clone() if kind == "kerv_silu_mul" else None
    if kind == "mm":
        call = lambda: torch.mm(*inputs).view(contract["output"]["shape"])
        reference = (cpu[0].float() @ cpu[1].float()).to(torch.bfloat16).view(contract["output"]["shape"])
    elif kind == "silu":
        call = lambda: F.silu(inputs[0])
        reference = F.silu(cpu[0].float()).to(torch.bfloat16)
    elif kind == "mul":
        call = lambda: torch.mul(*inputs)
        reference = (cpu[0].float() * cpu[1].float()).to(torch.bfloat16)
    else:
        if module is None:
            raise RuntimeError("KERV requires the source-checked GPU module")
        call = lambda: module.kerv_silu_mul(*inputs)
        reference = (F.silu(cpu[0].float()) * cpu[1].float()).to(torch.bfloat16)
    return {"call": call, "restore": (lambda: inputs[0].copy_(original)) if original is not None else lambda: None,
            "inputs": inputs, "cpu_inputs": cpu, "reference": reference,
            "contract": contract, "kind": kind}


def validate_operation(prepared, torch, module, numeric):
    from .probe_kerv_silu_mul import checked_launch_probe
    p = prepared
    p["restore"]()
    fused = p["kind"] == "kerv_silu_mul"
    actual = checked_launch_probe(module, *p["inputs"]) if fused else p["call"]()
    actual_cpu = actual.detach().cpu()
    if list(actual.shape) != p["contract"]["output"]["shape"]:
        raise ValueError("actual output shape differs from Key")
    if list(actual.stride()) != p["contract"]["output"]["strides"]:
        raise ValueError("actual output strides differ from Key")
    tolerance = numeric["mm" if p["kind"] == "mm" else "elementwise"]
    if not torch.isfinite(actual_cpu).all().item():
        raise ValueError("nonfinite Kernel output")
    torch.testing.assert_close(actual_cpu, p["reference"], **tolerance)
    alias = actual.data_ptr() == p["inputs"][0].data_ptr()
    if alias != fused:
        raise ValueError("output alias differs from implementation contract")
    for i, (value, original) in enumerate(zip(p["inputs"], p["cpu_inputs"])):
        if not (fused and i == 0) and not torch.equal(value.cpu(), original):
            raise ValueError("unexpected input mutation")
    result = {"passed": True, "finite": True, "rtol": tolerance["rtol"], "atol": tolerance["atol"],
              "max_abs_error": float((actual_cpu.float() - p["reference"].float()).abs().max()),
              "output_alias_gate": alias,
              "inputs_preserved_except_declared_gate": True,
              "observed_fused_JIT_dispatches": 1 if fused else None,
              "dispatch_evidence": "Python JIT-launch observation; not GPU-profiler kernel count" if fused else "explicit Torch primitive"}
    if fused:
        unfused = torch.nn.functional.silu(p["cpu_inputs"][0]) * p["cpu_inputs"][1]
        torch.testing.assert_close(actual_cpu, unfused, **tolerance)
        result["max_abs_error_unfused"] = float((actual_cpu.float() - unfused.float()).abs().max())
    return result


def telemetry(torch, device):
    result = {"memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
              "memory_reserved_bytes": int(torch.cuda.memory_reserved(device))}
    for name in ("utilization", "temperature", "clock_rate"):
        try:
            result[name] = {"value": float(getattr(torch.cuda, name)(device)), "status": "available"}
        except Exception as error:
            result[name] = {"status": "unavailable", "reason": str(error)}
    return result


def sample_block(prepared, torch, device, iterations):
    started = time.perf_counter()
    started_unix_s = time.time()
    with torch.cuda.device(device):
        before = telemetry(torch, device)
        torch.cuda.synchronize(device)
        events = []
        for _ in range(iterations):
            prepared["restore"]()
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            prepared["call"]()
            end.record()
            events.append((begin, end))
        torch.cuda.synchronize(device)
        samples = [b.elapsed_time(e) / 1000 for b, e in events]
        after = telemetry(torch, device)
    if any(not math.isfinite(v) or v <= 0 for v in samples):
        raise ValueError("invalid GPU event duration")
    return {"kernel_samples_s": samples, "block_wall_s": time.perf_counter() - started,
            "started_unix_s": started_unix_s,
            "telemetry_before": before, "telemetry_after": after,
            "timing_scope": "GPU events exclude gate restore; block wall includes restore, telemetry and synchronization"}


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def append_record(stream, value):
    stream.write(canonical(sealed(value, "record_id")) + "\n")
    stream.flush()


def read_records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_bundle(bundle):
    """Validate frozen identities and all raw blocks without Torch or label access."""
    bundle = Path(bundle)
    manifest = json.loads((bundle / "frozen_manifest.json").read_text(encoding="utf-8"))
    check_seal(manifest, "manifest_id")
    check_seal(manifest["draft"], "draft_id")
    check_seal(manifest["environment"], "environment_id")
    if manifest["format"] != FORMAT or manifest["draft_id"] != manifest["draft"]["draft_id"]:
        raise ValueError("invalid collection format or draft lineage")
    expected_manifest = bind_manifest(
        manifest["draft"], manifest["environment"], mode=manifest["recipe"]["mode"],
        order_seed=manifest["recipe"]["order_seed"],
        collector_sources=manifest["collector_source_sha256"])
    if manifest != expected_manifest:
        raise ValueError("bound manifest disagrees with draft/runtime/collector contract")
    entries = {e["slot_id"]: e for e in manifest["entries"]}
    if len(entries) != len(manifest["draft"]["entries"]):
        raise ValueError("collection dropped or duplicated draft slots")
    for e in entries.values():
        if e["kernel_id"] != key_id(e["kernel_key"]):
            raise ValueError("final Kernel Key hash mismatch")
        if e["kernel_key"]["hardware"]["source"] != manifest["environment"]["environment_id"]:
            raise ValueError("Kernel environment identity mismatch")
    recipe = manifest["recipe"]
    schedule = block_schedule(list(entries.values()), recipe["samples"], recipe["order_seed"])
    if schedule != manifest["schedule"]:
        raise ValueError("frozen block schedule mismatch")
    setups = read_records(bundle / "setup_records.jsonl")
    raw = read_records(bundle / "raw_measurements.jsonl")
    setup_by_slot = {}
    for row in setups:
        check_seal(row, "record_id")
        slot = row["slot_id"]
        if slot not in entries or slot in setup_by_slot:
            raise ValueError("unexpected or duplicate setup record")
        if row["manifest_id"] != manifest["manifest_id"] or row["kernel_id"] != entries[slot]["kernel_id"]:
            raise ValueError("setup lineage mismatch")
        if not math.isfinite(row["setup_wall_s"]) or row["setup_wall_s"] < 0:
            raise ValueError("invalid setup wall time")
        setup_by_slot[slot] = row
    by_slot = {slot: [] for slot in entries}
    if len(raw) > len(schedule):
        raise ValueError("too many measurement blocks")
    for expected, row in zip(schedule, raw):
        check_seal(row, "record_id")
        entry = entries[expected["slot_id"]]
        if any(row[k] != expected[k] for k in expected):
            raise ValueError("raw records violate frozen order")
        if row["manifest_id"] != manifest["manifest_id"] or row["kernel_id"] != entry["kernel_id"]:
            raise ValueError("raw record Key/manifest lineage mismatch")
        if row["environment_id"] != manifest["environment"]["environment_id"]:
            raise ValueError("raw environment mismatch")
        if row["status"] == "ok":
            values = row["kernel_samples_s"]
            if len(values) != recipe["iterations_per_sample"] or any(
                    not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError("invalid raw event samples")
            if not math.isfinite(row["block_wall_s"]) or row["block_wall_s"] <= 0:
                raise ValueError("invalid measurement wall time")
        by_slot[entry["slot_id"]].append(row)
    profiles = {}
    for slot, e in entries.items():
        setup = setup_by_slot.get(slot, {})
        blocks = by_slot[slot]
        numeric = setup.get("correctness", {})
        if setup.get("status") != "ok" or not numeric.get("passed"):
            continue
        if not numeric.get("finite") or not numeric.get("inputs_preserved_except_declared_gate"):
            raise ValueError("label lacks numeric/input-preservation evidence")
        if numeric.get("output_alias_gate") != (e["pattern_id"] == "silu_mul"):
            raise ValueError("label alias evidence contradicts Kernel contract")
        if e["pattern_id"] == "silu_mul" and numeric.get("observed_fused_JIT_dispatches") != 1:
            raise ValueError("fused label lacks dispatch evidence")
        if len(blocks) != recipe["samples"] or any(b["status"] != "ok" for b in blocks):
            continue
        medians = [statistics.median(b["kernel_samples_s"]) for b in blocks]
        quartiles = statistics.quantiles(medians, n=4, method="inclusive")
        profiles[e["kernel_id"]] = {
            "slot_id": slot, "kernel_key": e["kernel_key"], "kind": "measured",
            "duration_s": statistics.median(medians),
            "block_medians_s": medians, "block_median_iqr_s": quartiles[2] - quartiles[0],
            "first_to_last_relative_change": medians[-1] / medians[0] - 1,
            "profiling_wall_s": setup["setup_wall_s"] + sum(b["block_wall_s"] for b in blocks),
            "cost_scope": "per-Key setup + sampling; shared environment/source/IO overhead reported separately",
            "provenance": manifest["manifest_id"],
            "raw_record_ids": [setup["record_id"], *(b["record_id"] for b in blocks)],
        }
    complete = len(profiles) == len(entries) and len(raw) == len(schedule)
    return {"format": "fusion-placement-search.verified-group-key-profile", "version": "1.0",
            "manifest_id": manifest["manifest_id"], "scope": "development_isolated_kernels",
            "complete": complete, "expected_keys": len(entries), "accepted_keys": len(profiles),
            "expected_blocks": len(schedule), "recorded_blocks": len(raw),
            "profiles": profiles, "automatic_profile_writeback": False,
            "downstream_deployable": False}


def collect(draft, out_dir, capability_path, flagscale_root, *, mode="smoke", order_seed=0):
    started = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    launched = 0
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("no GPU available; group-Key collection was not run")
        devices = sorted({e["device_index"] for e in draft["entries"]})
        if not torch.version.hip or max(devices) >= torch.cuda.device_count():
            raise RuntimeError("draft requires its two HIP device indices; CPU/CUDA substitution is forbidden")
        from .probe_kerv_silu_mul import load_kernel_module
        module, capability = load_kernel_module(flagscale_root, capability_path)
        if capability["capability_id"] != draft["capability_id"]:
            raise ValueError("draft capability mismatch")
        module.configure_recording(None, False)
        module.configure_backend("triton")
        module.configure_enabled_operators({"kerv_silu_mul"})
        if module.triton is None or not module._use_triton("kerv_silu_mul"):
            raise RuntimeError("explicit enabled Triton implementation unavailable")
        environment = capture_environment(torch, module.triton, devices)
        manifest = bind_manifest(draft, environment, mode=mode, order_seed=order_seed)
        # Persist identities and concrete schedule BEFORE any operator launch.
        write_json(out_dir / "frozen_manifest.json", manifest)
        shared_preparation_wall = time.perf_counter() - started
        entries = {e["slot_id"]: e for e in manifest["entries"]}
        recipe, operations = manifest["recipe"], {}
        with (out_dir / "setup_records.jsonl").open("x", encoding="utf-8", newline="\n") as setup_stream, \
             (out_dir / "raw_measurements.jsonl").open("x", encoding="utf-8", newline="\n") as raw_stream, \
             torch.no_grad():
            for slot, e in entries.items():
                setup_started = time.perf_counter()
                row = {"slot_id": slot, "kernel_id": e["kernel_id"], "manifest_id": manifest["manifest_id"]}
                phases = {}
                try:
                    with torch.cuda.device(e["device_index"]):
                        phase = time.perf_counter()
                        p = prepare_operation(e, torch, module)
                        torch.cuda.synchronize(e["device_index"])
                        phases["input_setup_wall_s"] = time.perf_counter() - phase
                        phase = time.perf_counter()
                        launched += 1
                        row["correctness"] = validate_operation(p, torch, module, recipe["numeric"])
                        torch.cuda.synchronize(e["device_index"])
                        phases["validation_and_first_launch_wall_s"] = time.perf_counter() - phase
                        phase = time.perf_counter()
                        for _ in range(recipe["warmup"]):
                            p["restore"]()
                            p["call"]()
                        torch.cuda.synchronize(e["device_index"])
                        phases["warmup_wall_s"] = time.perf_counter() - phase
                        operations[slot] = p
                    row["status"] = "ok"
                except Exception as error:
                    row.update(status="error", error=f"{type(error).__name__}: {error}")
                row.update(setup_wall_s=time.perf_counter() - setup_started, phase_wall_s=phases)
                append_record(setup_stream, row)
            for block in manifest["schedule"]:
                e = entries[block["slot_id"]]
                row = {**block, "kernel_id": e["kernel_id"], "manifest_id": manifest["manifest_id"],
                       "environment_id": environment["environment_id"]}
                if e["slot_id"] not in operations:
                    row.update(status="not_run", error="Key setup/correctness failed; no timing label")
                else:
                    block_started = time.perf_counter()
                    try:
                        row.update(sample_block(operations[e["slot_id"]], torch, e["device_index"], recipe["iterations_per_sample"]))
                        row["status"] = "ok"
                    except Exception as error:
                        row.update(status="error", error=f"{type(error).__name__}: {error}",
                                   block_wall_s=time.perf_counter() - block_started)
                append_record(raw_stream, row)
        report = verify_bundle(out_dir)
        # The collection summary carries no latency table; export is explicit.
        summary = {k: v for k, v in report.items() if k != "profiles"}
        attributed = sum(p["profiling_wall_s"] for p in report["profiles"].values())
        elapsed = time.perf_counter() - started
        summary.update(shared_preparation_wall_s=shared_preparation_wall,
                       collection_wall_s=elapsed,
                       wall_scope="collect() entry through verification; excludes CLI checks, process/module startup and summary write",
                       accepted_key_attributed_wall_s=attributed,
                       remaining_wall_s=elapsed - attributed,
                       cost_use="observed development collection order/cache context; not an online SROP total-cost result",
                       accounting="remaining wall includes shared preparation, IO, verification and any failed-Key work; do not double-count nested phase timers")
        write_json(out_dir / "collection_summary.json", summary)
        return summary
    except Exception as error:
        write_json(out_dir / "failure.json", {
            "status": "failed_or_not_run", "error": f"{type(error).__name__}: {error}",
            "validation_launch_attempts": launched, "collection_wall_s": time.perf_counter() - started,
            "latency_labels_promoted": False})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("check", "collect"):
        p = sub.add_parser(name)
        p.add_argument("--draft", type=Path, default=DEFAULT_DRAFT)
        p.add_argument("--capability", type=Path, default=DEFAULT_BUNDLE / "capability.json")
        p.add_argument("--flagscale-root", type=Path, default=DEFAULT_FLAGSCALE)
        if name == "collect":
            p.add_argument("--out-dir", type=Path, required=True)
            p.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
            p.add_argument("--order-seed", type=int, default=0)
    p = sub.add_parser("verify")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--out", type=Path, help="optional new offline Profile file; no predictor update")
    args = parser.parse_args(argv)
    if args.action == "verify":
        report = verify_bundle(args.bundle)
        if args.out is not None:
            if not report["complete"]:
                raise ValueError("collection incomplete; retained records cannot form a full development Profile")
            write_json(args.out, report)
        print(json.dumps({k: v for k, v in report.items() if k != "profiles"}, indent=2))
        return 0 if report["complete"] else 1
    draft = load_draft(args.draft)
    capability = check_capability(args.capability, args.flagscale_root)
    if capability["capability_id"] != draft["capability_id"]:
        raise ValueError("draft capability mismatch")
    if args.action == "check":
        print(json.dumps({"draft_id": draft["draft_id"], "slots": len(draft["entries"]),
                          "source_and_capability_check": "PASS", "gpu_collection": "NOT RUN"}, indent=2))
        return 0
    report = collect(draft, args.out_dir, args.capability, args.flagscale_root,
                     mode=args.mode, order_seed=args.order_seed)
    print(json.dumps(report, indent=2))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
