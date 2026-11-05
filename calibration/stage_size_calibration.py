"""GPU0 SiLU/Mul size sweep, Native chain check and four cross-GPU Copies.

plan/review run on CPU; only `run` touches HIP. Records are calibration and
validation candidates. Nothing here updates a model, Profile or admission list.
The fit/holdout protocol is frozen into plan.json before any label exists.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

from calibration.lightop_profile_smoke import POOL, memory_snapshot, save
from cost_model.fusion_kernel_profile import key_id
from cost_model.measurement_labels import MeasurementContractError

ROOT = Path(__file__).resolve().parents[1]
SCOPE = "gpu0_stage_size_linear_calibration_and_copy_v1"
HIDDEN = 3072
SIZES = (64, 128, 256, 512, 1024)
FIT_SIZES = (64, 256, 1024)
HOLDOUT_SIZES = (128, 512)
STAGES = ("silu", "mul", "chain")
FITTED_STAGES = ("silu", "mul")
# FFN transfer classes: 0.5 MiB = [1,256,1024] BF16, 1.5 MiB = [1,256,3072] BF16.
COPY_SHAPES = {524288: (1, 256, 1024), 1572864: (1, 256, 3072)}
COPIES = tuple((size, src, dst) for size in COPY_SHAPES for src, dst in ((0, 1), (1, 0)))
CALLS, ROUNDS, WARMUP = 50, 7, 10
TOLERANCE = {"rtol": .01, "atol": .001}
QUALITY = {"attempts": ROUNDS, "extra_attempts": 0, "sample_replacement": False,
           "iqr_method": "inclusive", "max_iqr_over_median": .10,
           "anchor_policy": "none; same rule as native_stage_calibration_20261105_01"}
SOURCES = ("calibration/stage_size_calibration.py", "tools/run_stage_size_calibration.sh",
           "calibration/lightop_profile_smoke.py", "execution/reference_identity.py",
           "execution/ffn_transfer.py", "integration/flagscale/kerv_profile_runtime.py",
           "cost_model/fusion_kernel_profile.py", "cost_model/measurement_labels.py")
RECIPES = {
    "silu": {"id": "aten.silu.out.preallocated_stage.sized.v1",
             "timed_body": "aten.silu.out(gate, out=output)", "operands": ["gate"]},
    "mul": {"id": "aten.mul.out.preallocated_derived_left_stage.sized.v1",
            "timed_body": "aten.mul.out(left, up, out=output)", "operands": ["left", "up"],
            "left_origin": "aten.silu.out(gate, out=left), outside capture and timing"},
    "chain": {"id": "aten.silu_mul.out.preallocated_native_chain.sized.v1",
              "timed_body": "aten.silu.out(gate, out=mid); aten.mul.out(mid, up, out=output)",
              "operands": ["gate", "up"], "intermediate": "per-call preallocated mid"},
    "copy": {"id": "torch.Tensor.copy_.preallocated_cross_gpu.v1",
             "timed_body": "destination.copy_(source, non_blocking=True) on destination stream",
             "operands": ["source"], "binding": "execution/ffn_transfer.py:BoundTransfer",
             "streams": "destination stream forks/joins a source-side stream inside capture"},
}
FIT_PROTOCOL = {
    "model": "T_stage(N) = a_stage + b_stage * N; N = output elements = S * 3072",
    "relation_to_analytic_model": "local per-GPU/per-contract calibration layer; graph_analytic.py unchanged",
    "fit_sizes": list(FIT_SIZES), "holdout_sizes": list(HOLDOUT_SIZES),
    "point_estimate": "ordinary least squares on the per-size medians (us/call) of fit sizes",
    "round_sensitivity": "OLS over all 7^3 combinations of per-round fit values; min/max reported, not a confidence interval",
    "identification": {"b": "identified iff every round-combination slope > 0",
                       "a": "separately identified iff the round-combination range excludes 0"},
    "fit_admission": "every fit size quality_passed; otherwise the stage is not fitted and no substitute is used",
    "holdout_rule": "predictions written before holdout labels are read; holdout never refits",
    "holdout_error_threshold": None,
    "holdout_reports": ["per-case APE and signed error", "signed bias", "pairwise ordering of all holdout cases"],
    "extrapolation": "outside [64,1024] reported as unvalidated",
    "chain_check": "same-batch chain vs silu+mul measured and vs fitted sum; validation only, never fitted",
    "analytic_baseline": "frozen configs/graph_analytic_scenarios.json assumptions, stage core+h+c, no refit",
    "search_check_after_prediction": "48 FFN plans with measured copies; best plan unchanged is not accuracy evidence",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shape(size):
    return (1, size, HIDDEN)


def strides(dims):
    out, stride = [], 1
    for d in reversed(dims):
        out.insert(0, stride)
        stride *= d
    return out


def case_ids():
    return ([f"{stage}/S{size}" for size in SIZES for stage in STAGES]
            + [f"copy/{size}/{src}->{dst}" for size, src, dst in COPIES])


def describe(case_id):
    parts = case_id.split("/")
    if parts[0] == "copy":
        src, dst = map(int, parts[2].split("->"))
        size = int(parts[1])
        return {"kind": "copy", "bytes": size, "source": src, "destination": dst,
                "shape": list(COPY_SHAPES[size]), "timing_device": f"cuda:{dst}"}
    size = int(parts[1][1:])
    role = "fit" if size in FIT_SIZES else "holdout"
    if parts[0] == "chain":
        role = "chain_validation"
    return {"kind": "stage", "stage": parts[0], "S": size, "shape": list(shape(size)),
            "elements": size * HIDDEN, "role": role, "timing_device": "cuda:0"}


def contract():
    return {"mode": "graph_replay", "granularity": "batch_amortized", "calls_per_replay": CALLS,
            "rounds": ROUNDS, "extra_attempts": 0, "eager_preparation_calls": CALLS,
            "warmup_replays": WARMUP, "unit": "us/call", "statistic": "median",
            "correctness": dict(TOLERANCE),
            "timing_boundary": "start Event -> one Graph replay -> end Event on the timing stream",
            "synchronization": "all involved devices synchronized outside the Event interval",
            "excluded": ["allocation", "input generation", "left generation", "validation", "host submission"],
            "buffers": "shared read-only inputs; independent preallocated outputs per call",
            "cache_context": "isolated cases interleaved per round; no chain cache equivalence claimed"}


def plan():
    ids = case_ids()
    return {"scope": SCOPE, "primary_device": "cuda:0", "architecture": "gfx936", "seed": 20261107,
        "dtype": "bf16", "layout": "explicit_contiguous", "hidden": HIDDEN, "sizes": list(SIZES),
        "cases": {c: describe(c) for c in ids}, "recipes": copy.deepcopy(RECIPES),
        "order": [list(ids if i % 2 == 0 else ids[::-1]) for i in range(ROUNDS)],
        "contract": contract(), "quality_policy": copy.deepcopy(QUALITY), "pool": copy.deepcopy(POOL),
        "fit_protocol": copy.deepcopy(FIT_PROTOCOL),
        "collector": {"git_revision": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": {p: digest(ROOT / p) for p in SOURCES}},
        "out_of_scope": ["GEMM", "GPU1 stage calibration", "Fused multi-size prediction", "cross-GPU validation"],
        "existing_measurements_preserved": True, "automatic_admission": False,
        "model_parameters_updated": False}


class StageCall:
    """One preallocated call; CPU tensors allowed for numeric tests only."""
    def __init__(self, stage, gate, up, left):
        import torch
        from execution.reference_identity import tensor_observation
        if stage not in STAGES:
            raise ValueError("Unknown stage")
        dims = tuple(gate.shape)
        tensors = {"gate": gate, "up": up, "left": left}
        if (len(dims) != 3 or dims[0] != 1 or dims[2] != HIDDEN or any(
                tuple(t.shape) != dims or t.dtype != torch.bfloat16 or not t.is_contiguous()
                or t.requires_grad or t.device != gate.device for t in tensors.values())):
            raise ValueError("Explicit contiguous BF16 same-device [1,S,3072] inputs required")
        if any(torch._C._overlaps(a, b) for a, b in ((gate, up), (gate, left), (left, up))):
            raise ValueError("Stage inputs must have independent storage")
        self.stage, self.gate, self.up, self.left = stage, gate, up, left
        self.output = torch.empty_like(gate)
        self.mid = torch.empty_like(gate) if stage == "chain" else None
        observed = {**tensors, "output": self.output}
        if self.mid is not None:
            observed["mid"] = self.mid
        self.observation = {"recipe": RECIPES[stage],
            "tensors": {k: tensor_observation(t) for k, t in observed.items()},
            "source_sha256": digest(Path(__file__))}

    def __call__(self):
        import torch
        with torch.no_grad():
            if self.stage == "silu":
                torch.ops.aten.silu.out(self.gate, out=self.output)
            elif self.stage == "mul":
                torch.ops.aten.mul.out(self.left, self.up, out=self.output)
            else:
                torch.ops.aten.silu.out(self.gate, out=self.mid)
                torch.ops.aten.mul.out(self.mid, self.up, out=self.output)
        return self.output

    def invalidate(self):
        self.output.fill_(float("nan"))
        if self.mid is not None:
            self.mid.fill_(float("nan"))


def references(torch, stage, gate, up, left):
    import torch.nn.functional as F
    if stage == "silu":
        return F.silu(gate), F.silu(gate.float())
    # Chain and Mul both multiply the rounded BF16 SiLU result by up.
    return left * up, left.float() * up.float()


def check_stage(torch, calls, inputs, snapshots, refs):
    for name, tensor in inputs.items():
        if not torch.equal(tensor.view(torch.uint8), snapshots[name]):
            raise AssertionError("Read-only input changed: " + name)
    expected = torch.nn.functional.silu(inputs["gate"])
    if not torch.equal(inputs["left"].view(torch.uint8), expected.view(torch.uint8)):
        raise AssertionError("Mul left is not the BF16 SiLU result")
    pointers = set()
    for call in calls:
        for reference in refs:
            torch.testing.assert_close(call.output.float(), reference.float(), **TOLERANCE)
        if call.mid is not None and not torch.equal(call.mid.view(torch.uint8), expected.view(torch.uint8)):
            raise AssertionError("Chain intermediate is not the BF16 SiLU result")
        if any(torch._C._overlaps(call.output, t) for t in inputs.values()):
            raise AssertionError("Stage output aliases a read-only input")
        pointers.add(call.output.data_ptr())
    if len(pointers) != len(calls):
        raise AssertionError("Stage calls share output storage")
    return {"outputs_checked": len(calls), "references_per_output": len(refs),
            "input_bytes_unchanged": True, "outputs_independent": True}


def check_copy(torch, sources, destinations, snapshot):
    if not torch.equal(sources.view(torch.uint8), snapshot):
        raise AssertionError("Copy source bytes changed")
    host = sources.cpu()
    for index in range(destinations.shape[0]):
        if not torch.equal(destinations[index].cpu().view(torch.uint8), host[index].view(torch.uint8)):
            raise AssertionError("Copy destination differs from source")
    return {"outputs_checked": int(destinations.shape[0]), "references_per_output": 1,
            "input_bytes_unchanged": True, "outputs_independent": True}


def expected_checks(case):
    refs = 1 if case["kind"] == "copy" else 2
    return {"outputs_checked": CALLS, "references_per_output": refs,
            "input_bytes_unchanged": True, "outputs_independent": True}


def observe_runtime(torch):
    from integration.flagscale.kerv_profile_runtime import driver_version
    devices = {}
    for index in (0, 1):
        props = torch.cuda.get_device_properties(index)
        devices[str(index)] = {"name": props.name, "architecture": getattr(props, "gcnArchName", None),
                               "ordinal": index}
    return {"runtime": {"torch_distribution": importlib.metadata.version("torch"),
            "torch_module_version": str(torch.__version__), "torch_git_revision": torch.version.git_version,
            "torch_extension_sha256": digest(Path(torch._C.__file__)),
            "torch_build_sha256": hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
            "accelerator": "HIP", "hip": torch.version.hip, "driver": driver_version(),
            "grad_enabled": torch.is_grad_enabled(), "autocast_enabled": torch.is_autocast_enabled("cuda"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
        "devices": devices,
        "peer_capability": {f"{a}->{b}": bool(torch.cuda.can_device_access_peer(a, b)) for a, b in ((0, 1), (1, 0))},
        "provenance": {"uuid": {str(i): str(getattr(torch.cuda.get_device_properties(i), "uuid", "unavailable"))
                                for i in (0, 1)}, "torch_extension_path": str(torch._C.__file__)}}


def identity(frozen, prepared, runtime):
    rt = runtime["runtime"]
    if rt["accelerator"] != "HIP":
        raise MeasurementContractError("HIP runtime required")
    for ordinal in ("0", "1"):
        dev = runtime["devices"][ordinal]
        if dev["ordinal"] != int(ordinal) or str(dev.get("architecture", "")).split(":")[0] != frozen["architecture"]:
            raise MeasurementContractError("Runtime must identify gfx936 GPU0/GPU1")
    if (rt.get("grad_enabled") is not False or rt.get("autocast_enabled") is not False
            or rt.get("deterministic_algorithms") is not False):
        raise MeasurementContractError("Flags differ from no_grad BF16")
    for field in ("torch_distribution", "torch_git_revision", "torch_extension_sha256",
                  "torch_build_sha256", "hip", "driver"):
        if not rt.get(field):
            raise MeasurementContractError("Missing runtime identity: " + field)
    if set(prepared) != set(frozen["cases"]):
        raise MeasurementContractError("Prepared cases differ from plan")
    contract_id = key_id(frozen["contract"])
    keys = {}
    for case_id, case in frozen["cases"].items():
        obs = prepared[case_id]
        kind = case["stage"] if case["kind"] == "stage" else "copy"
        if obs["recipe"] != RECIPES[kind] or obs["source_sha256"] != frozen["collector"]["source_sha256"][SOURCES[0]]:
            raise MeasurementContractError("Implementation changed: " + case_id)
        expected_roles = {"silu": {"gate", "up", "left", "output"}, "mul": {"gate", "up", "left", "output"},
                          "chain": {"gate", "up", "left", "output", "mid"},
                          "copy": {"source", "destination"}}[kind]
        if set(obs["tensors"]) != expected_roles:
            raise MeasurementContractError("Incomplete tensor observation: " + case_id)
        for role, tensor in obs["tensors"].items():
            device = f"cuda:{case['destination']}" if role == "destination" else (
                f"cuda:{case['source']}" if kind == "copy" else "cuda:0")
            if (tensor["shape"] != case["shape"] or tensor["dtype"] != "bf16" or tensor["layout"] != "contiguous"
                    or tensor["strides"] != strides(case["shape"])
                    or tensor["layout_provenance"] != {"kind": "runtime_tensor", "device": device,
                        "observer": "Tensor.shape/dtype/stride/is_contiguous"}):
                raise MeasurementContractError("Actual layout differs from declared case: " + case_id)
        if kind == "copy":
            if not runtime["peer_capability"][f"{case['source']}->{case['destination']}"]:
                raise MeasurementContractError("Peer capability missing; no host-staging substitute")
            devices = [runtime["devices"][str(case["source"])], runtime["devices"][str(case["destination"])]]
        else:
            devices = [runtime["devices"]["0"]]
        key = {"implementation": obs, "runtime": rt, "devices": devices, "case": case,
               "contract": frozen["contract"], "pool": frozen["pool"]}
        keys[case_id] = {"case_identity": key, "case_identity_id": key_id(key), "contract_id": contract_id}
    return {"scope": SCOPE, "cases": keys, "uuid_is_provenance_only": True,
            "peer_capability_is_not_route_certificate": True}


def quality(case, rows):
    if [r["round"] for r in rows] != list(range(ROUNDS)):
        raise MeasurementContractError("Exactly seven chronological attempts required; no replacement")
    reasons, values = [], []
    for row in rows:
        if row["checks"] != expected_checks(case):
            raise MeasurementContractError("Correctness/storage checks incomplete")
        boundary = row["timing"]["event_boundary"]
        if (boundary["device"] != case["timing_device"] or boundary["start_device"] != case["timing_device"]
                or boundary["end_device"] != case["timing_device"] or boundary["calls_per_replay"] != CALLS
                or type(boundary["stream"]) is not int):
            raise MeasurementContractError("Event boundary does not cover the declared Graph")
        elapsed = row["timing"]["elapsed_ms_per_50_calls"]
        if (row["timing"]["error"] is not None or type(elapsed) not in (float, int)
                or not math.isfinite(elapsed) or elapsed <= 0):
            reasons.append("invalid_timing_round_" + str(row["round"]))
        else:
            values.append(elapsed * 1000 / CALLS)
    if len({r["timing"]["event_boundary"]["stream"] for r in rows}) != 1:
        raise MeasurementContractError("Formal rounds changed stream")
    stats = None
    if not reasons:
        median = statistics.median(values)
        q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
        stats = {"median_us": median, "min_us": min(values), "max_us": max(values),
                 "iqr_us": q3 - q1, "iqr_over_median": (q3 - q1) / median,
                 "n_rounds": ROUNDS, "n_calls": ROUNDS * CALLS}
        if stats["iqr_over_median"] > QUALITY["max_iqr_over_median"]:
            reasons.append("iqr_over_median_exceeds_10_percent")
    return {"status": "quality_failed" if reasons else "quality_passed",
            "reasons": reasons, "statistics": stats, "samples_us": values}


def evaluate(frozen, records):
    expected = [(i, c) for i, order in enumerate(frozen["order"]) for c in order]
    if [(r["round"], r["case"]) for r in records] != expected:
        raise MeasurementContractError("Chronological attempts must match the frozen order, no retries")
    for row in records:
        if type(row["started_monotonic_ns"]) is not int or type(row["finished_monotonic_ns"]) is not int \
                or row["finished_monotonic_ns"] <= row["started_monotonic_ns"]:
            raise MeasurementContractError("Missing attempt timestamps")
    if any(a["finished_monotonic_ns"] > b["started_monotonic_ns"] for a, b in zip(records, records[1:])):
        raise MeasurementContractError("Attempts overlap or were reordered")
    results = {c: quality(case, [r for r in records if r["case"] == c]) for c, case in frozen["cases"].items()}
    passed = all(q["status"] == "quality_passed" for q in results.values())
    return {"scope": SCOPE, "cases": results, "automatic_admission": False,
            "status": "calibration_candidates_ready" if passed else "partial_quality_failed",
            "quality_passed": sorted(c for c, q in results.items() if q["status"] == "quality_passed"),
            "quality_failed": sorted(c for c, q in results.items() if q["status"] != "quality_passed"),
            "data_role": "calibration_and_validation_candidates_only", "model_parameters_updated": False}


def stage_graphs(torch, frozen, device, stream, output, context):
    generator = torch.Generator(device=device).manual_seed(frozen["seed"])
    built, prepared, warmup = {}, {}, {}
    for size in SIZES:
        dims = shape(size)
        gate, up = [torch.randn(dims, device=device, dtype=torch.bfloat16, generator=generator) for _ in range(2)]
        left = torch.empty_like(gate)
        torch.ops.aten.silu.out(gate, out=left)
        inputs = {"gate": gate, "up": up, "left": left}
        snapshots = {k: t.view(torch.uint8).clone() for k, t in inputs.items()}
        for stage in STAGES:
            case_id = f"{stage}/S{size}"
            context.update(stage="capture_and_warmup", case=case_id, memory=memory_snapshot(torch, device))
            save(output / "last_stage.json", context)
            calls = [StageCall(stage, gate, up, left) for _ in range(CALLS)]
            if any(c.observation != calls[0].observation for c in calls):
                raise MeasurementContractError("Prepared call identities disagree")
            refs = references(torch, stage, gate, up, left)
            with torch.cuda.stream(stream):
                for call in calls:
                    call()
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream, pool=None):
                for call in calls:
                    call()
            with torch.cuda.stream(stream):
                for call in calls:
                    call.invalidate()
                for _ in range(WARMUP):
                    graph.replay()
            torch.cuda.synchronize(device)
            warmup[case_id] = check_stage(torch, calls, inputs, snapshots, refs)

            def invalidate(calls=calls):
                with torch.cuda.stream(stream):
                    for call in calls:
                        call.invalidate()

            def verify(calls=calls, inputs=inputs, snapshots=snapshots, refs=refs):
                return check_stage(torch, calls, inputs, snapshots, refs)

            built[case_id] = {"graph": graph, "stream": stream, "devices": (0,), "timing_device": device,
                              "invalidate": invalidate, "verify": verify}
            prepared[case_id] = calls[0].observation
    return built, prepared, warmup


def copy_graphs(torch, frozen, output, context):
    from execution.ffn_transfer import BoundTransfer, declared_carrier
    from execution.reference_identity import tensor_observation
    built, prepared, warmup = {}, {}, {}
    for size, src, dst in COPIES:
        case_id = f"copy/{size}/{src}->{dst}"
        context.update(stage="capture_and_warmup", case=case_id)
        save(output / "last_stage.json", context)
        dims = COPY_SHAPES[size]
        generator = torch.Generator(device=f"cuda:{src}").manual_seed(frozen["seed"] + size + src)
        base = torch.randn(dims, device=f"cuda:{src}", dtype=torch.bfloat16, generator=generator)
        sources = base.unsqueeze(0).repeat((CALLS, 1, 1, 1)).contiguous()
        destinations = torch.empty_like(sources, device=f"cuda:{dst}")
        snapshot = sources.view(torch.uint8).clone()
        source_stream, stream = torch.cuda.Stream(device=src), torch.cuda.Stream(device=dst)
        fork, join = torch.cuda.Event(), torch.cuda.Event()
        spec = declared_carrier(list(dims), "bf16")
        bindings = [BoundTransfer(a, b, spec, src, dst, stream)
                    for a, b in zip(sources.unbind(0), destinations.unbind(0))]

        def body(bindings=bindings, stream=stream, source_stream=source_stream, fork=fork, join=join):
            # Torch may enqueue cross-device copy work on the source side too.
            fork.record(stream)
            source_stream.wait_event(fork)
            with torch.cuda.stream(source_stream):
                for call in bindings:
                    call()
                join.record(source_stream)
            stream.wait_event(join)

        torch.cuda.synchronize(src)
        torch.cuda.synchronize(dst)
        with torch.cuda.device(dst), torch.cuda.stream(stream):
            body()
        torch.cuda.synchronize(src)
        torch.cuda.synchronize(dst)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.device(dst), torch.cuda.graph(graph, stream=stream, pool=None):
            body()
        destinations.fill_(float("nan"))
        torch.cuda.synchronize(dst)
        with torch.cuda.device(dst), torch.cuda.stream(stream):
            for _ in range(WARMUP):
                graph.replay()
        torch.cuda.synchronize(src)
        torch.cuda.synchronize(dst)
        warmup[case_id] = check_copy(torch, sources, destinations, snapshot)

        def invalidate(destinations=destinations):
            destinations.fill_(float("nan"))

        def verify(sources=sources, destinations=destinations, snapshot=snapshot):
            return check_copy(torch, sources, destinations, snapshot)

        built[case_id] = {"graph": graph, "stream": stream, "devices": (src, dst),
                          "timing_device": torch.device(f"cuda:{dst}"), "invalidate": invalidate, "verify": verify}
        prepared[case_id] = {"recipe": RECIPES["copy"], "source_sha256": digest(Path(__file__)),
            "tensors": {"source": tensor_observation(sources[0]), "destination": tensor_observation(destinations[0])}}
    return built, prepared, warmup


def timed(torch, entry):
    """One attempt; Event readout errors are recorded, GPU execution errors propagate."""
    stream, device = entry["stream"], entry["timing_device"]
    elapsed, error, start, end = None, None, None, None
    with torch.cuda.device(device), torch.cuda.stream(stream):
        try:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(stream)
        except Exception as exc:
            error = repr(exc)
        entry["graph"].replay()
        if error is None:
            try:
                end.record(stream)
                end.synchronize()
            except Exception as exc:
                error = repr(exc)
    for d in entry["devices"]:
        torch.cuda.synchronize(d)
    if error is None:
        try:
            elapsed = float(start.elapsed_time(end))
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError("nonpositive or nonfinite Event duration")
        except Exception as exc:
            elapsed, error = None, repr(exc)
    return {"elapsed_ms_per_50_calls": elapsed, "error": error,
            "event_boundary": {"device": str(device), "stream": int(stream.cuda_stream),
                               "start_device": str(start.device) if start and error is None else str(device),
                               "end_device": str(end.device) if end and error is None else str(device),
                               "calls_per_replay": CALLS}}


def run(output, frozen, context):
    import torch
    if not torch.version.hip or not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("HIP GPU0 and GPU1 required; no CPU/eager fallback")
    instrumentation = {k: v for k, v in os.environ.items() if v and (
        k in {"ROCP_TOOL_LIB", "ROCPROFILER_TOOL_LIBRARIES", "HSA_TOOLS_LIB"}
        or (k == "LD_PRELOAD" and any(s in v.lower() for s in ("prof", "trace"))))}
    if instrumentation or sys.getprofile() is not None or sys.gettrace() is not None:
        raise RuntimeError("Known profiling/debug instrumentation detected")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    with torch.no_grad():
        runtime = observe_runtime(torch)
        save(output / "runtime.json", runtime)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        entries, prepared, warmup = stage_graphs(torch, frozen, device, stream, output, context)
        more = copy_graphs(torch, frozen, output, context)
        entries.update(more[0]); prepared.update(more[1]); warmup.update(more[2])
        save(output / "preparation.json", prepared)
        save(output / "identity.json", identity(frozen, prepared, runtime))
        save(output / "warmup.json", warmup)
        records = []
        for attempt, order in enumerate(frozen["order"]):
            for case_id in order:
                entry = entries[case_id]
                context.update(stage="measurement", case=case_id, round=attempt)
                save(output / "last_stage.json", context)
                started = time.monotonic_ns()
                entry["invalidate"]()
                for d in entry["devices"]:
                    torch.cuda.synchronize(d)
                timing = timed(torch, entry)
                checks = entry["verify"]()
                row = {"case": case_id, "round": attempt, "started_monotonic_ns": started,
                       "finished_monotonic_ns": time.monotonic_ns(), "timing": timing, "checks": checks}
                records.append(row)
                save(output / f"attempt_{len(records)-1:03d}.json", row)
                print(json.dumps(row), flush=True)
        result = {"scope": SCOPE, "complete": True, "records": records,
                  "automatic_admission": False, "known_instrumentation_guard_passed": True}
        save(output / "result.json", result)
        return evaluate(frozen, records)


def review(directory):
    directory = Path(directory)
    if (directory / "failure.json").exists():
        raise MeasurementContractError("Collection failed; retain diagnostics")
    frozen, current = read(directory / "plan.json"), plan()
    current["collector"]["git_revision"] = frozen["collector"]["git_revision"]
    if frozen != current:
        raise MeasurementContractError("Plan or source identity changed")
    process, result = read(directory / "process.json"), read(directory / "result.json")
    if (process["python_returncode"] not in (0, 1) or process["log_returncode"] != 0
            or process["automatic_admission"] is not False or result["scope"] != SCOPE
            or result["complete"] is not True or result["automatic_admission"] is not False
            or result["known_instrumentation_guard_passed"] is not True):
        raise MeasurementContractError("Incomplete process or result")
    warm = read(directory / "warmup.json")
    if warm != {c: expected_checks(case) for c, case in frozen["cases"].items()}:
        raise MeasurementContractError("Incomplete warmup checks")
    count = sum(len(o) for o in frozen["order"])
    files = sorted(directory.glob("attempt_*.json"))
    if [p.name for p in files] != [f"attempt_{i:03d}.json" for i in range(count)]:
        raise MeasurementContractError("Missing or extra attempts")
    records = [read(p) for p in files]
    if records != result["records"]:
        raise MeasurementContractError("Raw attempts differ from summary")
    replay = identity(frozen, read(directory / "preparation.json"), read(directory / "runtime.json"))
    if replay != read(directory / "identity.json"):
        raise MeasurementContractError("Case identity or contract differs")
    evaluated = evaluate(frozen, records)
    expected_return = 0 if evaluated["status"] == "calibration_candidates_ready" else 1
    if process["python_returncode"] != expected_return:
        raise MeasurementContractError("Process result differs from recomputed quality")
    evaluated.update(identity=replay, plan_sha256=digest(directory / "plan.json"),
                     raw_sha256={p.name: digest(p) for p in sorted(directory.glob("*.json"))})
    return evaluated


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "run", "review"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args(argv)
    if args.phase == "review" and args.input is None:
        parser.error("review requires --input")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(p.name != "process.log" for p in output.iterdir()):
        parser.error("Fresh output directory required")
    context = {"stage": args.phase, "pool": POOL}
    try:
        if args.phase == "review":
            evaluated = review(args.input)
            save(output / "evaluation.json", evaluated)
        else:
            frozen = plan()
            save(output / "plan.json", frozen)
            if args.phase == "plan":
                print(json.dumps({"phase": "plan", "cases": len(frozen["cases"]),
                                  "attempts": sum(len(o) for o in frozen["order"]), "new_gpu_execution": False}))
                return 0
            evaluated = run(output, frozen, context)
            save(output / "collection_quality.json", evaluated)
        print(json.dumps({"status": evaluated["status"], "quality_failed": evaluated["quality_failed"],
                          "automatic_admission": False}))
        return 0 if evaluated["status"] == "calibration_candidates_ready" else 1
    except Exception as error:
        torch = sys.modules.get("torch")
        save(output / "failure.json", {"scope": SCOPE, "error": repr(error), "traceback": traceback.format_exc(),
            "context": context, "pool": POOL, "memory": memory_snapshot(torch, "cuda:0")
            if torch and args.phase == "run" else None, "automatic_admission": False})
        raise


if __name__ == "__main__":
    sys.exit(main())
