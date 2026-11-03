"""Small LightOP/hipprof exploratory trial. No Graph capture or Profile admission."""
import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
SCOPE = "hygon_lightop_exploratory_not_profile_admission"
ARMS = ("native_packed", "lightop_packed", "native_split", "lightop_split_with_pack")
SOURCE = "https://developer.sourcefind.cn/codes/OpenDAS/vllm_cscc/-/commit/3842b316979a3ca0c56ceecb7d0faf7262653950"


def save(path, value):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def file_info(path):
    path = Path(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def plan():
    return {"scope": SCOPE, "source_reference": SOURCE,
        "source_sha256": {n: file_info(ROOT / n)["sha256"] for n in (
            "calibration/hygon_silu_trial.py", "tools/run_hygon_silu_trial.sh")},
        "device": "cuda:0", "dtype": "bf16", "logical_shape": [1, 256, 3072],
        "call_shape_packed": [256, 6144], "call_shape_output": [256, 3072],
        "binding": "lightop.silu_and_mul_opt(out, packed_input)",
        "binding_status": "project trial of a vendor API; installed public export must be verified",
        "arms": list(ARMS), "warmup_calls_per_arm": 10, "timed_calls_per_arm": 100,
        "independent_timing_batches_per_arm": 1, "seed": 1234, "rtol": .01, "atol": .001,
        "timing": "synchronized eager loop; GPU Event and synchronized host wall reported separately",
        "allocation": "native allocates intermediate/output; LightOP wrapper allocates output on every call",
        "packing": "excluded only for already-packed input; included for split-input LightOP arm",
        "validation": "one pre-timing and final output per arm; input bytes checked after warmup/timing",
        "timed_intermediate_outputs_retained": False, "profiler": "separate processes after all benchmark timing",
        "profile_arms": ["native_packed", "lightop_packed"],
        "library_original_benchmark_executed": False, "kernel_dispatch_certified": False,
        "automatic_admission": False, "followup_matrix_open": False,
        "cross_batch_uuid_equality_required": False}


def inspect_stack(output):
    """Read installed packages without explicitly initializing GPU execution."""
    record = {"scope": SCOPE, "python": sys.version, "executable": sys.executable,
        "packages": {n: version(n) for n in ("torch", "triton", "lightop")},
        "hipprof": shutil.which("hipprof"), "torch": {}, "lightop": {},
        "package_files": [], "library_test_candidates": []}
    try:
        import torch
        record["torch"] = {"module": str(torch.__version__), "hip": torch.version.hip,
                            "git_revision": torch.version.git_version, "file": torch.__file__}
    except Exception:
        record["torch"]["import_traceback"] = traceback.format_exc()
    try:
        module = importlib.import_module("lightop")
        names = sorted(n for n in dir(module) if "silu" in n.lower() or "mul" in n.lower())
        binding = getattr(module, "silu_and_mul_opt", None)
        record["lightop"] = {"file": getattr(module, "__file__", None), "symbols": names,
            "required_export_callable": callable(binding), "doc": getattr(binding, "__doc__", None)}
        try:
            record["lightop"]["signature"] = str(inspect.signature(binding))
        except (TypeError, ValueError):
            record["lightop"]["signature"] = "not introspectable"
        if module.__file__:
            record["package_files"].append(file_info(module.__file__))
        owners = metadata.packages_distributions().get("lightop", [])
        for owner in owners:
            distribution = metadata.distribution(owner)
            for entry in distribution.files or []:
                path = Path(distribution.locate_file(entry))
                name = str(entry).lower()
                if "lightop" not in name or not path.is_file():
                    continue
                if path.suffix in (".so", ".pyd", ".py"):
                    record["package_files"].append(file_info(path))
                if ("test" in name or "bench" in name) and path.suffix == ".py":
                    record["library_test_candidates"].append(str(path))
    except Exception:
        record["lightop"]["import_traceback"] = traceback.format_exc()
    save(output / "inspection.json", record)
    return record


def functions(torch, op, gate, up):
    """Same logical expression; packing cost stays in the split-input arm."""
    width = gate.shape[-1]
    packed = torch.cat((gate, up), dim=-1)
    packed_backup = packed.clone()
    gate_backup, up_backup = gate.clone(), up.clone()
    last_split_input = [None]

    def native_packed():
        return torch.nn.functional.silu(packed[..., :width]) * packed[..., width:]

    def lightop_call(value):
        out = torch.empty(value.shape[:-1] + (width,), dtype=value.dtype, device=value.device)
        op(out, value)
        return out

    def split_call():
        value = torch.cat((gate, up), dim=-1)
        last_split_input[0] = value
        return lightop_call(value)

    arms = {"native_packed": native_packed, "lightop_packed": lambda: lightop_call(packed),
        "native_split": lambda: torch.nn.functional.silu(gate) * up,
        "lightop_split_with_pack": split_call}

    def unchanged():
        pairs = [(packed, packed_backup), (gate, gate_backup), (up, up_backup)]
        if last_split_input[0] is not None:
            pairs.append((last_split_input[0], torch.cat((gate_backup, up_backup), dim=-1)))
        return all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in pairs)

    return arms, unchanged


def runtime(output):
    import torch
    module = importlib.import_module("lightop")
    op = getattr(module, "silu_and_mul_opt", None)
    if not callable(op):
        raise RuntimeError("required public LightOP export missing; inspect installed sources before changing the binding")
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("this trial requires vendor HIP PyTorch and GPU0")
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    identity = {"scope": SCOPE, "device_index": 0, "name": props.name,
        "architecture": getattr(props, "gcnArchName", None), "uuid": str(getattr(props, "uuid", "unavailable")),
        "torch": str(torch.__version__), "torch_distribution": version("torch"),
        "hip": torch.version.hip, "lightop_distribution": version("lightop"),
        "lightop_file": module.__file__, "binding": "lightop.silu_and_mul_opt",
        "pool": "ordinary PyTorch eager allocator; no Graph pool",
        "input": {"logical_shape": [1, 256, 3072], "call_shape": [256, 3072], "dtype": "bf16"}}
    save(output / "runtime_identity.json", identity)
    torch.manual_seed(1234)
    gate = torch.randn((256, 3072), device="cuda:0", dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    arms, unchanged = functions(torch, op, gate, up)
    native = torch.nn.functional.silu(gate) * up
    fp32 = (torch.nn.functional.silu(gate.float()) * up.float()).to(torch.bfloat16)

    def check(value):
        if value.data_ptr() in (gate.data_ptr(), up.data_ptr()):
            raise AssertionError("output unexpectedly aliases input")
        torch.testing.assert_close(value, native, rtol=.01, atol=.001)
        torch.testing.assert_close(value, fp32, rtol=.01, atol=.001)
        if not unchanged():
            raise AssertionError("read-only input changed")

    return torch, arms, check


def benchmark(output):
    torch, arms, check = runtime(output)
    rows = {}
    with torch.no_grad():
        for name, function in arms.items():
            save(output / "last_stage.json", {"arm": name, "stage": "correctness_and_warmup",
                "memory": memory(torch), "pool": "eager allocator; no Graph pool"})
            check(function())
            for _ in range(10):
                value = function()
            torch.cuda.synchronize(0)
            check(value)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            save(output / "last_stage.json", {"arm": name, "stage": "timing", "memory": memory(torch)})
            torch.cuda.synchronize(0)
            begin = time.perf_counter_ns()
            start.record()
            for _ in range(100):
                value = function()
            end.record()
            end.synchronize()
            torch.cuda.synchronize(0)
            finish = time.perf_counter_ns()
            batch_ms = start.elapsed_time(end)
            check(value)
            if not (batch_ms > 0):
                raise RuntimeError("invalid GPU Event duration")
            rows[name] = {"gpu_batch_ms": batch_ms, "gpu_us_per_call": batch_ms * 1000 / 100,
                "synchronized_wall_us_per_call": (finish - begin) / 1000 / 100,
                "calls": 100, "warmup_calls": 10, "numeric_ok": True, "input_bytes_unchanged": True}
            save(output / "benchmark.json", {"scope": SCOPE, "arms": rows, "complete": len(rows) == 4,
                "profiler_active": False, "library_original_benchmark": False,
                "independent_batches_per_arm": 1, "automatic_admission": False})
            print(json.dumps({"arm": name, **rows[name]}), flush=True)


def profile(output, arm):
    torch, arms, check = runtime(output)
    save(output / "last_stage.json", {"arm": arm, "stage": "profile_correctness_and_calls",
        "memory": memory(torch), "pool": "eager allocator; no Graph pool"})
    with torch.no_grad():
        check(arms[arm]())
        for _ in range(10):
            value = arms[arm]()
        torch.cuda.synchronize(0)
        check(value)
    save(output / "profile_worker.json", {"scope": SCOPE, "arm": arm, "complete": True,
        "operator_calls": 11, "fixture_and_validation_also_in_trace": True,
        "timing_labels_emitted": False, "kernel_dispatch_certified": False})


def memory(torch):
    return {name: getattr(torch.cuda, name)(0) for name in (
        "memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved")}


def run(output, launcher, command=subprocess.run):
    frozen = plan()
    save(output / "plan.json", frozen)
    jobs = []
    script = str(Path(__file__).resolve())
    inspection_dir = output / "inspection"
    inspection_dir.mkdir()
    benchmark_dir = output / "benchmark"
    benchmark_dir.mkdir()
    jobs.append(("inspection", [launcher, "-X", "faulthandler", script, "inspect", "--output", str(inspection_dir)], inspection_dir))
    jobs.append(("benchmark", [launcher, "-X", "faulthandler", script, "benchmark", "--output", str(benchmark_dir)], benchmark_dir))
    summary = {"scope": SCOPE, "jobs": {}, "pipeline_passed": False, "automatic_admission": False,
               "followup_matrix_open": False, "no_retries": True}
    save(output / "summary.json", summary)
    for name, args, directory in jobs:
        with (directory / "worker.log").open("w", encoding="utf-8") as log:
            completed = command(args, cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=False)
        summary["jobs"][name] = {"returncode": completed.returncode, "attempts": 1}
        save(output / "summary.json", summary)
        print(json.dumps({"job": name, "returncode": completed.returncode}), flush=True)
        if completed.returncode:
            return summary
        if name == "inspection":
            observed = json.loads((directory / "inspection.json").read_text(encoding="utf-8"))
            if not observed["lightop"].get("required_export_callable") or not observed["torch"].get("hip"):
                summary["blocked_reason"] = "LightOP export or HIP PyTorch unavailable; return inspection before selecting a package/version"
                save(output / "summary.json", summary)
                return summary
    profiler = shutil.which("hipprof")
    if profiler:
        for arm in ("native_packed", "lightop_packed"):
            directory = output / ("hipprof_" + arm)
            directory.mkdir()
            args = [profiler, "--hip-trace", launcher, "-X", "faulthandler", script, "profile", "--output", str(directory), "--arm", arm]
            with (directory / "worker.log").open("w", encoding="utf-8") as log:
                completed = command(args, cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=False)
            record = directory / "profile_worker.json"
            trace_files = [p for p in directory.rglob("*") if p.is_file() and p.suffix in (".csv", ".db", ".json")
                           and p.name not in ("runtime_identity.json", "profile_worker.json", "failure.json")]
            summary["jobs"]["hipprof_" + arm] = {"returncode": completed.returncode, "attempts": 1,
                "worker_complete": record.is_file(), "raw_profiler_files": [str(p.relative_to(output)) for p in trace_files],
                "trace_audit_pending": True}
            save(output / "summary.json", summary)
            if completed.returncode or not record.is_file() or not trace_files:
                summary["blocked_reason"] = "hipprof failed or did not produce a complete worker/trace; preserve all output"
                save(output / "summary.json", summary)
                return summary
    else:
        summary["blocked_reason"] = "benchmark complete, hipprof not found; profiler stage not executed"
        save(output / "summary.json", summary)
        return summary
    summary["pipeline_passed"] = True
    save(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "inspect", "run", "benchmark", "profile"), nargs="?", default="plan")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--launcher")
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.phase == "run" and not args.launcher:
        parser.error("run requires --launcher")
    if args.phase == "profile" and args.arm is None:
        parser.error("profile requires --arm")
    if args.phase in ("plan", "run"):
        args.output.mkdir(parents=True, exist_ok=False)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
    try:
        if args.phase == "plan": save(args.output / "plan.json", plan())
        elif args.phase == "inspect": inspect_stack(args.output)
        elif args.phase == "benchmark": benchmark(args.output)
        elif args.phase == "profile": profile(args.output, args.arm)
        else:
            result = run(args.output.resolve(), str(Path(args.launcher).resolve()))
            print(json.dumps(result), flush=True)
            if not result["pipeline_passed"]:
                raise SystemExit(1)
    except Exception:
        save(args.output / "failure.json", {"scope": SCOPE, "phase": args.phase,
            "traceback": traceback.format_exc(), "last_stage": str(args.output / "last_stage.json")})
        raise


if __name__ == "__main__":
    main()
