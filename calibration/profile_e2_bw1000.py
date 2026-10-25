"""Profile E2 manifest objects on one homogeneous BW1000 device class."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - exercised on the GPU server
    torch = None
    F = None

from .e2_manifest import validate_shape_manifest
from .e2_records import (
    append_jsonl_record,
    build_environment_record,
    build_failed_measurement_record,
    build_raw_measurement_record,
    load_environment_registry,
    load_jsonl_records,
    upsert_environment_record,
    validate_raw_measurement_record,
)


def parse_int_csv(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("expected non-negative comma-separated device indices")
    if len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("device indices must be unique")
    return result


def _source_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _dtype(name: str):
    mapping = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unsupported manifest dtype: {name}") from exc


def _new_tensor(shape: Sequence[int], *, device: int, dtype):
    return torch.empty(tuple(int(value) for value in shape), device=device, dtype=dtype).normal_()


def _prepare_operation(item: Mapping[str, Any], device: int) -> Callable[[], object]:
    key = item["kernel_key"]
    implementation_id = key["implementation"]["id"]
    shape = key["shape"]
    input_dtype = _dtype(key["dtype"]["input"])
    with torch.cuda.device(device):
        seed = int(item["measurement"]["recipe"]["input_initialization"]["seed"])
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        if implementation_id == "torch.mm.reference":
            left = _new_tensor((shape["m"], shape["k"]), device=device, dtype=input_dtype)
            right = _new_tensor((shape["k"], shape["n"]), device=device, dtype=input_dtype)
            return lambda: torch.mm(left, right)

        if implementation_id == "torch.bmm.reference":
            left = _new_tensor(
                (shape["batch"], shape["m"], shape["k"]),
                device=device,
                dtype=input_dtype,
            )
            right = _new_tensor(
                (shape["batch"], shape["k"], shape["n"]),
                device=device,
                dtype=input_dtype,
            )
            return lambda: torch.bmm(left, right)

        input_shape = tuple(int(value) for value in shape["input"])
        value = _new_tensor(input_shape, device=device, dtype=input_dtype)
        if implementation_id == "torch.rms_norm.reference":
            normalized_size = int(shape["normalized_size"])
            weight = torch.ones(normalized_size, device=device, dtype=input_dtype)
            epsilon = float(key["runtime_flags"]["epsilon"])
            return lambda: F.rms_norm(value, (normalized_size,), weight, epsilon)

        if implementation_id == "fusion.rope.reference":
            if len(input_shape) != 4 or input_shape[-1] % 2:
                raise ValueError(f"RoPE expects [batch, sequence, heads, even head_dim]: {input_shape}")
            sequence = input_shape[1]
            head_dim = input_shape[-1]
            rotary_base = float(key["runtime_flags"]["rotary_base"])
            positions = torch.arange(sequence, device=device, dtype=torch.float32)
            frequencies = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
            frequencies = 1.0 / (rotary_base ** (frequencies / head_dim))
            angles = positions[:, None] * frequencies[None, :]
            cos = angles.cos().to(dtype=input_dtype)[None, :, None, :]
            sin = angles.sin().to(dtype=input_dtype)[None, :, None, :]

            def rope():
                even = value[..., 0::2]
                odd = value[..., 1::2]
                rotated_even = even * cos - odd * sin
                rotated_odd = even * sin + odd * cos
                return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)

            return rope

        if implementation_id == "torch.softmax.fp32_internal.reference":
            dim = int(key["runtime_flags"]["dim"])
            output_dtype = _dtype(key["dtype"]["output"])
            return lambda: F.softmax(value.float(), dim=dim).to(dtype=output_dtype)
        if implementation_id == "torch.silu.reference":
            return lambda: F.silu(value)
        if implementation_id == "torch.mul.reference":
            other = _new_tensor(input_shape, device=device, dtype=input_dtype)
            return lambda: value * other
        if implementation_id == "torch.add.reference":
            other = _new_tensor(input_shape, device=device, dtype=input_dtype)
            return lambda: value + other
    raise ValueError(f"unsupported measurement implementation: {implementation_id}")


def event_time_loop(
    operation: Callable[[], object],
    *,
    device: int,
    warmup: int,
    iterations: int,
    samples: int,
) -> list[float]:
    if warmup < 0 or iterations <= 0 or samples <= 0:
        raise ValueError("invalid timing counts")
    with torch.cuda.device(device):
        for _ in range(warmup):
            operation()
        torch.cuda.synchronize(device)
        durations: list[float] = []
        for _ in range(samples):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device)
            start.record()
            for _ in range(iterations):
                operation()
            end.record()
            end.synchronize()
            durations.append(float(start.elapsed_time(end)) / 1000.0 / iterations)
    return durations


def plan_measurements(
    manifest: Mapping[str, Any],
    records: Iterable[Mapping[str, Any]],
    *,
    environment_by_device: Mapping[int, str],
    assignment: str,
    resume: bool,
    include_ops: set[str] | None = None,
    limit: int | None = None,
) -> tuple[tuple[int, Mapping[str, Any]], ...]:
    devices = tuple(sorted(environment_by_device))
    if not devices:
        raise ValueError("measurement planning requires at least one device")
    if assignment not in {"shard", "replicate"}:
        raise ValueError(f"unknown device assignment: {assignment}")
    successful = {
        (
            str(record.get("object_id")),
            str(record.get("environment_id")),
            int(record.get("physical_device_index", -1)),
        )
        for record in records
        if record.get("status") == "ok"
    }
    successful_by_environment = {(object_id, environment_id) for object_id, environment_id, _ in successful}
    objects = [
        item
        for item in manifest["objects"]
        if bool(item["measurement"]["enabled"])
        and (include_ops is None or item["kernel_key"]["operator"] in include_ops)
    ]
    if assignment == "shard":
        if resume:
            objects = [
                item
                for item in objects
                if not any(
                    (item["object_id"], environment_by_device[device])
                    in successful_by_environment
                    for device in devices
                )
            ]
        if limit is not None:
            objects = objects[:limit]
        return tuple((devices[index % len(devices)], item) for index, item in enumerate(objects))

    pairs = [(device, item) for item in objects for device in devices]
    if resume:
        pairs = [
            (device, item)
            for device, item in pairs
            if (item["object_id"], environment_by_device[device], device) not in successful
        ]
    if limit is not None:
        pairs = pairs[:limit]
    return tuple(pairs)


def _validate_runtime(manifest: Mapping[str, Any], devices: Sequence[int]) -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for E2 profiling")
    if not torch.cuda.is_available():
        raise RuntimeError("no HIP/CUDA-compatible accelerator is visible")
    if max(devices) >= torch.cuda.device_count():
        raise RuntimeError(
            f"requested devices {list(devices)}, visible count is {torch.cuda.device_count()}"
        )
    expected = manifest["environment_contract"]
    actual_torch = str(torch.__version__).split("+")[0]
    actual_hip = str(torch.version.hip)
    if actual_torch != str(expected["framework_version"]):
        raise RuntimeError(
            f"PyTorch version mismatch: manifest={expected['framework_version']} runtime={actual_torch}"
        )
    if actual_hip != str(expected["accelerator_version"]):
        raise RuntimeError(
            f"HIP version mismatch: manifest={expected['accelerator_version']} runtime={actual_hip}"
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/bw1000_e2/shape_manifest.json")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/bw1000_e2"))
    parser.add_argument("--devices", type=parse_int_csv, default=(0, 1))
    parser.add_argument("--assignment", choices=("shard", "replicate"), default="shard")
    parser.add_argument("--include-ops", default="")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--fail-fast", action="store_true")
    parser.set_defaults(resume=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_shape_manifest(manifest)
    _validate_runtime(manifest, args.devices)

    suffix = ".quick" if args.quick else ""
    raw_path = args.out_dir / f"raw_measurements{suffix}.jsonl"
    environments_path = args.out_dir / f"environments{suffix}.json"
    source_revision = _source_revision()
    environment_by_device: dict[int, str] = {}
    environments: dict[str, Mapping[str, Any]] = load_environment_registry(environments_path)
    for device in args.devices:
        properties = torch.cuda.get_device_properties(device)
        environment = build_environment_record(
            hardware_class=manifest["environment_contract"]["hardware_class"],
            device_name=str(properties.name),
            total_memory_bytes=int(properties.total_memory),
            framework_version=str(torch.__version__).split("+")[0],
            accelerator_version=str(torch.version.hip),
            python_version=sys.version,
            platform=platform.platform(),
            source_revision=source_revision,
        )
        upsert_environment_record(environments_path, environment)
        environments[environment["environment_id"]] = environment
        environment_by_device[int(device)] = environment["environment_id"]

    records = load_jsonl_records(raw_path)
    for record in records:
        validate_raw_measurement_record(record, manifest=manifest, environments=environments)
    include_ops = {value.strip() for value in args.include_ops.split(",") if value.strip()} or None
    assignments = plan_measurements(
        manifest,
        records,
        environment_by_device=environment_by_device,
        assignment=args.assignment,
        resume=args.resume,
        include_ops=include_ops,
        limit=args.limit,
    )
    run_id = (
        f"{socket.gethostname()}:{datetime.now().astimezone().isoformat()}:"
        f"{uuid.uuid4().hex[:8]}"
    )
    print(f"manifest_id={manifest['manifest_id']}")
    print(f"source_revision={source_revision}")
    print(f"devices={list(args.devices)} assignment={args.assignment} quick={args.quick}")
    print(f"scheduled={len(assignments)} existing_records={len(records)}")
    if not assignments:
        print("no pending measurement objects")
        print("Fusion E2a BW1000 profiling: OK")
        return 0

    failures = 0
    for index, (device, item) in enumerate(assignments, 1):
        key = item["kernel_key"]
        recipe = item["measurement"]["recipe"]
        warmup = 2 if args.quick else int(recipe["warmup"])
        iterations = 5 if args.quick else int(recipe["iterations_per_sample"])
        samples = 3 if args.quick else int(recipe["samples"])
        object_id = item["object_id"]
        started = time.perf_counter()
        measured_at = datetime.now().astimezone().isoformat()
        operation = None
        try:
            free_bytes, _ = torch.cuda.mem_get_info(device)
            estimate = int(item["measurement"]["estimated_memory_bytes"])
            if estimate > int(free_bytes * 0.80):
                raise MemoryError(
                    f"estimated working set {estimate} exceeds 80% of free memory {free_bytes}"
                )
            operation = _prepare_operation(item, device)
            durations = event_time_loop(
                operation,
                device=device,
                warmup=warmup,
                iterations=iterations,
                samples=samples,
            )
            wall_time = time.perf_counter() - started
            record = build_raw_measurement_record(
                object_id=object_id,
                environment_id=environment_by_device[device],
                physical_device_index=device,
                samples_s=durations,
                warmup=warmup,
                iterations_per_sample=iterations,
                run_id=run_id,
                measured_at=measured_at,
                profiling_wall_time_s=wall_time,
            )
            validate_raw_measurement_record(record, manifest=manifest, environments=environments)
            append_jsonl_record(raw_path, record)
            median_us = record["timing"]["summary"]["median_s"] * 1e6
            print(
                f"[{index}/{len(assignments)}] device={device} op={key['operator']} "
                f"impl={key['implementation']['id']} median={median_us:.3f} us"
            )
        except Exception as exc:
            failures += 1
            wall_time = time.perf_counter() - started
            failure = build_failed_measurement_record(
                object_id=object_id,
                environment_id=environment_by_device[device],
                physical_device_index=device,
                run_id=run_id,
                measured_at=measured_at,
                error_type=type(exc).__name__,
                error_message=str(exc),
                profiling_wall_time_s=wall_time,
            )
            validate_raw_measurement_record(failure, manifest=manifest, environments=environments)
            append_jsonl_record(raw_path, failure)
            print(
                f"[{index}/{len(assignments)}] device={device} op={key['operator']} "
                f"ERROR {type(exc).__name__}: {exc}"
            )
            if args.fail_fast:
                raise
        finally:
            operation = None
            if torch is not None and torch.cuda.is_available():
                try:
                    torch.cuda.synchronize(device)
                except Exception:
                    pass
                gc.collect()
                torch.cuda.empty_cache()

    print(f"raw_measurements={raw_path}")
    print(f"environments={environments_path}")
    print(f"completed={len(assignments) - failures} failures={failures}")
    if failures:
        print("Fusion E2a BW1000 profiling: INCOMPLETE")
        return 1
    print("Fusion E2a BW1000 profiling: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
