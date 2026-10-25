"""Raw E2 measurement records, validation, and resumable JSONL storage."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .e2_manifest import content_id, validate_shape_manifest


ENVIRONMENT_FORMAT = "fusion-placement-search.e2-environment"
ENVIRONMENT_VERSION = "1.0"
ENVIRONMENT_REGISTRY_FORMAT = "fusion-placement-search.e2-environment-registry"
ENVIRONMENT_REGISTRY_VERSION = "1.0"
MEASUREMENT_FORMAT = "fusion-placement-search.e2-raw-measurement"
MEASUREMENT_VERSION = "1.0"


def build_environment_record(
    *,
    hardware_class: str,
    device_name: str,
    total_memory_bytes: int,
    framework_version: str,
    accelerator_version: str,
    python_version: str,
    platform: str,
    source_revision: str,
    implementation_catalog_version: str = "e2-v1",
) -> dict[str, Any]:
    """Build a stable environment fingerprint without run-specific timestamps."""

    record = {
        "format": ENVIRONMENT_FORMAT,
        "version": ENVIRONMENT_VERSION,
        "hardware": {
            "class": str(hardware_class),
            "device_name": str(device_name),
            "total_memory_bytes": int(total_memory_bytes),
        },
        "software": {
            "framework": "pytorch",
            "framework_version": str(framework_version),
            "accelerator": "hip",
            "accelerator_version": str(accelerator_version),
            "python_version": str(python_version),
            "platform": str(platform),
        },
        "source_revision": str(source_revision),
        "implementation_catalog_version": str(implementation_catalog_version),
    }
    record["environment_id"] = content_id("measurement-environment", record)
    return record


def validate_environment_record(record: Mapping[str, Any]) -> None:
    if record.get("format") != ENVIRONMENT_FORMAT:
        raise ValueError("unexpected E2 environment format")
    unhashed = dict(record)
    claimed = unhashed.pop("environment_id", None)
    if claimed != content_id("measurement-environment", unhashed):
        raise ValueError("environment content hash mismatch")
    if int(record["hardware"]["total_memory_bytes"]) <= 0:
        raise ValueError("environment total memory must be positive")


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _timing_summary(samples_s: Iterable[float]) -> tuple[list[float], dict[str, float]]:
    samples = [float(value) for value in samples_s]
    if not samples or any(not math.isfinite(value) or value <= 0.0 for value in samples):
        raise ValueError("timing samples must be finite positive seconds")
    ordered = sorted(samples)
    return samples, {
        "median_s": float(statistics.median(samples)),
        "mean_s": float(statistics.fmean(samples)),
        "std_s": float(statistics.pstdev(samples)),
        "p10_s": _percentile(ordered, 0.10),
        "p90_s": _percentile(ordered, 0.90),
        "min_s": float(ordered[0]),
        "max_s": float(ordered[-1]),
    }


def build_raw_measurement_record(
    *,
    object_id: str,
    environment_id: str,
    physical_device_index: int,
    samples_s: Iterable[float],
    warmup: int,
    iterations_per_sample: int,
    run_id: str,
    measured_at: str,
    profiling_wall_time_s: float | None = None,
) -> dict[str, Any]:
    samples, summary = _timing_summary(samples_s)
    record = {
        "format": MEASUREMENT_FORMAT,
        "version": MEASUREMENT_VERSION,
        "object_id": str(object_id),
        "environment_id": str(environment_id),
        "physical_device_index": int(physical_device_index),
        "timing": {
            "unit": "seconds",
            "warmup": int(warmup),
            "iterations_per_sample": int(iterations_per_sample),
            "samples": samples,
            "summary": summary,
        },
        "provenance": {
            "run_id": str(run_id),
            "measured_at": str(measured_at),
            "profiling_wall_time_s": (
                float(profiling_wall_time_s) if profiling_wall_time_s is not None else None
            ),
        },
        "status": "ok",
    }
    if int(warmup) < 0 or int(iterations_per_sample) <= 0:
        raise ValueError("invalid timing repetition counts")
    record["record_id"] = content_id("raw-measurement", record)
    return record


def build_failed_measurement_record(
    *,
    object_id: str,
    environment_id: str,
    physical_device_index: int,
    run_id: str,
    measured_at: str,
    error_type: str,
    error_message: str,
    profiling_wall_time_s: float,
) -> dict[str, Any]:
    record = {
        "format": MEASUREMENT_FORMAT,
        "version": MEASUREMENT_VERSION,
        "object_id": str(object_id),
        "environment_id": str(environment_id),
        "physical_device_index": int(physical_device_index),
        "timing": None,
        "provenance": {
            "run_id": str(run_id),
            "measured_at": str(measured_at),
            "profiling_wall_time_s": float(profiling_wall_time_s),
        },
        "status": "error",
        "error": {"type": str(error_type), "message": str(error_message)},
    }
    record["record_id"] = content_id("raw-measurement", record)
    return record


def validate_raw_measurement_record(
    record: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    environments: Mapping[str, Mapping[str, Any]],
) -> None:
    if record.get("format") != MEASUREMENT_FORMAT:
        raise ValueError("unexpected E2 raw-measurement format")
    known_objects = {
        item["object_id"]
        for item in manifest["objects"]
        if bool(item["measurement"]["enabled"])
    }
    if record.get("object_id") not in known_objects:
        raise ValueError("raw measurement references unknown or disabled object")
    environment_id = record.get("environment_id")
    if environment_id not in environments:
        raise ValueError("raw measurement references unknown environment")
    validate_environment_record(environments[environment_id])
    status = record.get("status")
    if status == "ok":
        timing = record["timing"]
        samples, expected_summary = _timing_summary(timing["samples"])
        if timing.get("unit") != "seconds":
            raise ValueError("raw measurement timing unit must be seconds")
        if int(timing.get("warmup", -1)) < 0 or int(timing.get("iterations_per_sample", 0)) <= 0:
            raise ValueError("invalid timing repetition counts")
        if samples != [float(value) for value in timing["samples"]]:
            raise AssertionError("timing normalization failed")
        for name, expected in expected_summary.items():
            if not math.isclose(
                float(timing["summary"][name]), expected, rel_tol=1e-12, abs_tol=0.0
            ):
                raise ValueError(f"raw measurement summary mismatch: {name}")
    elif status == "error":
        if record.get("timing") is not None or not record.get("error", {}).get("type"):
            raise ValueError("invalid failed raw measurement record")
    else:
        raise ValueError(f"unknown raw measurement status: {status}")
    wall_time = record.get("provenance", {}).get("profiling_wall_time_s")
    if wall_time is not None and (not math.isfinite(float(wall_time)) or float(wall_time) < 0.0):
        raise ValueError("profiling wall time must be finite and non-negative")
    unhashed = dict(record)
    claimed = unhashed.pop("record_id", None)
    if claimed != content_id("raw-measurement", unhashed):
        raise ValueError("raw measurement content hash mismatch")


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return records


def append_jsonl_record(path: Path, record: Mapping[str, Any]) -> bool:
    """Append once by record_id; return False for an already persisted record."""

    record_id = record.get("record_id")
    if not record_id:
        raise ValueError("JSONL record requires record_id")
    existing = load_jsonl_records(path)
    by_id = {item.get("record_id"): item for item in existing}
    if record_id in by_id:
        if by_id[record_id] != dict(record):
            raise ValueError(f"record_id collision in {path}: {record_id}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return True


def load_environment_registry(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != ENVIRONMENT_REGISTRY_FORMAT:
        raise ValueError("unexpected E2 environment registry format")
    records = payload.get("environments", [])
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        validate_environment_record(record)
        environment_id = record["environment_id"]
        if environment_id in result:
            raise ValueError(f"duplicate environment record: {environment_id}")
        result[environment_id] = record
    return result


def upsert_environment_record(path: Path, record: Mapping[str, Any]) -> bool:
    validate_environment_record(record)
    environments = load_environment_registry(path)
    environment_id = str(record["environment_id"])
    if environment_id in environments:
        if environments[environment_id] != dict(record):
            raise ValueError(f"environment_id collision: {environment_id}")
        return False
    environments[environment_id] = dict(record)
    payload = {
        "format": ENVIRONMENT_REGISTRY_FORMAT,
        "version": ENVIRONMENT_REGISTRY_VERSION,
        "environments": [environments[key] for key in sorted(environments)],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)
    return True


def pending_measurement_object_ids(
    manifest: Mapping[str, Any],
    records: Iterable[Mapping[str, Any]],
    *,
    environment_id: str,
    physical_device_index: int | None = None,
) -> tuple[str, ...]:
    """Return enabled objects lacking a successful record in this environment."""

    validate_shape_manifest(manifest)
    enabled = {
        item["object_id"]
        for item in manifest["objects"]
        if bool(item["measurement"]["enabled"])
    }
    completed = {
        str(record["object_id"])
        for record in records
        if record.get("environment_id") == environment_id and record.get("status") == "ok"
        and (
            physical_device_index is None
            or int(record.get("physical_device_index", -1)) == int(physical_device_index)
        )
    }
    unknown = completed - enabled
    if unknown:
        raise ValueError(f"completed records contain unknown objects: {sorted(unknown)}")
    return tuple(sorted(enabled - completed))
