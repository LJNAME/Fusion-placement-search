"""Build the E2 residual-learning dataset and enforce label visibility.

The persisted JSONL file is an oracle-owned label store.  Callers that take
part in prediction or sampling must use :func:`make_dataset_view`; it masks
hidden labels according to the strict split contract before returning data.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from simulator.hardware import HardwareSpec, load_hardware
from simulator.schema import OpNode
from simulator.shape_model import compute_efficiency

from .e2_contract import validate_split_contract
from .e2_manifest import content_id, validate_shape_manifest
from .e2_records import (
    load_environment_registry,
    load_jsonl_records,
    validate_raw_measurement_record,
)


DATASET_FORMAT = "fusion-placement-search.e2-profiling-dataset"
DATASET_VERSION = "1.0"
DATASET_RECORD_FORMAT = "fusion-placement-search.e2-profiling-record"


def _numel(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def _shape_dimensions(kernel_key: Mapping[str, Any]) -> dict[str, int]:
    shape = kernel_key["shape"]
    dimensions = {
        "batch": int(shape.get("batch", 1)),
        "m": int(shape.get("m", 0)),
        "n": int(shape.get("n", 0)),
        "k": int(shape.get("k", 0)),
        "normalized_size": int(shape.get("normalized_size", 0)),
    }
    input_shape = [int(value) for value in shape.get("input", ())]
    dimensions["numel"] = int(
        shape.get(
            "numel",
            _numel(input_shape) if input_shape else dimensions["batch"] * dimensions["m"] * dimensions["n"],
        )
    )
    for index in range(4):
        dimensions[f"d{index}"] = input_shape[index] if index < len(input_shape) else 0
    return dimensions


def _operation_flops(kernel_key: Mapping[str, Any], dimensions: Mapping[str, int]) -> float:
    operator = str(kernel_key["operator"])
    if operator == "matmul":
        return float(
            2
            * dimensions["batch"]
            * dimensions["m"]
            * dimensions["n"]
            * dimensions["k"]
        )
    coefficients = {
        "rmsnorm": 5.0,
        "rope": 4.0,
        "softmax": 5.0,
        "silu": 4.0,
        "mul": 1.0,
        "add": 1.0,
    }
    return float(dimensions["numel"]) * coefficients.get(operator, 1.0)


def _mean_hardware(hw: HardwareSpec) -> tuple[float, float, float]:
    return (
        statistics.fmean(device.flops for device in hw.devices),
        statistics.fmean(device.mem_bw for device in hw.devices),
        float(hw.launch_overhead),
    )


def analytic_kernel_estimate(
    manifest_object: Mapping[str, Any], hw: HardwareSpec
) -> dict[str, float | str]:
    """Return a hardware-normalized roofline prior for one measured kernel."""

    kernel_key = manifest_object["kernel_key"]
    dimensions = _shape_dimensions(kernel_key)
    flops = _operation_flops(kernel_key, dimensions)
    memory_bytes = float(manifest_object["measurement"]["estimated_memory_bytes"])
    peak_flops, memory_bandwidth, launch_s = _mean_hardware(hw)
    node = OpNode(
        id="measured-kernel",
        op_type=str(kernel_key["operator"]),
        flops=flops,
        out_bytes=float(manifest_object["measurement"]["estimated_output_bytes"]),
        extra={"shape": dict(kernel_key["shape"])},
    )
    efficiency = float(compute_efficiency(hw, node))
    compute_s = flops / (peak_flops * efficiency) if flops > 0.0 and efficiency > 0.0 else 0.0
    memory_s = memory_bytes / memory_bandwidth
    estimate_s = max(compute_s, memory_s) + launch_s
    return {
        "source": "hardware-class-mean-shape-aware-roofline",
        "flops": flops,
        "memory_bytes": memory_bytes,
        "arithmetic_intensity": flops / memory_bytes if memory_bytes > 0.0 else 0.0,
        "peak_flops_per_s": peak_flops,
        "memory_bandwidth_bytes_per_s": memory_bandwidth,
        "compute_efficiency": efficiency,
        "compute_s": compute_s,
        "memory_s": memory_s,
        "launch_s": launch_s,
        "estimate_s": estimate_s,
    }


def _split_roles(contract: Mapping[str, Any], object_id: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for split in contract["splits"]:
        if object_id in split["hidden_object_ids"]:
            role = "hidden"
        elif object_id in split["shared_seen_object_ids"]:
            role = "shared_seen"
        elif object_id in split["train_object_ids"]:
            role = "train"
        else:
            role = "unused"
        result[split["split_id"]] = role
    return result


def _feature_record(
    manifest_object: Mapping[str, Any], analytic: Mapping[str, Any], hw: HardwareSpec
) -> dict[str, Any]:
    kernel_key = manifest_object["kernel_key"]
    dimensions = _shape_dimensions(kernel_key)
    usages = manifest_object["usages"]
    occurrence_count = sum(int(usage["occurrence_count"]) for usage in usages)
    numeric = {
        **dimensions,
        "flops": float(analytic["flops"]),
        "memory_bytes": float(analytic["memory_bytes"]),
        "arithmetic_intensity": float(analytic["arithmetic_intensity"]),
        "compute_demand_s": float(analytic["flops"]) / float(analytic["peak_flops_per_s"]),
        "memory_demand_s": float(analytic["memory_bytes"])
        / float(analytic["memory_bandwidth_bytes_per_s"]),
        "launch_s": float(analytic["launch_s"]),
        "usage_occurrences": occurrence_count,
        "usage_models": len({usage["model_id"] for usage in usages}),
        "usage_sequence_lengths": len({int(usage["sequence_length"]) for usage in usages}),
    }
    log_numeric = {
        f"log1p_{name}": math.log1p(float(value))
        for name, value in numeric.items()
        if float(value) >= 0.0
    }
    return {
        "categorical": {
            "operator": kernel_key["operator"],
            "implementation_id": kernel_key["implementation"]["id"],
            "input_dtype": kernel_key["dtype"]["input"],
            "compute_dtype": kernel_key["dtype"]["compute"],
            "output_dtype": kernel_key["dtype"]["output"],
            "hardware_class": kernel_key["hardware"]["class"],
        },
        "numeric": numeric,
        "log_numeric": log_numeric,
        "hardware": {
            "device_count": hw.n_devs,
            "peak_flops_per_s": float(analytic["peak_flops_per_s"]),
            "memory_bandwidth_bytes_per_s": float(analytic["memory_bandwidth_bytes_per_s"]),
            "launch_s": float(analytic["launch_s"]),
        },
    }


def build_profiling_dataset(
    manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
    environments: Mapping[str, Mapping[str, Any]],
    raw_records: Sequence[Mapping[str, Any]],
    hw: HardwareSpec,
    *,
    require_complete: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Aggregate raw replicas and build one residual-learning row per object."""

    validate_shape_manifest(manifest)
    validate_split_contract(contract, manifest)
    enabled = {
        item["object_id"]: item
        for item in manifest["objects"]
        if bool(item["measurement"]["enabled"])
    }
    by_object: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for raw in raw_records:
        validate_raw_measurement_record(raw, manifest=manifest, environments=environments)
        if raw["status"] == "ok":
            by_object[str(raw["object_id"])].append(raw)
    missing = sorted(set(enabled) - set(by_object))
    if require_complete and missing:
        raise ValueError(f"profiling dataset is missing {len(missing)} enabled objects")

    records: list[dict[str, Any]] = []
    for object_id in sorted(by_object):
        item = enabled[object_id]
        source_records = sorted(by_object[object_id], key=lambda record: record["record_id"])
        medians = [float(record["timing"]["summary"]["median_s"]) for record in source_records]
        measured_s = float(statistics.median(medians))
        analytic = analytic_kernel_estimate(item, hw)
        estimate_s = float(analytic["estimate_s"])
        record = {
            "format": DATASET_RECORD_FORMAT,
            "version": DATASET_VERSION,
            "object_id": object_id,
            "kernel_key": item["kernel_key"],
            "features": _feature_record(item, analytic, hw),
            "analytic": analytic,
            "measurement": {
                "duration_s": measured_s,
                "aggregation": "median-of-record-medians",
                "replicas": len(source_records),
                "sample_count": sum(len(record["timing"]["samples"]) for record in source_records),
                "physical_device_indices": sorted(
                    {int(record["physical_device_index"]) for record in source_records}
                ),
                "environment_ids": sorted({str(record["environment_id"]) for record in source_records}),
                "raw_record_ids": [str(record["record_id"]) for record in source_records],
            },
            "target": {
                "ratio": measured_s / estimate_s,
                "log_residual": math.log(measured_s / estimate_s),
            },
            "usages": item["usages"],
            "split_roles": _split_roles(contract, object_id),
        }
        record["record_id"] = content_id("profiling-dataset-record", record)
        records.append(record)

    metadata = {
        "format": DATASET_FORMAT,
        "version": DATASET_VERSION,
        "manifest_id": manifest["manifest_id"],
        "contract_id": contract["contract_id"],
        "hardware_profile": hw.name,
        "aggregation": "one object row; median of raw-record medians",
        "label_target": "log(measured_duration_s / analytic_estimate_s)",
        "access_rule": "persisted labels are oracle-owned; predictor and sampler must use make_dataset_view",
        "record_count": len(records),
        "missing_enabled_object_ids": missing,
        "record_ids": [record["record_id"] for record in records],
    }
    metadata["dataset_id"] = content_id("profiling-dataset", metadata)
    return metadata, records


def make_dataset_view(
    records: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    *,
    split_id: str,
    mode: str,
    role: str,
    revealed_object_ids: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Return all feature rows while masking labels forbidden to a caller."""

    matches = [split for split in contract["splits"] if split["split_id"] == split_id]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate split_id: {split_id}")
    split = matches[0]
    train = set(split["train_object_ids"])
    hidden = set(split["hidden_object_ids"])
    revealed = set(revealed_object_ids)
    if mode not in {"zero_shot", "active_calibration"}:
        raise ValueError(f"unknown E2 mode: {mode}")
    if role not in {"predictor", "sampler", "oracle_evaluator"}:
        raise ValueError(f"unknown E2 role: {role}")
    if mode == "zero_shot" and revealed:
        raise ValueError("zero-shot mode forbids revealing hidden labels")
    if not revealed <= hidden:
        raise ValueError("active calibration can reveal hidden objects only")

    visible_objects = set(record["object_id"] for record in records) if role == "oracle_evaluator" else train
    if role != "oracle_evaluator" and mode == "active_calibration":
        visible_objects |= revealed
    view: list[dict[str, Any]] = []
    for source in records:
        row = copy.deepcopy(dict(source))
        visible = row["object_id"] in visible_objects
        if not visible:
            row.pop("measurement", None)
            row.pop("target", None)
        row["label_access"] = {
            "split_id": split_id,
            "mode": mode,
            "role": role,
            "visible": visible,
        }
        view.append(row)
    return view


def write_profiling_dataset(
    output: Path, metadata_output: Path, metadata: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    with metadata_output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--environments", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--hardware", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    environments = load_environment_registry(args.environments)
    raw_records = load_jsonl_records(args.raw)
    metadata, records = build_profiling_dataset(
        manifest,
        contract,
        environments,
        raw_records,
        load_hardware(args.hardware),
    )
    metadata_output = args.metadata_output or args.output.with_suffix(".meta.json")
    write_profiling_dataset(args.output, metadata_output, metadata, records)
    operators: dict[str, int] = defaultdict(int)
    for record in records:
        operators[str(record["kernel_key"]["operator"])] += 1
    print(f"records: {len(records)}")
    print("operators: " + ",".join(f"{key}={operators[key]}" for key in sorted(operators)))
    print(f"dataset_id: {metadata['dataset_id']}")
    print(f"output: {args.output}")
    print(f"metadata: {metadata_output}")
    print("Fusion E2b profiling dataset: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
