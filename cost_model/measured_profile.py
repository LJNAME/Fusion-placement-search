"""Measured full-shape hardware profile with analytic fallback.

The raw calibration artifact is intentionally kept separate from HardwareSpec:
HardwareSpec stores compact aggregate rates, while this profile preserves exact
operator shapes and message-size curves for ranking placed graphs.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from simulator.hardware import HardwareSpec, load_hardware
from simulator.schema import OpGraph

from .hardware_profile import AnalyticHardwareProfile, GroupCost, TransferCost
from .structure import GroupStructure


@dataclass(frozen=True)
class _KernelMeasurement:
    device: int
    op_type: str
    dtype: str
    duration: float
    output_numel: int | None = None
    inner_size: int | None = None
    m: int | None = None
    n: int | None = None
    k: int | None = None

    @property
    def is_matmul(self) -> bool:
        return self.m is not None and self.n is not None and self.k is not None


@dataclass(frozen=True)
class _LinkPoint:
    size_bytes: float
    duration: float


@dataclass(frozen=True)
class _LinkTable:
    src_device: int
    dst_device: int
    points: tuple[_LinkPoint, ...]
    fitted_bandwidth: float


def _product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def _node_dtype(node) -> str:
    return str((node.extra or {}).get("dtype", "unknown")).lower()


def _node_output_numel(node) -> int | None:
    shape = (node.extra or {}).get("output_shape")
    if shape is None:
        return None
    return _product(int(value) for value in shape)


def _node_matmul_shape(node) -> tuple[int, int, int] | None:
    shape = (node.extra or {}).get("shape") or {}
    try:
        m = int(shape["m"])
        n = int(shape["n"])
        k = int(shape["k"])
    except (KeyError, TypeError, ValueError):
        return None
    if min(m, n, k) <= 0:
        return None
    return (m, n, k)


def _raw_reference_shapes(raw: Mapping) -> dict[str, tuple[int, int | None]]:
    model = raw.get("model") or {}
    tokens = int(model["seq_length"]) * int(model["micro_batch_size"])
    hidden = int(model["hidden_size"])
    ffn = int(model["ffn_hidden_size"])
    return {
        "rmsnorm_reference": (tokens * hidden, hidden),
        "silu": (tokens * ffn, None),
        "elementwise_mul": (tokens * ffn, None),
        "residual_add": (tokens * hidden, None),
    }


def _normalize_raw_matmul_shape(item: Mapping) -> tuple[int, int, int]:
    shape = item.get("shape") or {}
    m = int(shape["m"])
    n = int(shape["n"])
    k = int(shape["k"])
    batch = int(shape.get("batch", 1))
    return (batch * m, n, k)


def _load_kernel_measurements(raw: Mapping, physical_to_logical: Mapping[int, int]) -> tuple[_KernelMeasurement, ...]:
    default_dtype = str(raw["dtype"]).lower()
    reference_shapes = _raw_reference_shapes(raw)
    op_name_to_type = {
        "rmsnorm_reference": "rmsnorm",
        "silu": "silu",
        "elementwise_mul": "mul",
        "residual_add": "add",
    }
    measurements = []
    for device in raw.get("devices") or []:
        physical = int(device["index"])
        if physical not in physical_to_logical:
            continue
        logical = int(physical_to_logical[physical])
        for item in device.get("matmul") or []:
            m, n, k = _normalize_raw_matmul_shape(item)
            measurements.append(
                _KernelMeasurement(
                    device=logical,
                    op_type="matmul",
                    dtype=str(item.get("dtype", default_dtype)).lower(),
                    duration=float(item["duration_s"]["median"]),
                    m=m,
                    n=n,
                    k=k,
                )
            )
        for item in device.get("reference_ops") or []:
            name = str(item.get("name", ""))
            # An eager expression is not evidence for a real fused pattern.
            if name not in op_name_to_type:
                continue
            fallback_numel, fallback_inner = reference_shapes[name]
            output_numel = int(item.get("output_numel", fallback_numel))
            raw_inner = item.get("inner_size", fallback_inner)
            inner_size = int(raw_inner) if raw_inner is not None else None
            measurements.append(
                _KernelMeasurement(
                    device=logical,
                    op_type=op_name_to_type[name],
                    dtype=str(item.get("dtype", default_dtype)).lower(),
                    duration=float(item["duration_s"]["median"]),
                    output_numel=output_numel,
                    inner_size=inner_size,
                )
            )
        for item in device.get("supplemental_ops") or []:
            status = str(item.get("implementation_status", ""))
            if status not in {"reference", "implemented"}:
                continue
            op_type = str(item.get("op_type", "")).lower()
            item_dtype = str(item.get("dtype", default_dtype)).lower()
            if op_type in {"matmul", "batched_matmul"}:
                m, n, k = _normalize_raw_matmul_shape(item)
                measurements.append(
                    _KernelMeasurement(
                        device=logical,
                        op_type="matmul",
                        dtype=item_dtype,
                        duration=float(item["duration_s"]["median"]),
                        m=m,
                        n=n,
                        k=k,
                    )
                )
                continue
            if op_type not in {"rmsnorm", "rope", "softmax", "silu", "mul", "add"}:
                continue
            output_numel = int(item["output_numel"])
            raw_inner = item.get("inner_size")
            inner_size = int(raw_inner) if raw_inner is not None else None
            measurements.append(
                _KernelMeasurement(
                    device=logical,
                    op_type=op_type,
                    dtype=item_dtype,
                    duration=float(item["duration_s"]["median"]),
                    output_numel=output_numel,
                    inner_size=inner_size,
                )
            )
    return tuple(measurements)


def _load_link_tables(raw: Mapping, physical_to_logical: Mapping[int, int]) -> tuple[_LinkTable, ...]:
    tables = []
    for item in raw.get("links") or []:
        src_physical = int(item["src"])
        dst_physical = int(item["dst"])
        if src_physical not in physical_to_logical or dst_physical not in physical_to_logical:
            continue
        points = tuple(
            sorted(
                (
                    _LinkPoint(
                        size_bytes=float(point["size_bytes"]),
                        duration=float(point["duration_s"]["median"]),
                    )
                    for point in item.get("points") or []
                ),
                key=lambda point: point.size_bytes,
            )
        )
        if not points:
            continue
        tables.append(
            _LinkTable(
                src_device=int(physical_to_logical[src_physical]),
                dst_device=int(physical_to_logical[dst_physical]),
                points=points,
                fitted_bandwidth=float(item["fit"]["bandwidth_bytes_per_s"]),
            )
        )
    return tuple(tables)


def _interpolate_link(table: _LinkTable, nbytes: float) -> tuple[float, bool]:
    size = max(float(nbytes), 0.0)
    points = table.points
    if size <= points[0].size_bytes:
        if points[0].size_bytes <= 0:
            return points[0].duration, False
        return points[0].duration * size / points[0].size_bytes, size != points[0].size_bytes
    if size > points[-1].size_bytes:
        tail = (size - points[-1].size_bytes) / table.fitted_bandwidth
        return points[-1].duration + tail, True
    sizes = [point.size_bytes for point in points]
    right = bisect_left(sizes, size)
    if sizes[right] == size:
        return points[right].duration, False
    left = right - 1
    lo = points[left]
    hi = points[right]
    # Interpolate latency against log(message size), which tracks the measured
    # regime transitions better than one global bandwidth/latency pair.
    lo_x = math.log2(lo.size_bytes)
    hi_x = math.log2(hi.size_bytes)
    x = math.log2(size)
    weight = (x - lo_x) / (hi_x - lo_x)
    return lo.duration + weight * (hi.duration - lo.duration), False


def _shape_distance(measurement: _KernelMeasurement, shape: tuple[int, int, int]) -> float:
    assert measurement.m is not None and measurement.n is not None and measurement.k is not None
    distance = 0.0
    for measured, requested in zip((measurement.m, measurement.n, measurement.k), shape):
        distance += abs(math.log2(float(requested) / float(measured)))
    return distance


class MeasuredHardwareProfile:
    """Full-shape/table profile backed by a calibration raw artifact."""

    def __init__(
        self,
        hardware: HardwareSpec,
        raw: Mapping,
        *,
        interpolation_radius: float = 1.0,
    ) -> None:
        if str(raw.get("format")) != "fusion-placement-search.bw1000-calibration":
            raise ValueError("unsupported measured calibration format")
        if str(raw.get("version")) != "1.0":
            raise ValueError("unsupported measured calibration version")
        if hardware.n_devs != 2:
            raise ValueError("measured BW1000 profile expects exactly two logical devices")
        source = str(raw.get("source_id", ""))
        if not source:
            raise ValueError("raw calibration is missing source_id")
        physical_to_logical = {}
        for logical, device in enumerate(hardware.devices):
            physical = int(device.extra.get("physical_index", logical))
            if physical in physical_to_logical:
                raise ValueError(f"duplicate physical device index: {physical}")
            physical_to_logical[physical] = logical
            configured_source = str(device.extra.get("source", source))
            if configured_source != source:
                raise ValueError("raw calibration and hardware config source mismatch")
        self.hardware = hardware
        self.raw = dict(raw)
        self.source_id = source
        self.interpolation_radius = float(interpolation_radius)
        self._analytic = AnalyticHardwareProfile(hardware)
        self._kernels = _load_kernel_measurements(raw, physical_to_logical)
        self._links = {
            (table.src_device, table.dst_device): table
            for table in _load_link_tables(raw, physical_to_logical)
        }
        if not self._kernels:
            raise ValueError("raw calibration contains no usable kernel measurements")
        if len(self._links) != 2:
            raise ValueError("raw calibration must contain both link directions")

    @classmethod
    def from_files(
        cls,
        hardware_path: str | Path,
        raw_path: str | Path,
        *,
        interpolation_radius: float = 1.0,
    ) -> "MeasuredHardwareProfile":
        hardware = load_hardware(hardware_path)
        with open(raw_path, "r", encoding="utf-8") as stream:
            raw = json.load(stream)
        return cls(hardware, raw, interpolation_radius=interpolation_radius)

    @property
    def n_devs(self) -> int:
        return self.hardware.n_devs

    @property
    def overlap_send(self) -> bool:
        return bool(self.hardware.overlap_send)

    @property
    def interference(self) -> float:
        return float(self.hardware.interference)

    def max_memory(self, device: int) -> float:
        return float(self.hardware.devices[device].max_mem)

    def _exact_kernel(self, device: int, node) -> _KernelMeasurement | None:
        dtype = _node_dtype(node)
        if node.op_type == "matmul":
            shape = _node_matmul_shape(node)
            if shape is None:
                return None
            for item in self._kernels:
                if (
                    item.device == device
                    and item.op_type == "matmul"
                    and item.dtype == dtype
                    and (item.m, item.n, item.k) == shape
                ):
                    return item
            return None
        output_numel = _node_output_numel(node)
        output_shape = (node.extra or {}).get("output_shape") or ()
        inner_size = int(output_shape[-1]) if output_shape else None
        for item in self._kernels:
            if (
                item.device == device
                and item.op_type == node.op_type
                and item.dtype == dtype
                and item.output_numel == output_numel
                and (node.op_type != "rmsnorm" or item.inner_size == inner_size)
            ):
                return item
        return None

    def _interpolated_matmul(self, device: int, node) -> tuple[float, float] | None:
        shape = _node_matmul_shape(node)
        if shape is None:
            return None
        dtype = _node_dtype(node)
        candidates = [
            item
            for item in self._kernels
            if item.device == device and item.op_type == "matmul" and item.dtype == dtype
        ]
        if not candidates:
            return None
        nearest = min(candidates, key=lambda item: _shape_distance(item, shape))
        distance = _shape_distance(nearest, shape)
        if distance > self.interpolation_radius:
            return None
        assert nearest.m is not None and nearest.n is not None and nearest.k is not None
        measured_flops = 2.0 * nearest.m * nearest.n * nearest.k
        throughput = measured_flops / nearest.duration
        requested_flops = 2.0 * shape[0] * shape[1] * shape[2]
        return requested_flops / throughput, distance

    def estimate_group(self, graph: OpGraph, group: GroupStructure) -> GroupCost:
        if len(group.node_ids) == 1:
            node = graph.nodes[group.node_ids[0]]
            if node.op_type == "input":
                return GroupCost(0.0, 0.0, 0.0, 0.0, 1.0, "logical_input")
            exact = self._exact_kernel(group.device, node)
            if exact is not None:
                launch = min(float(self.hardware.launch_overhead), exact.duration)
                return GroupCost(
                    duration=exact.duration,
                    compute_time=max(0.0, exact.duration - launch),
                    memory_time=0.0,
                    launch_time=launch,
                    confidence=0.95,
                    source="measured_exact_shape",
                )
            if node.op_type == "matmul":
                interpolated = self._interpolated_matmul(group.device, node)
                if interpolated is not None:
                    duration, distance = interpolated
                    launch = min(float(self.hardware.launch_overhead), duration)
                    return GroupCost(
                        duration=duration,
                        compute_time=max(0.0, duration - launch),
                        memory_time=0.0,
                        launch_time=launch,
                        confidence=0.65,
                        source="measured_nearest_shape",
                        ood_reasons=(
                            f"kernel:{group.device}:{node.id}:shape_interpolated:{distance:.3f}",
                        ),
                    )

        fallback = self._analytic.estimate_group(graph, group)
        reasons = set(fallback.ood_reasons)
        if group.implementation_status in {"planned", "implemented"} and len(group.node_ids) > 1:
            reasons.add(f"pattern:{group.pattern_id}:no_measured_fused_kernel")
            confidence = min(fallback.confidence, 0.15)
        else:
            reasons.add(f"kernel:{group.device}:{group.group_id}:no_measured_shape")
            confidence = min(fallback.confidence, 0.25)
        return GroupCost(
            duration=fallback.duration,
            compute_time=fallback.compute_time,
            memory_time=fallback.memory_time,
            launch_time=fallback.launch_time,
            confidence=confidence,
            source="analytic_fallback",
            ood_reasons=tuple(sorted(reasons)),
        )

    def estimate_transfer(self, src_device: int, dst_device: int, nbytes: float) -> TransferCost:
        table = self._links.get((int(src_device), int(dst_device)))
        if table is None:
            fallback = self._analytic.estimate_transfer(src_device, dst_device, nbytes)
            reasons = set(fallback.ood_reasons)
            reasons.add(f"link:{src_device}-{dst_device}:no_measured_direction")
            return TransferCost(
                fallback.duration,
                min(fallback.confidence, 0.2),
                "analytic_link_fallback",
                tuple(sorted(reasons)),
            )
        duration, extrapolated = _interpolate_link(table, nbytes)
        reasons = (
            (f"link:{src_device}-{dst_device}:size_extrapolated",)
            if extrapolated
            else ()
        )
        return TransferCost(
            duration=float(duration),
            confidence=0.7 if extrapolated else 0.95,
            source="measured_link_table",
            ood_reasons=reasons,
        )

    def link_key(self, src_device: int, dst_device: int) -> tuple[int, int]:
        return (
            (src_device, dst_device)
            if src_device <= dst_device
            else (dst_device, src_device)
        )


__all__ = ["MeasuredHardwareProfile"]
