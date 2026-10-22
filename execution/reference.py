"""Torch reference executor for an executable placed operator graph.

This module validates placement and operator semantics.  It deliberately uses
deterministic synthetic inputs and weights because placed-op-graph artifacts do
not carry model parameters.  Measurements therefore describe this operator
graph invocation, not end-to-end training throughput.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from simulator.fusion_rules import default_fusion_registry
from simulator.placement_artifacts import PlacedOpGraph
from simulator.validate import validate_fusion_plan


EXECUTION_REPORT_FORMAT = "fusion-placement-search.execution-report"
EXECUTION_REPORT_VERSION = "1.1"


@dataclass(frozen=True)
class GroupMeasurement:
    group_index: int
    node_ids: tuple[str, ...]
    device: str
    pattern_id: str
    transfer_median_s: float
    kernel_median_s: float


@dataclass(frozen=True)
class ExecutionReport:
    format: str
    version: str
    scope: str
    graph_hash: str
    hardware_hash: str
    candidate_id: str
    warmup_iterations: int
    measured_iterations: int
    devices: tuple[str, ...]
    group_timing_method: str
    predicted_makespan_s: float | None
    predicted_peak_memory_bytes: tuple[float, ...]
    total_median_s: float
    groups: tuple[GroupMeasurement, ...]
    peak_memory_bytes: tuple[int, ...]
    output_node: str
    output_shape: tuple[int, ...]
    output_dtype: str
    output_mean: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def save_execution_report(path: str | Path, report: ExecutionReport) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _torch_dtype(name: str) -> torch.dtype:
    mapping = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported tensor dtype: {name}") from exc


def _sync(devices: Sequence[torch.device]) -> None:
    for device in dict.fromkeys(devices):
        if device.type == "cuda":
            torch.cuda.synchronize(device)


class _ReferenceExecutor:
    def __init__(
        self,
        placed: PlacedOpGraph,
        devices: Sequence[str | torch.device],
        seed: int,
    ) -> None:
        self.placed = placed
        self.graph = placed.graph
        self.devices = tuple(torch.device(device) for device in devices)
        self.seed = int(seed)
        self._weights: dict[tuple[str, str], torch.Tensor] = {}
        self._inputs: dict[tuple[str, str], torch.Tensor] = {}
        self._norm_weights: dict[tuple[str, str], torch.Tensor] = {}
        self._rope_tables: dict[
            tuple[str, str], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._validate()

    def _validate(self) -> None:
        if len(self.devices) != self.placed.hardware.n_devs:
            raise ValueError(
                f"received {len(self.devices)} runtime devices, expected "
                f"{self.placed.hardware.n_devs}"
            )
        validate_fusion_plan(
            self.graph,
            self.placed.plan,
            self.placed.hardware.n_devs,
            registry=default_fusion_registry(),
            require_executable=True,
        )
        for index, group in enumerate(self.placed.plan.groups):
            if len(group.node_ids) != 1:
                raise ValueError(
                    f"reference executor currently requires singleton groups; "
                    f"group {index} has {len(group.node_ids)} nodes"
                )
        for device in self.devices:
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA/HIP runtime is unavailable")

    def _random(self, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device, salt: int) -> torch.Tensor:
        generator = torch.Generator(device=device)
        generator.manual_seed(self.seed + salt)
        return torch.randn(shape, dtype=dtype, device=device, generator=generator) * 0.02

    def _input(self, node_id: str, device: torch.device) -> torch.Tensor:
        key = (node_id, str(device))
        if key not in self._inputs:
            node = self.graph.nodes[node_id]
            shape = tuple(int(value) for value in node.extra["output_shape"])
            dtype = _torch_dtype(str(node.extra["dtype"]))
            self._inputs[key] = self._random(shape, dtype, device, salt=11)
        return self._inputs[key]

    def _weight(self, node_id: str, device: torch.device, k: int, n: int, dtype: torch.dtype) -> torch.Tensor:
        key = (node_id, str(device))
        if key not in self._weights:
            salt = 1000 + sum(ord(char) for char in node_id)
            self._weights[key] = self._random((k, n), dtype, device, salt=salt)
        return self._weights[key]

    def _rmsnorm(self, node_id: str, value: torch.Tensor) -> torch.Tensor:
        key = (node_id, str(value.device))
        width = int(value.shape[-1])
        if key not in self._norm_weights:
            self._norm_weights[key] = torch.ones(width, dtype=value.dtype, device=value.device)
        return F.rms_norm(value, (width,), self._norm_weights[key], eps=1e-6)

    def _rope(self, node_id: str, value: torch.Tensor) -> torch.Tensor:
        width = int(value.shape[-1])
        if width % 2:
            raise ValueError(f"RoPE head dimension must be even, got {width}")
        key = (node_id, str(value.device))
        if key not in self._rope_tables:
            positions = torch.arange(
                value.shape[1], device=value.device, dtype=torch.float32
            )
            frequencies = torch.arange(
                0, width, 2, device=value.device, dtype=torch.float32
            )
            frequencies = 1.0 / (1000000.0 ** (frequencies / width))
            angles = torch.outer(positions, frequencies)
            self._rope_tables[key] = (
                angles.cos().to(value.dtype)[None, :, None, :],
                angles.sin().to(value.dtype)[None, :, None, :],
            )
        cos, sin = self._rope_tables[key]
        even, odd = value[..., 0::2], value[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def _matmul(self, node_id: str, inputs: list[torch.Tensor]) -> torch.Tensor:
        node = self.graph.nodes[node_id]
        local_name = str(node.extra.get("local_name", ""))
        output_shape = tuple(int(value) for value in node.extra["output_shape"])
        if local_name.endswith("qk_matmul"):
            query, key = inputs
            repeat = query.shape[2] // key.shape[2]
            key = key.repeat_interleave(repeat, dim=2)
            return torch.matmul(query.permute(0, 2, 1, 3), key.permute(0, 2, 3, 1))
        if local_name.endswith("av_matmul"):
            probability, value = inputs
            repeat = probability.shape[1] // value.shape[2]
            value = value.repeat_interleave(repeat, dim=2).permute(0, 2, 1, 3)
            return torch.matmul(probability, value).permute(0, 2, 1, 3)

        shape = dict(node.extra["shape"])
        k, n = int(shape["k"]), int(shape["n"])
        source = inputs[0].reshape(-1, k)
        weight = self._weight(node_id, source.device, k, n, source.dtype)
        return torch.matmul(source, weight).reshape(output_shape)

    def _execute_node(self, node_id: str, inputs: list[torch.Tensor], device: torch.device) -> torch.Tensor:
        node = self.graph.nodes[node_id]
        op_type = node.op_type
        if op_type == "input":
            result = self._input(node_id, device)
        elif op_type == "matmul":
            result = self._matmul(node_id, inputs)
        elif op_type == "rmsnorm":
            result = self._rmsnorm(node_id, inputs[0])
        elif op_type == "rope":
            result = self._rope(node_id, inputs[0])
        elif op_type == "softmax":
            compute_dtype = str(node.extra.get("compute_dtype", node.extra["dtype"]))
            source = inputs[0].float() if compute_dtype == "fp32" else inputs[0]
            result = F.softmax(source, dim=-1).to(_torch_dtype(str(node.extra["dtype"])))
        elif op_type == "silu":
            result = F.silu(inputs[0])
        elif op_type == "mul":
            result = inputs[0] * inputs[1]
        elif op_type == "add":
            result = inputs[0] + inputs[1]
        else:
            raise NotImplementedError(f"reference operator is not implemented: {op_type}")

        expected = tuple(int(value) for value in node.extra["output_shape"])
        if tuple(result.shape) != expected:
            raise RuntimeError(
                f"operator {node_id} produced shape {tuple(result.shape)}, expected {expected}"
            )
        return result

    def run_once(
        self,
        measure_groups: bool,
    ) -> tuple[torch.Tensor, list[float], list[float], float]:
        outputs: dict[str, torch.Tensor] = {}
        remaining = {node_id: len(self.graph.out_edges(node_id)) for node_id in self.graph.nodes}
        transfer_times = [0.0] * len(self.placed.plan.groups)
        kernel_times = [0.0] * len(self.placed.plan.groups)
        _sync(self.devices)
        total_start = time.perf_counter()

        for index, group in enumerate(self.placed.plan.groups):
            node_id = group.node_ids[0]
            target = self.devices[group.device]
            predecessors = self.graph.predecessors(node_id)
            inputs: list[torch.Tensor] = []
            cross_device = any(outputs[item].device != target for item in predecessors)
            if measure_groups and cross_device:
                _sync(self.devices)
                transfer_start = time.perf_counter()
            for predecessor in predecessors:
                value = outputs[predecessor]
                inputs.append(value if value.device == target else value.to(target))
            if measure_groups and cross_device:
                _sync(self.devices)
                transfer_times[index] = time.perf_counter() - transfer_start

            if measure_groups and target.type == "cuda":
                with torch.cuda.device(target):
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    outputs[node_id] = self._execute_node(node_id, inputs, target)
                    end_event.record()
                    end_event.synchronize()
                    kernel_times[index] = start_event.elapsed_time(end_event) / 1e3
            else:
                kernel_start = time.perf_counter()
                outputs[node_id] = self._execute_node(node_id, inputs, target)
                if measure_groups:
                    kernel_times[index] = time.perf_counter() - kernel_start

            for predecessor in predecessors:
                remaining[predecessor] -= 1
                if remaining[predecessor] == 0:
                    del outputs[predecessor]

        _sync(self.devices)
        total = time.perf_counter() - total_start
        output_node = str(self.graph.extra.get("output_node") or self.placed.plan.groups[-1].node_ids[-1])
        return outputs[output_node], transfer_times, kernel_times, total


def execute_reference_plan(
    placed: PlacedOpGraph,
    devices: Sequence[str | torch.device],
    *,
    warmup_iterations: int = 2,
    measured_iterations: int = 5,
    seed: int = 1234,
) -> ExecutionReport:
    """Execute one placed graph and return median per-group and end-to-end timings."""

    if warmup_iterations < 0 or measured_iterations <= 0:
        raise ValueError("warmup_iterations must be non-negative and measured_iterations positive")
    executor = _ReferenceExecutor(placed, devices, seed)
    with torch.no_grad():
        for _ in range(warmup_iterations):
            executor.run_once(measure_groups=False)

        for device in executor.devices:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

        transfer_samples = [[] for _ in placed.plan.groups]
        kernel_samples = [[] for _ in placed.plan.groups]
        totals: list[float] = []
        output: torch.Tensor | None = None
        for _ in range(measured_iterations):
            output, transfers, kernels, _instrumented_total = executor.run_once(
                measure_groups=True
            )
            for index, value in enumerate(transfers):
                transfer_samples[index].append(value)
            for index, value in enumerate(kernels):
                kernel_samples[index].append(value)
            output, _transfers, _kernels, total = executor.run_once(
                measure_groups=False
            )
            totals.append(total)

    assert output is not None
    peaks = tuple(
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        for device in executor.devices
    )
    measurements = tuple(
        GroupMeasurement(
            group_index=index,
            node_ids=tuple(group.node_ids),
            device=str(executor.devices[group.device]),
            pattern_id=str(group.extra["pattern_id"]),
            transfer_median_s=float(statistics.median(transfer_samples[index])),
            kernel_median_s=float(statistics.median(kernel_samples[index])),
        )
        for index, group in enumerate(placed.plan.groups)
    )
    return ExecutionReport(
        format=EXECUTION_REPORT_FORMAT,
        version=EXECUTION_REPORT_VERSION,
        scope="synthetic-forward-operator-graph",
        graph_hash=placed.graph_hash,
        hardware_hash=placed.hardware_hash,
        candidate_id=placed.candidate_id,
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
        devices=tuple(str(device) for device in executor.devices),
        group_timing_method=(
            "cuda-event" if all(device.type == "cuda" for device in executor.devices)
            else "perf-counter"
        ),
        predicted_makespan_s=placed.predicted_makespan,
        predicted_peak_memory_bytes=tuple(placed.predicted_peak_memory),
        total_median_s=float(statistics.median(totals)),
        groups=measurements,
        peak_memory_bytes=peaks,
        output_node=str(placed.graph.extra.get("output_node") or placed.plan.groups[-1].node_ids[-1]),
        output_shape=tuple(int(value) for value in output.shape),
        output_dtype=str(output.dtype).removeprefix("torch."),
        output_mean=float(output.float().mean().cpu()),
    )
