"""Hardware-specific latency boundary used by the structural cost model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple, runtime_checkable

from simulator.hardware import HardwareSpec
from simulator.schema import OpGraph
from simulator.shape_model import group_compute_time

from .structure import GroupStructure


@dataclass(frozen=True)
class GroupCost:
    duration: float
    compute_time: float
    memory_time: float
    launch_time: float
    confidence: float
    source: str
    ood_reasons: Tuple[str, ...] = ()


@dataclass(frozen=True)
class TransferCost:
    duration: float
    confidence: float
    source: str
    ood_reasons: Tuple[str, ...] = ()


@runtime_checkable
class HardwareCostProfile(Protocol):
    @property
    def n_devs(self) -> int:
        ...

    @property
    def overlap_send(self) -> bool:
        ...

    @property
    def interference(self) -> float:
        ...

    def max_memory(self, device: int) -> float:
        ...

    def estimate_group(
        self,
        graph: OpGraph,
        group: GroupStructure,
    ) -> GroupCost:
        ...

    def estimate_transfer(
        self,
        src_device: int,
        dst_device: int,
        nbytes: float,
    ) -> TransferCost:
        ...

    def link_key(self, src_device: int, dst_device: int) -> tuple[int, int]:
        ...


class AnalyticHardwareProfile:
    """Roofline/shape-curve fallback backed by the existing HardwareSpec."""

    def __init__(self, hardware: HardwareSpec) -> None:
        self.hardware = hardware

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

    def _device_calibrated(self, device: int) -> bool:
        spec = self.hardware.devices[device]
        return bool(spec.extra.get("calibrated", self.hardware.extra.get("calibrated", False)))

    def estimate_group(
        self,
        graph: OpGraph,
        group: GroupStructure,
    ) -> GroupCost:
        device = self.hardware.devices[group.device]
        compute_time = group_compute_time(
            graph,
            group.node_ids,
            device,
            self.hardware,
        )
        memory_time = (group.input_bytes + group.output_bytes) / device.mem_bw
        launch_time = self.hardware.launch_overhead + self.hardware.fusion_overhead * max(
            0, len(group.node_ids) - 1
        )
        duration = max(compute_time, memory_time) + launch_time

        reasons = []
        calibrated = self._device_calibrated(group.device)
        confidence = 0.55 if calibrated else 0.2
        if not calibrated:
            reasons.append(f"device:{group.device}:uncalibrated")
        if group.implementation_status == "planned":
            reasons.append(f"pattern:{group.pattern_id}:no_measured_latency")
            confidence = min(confidence, 0.15)
        shape_model = self.hardware.extra.get("shape_model") or {}
        for node_id in group.node_ids:
            node = graph.nodes[node_id]
            if node.op_type == "matmul" and node.op_type not in shape_model:
                reasons.append("shape_model:matmul:analytic_default")
                confidence = min(confidence, 0.25)

        return GroupCost(
            duration=float(duration),
            compute_time=float(compute_time),
            memory_time=float(memory_time),
            launch_time=float(launch_time),
            confidence=float(confidence),
            source="analytic_roofline",
            ood_reasons=tuple(sorted(set(reasons))),
        )

    def estimate_transfer(
        self,
        src_device: int,
        dst_device: int,
        nbytes: float,
    ) -> TransferCost:
        bandwidth = self.hardware.link_bandwidth(src_device, dst_device)
        duration = float(nbytes) / bandwidth + self.hardware.comm_latency
        calibrated = (
            self._device_calibrated(src_device)
            and self._device_calibrated(dst_device)
            and self.hardware.has_links
        )
        reasons = () if calibrated else (f"link:{src_device}-{dst_device}:uncalibrated",)
        return TransferCost(
            duration=float(duration),
            confidence=0.6 if calibrated else 0.2,
            source="analytic_link",
            ood_reasons=reasons,
        )

    def link_key(self, src_device: int, dst_device: int) -> tuple[int, int]:
        if not self.hardware.has_links:
            return (0, 0)
        return (
            (src_device, dst_device)
            if src_device <= dst_device
            else (dst_device, src_device)
        )


def as_hardware_profile(
    hardware: HardwareSpec | HardwareCostProfile,
) -> HardwareCostProfile:
    if isinstance(hardware, HardwareSpec):
        return AnalyticHardwareProfile(hardware)
    if isinstance(hardware, HardwareCostProfile):
        return hardware
    raise TypeError("hardware must be HardwareSpec or HardwareCostProfile")


__all__ = [
    "AnalyticHardwareProfile",
    "GroupCost",
    "HardwareCostProfile",
    "TransferCost",
    "as_hardware_profile",
]
