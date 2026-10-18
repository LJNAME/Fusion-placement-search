"""Stable output contract for Fusion cost models."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, Tuple, runtime_checkable

from simulator.schema import OpGraph, Plan


@dataclass(frozen=True)
class CostEstimate:
    """A physical-unit prediction returned by every new cost model."""

    makespan: float
    peak_memory: Tuple[float, ...]
    critical_path: Tuple[str, ...]
    communication_time: float
    computation_time: float
    per_device_time: Tuple[float, ...]
    cross_device_bytes: float
    confidence: float
    out_of_distribution: bool
    ood_reasons: Tuple[str, ...] = ()
    model_name: str = "cost-model"
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        non_negative = {
            "makespan": self.makespan,
            "communication_time": self.communication_time,
            "computation_time": self.computation_time,
            "cross_device_bytes": self.cross_device_bytes,
        }
        for name, value in non_negative.items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name, values in (
            ("peak_memory", self.peak_memory),
            ("per_device_time", self.per_device_time),
        ):
            if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in values):
                raise ValueError(f"{name} must contain finite non-negative values")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.out_of_distribution != bool(self.ood_reasons):
            raise ValueError("out_of_distribution must match whether ood_reasons is empty")

    @property
    def peak_memory_max(self) -> float:
        return max(self.peak_memory, default=0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "makespan": float(self.makespan),
            "peak_memory": list(self.peak_memory),
            "critical_path": list(self.critical_path),
            "communication_time": float(self.communication_time),
            "computation_time": float(self.computation_time),
            "per_device_time": list(self.per_device_time),
            "cross_device_bytes": float(self.cross_device_bytes),
            "confidence": float(self.confidence),
            "out_of_distribution": bool(self.out_of_distribution),
            "ood_reasons": list(self.ood_reasons),
            "model_name": self.model_name,
            "details": self.details,
        }

    def as_search_candidate(
        self,
        rank: int,
        plan: Plan,
        extra: dict[str, Any] | None = None,
    ):
        """Bridge the M3 result into the versioned M2 SearchResult schema."""

        from simulator.placement_artifacts import SearchCandidate

        metadata = {
            "cost_model": self.model_name,
            "confidence": float(self.confidence),
            "out_of_distribution": bool(self.out_of_distribution),
            "ood_reasons": list(self.ood_reasons),
            "communication_time": float(self.communication_time),
            "cross_device_bytes": float(self.cross_device_bytes),
        }
        metadata.update(extra or {})
        return SearchCandidate(
            rank=int(rank),
            plan=plan,
            predicted_makespan=float(self.makespan),
            predicted_peak_memory=tuple(self.peak_memory),
            extra=metadata,
        )


@runtime_checkable
class CostModel(Protocol):
    """Official M3+ cost-model interface."""

    def predict(self, graph: OpGraph, plan: Plan, hardware: Any) -> CostEstimate:
        ...


__all__ = ["CostEstimate", "CostModel"]
