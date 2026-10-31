"""Measurement contracts belong to labels, never to canonical Kernel Keys."""
from dataclasses import dataclass
import math
from typing import Mapping
import warnings

from calibration.e2_manifest import content_id


def measurement_contract_id(contract):
    required = {"executor", "submission_included", "allocation_included", "granularity", "unit", "environment"}
    if required - contract.keys():
        raise ValueError("measurement contract is incomplete")
    if not all(isinstance(contract[k], bool) for k in ("submission_included", "allocation_included")):
        raise ValueError("measurement inclusion flags must be boolean")
    if not all(contract[k] for k in ("executor", "granularity", "unit")):
        raise ValueError("measurement contract semantics must be explicit")
    if {"torch", "hip", "triton", "device_model"} - contract["environment"].keys():
        raise ValueError("measurement contract environment is incomplete")
    return content_id("measurement-contract", contract)


class MeasurementContractError(ValueError):
    """A matched measurement cannot be consumed under the requested contract."""


@dataclass(frozen=True)
class MeasurementLabel:
    key_id: str
    measurement_contract_id: str | None
    statistics: Mapping
    admission: str = "legacy_unbound"

    def __post_init__(self):
        if not self.key_id or self.admission not in {"legacy_unbound", "conditional_dev", "accepted"}:
            raise ValueError("invalid measurement label identity/admission")
        if not math.isfinite(self.statistics["median_us"]) or self.statistics["median_us"] <= 0:
            raise ValueError("invalid measurement label duration")

    @classmethod
    def from_duration(cls, key_id, duration_s, metadata=None):
        metadata = metadata or {}
        statistics = metadata.get("statistics") or {
            "median_us": duration_s * 1e6, "mad_rel": None, "n_rounds": None, "n_calls": None}
        if not math.isclose(statistics["median_us"] / 1e6, duration_s, rel_tol=1e-12):
            raise ValueError("label statistics disagree with original duration")
        return cls(key_id, metadata.get("measurement_contract_id"), statistics,
                   metadata.get("admission", "legacy_unbound"))


@dataclass(frozen=True)
class LabelMatch:
    label: MeasurementLabel
    diagnostic: bool
    reasons: tuple[str, ...]


class MeasurementLabelLookup:
    def __init__(self, labels):
        self.labels = dict(labels)
        if any(key != label.key_id for key, label in self.labels.items()):
            raise ValueError("measurement label mapping changes a Kernel Key identity")

    def lookup(self, key_id, contract_id, *, allow_unbound=False):
        label = self.labels[key_id]
        reasons = []
        if label.admission != "accepted":
            reasons.append("admission=" + label.admission)
        if not contract_id or not label.measurement_contract_id or contract_id != label.measurement_contract_id:
            reasons.append("contract_mismatch")
        if reasons:
            message = (f"measurement contract rejected key={key_id}; expected={contract_id}; "
                       f"actual={label.measurement_contract_id}; admission={label.admission}")
            if not allow_unbound:
                raise MeasurementContractError(message)
            warnings.warn("diagnostic allow_unbound: " + message, RuntimeWarning, stacklevel=2)
        return LabelMatch(label, bool(reasons), tuple("measurement_contract:" + r for r in reasons))
