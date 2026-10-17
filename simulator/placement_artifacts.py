"""Versioned artifacts for ranked search results and placed operator graphs."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from .dataset import graph_from_dict, graph_to_dict, plan_from_dict, plan_to_dict
from .fusion_rules import FusionPatternRegistry, default_fusion_registry
from .hardware import HardwareSpec, hardware_from_dict, hardware_to_dict
from .schema import OpGraph, Plan
from .validate import validate_fusion_plan


SEARCH_RESULT_FORMAT = "fusion-placement-search.search-result"
SEARCH_RESULT_VERSION = "1.0"
PLACED_OP_GRAPH_FORMAT = "fusion-placement-search.placed-op-graph"
PLACED_OP_GRAPH_VERSION = "1.0"


def _normalize_json_numbers(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_json_numbers(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json_numbers(item) for item in value]
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _normalize_json_numbers(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_graph_dict(graph: OpGraph) -> dict[str, Any]:
    data = graph_to_dict(graph)
    data["nodes"] = sorted(data["nodes"], key=lambda node: node["id"])
    data["edges"] = sorted(
        data["edges"],
        key=lambda edge: (
            edge["src"],
            edge["dst"],
            _canonical_json(edge.get("bytes")),
            _canonical_json(edge.get("extra", {})),
        ),
    )
    return data


def graph_fingerprint(graph: OpGraph) -> str:
    """Return an insertion-order-independent graph fingerprint."""

    graph.topological_order()
    return _sha256(_canonical_graph_dict(graph))


def hardware_fingerprint(hardware: HardwareSpec) -> str:
    """Return a fingerprint whose device order still carries device-id meaning."""

    return _sha256(hardware_to_dict(hardware))


def plan_fingerprint(plan: Plan) -> str:
    """Return a stable identity for a plan, excluding its display name."""

    value = {
        "groups": [
            {
                "node_ids": list(group.node_ids),
                "device": int(group.device),
                "pattern_id": group.extra.get("pattern_id"),
                "backend": group.extra.get("backend"),
                "workspace_bytes": float(group.extra.get("workspace_bytes", 0.0)),
            }
            for group in plan.groups
        ]
    }
    return _sha256(value)


@dataclass
class SearchCandidate:
    """One ranked, predicted candidate in a search-result artifact."""

    rank: int
    plan: Plan
    predicted_makespan: float
    predicted_peak_memory: Tuple[float, ...]
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def candidate_id(self) -> str:
        return plan_fingerprint(self.plan)


@dataclass
class SearchResult:
    """Self-contained result of one reproducible search invocation."""

    graph: OpGraph
    hardware: HardwareSpec
    objective: dict[str, Any]
    constraints: dict[str, Any]
    search_method: str
    budget: dict[str, Any]
    seed: int
    candidates: list[SearchCandidate]
    baselines: list[SearchCandidate] = field(default_factory=list)
    name: str = "search-result"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def graph_hash(self) -> str:
        return graph_fingerprint(self.graph)

    @property
    def hardware_hash(self) -> str:
        return hardware_fingerprint(self.hardware)


@dataclass
class PlacedOpGraph:
    """A graph plus one concrete fusion and device-placement plan."""

    graph: OpGraph
    hardware: HardwareSpec
    plan: Plan
    rank: Optional[int] = None
    predicted_makespan: Optional[float] = None
    predicted_peak_memory: Tuple[float, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def graph_hash(self) -> str:
        return graph_fingerprint(self.graph)

    @property
    def hardware_hash(self) -> str:
        return hardware_fingerprint(self.hardware)

    @property
    def candidate_id(self) -> str:
        return plan_fingerprint(self.plan)


def _validate_number(value: float, field_name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return number


def _validate_peak_memory(
    values: Sequence[float],
    hardware: HardwareSpec,
    field_name: str,
    *,
    required: bool,
) -> Tuple[float, ...]:
    if not values and not required:
        return ()
    if len(values) != hardware.n_devs:
        raise ValueError(
            f"{field_name} has {len(values)} values, expected {hardware.n_devs}"
        )
    peaks = tuple(
        _validate_number(value, f"{field_name}[{device}]")
        for device, value in enumerate(values)
    )
    for device, (peak, spec) in enumerate(zip(peaks, hardware.devices)):
        if peak > spec.max_mem:
            raise ValueError(
                f"{field_name}[{device}] exceeds device memory: "
                f"{peak} > {spec.max_mem}"
            )
    return peaks


def validate_placed_op_graph(
    placed: PlacedOpGraph,
    registry: Optional[FusionPatternRegistry] = None,
) -> None:
    placed.graph.topological_order()
    rules = registry or default_fusion_registry()
    validate_fusion_plan(
        placed.graph,
        placed.plan,
        placed.hardware.n_devs,
        registry=rules,
    )
    if placed.predicted_makespan is not None:
        _validate_number(placed.predicted_makespan, "predicted_makespan")
    _validate_peak_memory(
        placed.predicted_peak_memory,
        placed.hardware,
        "predicted_peak_memory",
        required=False,
    )
    for group_index, group in enumerate(placed.plan.groups):
        workspace = _validate_number(
            group.extra.get("workspace_bytes", 0.0),
            f"fusion_groups[{group_index}].workspace_bytes",
        )
        if workspace > placed.hardware.devices[group.device].max_mem:
            raise ValueError(
                f"fusion_groups[{group_index}].workspace_bytes exceeds device memory"
            )


def validate_search_result(
    result: SearchResult,
    registry: Optional[FusionPatternRegistry] = None,
) -> None:
    result.graph.topological_order()
    if not result.search_method:
        raise ValueError("search_method must not be empty")
    if not result.objective:
        raise ValueError("objective must not be empty")
    if not result.candidates:
        raise ValueError("candidates must not be empty")

    declared_limits = result.constraints.get("max_memory_bytes")
    actual_limits = tuple(float(device.max_mem) for device in result.hardware.devices)
    if declared_limits is not None and tuple(map(float, declared_limits)) != actual_limits:
        raise ValueError("constraints.max_memory_bytes does not match hardware")

    rules = registry or default_fusion_registry()
    expected_ranks = list(range(1, len(result.candidates) + 1))
    actual_ranks = [candidate.rank for candidate in result.candidates]
    if actual_ranks != expected_ranks:
        raise ValueError(
            f"candidate ranks must be contiguous and ordered: {expected_ranks}"
        )

    for collection_name, candidates in (
        ("candidates", result.candidates),
        ("baselines", result.baselines),
    ):
        seen_ids: set[str] = set()
        for index, candidate in enumerate(candidates):
            validate_fusion_plan(
                result.graph,
                candidate.plan,
                result.hardware.n_devs,
                registry=rules,
            )
            _validate_number(
                candidate.predicted_makespan,
                f"{collection_name}[{index}].predicted_makespan",
            )
            _validate_peak_memory(
                candidate.predicted_peak_memory,
                result.hardware,
                f"{collection_name}[{index}].predicted_peak_memory",
                required=True,
            )
            if candidate.candidate_id in seen_ids:
                raise ValueError(f"duplicate candidate plan: {candidate.candidate_id}")
            seen_ids.add(candidate.candidate_id)


def _candidate_to_dict(candidate: SearchCandidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "rank": int(candidate.rank),
        "predicted_makespan": float(candidate.predicted_makespan),
        "predicted_peak_memory": list(candidate.predicted_peak_memory),
        "plan": plan_to_dict(candidate.plan),
        "extra": candidate.extra,
    }


def _candidate_from_dict(data: Mapping[str, Any]) -> SearchCandidate:
    candidate = SearchCandidate(
        rank=int(data["rank"]),
        plan=plan_from_dict(dict(data["plan"])),
        predicted_makespan=float(data["predicted_makespan"]),
        predicted_peak_memory=tuple(
            float(value) for value in data["predicted_peak_memory"]
        ),
        extra=dict(data.get("extra", {})),
    )
    if data.get("candidate_id") != candidate.candidate_id:
        raise ValueError("candidate_id does not match plan contents")
    return candidate


def search_result_artifact(result: SearchResult) -> dict[str, Any]:
    validate_search_result(result)
    constraints = copy.deepcopy(result.constraints)
    constraints.setdefault(
        "max_memory_bytes",
        [float(device.max_mem) for device in result.hardware.devices],
    )
    return {
        "format": SEARCH_RESULT_FORMAT,
        "version": SEARCH_RESULT_VERSION,
        "name": result.name,
        "inputs": {
            "graph_hash": result.graph_hash,
            "hardware_hash": result.hardware_hash,
            "graph": graph_to_dict(result.graph),
            "hardware": hardware_to_dict(result.hardware),
        },
        "objective": result.objective,
        "constraints": constraints,
        "search": {
            "method": result.search_method,
            "budget": result.budget,
            "seed": int(result.seed),
        },
        "candidates": [_candidate_to_dict(item) for item in result.candidates],
        "baselines": [_candidate_to_dict(item) for item in result.baselines],
        "extra": result.extra,
    }


def save_search_result(path: str | Path, result: SearchResult) -> None:
    artifact = search_result_artifact(result)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_search_result(
    path: str | Path,
    registry: Optional[FusionPatternRegistry] = None,
) -> SearchResult:
    data: Mapping[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format") != SEARCH_RESULT_FORMAT:
        raise ValueError(f"unsupported artifact format: {data.get('format')}")
    if data.get("version") != SEARCH_RESULT_VERSION:
        raise ValueError(f"unsupported artifact version: {data.get('version')}")

    inputs = dict(data["inputs"])
    graph = graph_from_dict(dict(inputs["graph"]))
    hardware = hardware_from_dict(dict(inputs["hardware"]))
    if inputs.get("graph_hash") != graph_fingerprint(graph):
        raise ValueError("graph_hash does not match embedded graph")
    if inputs.get("hardware_hash") != hardware_fingerprint(hardware):
        raise ValueError("hardware_hash does not match embedded hardware")

    search = dict(data["search"])
    result = SearchResult(
        graph=graph,
        hardware=hardware,
        objective=dict(data["objective"]),
        constraints=dict(data.get("constraints", {})),
        search_method=str(search["method"]),
        budget=dict(search.get("budget", {})),
        seed=int(search["seed"]),
        candidates=[_candidate_from_dict(item) for item in data["candidates"]],
        baselines=[_candidate_from_dict(item) for item in data.get("baselines", [])],
        name=str(data.get("name", "search-result")),
        extra=dict(data.get("extra", {})),
    )
    validate_search_result(result, registry=registry)
    return result


def placed_op_graph_artifact(placed: PlacedOpGraph) -> dict[str, Any]:
    validate_placed_op_graph(placed)
    rules = default_fusion_registry()
    graph_data = graph_to_dict(placed.graph)
    node_by_id = {node["id"]: node for node in graph_data["nodes"]}
    groups = []
    for index, group in enumerate(placed.plan.groups):
        group_id = f"fg{index:04d}"
        pattern_id = str(group.extra["pattern_id"])
        backend = str(group.extra.get("backend", rules.get(pattern_id).backend))
        workspace_bytes = float(group.extra.get("workspace_bytes", 0.0))
        groups.append(
            {
                "fusion_group_id": group_id,
                "node_ids": list(group.node_ids),
                "device_id": int(group.device),
                "pattern_id": pattern_id,
                "backend": backend,
                "workspace_bytes": workspace_bytes,
                "extra": group.extra,
            }
        )
        for node_id in group.node_ids:
            node_by_id[node_id]["fusion_group_id"] = group_id
            node_by_id[node_id]["device_id"] = int(group.device)

    return {
        "format": PLACED_OP_GRAPH_FORMAT,
        "version": PLACED_OP_GRAPH_VERSION,
        "graph_hash": placed.graph_hash,
        "hardware_hash": placed.hardware_hash,
        "candidate_id": placed.candidate_id,
        "rank": placed.rank,
        "predicted_makespan": placed.predicted_makespan,
        "predicted_peak_memory": list(placed.predicted_peak_memory),
        "hardware": hardware_to_dict(placed.hardware),
        "graph": graph_data,
        "fusion_groups": groups,
        "plan": {
            "name": placed.plan.name,
            "extra": placed.plan.extra,
        },
        "extra": placed.extra,
    }


def save_placed_op_graph(path: str | Path, placed: PlacedOpGraph) -> None:
    artifact = placed_op_graph_artifact(placed)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_placed_op_graph(
    path: str | Path,
    registry: Optional[FusionPatternRegistry] = None,
) -> PlacedOpGraph:
    data: Mapping[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format") != PLACED_OP_GRAPH_FORMAT:
        raise ValueError(f"unsupported artifact format: {data.get('format')}")
    if data.get("version") != PLACED_OP_GRAPH_VERSION:
        raise ValueError(f"unsupported artifact version: {data.get('version')}")

    graph_data = copy.deepcopy(dict(data["graph"]))
    node_annotations: dict[str, tuple[str, int]] = {}
    for node in graph_data["nodes"]:
        group_id = str(node.pop("fusion_group_id"))
        device_id = int(node.pop("device_id"))
        node_annotations[str(node["id"])] = (group_id, device_id)
    graph = graph_from_dict(graph_data)
    hardware = hardware_from_dict(dict(data["hardware"]))
    if data.get("graph_hash") != graph_fingerprint(graph):
        raise ValueError("graph_hash does not match embedded graph")
    if data.get("hardware_hash") != hardware_fingerprint(hardware):
        raise ValueError("hardware_hash does not match embedded hardware")

    group_data = list(data["fusion_groups"])
    plan_data = dict(data.get("plan", {}))
    plan = plan_from_dict(
        {
            "name": plan_data.get("name", "placed-plan"),
            "extra": plan_data.get("extra", {}),
            "groups": [
                {
                    "node_ids": group["node_ids"],
                    "device": group["device_id"],
                    "extra": group.get("extra", {}),
                }
                for group in group_data
            ],
        }
    )
    for index, group in enumerate(group_data):
        expected_id = f"fg{index:04d}"
        if group.get("fusion_group_id") != expected_id:
            raise ValueError("fusion_group_id sequence is not stable")
        plan_group = plan.groups[index]
        if str(group.get("pattern_id")) != str(plan_group.extra.get("pattern_id")):
            raise ValueError(f"fusion group pattern annotation mismatch: {expected_id}")
        expected_backend = str(
            plan_group.extra.get(
                "backend",
                (registry or default_fusion_registry())
                .get(str(plan_group.extra.get("pattern_id")))
                .backend,
            )
        )
        if str(group.get("backend")) != expected_backend:
            raise ValueError(f"fusion group backend annotation mismatch: {expected_id}")
        if float(group.get("workspace_bytes", 0.0)) != float(
            plan_group.extra.get("workspace_bytes", 0.0)
        ):
            raise ValueError(f"fusion group workspace annotation mismatch: {expected_id}")
        for node_id in group["node_ids"]:
            if node_annotations.get(str(node_id)) != (
                expected_id,
                int(group["device_id"]),
            ):
                raise ValueError(f"node placement annotation mismatch: {node_id}")

    placed = PlacedOpGraph(
        graph=graph,
        hardware=hardware,
        plan=plan,
        rank=None if data.get("rank") is None else int(data["rank"]),
        predicted_makespan=(
            None
            if data.get("predicted_makespan") is None
            else float(data["predicted_makespan"])
        ),
        predicted_peak_memory=tuple(
            float(value) for value in data.get("predicted_peak_memory", [])
        ),
        extra=dict(data.get("extra", {})),
    )
    if data.get("candidate_id") != placed.candidate_id:
        raise ValueError("candidate_id does not match placed plan")
    validate_placed_op_graph(placed, registry=registry)
    return placed


def placed_from_search_result(result: SearchResult, rank: int = 1) -> PlacedOpGraph:
    validate_search_result(result)
    try:
        candidate = next(item for item in result.candidates if item.rank == rank)
    except StopIteration as exc:
        raise ValueError(f"candidate rank not found: {rank}") from exc
    return PlacedOpGraph(
        graph=result.graph,
        hardware=result.hardware,
        plan=candidate.plan,
        rank=candidate.rank,
        predicted_makespan=candidate.predicted_makespan,
        predicted_peak_memory=candidate.predicted_peak_memory,
        extra={"source_search": result.name},
    )


__all__ = [
    "PLACED_OP_GRAPH_FORMAT",
    "PLACED_OP_GRAPH_VERSION",
    "SEARCH_RESULT_FORMAT",
    "SEARCH_RESULT_VERSION",
    "PlacedOpGraph",
    "SearchCandidate",
    "SearchResult",
    "graph_fingerprint",
    "hardware_fingerprint",
    "load_placed_op_graph",
    "load_search_result",
    "placed_from_search_result",
    "placed_op_graph_artifact",
    "plan_fingerprint",
    "save_placed_op_graph",
    "save_search_result",
    "search_result_artifact",
    "validate_placed_op_graph",
    "validate_search_result",
]
