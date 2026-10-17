"""Fusion pattern definitions and deterministic legal-candidate enumeration.

The legacy simulator accepts any topologically valid grouping.  This module adds
the stricter contract used by the standalone Fusion search without changing that
legacy behavior: every group names a pattern, and the pattern validates graph
structure, tensor metadata, and implementation status.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .schema import FusionGroup, OpGraph, Plan


class FusionRuleError(ValueError):
    """Raised when a fusion group does not satisfy its declared pattern."""


def _node_dtype(graph: OpGraph, node_id: str) -> Optional[str]:
    value = graph.nodes[node_id].extra.get("dtype")
    return str(value).lower() if value is not None else None


def _node_shape(graph: OpGraph, node_id: str) -> Optional[Tuple[int, ...]]:
    value = graph.nodes[node_id].extra.get("output_shape")
    if value is None:
        return None
    return tuple(int(dim) for dim in value)


@dataclass(frozen=True)
class FusionPattern:
    """A graph-local fusion rule plus its current backend availability."""

    pattern_id: str
    backend: str
    implementation_status: str
    op_types: Tuple[str, ...] = ()
    allowed_op_types: Tuple[str, ...] = ()
    min_nodes: int = 1
    max_nodes: int = 1
    supported_dtypes: Tuple[str, ...] = ("bf16", "fp16", "fp32")
    require_same_shape: bool = False
    require_same_layer: bool = True
    require_linear_chain: bool = True
    fallback: bool = False
    priority: int = 0

    @property
    def executable(self) -> bool:
        return self.implementation_status in {"reference", "implemented"}

    def validate(self, graph: OpGraph, node_ids: Sequence[str]) -> None:
        ids = tuple(node_ids)
        if not (self.min_nodes <= len(ids) <= self.max_nodes):
            raise FusionRuleError(
                f"pattern {self.pattern_id} expects {self.min_nodes}..{self.max_nodes} "
                f"nodes, got {len(ids)}"
            )
        if len(set(ids)) != len(ids):
            raise FusionRuleError(f"pattern {self.pattern_id} contains duplicate nodes")
        missing = [node_id for node_id in ids if node_id not in graph.nodes]
        if missing:
            raise FusionRuleError(
                f"pattern {self.pattern_id} references missing nodes: {missing}"
            )

        actual_ops = tuple(graph.nodes[node_id].op_type for node_id in ids)
        if self.op_types and actual_ops != self.op_types:
            raise FusionRuleError(
                f"pattern {self.pattern_id} expects ops {self.op_types}, got {actual_ops}"
            )
        if self.allowed_op_types:
            invalid = [op for op in actual_ops if op not in self.allowed_op_types]
            if invalid:
                raise FusionRuleError(
                    f"pattern {self.pattern_id} does not allow ops {invalid}"
                )

        known_dtypes = {
            dtype for dtype in (_node_dtype(graph, node_id) for node_id in ids) if dtype
        }
        if self.supported_dtypes:
            unsupported = sorted(known_dtypes - set(self.supported_dtypes))
            if unsupported:
                raise FusionRuleError(
                    f"pattern {self.pattern_id} does not support dtype(s) {unsupported}"
                )
        if len(known_dtypes) > 1:
            raise FusionRuleError(
                f"pattern {self.pattern_id} requires one dtype, got {sorted(known_dtypes)}"
            )

        phases = {
            graph.nodes[node_id].extra.get("phase")
            for node_id in ids
            if graph.nodes[node_id].extra.get("phase") is not None
        }
        if len(phases) > 1:
            raise FusionRuleError(
                f"pattern {self.pattern_id} crosses execution phases: {sorted(phases)}"
            )

        if self.require_same_layer:
            layers = {
                graph.nodes[node_id].extra.get("layer")
                for node_id in ids
                if graph.nodes[node_id].extra.get("layer") is not None
            }
            if len(layers) > 1:
                raise FusionRuleError(
                    f"pattern {self.pattern_id} crosses layers: {sorted(layers)}"
                )

        if self.require_same_shape:
            shapes = {
                shape for shape in (_node_shape(graph, node_id) for node_id in ids) if shape
            }
            if len(shapes) > 1:
                raise FusionRuleError(
                    f"pattern {self.pattern_id} requires equal output shapes, got {sorted(shapes)}"
                )

        if self.require_linear_chain and len(ids) > 1:
            for src, dst in zip(ids, ids[1:]):
                if dst not in graph.successors(src):
                    raise FusionRuleError(
                        f"pattern {self.pattern_id} requires edge {src} -> {dst}"
                    )

            group_nodes = set(ids)
            for node_id in ids[:-1]:
                external = sorted(set(graph.successors(node_id)) - group_nodes)
                if external:
                    raise FusionRuleError(
                        f"pattern {self.pattern_id} internal node {node_id} has "
                        f"external consumer(s) {external}"
                    )

    def matches(self, graph: OpGraph, node_ids: Sequence[str]) -> bool:
        try:
            self.validate(graph, node_ids)
        except FusionRuleError:
            return False
        return True


@dataclass(frozen=True)
class FusionCandidate:
    pattern_id: str
    node_ids: Tuple[str, ...]
    backend: str
    implementation_status: str

    @property
    def executable(self) -> bool:
        return self.implementation_status in {"reference", "implemented"}

    def to_group(self, device: int) -> FusionGroup:
        return FusionGroup(
            node_ids=list(self.node_ids),
            device=int(device),
            extra={
                "pattern_id": self.pattern_id,
                "backend": self.backend,
                "implementation_status": self.implementation_status,
            },
        )


class FusionPatternRegistry:
    def __init__(self, patterns: Iterable[FusionPattern] = ()) -> None:
        self._patterns: Dict[str, FusionPattern] = {}
        for pattern in patterns:
            self.register(pattern)

    def register(self, pattern: FusionPattern) -> None:
        if pattern.pattern_id in self._patterns:
            raise ValueError(f"duplicate fusion pattern: {pattern.pattern_id}")
        self._patterns[pattern.pattern_id] = pattern

    def get(self, pattern_id: str) -> FusionPattern:
        try:
            return self._patterns[pattern_id]
        except KeyError as exc:
            raise FusionRuleError(f"unknown fusion pattern: {pattern_id}") from exc

    def patterns(self) -> Tuple[FusionPattern, ...]:
        return tuple(
            sorted(self._patterns.values(), key=lambda p: (-p.priority, p.pattern_id))
        )


_DEFAULT_PATTERNS = (
    FusionPattern(
        pattern_id="silu_mul",
        backend="template",
        implementation_status="planned",
        op_types=("silu", "mul"),
        min_nodes=2,
        max_nodes=2,
        require_same_shape=True,
        priority=30,
    ),
    FusionPattern(
        pattern_id="elementwise_chain",
        backend="template",
        implementation_status="planned",
        allowed_op_types=("add", "silu", "mul"),
        min_nodes=2,
        max_nodes=4,
        require_same_shape=True,
        priority=20,
    ),
    FusionPattern(
        pattern_id="rmsnorm_reference",
        backend="reference",
        implementation_status="reference",
        op_types=("rmsnorm",),
        min_nodes=1,
        max_nodes=1,
        priority=10,
    ),
    FusionPattern(
        pattern_id="unfused",
        backend="reference",
        implementation_status="reference",
        supported_dtypes=(),
        min_nodes=1,
        max_nodes=1,
        require_same_layer=False,
        fallback=True,
        priority=-100,
    ),
)


def default_fusion_registry() -> FusionPatternRegistry:
    return FusionPatternRegistry(_DEFAULT_PATTERNS)


def _candidate_paths(graph: OpGraph, pattern: FusionPattern) -> Iterable[Tuple[str, ...]]:
    if pattern.fallback:
        for node_id in graph.topological_order():
            yield (node_id,)
        return

    order = graph.topological_order()
    if pattern.op_types:
        starts = [
            node_id
            for node_id in order
            if graph.nodes[node_id].op_type == pattern.op_types[0]
        ]
    else:
        starts = [
            node_id
            for node_id in order
            if graph.nodes[node_id].op_type in pattern.allowed_op_types
        ]

    def walk(path: Tuple[str, ...]) -> Iterable[Tuple[str, ...]]:
        if len(path) >= pattern.min_nodes and pattern.matches(graph, path):
            yield path
        if len(path) >= pattern.max_nodes:
            return
        next_index = len(path)
        for successor in sorted(graph.successors(path[-1])):
            op_type = graph.nodes[successor].op_type
            if pattern.op_types:
                if next_index >= len(pattern.op_types) or op_type != pattern.op_types[next_index]:
                    continue
            elif op_type not in pattern.allowed_op_types:
                continue
            yield from walk(path + (successor,))

    for start in starts:
        yield from walk((start,))


def enumerate_fusion_candidates(
    graph: OpGraph,
    registry: Optional[FusionPatternRegistry] = None,
) -> List[FusionCandidate]:
    """Enumerate deterministic graph-local candidates for small-graph oracles/tests.

    Specialized singleton patterns replace the generic ``unfused`` candidate for
    the same node.  Multi-node candidates may overlap; the later plan search is
    responsible for choosing a disjoint covering.
    """

    patterns = (registry or default_fusion_registry()).patterns()
    candidates: Dict[Tuple[str, ...], FusionCandidate] = {}
    specialized_singletons: set[str] = set()

    for pattern in patterns:
        if pattern.fallback:
            continue
        for node_ids in _candidate_paths(graph, pattern):
            candidate = FusionCandidate(
                pattern_id=pattern.pattern_id,
                node_ids=node_ids,
                backend=pattern.backend,
                implementation_status=pattern.implementation_status,
            )
            # Patterns are processed by descending priority.  Keep the most
            # specific rule when two rules describe exactly the same group.
            candidates.setdefault(candidate.node_ids, candidate)
            if len(node_ids) == 1:
                specialized_singletons.add(node_ids[0])

    for pattern in patterns:
        if not pattern.fallback:
            continue
        for node_ids in _candidate_paths(graph, pattern):
            if node_ids[0] in specialized_singletons:
                continue
            candidate = FusionCandidate(
                pattern_id=pattern.pattern_id,
                node_ids=node_ids,
                backend=pattern.backend,
                implementation_status=pattern.implementation_status,
            )
            candidates.setdefault(candidate.node_ids, candidate)

    position = {node_id: index for index, node_id in enumerate(graph.topological_order())}
    return sorted(
        candidates.values(),
        key=lambda candidate: (
            min(position[node_id] for node_id in candidate.node_ids),
            len(candidate.node_ids),
            candidate.pattern_id,
            candidate.node_ids,
        ),
    )


def make_reference_plan(
    graph: OpGraph,
    device: int = 0,
    name: str = "unfused-reference",
    registry: Optional[FusionPatternRegistry] = None,
) -> Plan:
    """Build a strictly annotated all-singleton reference plan."""

    candidates = enumerate_fusion_candidates(graph, registry)
    singleton_by_node = {
        candidate.node_ids[0]: candidate
        for candidate in candidates
        if len(candidate.node_ids) == 1
    }
    groups = [
        singleton_by_node[node_id].to_group(device)
        for node_id in graph.topological_order()
    ]
    return Plan(groups=groups, name=name)
