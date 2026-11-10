"""Fusion Library manifest (B): the searcher consumes registered library capabilities.

Division of work: the project group ships the fusion-op library and its
interfaces; this manifest registers them; search reads it to decide which
fusions may compete; downstream builds and executes the chosen configuration.
The manifest is a capability registry. Its declarations (dtype, shape range,
layout, training support) come from the provider and are recorded, not derived.

Search scopes:
  full        - every ``implemented`` fusion competes; unmeasured ones are priced
                analytically and flagged ``no_calibration``;
  development - only implemented fusions with measured calibration compete.
Measurements only calibrate the fused cost: fused = c * analytic_fused, with
c fitted from measured native/fused pairs. They never decide eligibility in
``full`` scope.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Optional

from simulator import FusionGroup, OpGraph, Plan
from simulator.fusion_rules import FusionPattern, FusionPatternRegistry, FusionRuleError, default_fusion_registry

from .structure import analyze_plan_structure
from .training_memory import BACKWARD_SAVE

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/fusion_library_lightop_v1/manifest.json"
DEFAULT_CALIBRATION = ROOT / "data/fusion_library_lightop_v1/calibration.json"
STATUSES = ("planned", "implemented")
SCOPES = ("full", "development")
_SHAPE_KEYS = {"dim", "min", "max", "multiple_of"}


class FusionLibraryError(ValueError):
    pass


@dataclass(frozen=True)
class LibraryPattern(FusionPattern):
    """A FusionPattern with provider-declared shape ranges.

    ``intermediate_outputs`` lets earlier ops in the chain also feed nodes outside
    the group (e.g. add -> rmsnorm where the sum is the next residual); the
    chain edges themselves are still required.
    """

    shape_constraints: tuple = ()
    intermediate_outputs: bool = False

    def validate(self, graph: OpGraph, node_ids) -> None:
        super().validate(graph, node_ids)
        ids = tuple(node_ids)
        if self.intermediate_outputs:
            for src, dst in zip(ids, ids[1:]):
                if dst not in graph.successors(src):
                    raise FusionRuleError(f"pattern {self.pattern_id} requires edge {src} -> {dst}")
        for node_id in ids:
            shape = graph.nodes[node_id].extra.get("output_shape")
            for rule in self.shape_constraints:
                if shape is None:
                    raise FusionRuleError(f"pattern {self.pattern_id} needs a known shape at {node_id}")
                size = int(shape[rule["dim"]])
                if (size < rule.get("min", size) or size > rule.get("max", size)
                        or size % rule.get("multiple_of", 1)):
                    raise FusionRuleError(f"pattern {self.pattern_id}: dim {rule['dim']}={size} outside {rule}")


class JointQKRmsRopePattern(LibraryPattern):
    """One joint Q/K call, never two independent RMSNorm/RoPE labels.

    The graph must supply an explicit semantic declaration. Measurements at a
    chosen rotary convention do not fill in missing graph semantics.
    """

    def candidate_node_ids(self, graph):
        for join in graph.topological_order():
            if graph.nodes[join].op_type != "matmul":
                continue
            branches = []
            for rope in graph.predecessors(join):
                if graph.nodes[rope].op_type != "rope":
                    continue
                parents = list(graph.predecessors(rope))
                if len(parents) == 1 and graph.nodes[parents[0]].op_type == "rmsnorm":
                    branches.append((parents[0], rope))
            if len(branches) == 2:
                # Shape distinguishes Q (16 heads) and K (8) in the measured case.
                branches.sort(key=lambda pair: tuple(graph.nodes[pair[0]].extra.get("output_shape", ())), reverse=True)
                yield tuple(n for pair in branches for n in pair)

    def validate(self, graph, node_ids):
        super().validate(graph, node_ids)
        qn, qr, kn, kr = node_ids
        shapes = ((1, 256, 16, 128), (1, 256, 8, 128))
        for (norm, rope), shape in zip(((qn, qr), (kn, kr)), shapes):
            if any(tuple(graph.nodes[n].extra.get("output_shape", ())) != shape for n in (norm, rope)):
                raise FusionRuleError("joint Q/K RMSRoPE requires its measured Q/K shapes")
            if list(graph.predecessors(rope)) != [norm] or set(graph.successors(norm)) != {rope}:
                raise FusionRuleError("joint RMSRoPE cannot remove an externally consumed norm")
            if len(graph.in_edges(norm)) != 1 or graph.nodes[norm].param_bytes != 256:
                raise FusionRuleError("joint RMSRoPE requires one activation and BF16 head weights")
        joins = set(graph.successors(qr)) & set(graph.successors(kr))
        if not any(graph.nodes[n].op_type == "matmul" for n in joins):
            raise FusionRuleError("Q and K must feed the same attention product")
        nodes = [graph.nodes[n] for n in node_ids]
        if any(n.extra.get("dtype") != "bf16" or n.extra.get("phase") != "forward" for n in nodes):
            raise FusionRuleError("joint RMSRoPE supports BF16 forward only")
        if len({n.extra.get("layer") for n in nodes}) != 1 or nodes[0].extra.get("layer") is None:
            raise FusionRuleError("joint RMSRoPE requires explicit same-layer metadata")
        declared = graph.extra.get("fusion_contracts", {}).get(self.pattern_id, {})
        expected = {"theta": 1e6, "rotary_dim": 128, "is_neox": True, "eps": 1e-6,
                    "residual_q": None, "residual_k": None}
        if not declared.get("source") or any(k not in declared or declared[k] != v for k, v in expected.items()):
            raise FusionRuleError("missing or incompatible graph rotary/normalization semantic declaration")


@dataclass(frozen=True)
class FusionEntry:
    fusion_id: str
    implementation: str
    status: str
    training_supported: Optional[bool]
    calibration: bool
    layout_constraints: Optional[str]
    pattern: LibraryPattern


@dataclass(frozen=True)
class FusionLibrary:
    name: str
    manifest_sha256: str
    entries: tuple[FusionEntry, ...]
    not_registered: tuple[dict, ...]

    @property
    def patterns(self) -> tuple[LibraryPattern, ...]:
        """Patterns allowed to compete in full scope (implemented only)."""
        return tuple(e.pattern for e in self.entries if e.status == "implemented")

    def eligible(self, scope: str = "full") -> tuple[FusionEntry, ...]:
        if scope not in SCOPES:
            raise FusionLibraryError(f"scope must be one of {SCOPES}")
        return tuple(e for e in self.entries if e.status == "implemented"
                     and (scope == "full" or e.calibration))

    def registry(self, scope: str = "full") -> FusionPatternRegistry:
        """Eligible library fusions plus the default singletons; no template fusions."""
        singletons = [p for p in default_fusion_registry().patterns() if p.max_nodes == 1]
        return FusionPatternRegistry([*singletons, *(e.pattern for e in self.eligible(scope))])

    def training_backward_policies(self) -> dict[str, str]:
        """Fusions the provider declares trainable. The memory model assumes the
        fused op keeps its inputs for backward (save), the conservative choice."""
        return {e.fusion_id: BACKWARD_SAVE for e in self.entries if e.training_supported is True}

    def describe(self, scope: str = "full") -> dict:
        eligible = {e.fusion_id for e in self.eligible(scope)}
        return {"library": self.name, "manifest_sha256": self.manifest_sha256, "scope": scope,
                "fusions": {e.fusion_id: {"implementation": e.implementation, "status": e.status,
                                          "op_types": list(e.pattern.op_types),
                                          "training_supported": e.training_supported,
                                          "calibration": e.calibration,
                                          "eligible": e.fusion_id in eligible}
                            for e in self.entries},
                "not_registered": [x["implementation"] for x in self.not_registered]}

    def check_calibration(self, observations) -> None:
        """Declared calibration flags must match the observations actually supplied."""
        measured = {o["pattern_id"] for o in observations}
        known = {e.fusion_id: e for e in self.entries}
        for pid in measured:
            if pid not in known:
                raise FusionLibraryError(f"calibration for unregistered fusion {pid}")
            if not known[pid].calibration:
                raise FusionLibraryError(f"{pid} has observations but declares calibration=false")
        missing = sorted(e.fusion_id for e in self.entries if e.calibration and e.fusion_id not in measured)
        if missing:
            raise FusionLibraryError(f"calibration=true without observations: {missing}")


def _read(path) -> tuple[dict, str]:
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _shape_constraints(value, fid):
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(r, dict) or "dim" not in r
                                          or set(r) - _SHAPE_KEYS for r in value):
        raise FusionLibraryError(f"{fid}: supported_shapes must be null or a list of "
                                 f"{{dim, min, max, multiple_of}}")
    return tuple(dict(r) for r in value)


def load_library(path=DEFAULT_MANIFEST) -> FusionLibrary:
    """Register library fusions. Checks only what the registry needs: the library
    build matches the catalog, and every ``implemented`` fusion names an interface
    the catalog lists as implemented."""
    manifest, digest = _read(path)
    if manifest.get("schema_version") != 2:
        raise FusionLibraryError("unsupported manifest schema_version (expected 2)")
    catalog, _ = _read(ROOT / manifest["catalog"])
    if catalog["wheel_sha256"] != manifest["wheel_sha256"]:
        raise FusionLibraryError("manifest and catalog describe different library builds")
    records = {c["id"]: c for c in catalog["capabilities"]}
    entries, seen = [], set()
    for item in manifest["fusions"]:
        fid, impl, status = item["fusion_id"], item["implementation"], item["status"]
        if fid in seen:
            raise FusionLibraryError(f"duplicate fusion_id {fid}")
        seen.add(fid)
        if status not in STATUSES:
            raise FusionLibraryError(f"{fid}: status must be one of {STATUSES}")
        if status == "implemented":
            record = records.get(impl)
            if record is None:
                raise FusionLibraryError(f"{fid}: interface {impl} not in library catalog")
            if record["implementation_status"] != "implemented":
                raise FusionLibraryError(f"{fid}: catalog lists {impl} as {record['implementation_status']}")
        training = item.get("training_supported")
        if training not in (None, True, False):
            raise FusionLibraryError(f"{fid}: training_supported must be true, false or null")
        calibration = item.get("calibration", False)
        if not isinstance(calibration, bool):
            raise FusionLibraryError(f"{fid}: calibration must be a boolean")
        op = item["operator_pattern"]
        ops = tuple(op["op_types"])
        if len(ops) < 2:
            raise FusionLibraryError(f"{fid}: a fusion covers at least two ops")
        intermediate = bool(op.get("intermediate_outputs", False))
        topology = op.get("topology", "chain")
        if topology not in ("chain", "joint_qk_rms_rope"):
            raise FusionLibraryError(f"{fid}: unknown topology {topology}")
        pattern_type = JointQKRmsRopePattern if topology == "joint_qk_rms_rope" else LibraryPattern
        pattern = pattern_type(
            pattern_id=fid, backend=manifest["library"], implementation_status=status,
            op_types=ops, min_nodes=len(ops), max_nodes=len(ops),
            supported_dtypes=tuple(item["supported_dtypes"]),
            require_same_shape=bool(op.get("same_shape", True)),
            require_linear_chain=topology == "chain" and not intermediate, priority=int(item.get("priority", 40)),
            shape_constraints=_shape_constraints(item.get("supported_shapes"), fid),
            intermediate_outputs=intermediate)
        entries.append(FusionEntry(fid, impl, status, training, calibration,
                                   item.get("layout_constraints"), pattern))
    return FusionLibrary(manifest["library"], digest, tuple(entries),
                         tuple(manifest.get("not_registered", ())))


def fit_corrections(library: FusionLibrary, profile, observations, build_graph: Callable[[str], OpGraph]):
    """Per-pattern factor c with  fused = c * analytic_fused * (measured_native / analytic_native).

    Dividing out the native ratio removes the analytic model's per-shape bias,
    so c only carries what is specific to the fused kernel. Several observations
    of one pattern pool by geometric mean.
    """
    registry = library.registry()
    known = {p.pattern_id for p in library.patterns}
    logs: dict[str, list[float]] = {}
    fits = []
    graphs: dict[str, OpGraph] = {}
    for obs in observations:
        pid = obs["pattern_id"]
        if pid not in known:
            raise FusionLibraryError(f"calibration for pattern {pid} not in the library")
        graph = graphs.setdefault(obs["scale"], build_graph(obs["scale"]))
        ids = tuple(obs["node_ids"])
        registry.get(pid).validate(graph, ids)
        n = profile.n_devs
        fused_plan = [g for g in analyze_plan_structure(graph, _one(graph, ids, pid, True), n,
                      registry=registry).groups if set(g.node_ids) == set(ids)]
        native_plan = [g for g in analyze_plan_structure(graph, _one(graph, ids, pid, False), n,
                       registry=registry).groups if set(g.node_ids) <= set(ids)]
        a_fused = sum(profile.estimate_group(graph, g).duration for g in fused_plan)
        a_native = sum(profile.estimate_group(graph, g).duration for g in native_plan)
        c = (obs["fused_us"] / obs["native_us"]) / (a_fused / a_native)
        if not (math.isfinite(c) and c > 0):
            raise FusionLibraryError(f"non-positive correction for {pid}")
        logs.setdefault(pid, []).append(math.log(c))
        fits.append({"pattern_id": pid, "scale": obs["scale"], "node_ids": list(ids),
                     "measured_fused_over_native": obs["fused_us"] / obs["native_us"],
                     "analytic_fused_over_native": a_fused / a_native, "correction": c,
                     "evidence_grade": obs.get("evidence_grade")})
    corrections = {pid: math.exp(sum(v) / len(v)) for pid, v in logs.items()}
    return corrections, fits


def _one(graph: OpGraph, ids, pid: str, fused: bool) -> Plan:
    inside = set(ids)
    groups = []
    for nid in graph.topological_order():
        if nid in inside and fused:
            if nid == ids[-1]:
                groups.append(FusionGroup(list(ids), 0, {"pattern_id": pid}))
        else:
            groups.append(FusionGroup([nid], 0, {"pattern_id": "unfused"}))
    return Plan(groups=groups)


class CalibratedFusionProfile:
    """Delegates everything to ``base``; scales library-fused group durations."""

    def __init__(self, base, library: FusionLibrary, corrections: Mapping[str, float]):
        self.base, self.library = base, library
        self.corrections = dict(corrections)
        self._library_ids = {p.pattern_id for p in library.patterns}

    def __getattr__(self, name):
        return getattr(self.base, name)

    def estimate_group(self, graph, group):
        cost = self.base.estimate_group(graph, group)
        if group.pattern_id not in self._library_ids:
            return cost
        c = self.corrections.get(group.pattern_id)
        if c is None:
            return replace(cost, confidence=min(cost.confidence, .15), source=cost.source + "+uncalibrated_fusion",
                           ood_reasons=tuple(sorted({*cost.ood_reasons,
                                                     f"pattern:{group.pattern_id}:no_calibration"})))
        return replace(cost, duration=cost.duration * c, compute_time=cost.compute_time * c,
                       memory_time=cost.memory_time * c, source=cost.source + "+fusion_calibrated")

    def describe(self) -> dict:
        return {"corrections": dict(self.corrections)}


def load_calibration(path=DEFAULT_CALIBRATION) -> tuple[list[dict], str]:
    data, digest = _read(path)
    if data.get("schema_version") != 1:
        raise FusionLibraryError("unsupported calibration schema_version")
    for obs in data["observations"]:
        src = ROOT / obs["source"]
        if hashlib.sha256(src.read_bytes()).hexdigest() != obs["source_sha256"]:
            raise FusionLibraryError(f"calibration source changed: {obs['source']}")
    return data["observations"], digest
