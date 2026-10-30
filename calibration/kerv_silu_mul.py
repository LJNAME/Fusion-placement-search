"""Source-pinned, development-only KERV SiLU-Mul capability and graph guards.

No whole-graph executor or downstream capability is created here. GPU validation
is separate from source availability; native/auto fallback is not fused evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from simulator import FusionPattern, FusionPatternRegistry, default_fusion_registry
from simulator.fusion_rules import FusionRuleError, enumerate_fusion_candidates

from cost_model.fusion_kernel_profile import key_id


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FLAGSCALE = ROOT.parent / "FlagScale-main" / "FlagScale-main"
DEFAULT_BUNDLE = ROOT / "data/kerv_silu_mul_proxy_v1"
PACKAGE = Path("flagscale/models/kerv/ops/KERVRuntimeOptimization")
CORE = PACKAGE / "embodied_ops/kerv_ops.py"
FORMAT = "fusion-placement-search.development-fusion-capability"
DTYPES = ("bf16", "fp16", "fp32")


def source_hashes(flagscale_root):
    root = Path(flagscale_root)
    files = sorted((root / PACKAGE).rglob("*.py"))
    if not files or not (root / CORE).is_file():
        raise FileNotFoundError("FlagScale KERV sources missing; set --flagscale-root")
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files}


def build_capability(flagscale_root=DEFAULT_FLAGSCALE):
    root = Path(flagscale_root)
    hashes = source_hashes(root)
    revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True)
    # This contract was inspected against this source revision, not inferred from
    # the public operator's name. Refuse silently changed guards in another tree.
    if revision.returncode or revision.stdout.strip() != "f8e9a62c499d526a6ffa47f533bcf342330b7326":
        raise ValueError("unreviewed FlagScale revision; review KERV guards before creating a new capability")
    original = subprocess.run(["git", "-C", str(root), "show", "HEAD:" + CORE.as_posix()],
                              capture_output=True)
    if original.returncode or hashlib.sha256(original.stdout).hexdigest() != hashes[CORE.as_posix()]:
        # Git stores LF; accept a checkout with CRLF but reject semantic edits.
        stored = original.stdout.replace(b"\r\n", b"\n")
        checked = (root / CORE).read_bytes().replace(b"\r\n", b"\n")
        if original.returncode or stored != checked:
            raise ValueError("KERV core differs from the inspected revision")
    payload = {
        "format": FORMAT, "version": "1.0",
        "authority": "development_surrogate_not_project_downstream_contract",
        "flagscale_revision": revision.stdout.strip(), "source_sha256": hashes,
        "binding": {
            "id": "flagos_embodied.kerv_silu_mul.triton",
            "version": "sha256:" + hashes[CORE.as_posix()],
            "public_operator": "torch.ops.flagos_embodied.kerv_silu_mul",
            "backend": "flagscale_kerv_triton",
            "compile_flags": {},
            "runtime_flags": {"backend": "triton", "enabled_operator": "kerv_silu_mul",
                              "output_alias_input": "gate", "inplace": True},
        },
        "static_contract": {
            "operators": ["silu", "mul"], "same_shape": True, "same_device": True,
            "same_dtype": True, "dtypes": list(DTYPES), "minimum_rank": 1,
            "minimum_width": 1, "maximum_width": 8192,
            "layout": "both_inputs_contiguous", "runtime_device": "cuda_API_including_HIP",
            "gate_original_must_have_no_consumers_outside_group": True,
            "gate_must_not_be_an_exported_graph_output": True,
        },
        "numeric_contract": {
            "scope": "forward_only", "internal_compute": "fp32", "final_cast": "gate_dtype",
            "equivalence": "tolerance_based_not_bitwise_unfused_rounding",
            "bf16_fp16_rtol": .01, "bf16_fp16_atol": .001,
            "fp32_rtol": .0001, "fp32_atol": .000001,
        },
        "validation": {"source_inspected": True, "GPU_validated": False,
                       "latency_measured": False, "downstream_deployable": False},
        "fallback_policy": "reject_from_fused_candidates_and_keep_unfused_alternative",
        "notes": [
            "auto mode does not enable kerv_silu_mul Triton in the pinned source.",
            "width 11008 takes the native fallback, not a single fused Kernel.",
            "The supplied graph must satisfy the inplace gate liveness constraint.",
            "Unknown graph layouts can only be conditional runtime requirements, not verified facts.",
        ],
    }
    return {**payload, "capability_id": key_id(payload)}


def check_capability(path, flagscale_root=DEFAULT_FLAGSCALE):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    payload = {key: value for key, value in manifest.items() if key != "capability_id"}
    if manifest.get("format") != FORMAT or manifest.get("capability_id") != key_id(payload):
        raise ValueError("development capability content hash mismatch")
    if source_hashes(flagscale_root) != manifest["source_sha256"]:
        raise ValueError("FlagScale KERV source hashes changed")
    if manifest != build_capability(flagscale_root):
        raise ValueError("development capability no longer matches the reviewed contract")
    return manifest


def _contiguous(node):
    shape = node.extra.get("output_shape")
    strides = node.extra.get("strides")
    if strides is not None and shape is not None:
        expected = []
        stride = 1
        for size in reversed(shape):
            expected.append(stride)
            stride *= int(size)
        return tuple(strides) == tuple(reversed(expected))
    layout = node.extra.get("layout")
    if isinstance(layout, dict):
        if layout.get("order") not in {None, "row_major"}:
            return False
        value = layout.get("contiguity")
        return True if value == "contiguous" else False if value in {"strided", "noncontiguous"} else None
    return True if layout == "contiguous" else None


def candidate_applicability(graph, node_ids, *, layout_policy="require_explicit"):
    if layout_policy not in {"require_explicit", "require_contiguous_at_runtime"}:
        raise ValueError("unsupported KERV layout policy")
    reasons, requirements = [], []
    try:
        default_fusion_registry().get("silu_mul").validate(graph, node_ids)
    except FusionRuleError as exc:
        return {"eligible": False, "reasons": [str(exc)], "requirements": []}
    silu, mul = node_ids
    gate_edges = graph.in_edges(silu)
    mul_edges = graph.in_edges(mul)
    other = [edge for edge in mul_edges if edge.src != silu]
    if len(gate_edges) != 1 or len(mul_edges) != 2 or len(other) != 1:
        return {"eligible": False, "reasons": ["unsupported_input_arity"], "requirements": []}
    gate, up = gate_edges[0].src, other[0].src
    names = (gate, up, silu, mul)
    if any(graph.nodes[name].extra.get("phase") != "forward" for name in names):
        reasons.append("forward_only_or_unknown_execution_phase")
    shapes = [graph.nodes[name].extra.get("output_shape") for name in names]
    if any(shape is None for shape in shapes) or len({tuple(shape) for shape in shapes if shape is not None}) != 1:
        reasons.append("input_or_output_shape_mismatch_or_unknown")
    shape = shapes[0]
    if shape is not None and (not shape or any(int(size) <= 0 for size in shape) or int(shape[-1]) > 8192):
        reasons.append("width_or_rank_outside_single_kernel_guard")
    dtypes = {graph.nodes[name].extra.get("dtype") for name in names}
    if len(dtypes) != 1 or not dtypes <= set(DTYPES):
        reasons.append("unsupported_or_mixed_dtype")
    group = set(node_ids)
    if any(edge.dst not in group for edge in graph.out_edges(gate)):
        reasons.append("gate_original_has_external_consumers")
    exported = set(graph.extra.get("output_nodes", []))
    if graph.extra.get("output_node"):
        exported.add(graph.extra["output_node"])
    if gate in exported:
        reasons.append("gate_original_is_exported_output")
    for label, name in (("gate", gate), ("up", up)):
        contiguous = _contiguous(graph.nodes[name])
        if contiguous is False:
            reasons.append(label + "_is_not_contiguous")
        elif contiguous is None:
            if layout_policy == "require_explicit":
                reasons.append(label + "_layout_unknown")
            else:
                requirements.append(label + "_must_be_contiguous_at_runtime")
    return {"eligible": not reasons, "reasons": reasons, "requirements": requirements,
            "gate_node": gate, "up_node": up, "shape": list(shape) if shape else None,
            "dtype": next(iter(dtypes)) if len(dtypes) == 1 else None}


class _KervDevelopmentPattern(FusionPattern):
    def __init__(self, layout_policy):
        # Preserve the old pattern/backend annotations for artifact compatibility;
        # the actual source binding lives in the group Kernel Key. No deployable
        # status is fabricated before GPU and downstream validation.
        super().__init__(pattern_id="silu_mul", backend="template", implementation_status="planned",
                         op_types=("silu", "mul"), min_nodes=2, max_nodes=2,
                         require_same_shape=True, priority=30)
        object.__setattr__(self, "layout_policy", layout_policy)

    def validate(self, graph, node_ids):
        super().validate(graph, node_ids)
        result = candidate_applicability(graph, node_ids, layout_policy=self.layout_policy)
        if not result["eligible"]:
            raise FusionRuleError("KERV development guard: " + ",".join(result["reasons"]))


def development_registry(manifest, *, layout_policy="require_contiguous_at_runtime"):
    if manifest.get("authority") != "development_surrogate_not_project_downstream_contract":
        raise ValueError("not a KERV development capability")
    singletons = [p for p in default_fusion_registry().patterns() if p.max_nodes == 1]
    return FusionPatternRegistry([*singletons, _KervDevelopmentPattern(layout_policy)])


def graph_applicability_report(manifest):
    from cost_model.search_evaluation import build_evaluation_graph
    rows = []
    for scale in ("layer", "model", "model_3b"):
        graph = build_evaluation_graph(scale)
        candidates = [c for c in enumerate_fusion_candidates(graph) if c.pattern_id == "silu_mul"]
        rows.append({"scale": scale, "candidates": [
            {"node_ids": list(candidate.node_ids),
             "strict_layout": candidate_applicability(graph, candidate.node_ids),
             "conditional_layout": candidate_applicability(
                 graph, candidate.node_ids, layout_policy="require_contiguous_at_runtime")}
            for candidate in candidates]})
    return {"capability_id": manifest["capability_id"], "scope": "static_source_guards_only",
            "GPU_validated": False, "graphs": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "check"))
    parser.add_argument("--flagscale-root", type=Path, default=DEFAULT_FLAGSCALE)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args(argv)
    path = args.bundle / "capability.json"
    if args.action == "prepare":
        manifest = build_capability(args.flagscale_root)
        if path.exists():
            raise FileExistsError("capability already exists; use check or a new --bundle")
        args.bundle.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        (args.bundle / "applicability.json").write_text(
            json.dumps(graph_applicability_report(manifest), indent=2) + "\n", encoding="utf-8")
    manifest = check_capability(path, args.flagscale_root)
    print("capability_id:", manifest["capability_id"])
    print("KERV source contract: PASS; explicit Triton, width <= 8192, dead gate input")
    print("Scope: development surrogate; GPU validation and latency: NOT RUN")


if __name__ == "__main__":
    main()
