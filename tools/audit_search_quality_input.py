"""CPU input audit for Fixed-64 model expansion; no search or GPU calls."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibration.kerv_silu_mul import candidate_applicability
from cost_model.search_evaluation import qwen25_3b_spec
from cost_model.structure import analyze_plan_structure
from simulator.fusion_rules import FusionRuleError, enumerate_fusion_candidates, make_reference_plan
from simulator.model_graphs import build_model_graph
from simulator.placement_artifacts import graph_fingerprint
from tools.evaluate_executable_search import setup


def coverage(graph, registry, profile):
    devices = {}
    for device in range(profile.n_devs):
        plan = make_reference_plan(graph, device=device, registry=registry)
        structure = analyze_plan_structure(graph, plan, profile.n_devs, registry=registry)
        counts = Counter(exact_profile=0, nearest_shape=0, analytic_fallback=0, logical_input=0, unavailable=0)
        rows = []
        for group in structure.groups:
            cost = profile.estimate_group(graph, group)
            if not math.isfinite(cost.duration) or cost.duration < 0:
                category = "unavailable"
            else:
                category = {"measured_layer_invariant": "exact_profile", "analytic_roofline": "analytic_fallback",
                            "logical_input": "logical_input"}[cost.source]
            counts[category] += 1
            rows.append({"nodes": list(group.node_ids), "source": cost.source, "category": category,
                         "duration_us": cost.duration * 1e6, "reasons": list(cost.ood_reasons)})
        devices[str(device)] = {"counts": dict(counts), "queries": rows}
    return devices


def audit(graph, registry, profile):
    executable = [c for c in enumerate_fusion_candidates(graph, registry, require_executable=True)
                  if len(c.node_ids) > 1]
    structural, rejected = Counter(), []
    for layer in sorted({n.extra["layer"] for n in graph.nodes.values() if type(n.extra.get("layer")) is int}):
        for pattern, local in (("silu_mul", ("mlp.silu", "mlp.gated_mul")),
                               ("add_rmsnorm", ("residual_add_1", "pre_mlp_norm"))):
            ids = tuple(f"L{layer:03d}.{name}" for name in local)
            if not all(n in graph.nodes for n in ids) or ids[1] not in graph.successors(ids[0]):
                continue
            structural[pattern] += 1
            try:
                registry.get(pattern).validate(graph, ids)
            except FusionRuleError as error:
                rejected.append({"pattern": pattern, "nodes": list(ids), "reason": str(error)})
    return {"graph_name": graph.name, "graph_fingerprint": graph_fingerprint(graph),
            "runtime_captured": False, "nodes": len(graph.nodes), "edges": len(graph.edges),
            "op_counts": dict(Counter(n.op_type for n in graph.nodes.values())),
            "layout_states": dict(Counter(str(n.extra.get("layout", "unknown")) for n in graph.nodes.values())),
            "structural_positions": dict(structural), "executable_fusion_count": len(executable),
            "executable_patterns": dict(Counter(c.pattern_id for c in executable)),
            "rejected_fusions": rejected, "cost_coverage": coverage(graph, registry, profile)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    registry, hardware, baseline, scorer = setup()
    spec = qwen25_3b_spec()
    graph = build_model_graph(spec)
    original, target = (audit(g, registry, scorer.profile) for g in (baseline, graph))
    kerv = [candidate_applicability(graph, (f"L{i:03d}.mlp.silu", f"L{i:03d}.mlp.gated_mul"),
                                   layout_policy="require_contiguous_at_runtime") for i in range(spec.num_layers)]
    ready = target["executable_fusion_count"] > 0 and all(
        d["counts"]["unavailable"] == 0 for d in target["cost_coverage"].values())
    report = {"scope": "Fixed-64 input audit, same existing registry and cost-query path",
              "selected_model": "Qwen2.5-3B", "spec": asdict(spec),
              "hardware_devices": len(hardware.devices), "store_records": len(scorer.profile.store.records),
              "coverage_unit": "singleton node/device queries; excludes rejected fusion groups",
              "nearest_shape_enabled": False, "legacy_eager_profiles_imported": False,
              "baseline_qwen3_06b": original, "target_qwen25_3b": target,
              "kerv_silu_mul": {"eligible": sum(r["eligible"] for r in kerv),
                                "rejected": sum(not r["eligible"] for r in kerv), "example": kerv[0]},
              "fusion_placement_comparison_ready": ready,
              "decision": "ready" if ready else "no executable fusion; do not run a joint-fusion comparison",
              "search_executed": False, "new_gpu_execution": False,
              "tree_metrics": None, "shape_guards_changed": False}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"model": report["selected_model"], "nodes": target["nodes"],
                      "executable_fusions": target["executable_fusion_count"],
                      "cost_counts": {d: v["counts"] for d, v in target["cost_coverage"].items()},
                      "decision": report["decision"]}), flush=True)


if __name__ == "__main__":
    main()
