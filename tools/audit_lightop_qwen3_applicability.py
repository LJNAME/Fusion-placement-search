"""CPU structural enumeration plus source-reviewed LightOP applicability.

This is a frozen Qwen3 reference-builder audit, not a generic kernel matcher.
Source-reviewed API judgments below are not inferred from function names.
No LightOP import, runtime adapter, cost query, search, or GPU execution.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from cost_model.search_evaluation import build_evaluation_graph
from simulator.fusion_rules import default_fusion_registry
from simulator.placement_artifacts import graph_fingerprint


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/lightop_qwen3_applicability_v1"
GRAPH_HASH = "sha256:fc718571cb1f7e3d17d32e50401d81f6d465d410dd8f493e3e81305be3ba79ea"
REVIEW = "data/lightop_vendor_20261104_02_review/report.json"
TRIAL = "data/hygon_silu_trial02_review/report.json"
EVIDENCE = "data/lightop_qwen3_applicability_v1/source_evidence.json"


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def layout_state(node):
    """Do not turn logical shape or shared module membership into strides."""
    return {"state": node.extra.get("layout", "unknown"),
            "source": node.extra.get("layout_source", "not_recorded_in_logical_graph")}


def enumerate_matches(graph):
    pattern = default_fusion_registry().get("silu_mul")
    silu, qk = [], []
    for edge in sorted(graph.edges, key=lambda e: (e.src, e.dst)):
        ids = [edge.src, edge.dst]
        if not pattern.matches(graph, ids):
            continue
        nodes = [graph.nodes[i] for i in ids]
        # The existing rule tolerates absent shape/dtype; audit candidates do not.
        if any(not n.extra.get("output_shape") or not n.extra.get("dtype") for n in nodes):
            continue
        gate_inputs = sorted(graph.predecessors(ids[0]))
        up_inputs = sorted(set(graph.predecessors(ids[1])) - {ids[0]})
        if len(gate_inputs) != 1 or len(up_inputs) != 1:
            continue
        gate, up = (graph.nodes[i] for i in (gate_inputs[0], up_inputs[0]))
        shape, dtype = nodes[0].extra["output_shape"], nodes[0].extra["dtype"]
        if any(n.extra.get("output_shape") != shape or n.extra.get("dtype") != dtype
               for n in (gate, up)):
            continue
        if any(n.extra.get("layer") != nodes[0].extra.get("layer") for n in (gate, up)):
            continue
        silu.append({"layer": nodes[0].extra["layer"], "nodes": ids,
                     "boundary_inputs": {"gate": gate.id, "up": up.id},
                     "logical_shape": list(shape), "dtype": dtype,
                     "input_layout": {"gate": layout_state(gate), "up": layout_state(up)},
                     "output_layout": layout_state(nodes[-1]),
                     "packed_storage_proven": False,
                     "graph_adapter_available": False})
    # A joint Q/K call is two separate branches, not an existing linear-chain rule.
    for layer in sorted({n.extra["layer"] for n in graph.nodes.values() if "layer" in n.extra}):
        names = ("q_norm", "q_rope", "k_norm", "k_rope")
        ids = [f"L{layer:03d}.self_attention.{name}" for name in names]
        if not all(i in graph.nodes for i in ids):
            continue
        if [graph.nodes[i].op_type for i in ids] != ["rmsnorm", "rope", "rmsnorm", "rope"]:
            continue
        if not all(dst in graph.successors(src) for src, dst in ((ids[0], ids[1]), (ids[2], ids[3]))):
            continue
        if any(set(graph.successors(ids[i])) != {ids[i + 1]} for i in (0, 2)):
            continue
        qk.append({"layer": layer, "nodes": ids,
                   "q_shape": list(graph.nodes[ids[0]].extra["output_shape"]),
                   "k_shape": list(graph.nodes[ids[2]].extra["output_shape"]),
                   "dtype": graph.nodes[ids[0]].extra.get("dtype"),
                   "structural_only": True, "fusion_group_legality_proven": False,
                   "semantic_equivalence_proven": False})
    return {"silu_mul": silu, "qk_rms_rope": qk}


def reviewed_rows():
    """Manual semantic judgments for the pinned graph and verified source excerpts."""
    rows = []
    for activation in ("silu", "gelu", "gelu_tanh"):
        rows.append({
            "job_id": f"test_act_and_mul_{activation}",
            "capability_id": f"op.{activation}_and_mul_opt",
            "status": "needs_adapter" if activation == "silu" else "not_applicable",
            "match_family": "silu_mul" if activation == "silu" else None,
            "verified_case": "BF16 [tokens,8192] packed input -> [tokens,4096] output; tokens=32..16384",
            "target": "BF16 gate/up [1,256,3072]; independent logical producers",
            "semantic_check": "SiLU(gate)*up" if activation == "silu" else "GELU is not interchangeable with the graph's SiLU",
            "layout_alias_check": "Packed [gate|up], separate preallocated output; graph storage/strides/alias unknown",
            "cost_boundary": "Vendor eager Event mean, vLLM fused baseline; not a Native two-kernel baseline or Graph Replay",
            "missing_evidence": (["PlacedOpGraph binding including pack and output allocation/lifetime",
                                  "actual producer strides, alias, co-location and packing boundary",
                                  "Canonical implementation+adapter identity and target measurement contract"]
                                 if activation == "silu" else ["no GELU nodes in this graph"]),
            "source_refs": ["test/test_act_and_mul.py:40-70"]})
    rows.extend([
        {"job_id": "test_fuse_fill_rms_x2_concat", "capability_id": "fuse_fill_rms_x2_concat",
         "status": "not_applicable", "match_family": None,
         "verified_case": "BF16 hidden=7168, ten batch sizes; two zero-difference reports; no timing",
         "target": "Dense hidden=1024; no position-zero fill or two-hidden-state concat",
         "semantic_check": "Masks positions==0, normalizes both inputs in place, concatenates outputs",
         "layout_alias_check": "Mutates two inputs and writes concat output; not a standalone RMSNorm replacement",
         "cost_boundary": "Correctness-only; no measured cost to reuse",
         "missing_evidence": ["required fill+two-branch+concat computation absent"],
         "source_refs": ["test/test_fuse_fill_rms_x2_concat.py:17-39", "lightop/rmsnorm.py:319-330"]},
        {"job_id": "test_fuse_groupgemm_permute_quant", "capability_id": "fuse_groupgemm_permute_quant",
         "status": "not_applicable", "match_family": None,
         "verified_case": "BF16 tokens x 7168, 256 experts, top-k=8, INT8 quantized output",
         "target": "Dense non-MoE graph; no routing map or quantization/scales",
         "semantic_check": "Permutes selected tokens then quantizes; no GroupGEMM executed",
         "layout_alias_check": "Produces reordered quantized tokens, scales and expert indices/counts",
         "cost_boundary": "Synchronized wall-clock mean 100 calls after 10 warmups",
         "missing_evidence": ["expert routing+permute+quant computation absent"],
         "source_refs": ["test/test_fuse_groupgemm_permute_quant.py:20-38", "test/test_fuse_groupgemm_permute_quant.py:49-78", "lightop/permute_unpermute.py:22-27"]},
        {"job_id": "test_fuse_rmsnorm_rope_contiguous", "capability_id": "fused_rms_norm_rope_contiguous",
         "status": "not_applicable", "match_family": None,
         "verified_case": "MLA BF16 inputs; compressed-KV dim=512, rotary dim=64, FP8 KV cache; zero differences only",
         "target": "Qwen GQA Q/K per-head norms, no compressed MLA KV or cache writes",
         "semantic_check": "DeepSeek rotary path + compressed KV RMSNorm + KV cache storage",
         "layout_alias_check": "Requires kv_c, normalized output, slot mapping and KV-cache buffers",
         "cost_boundary": "Correctness-only; no measured cost to reuse",
         "missing_evidence": ["MLA and KV-cache storage computation absent"],
         "source_refs": ["test/test_fuse_rmsnorm_rope_contiguous.py:48-54", "test/test_fuse_rmsnorm_rope_contiguous.py:96-114", "lightop/rmsnorm.py:279-294"]},
        {"job_id": "test_fused_rms_rope", "capability_id": "rms_rotary_embedding_fuse",
         "status": "evidence_insufficient", "match_family": "qk_rms_rope",
         "verified_case": "FP16 Q/K heads=8/1, head_dim=128, rotary_dim=64, epsilon=1e-6, theta=10000, neox=True",
         "target": "BF16 Q/K heads=16/8, head_dim=128; epsilon/rotary semantics not recorded",
         "semantic_check": "Two Q/K RMSNorm->RoPE branches structurally match; numerical and group legality not established",
         "layout_alias_check": "API overwrites query/key; graph alias/strides and outside consumers require validation",
         "cost_boundary": "Event benchmark includes Q/K clones inside each timed lambda; not pure kernel or Graph Replay",
         "missing_evidence": ["graph RMS epsilon, rotation dimension/theta/neox, positions/cache construction and weights",
                              "BF16 and 16/8 GQA support for this implementation",
                              "actual layout/alias safety and legal joint group for two branches",
                              "adapter identity and compatible timing boundary"],
         "source_refs": ["test/test_fused_rms_rope.py:14-27", "test/test_fused_rms_rope.py:43-68", "test/test_fused_rms_rope.py:145-156", "lightop/fuse_rmsnorm_rope.py:6-43"]},
        {"job_id": "test_fused_trans_bmm_trans_benchmark", "capability_id": "fused_bmm",
         "status": "not_applicable", "match_family": None,
         "verified_case": "INT8 x/w + scales -> BF16; 32 unique shapes, four repeats each; quantization cost forced to zero",
         "target": "BF16 attention QK/AV matmuls; no INT8 inputs or scales",
         "semantic_check": "Verified quantized path cannot silently replace BF16 matmul or prove pure fusion gain",
         "layout_alias_check": "Input/output permutations and scale ownership would require a different graph contract",
         "cost_boundary": "Triton do_bench quantiles versus BF16 Torch; excludes quantization; workbook FP8 label is misleading",
         "missing_evidence": ["matching precision and graph semantics; untested BF16 API support is not inferred"],
         "source_refs": ["test/test_fused_trans_bmm_trans.py:484-508", "test/test_fused_trans_bmm_trans.py:896-924", "lightop/gemmopt.py:727-765"]},
        {"job_id": "test_reshape_cache_cuda", "capability_id": "reshape_and_cache_cuda",
         "status": "not_applicable", "match_family": None,
         "verified_case": "FP16 key/value, heads=1, head_dim=128, 17 token sizes, paged-cache writes",
         "target": "BF16 explicit forward DAG; no KV-cache nodes or slot mapping",
         "semantic_check": "Cache storage side effect cannot be introduced as a replacement for a pure attention node",
         "layout_alias_check": "Writes preallocated key/value cache using slot_mapping",
         "cost_boundary": "Eager Event benchmark; not a Graph Replay cache profile",
         "missing_evidence": ["cache storage and slot-mapping computation absent"],
         "source_refs": ["test/test_reshape_cache_cuda.py:35-45", "test/test_reshape_cache_cuda.py:86-101", "test/test_reshape_cache_cuda.py:150-180", "lightop/reshape_and_cache_cuda.py:7-24"]},
    ])
    return rows


def build():
    catalog_path = "data/lightop_fusion_catalog_v1/capabilities.json"
    matrix_path = "data/lightop_fusion_catalog_v1/vendor_jobs.json"
    catalog, matrix, review, trial, evidence = map(read, (catalog_path, matrix_path, REVIEW, TRIAL, EVIDENCE))
    caps = {c["id"]: c for c in catalog["capabilities"]}
    completed = {r["job_id"]: r for r in review["records"] if r["returncode"] == 0}
    if evidence["wheel_sha256"] != catalog["wheel_sha256"] or evidence["receipt_archive_sha256"] != review["archive_sha256"]:
        raise ValueError("source evidence belongs to a different wheel or receipt")
    for name, source in evidence["source_files"].items():
        expected = (matrix["test_tree_sha256"][name] if name.startswith("test/") else
                    next(c["source_sha256"] for c in catalog["public_exports"] if c["source"] == name))
        if source["source_sha256"] != expected:
            raise ValueError(f"source identity mismatch: {name}")
    graph = build_evaluation_graph("model")
    if graph_fingerprint(graph) != GRAPH_HASH:
        raise ValueError("reference graph changed; source-reviewed judgments need a new audit")
    matches = enumerate_matches(graph)
    rows = reviewed_rows()
    if set(completed) != {r["job_id"] for r in rows}:
        raise ValueError("audit must cover exactly the completed jobs, excluding deferred failures")
    for row in rows:
        cap = caps[row["capability_id"]]
        record = completed[row["job_id"]]
        if row["capability_id"] not in record["primary_capabilities"]:
            raise ValueError("API is not the reviewed job's primary capability")
        if evidence["source_files"][record["source"]]["source_sha256"] != record["source_sha256"]:
            raise ValueError("job source differs from the reviewed receipt")
        row.update({"binding": cap["binding"], "vendor_review_status": record["review_status"],
                    "structural_group_count": len(matches.get(row["match_family"], [])),
                    "project_adapter_available": cap["placed_graph_adapter"],
                    "direct_mapping_ready": False, "profile_reusable_for_search": False})
    split = trial["comparisons"]["split"]
    report = {
        "scope": "cpu_applicability_audit_of_frozen_logical_reference_graph_not_gpu_validation",
        "graph": {"builder": "cost_model.search_evaluation.build_evaluation_graph(model)",
                  "fingerprint": GRAPH_HASH, "runtime_captured": False,
                  "nodes": len(graph.nodes), "edges": len(graph.edges),
                  "op_counts": dict(sorted(Counter(n.op_type for n in graph.nodes.values()).items())),
                  "runtime_layout": "unknown; logical shape does not establish contiguous storage",
                  "b1_graph_fingerprint_matches": read("data/ffn_placement_b1/identity.json")["source_graph_hash"] == GRAPH_HASH},
        "status_counts": {status: sum(r["status"] == status for r in rows) for status in
                          ("direct_mapping", "needs_adapter", "evidence_insufficient", "not_applicable")},
        "rows": rows,
        "existing_width_3072_evidence": {
            "source": TRIAL, "binding": trial["identity"]["binding"],
            "input": trial["identity"]["input"], "contract": "eager Event batch; 10 warmup + 100 calls; one independent batch per arm",
            "native_split_us": split["native_us"], "lightop_split_with_pack_us": split["lightop_path_us"],
            "relative_duration_change_percent": split["relative_duration_change_percent"],
            "packed_storage_kernel_us": trial["benchmark"]["arms"]["lightop_packed"]["gpu_us_per_call"],
            "numeric_checks_passed": all(a["numeric_ok"] and a["input_bytes_unchanged"] for a in trial["benchmark"]["arms"].values()),
            "formal_profile_reuse": False,
            "limits": "Exploratory single-batch evidence, no target Graph contract or graph adapter identity; separate dispatch trace not all timed calls",
        },
        "new_accepted_profiles": 0,
        "historical_ffn_cost_coverage": {"source": REVIEW, "priceable_plans": review["priceable_ffn_plans"], "recomputed_in_this_audit": False},
        "gpu_executed": False, "adapter_implemented": False,
        "deferred_jobs_resumed": False, "search_executed": False,
        "source_sha256": {p: sha((ROOT / p).read_bytes()) for p in (
            catalog_path, matrix_path, REVIEW, TRIAL, EVIDENCE,
            "cost_model/search_evaluation.py", "simulator/model_graphs/dense_layer.py",
            "simulator/fusion_rules.py", "tools/audit_lightop_qwen3_applicability.py")},
    }
    return {"applicability.json": report, "node_matches.json": matches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    first, second = build(), build()
    if encoded(first) != encoded(second):
        raise ValueError("non-reproducible CPU enumeration")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in first.items():
        (args.output / name).write_bytes(encoded(value))
    verification = {"scope": "CPU structural enumeration and source-identity checks only",
                    "repeat_byte_identical": True,
                    "output_sha256": {n: sha(encoded(v)) for n, v in first.items()}}
    (args.output / "verification.json").write_bytes(encoded(verification))
    print(json.dumps({"status_counts": first["applicability.json"]["status_counts"],
                      "structural_groups": {k: len(v) for k, v in first["node_matches.json"].items()},
                      "output": str(args.output)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
