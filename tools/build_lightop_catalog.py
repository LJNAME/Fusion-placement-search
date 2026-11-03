"""Inventory the pinned vendor wheel without importing its native extension."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/hygon_lightop_wheel_v1/manifest.json"
OUTPUT = ROOT / "data/lightop_fusion_catalog_v1"

# Explicit composite interfaces; variants are counted as interfaces, not patterns.
GROUPS = {
    "activation_elementwise": "FuseBiasSwiGLU FuseBiasGelu AddDropout FuseBiasDropoutAdd ReluDropout fuse_silu_and_mul",
    "rmsnorm": "fused_add_rms_norm RMSNormDropout RMSNormAddDropout RMSNormAddDropoutBias RMSNormAdd gemma_fused_add_rmsnorm",
    "norm_quantization": "rms_norm_dynamic_per_token_quant rms_norm_per_token_fp8_quant gemma_rms_norm_per_token_fp8_quant",
    "layernorm": "AddLn LnDropout LnAddDropout LnAddDropoutBias LnDropPath",
    "groupnorm": "CeluGN GNSilu",
    "norm_rope_cache": "rms_rotary_embedding_fuse rms_rotary_embedding_fuse_with_kv_store gammarms_rotary_embedding_fuse_with_kv_store split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant fused_rms_norm_rope_contiguous fuse_rmsnorm_rope_quant_qkv fuse_rms_mrope_cuda",
    "activation_quantization": "fuse_silu_mul_quant fuse_silu_mul_quant_ep fuse_silu_mul_fp8_quant fuse_silu_mul_fp8_quant_ep moe_swiglu_dynamic_quant",
    "softmax": "FusedScaleMaskSoftmax FusedScaleMaskSoftmaxDropout",
    "data_movement_quantization": "fuse_groupgemm_permute_quant fuse_fill_rms_x2_concat reshape_and_cache_cuda destindex_copy_quantize_int8_cuda blockwise_scaled_cast_transpose",
    "moe_routing": "fast_topk_transform_fused fast_topk_transform_ragged_fused moe_fused_gate",
    "bmm": "fused_bmm",
}
FUSIONS = {name: group for group, names in GROUPS.items() for name in names.split()}
OPTIMIZED = {"silu_and_mul_opt": "silu", "gelu_and_mul_opt": "gelu", "gelu_tanh_and_mul_opt": "gelu_tanh"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")


def calls(tree):
    return {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}


def build(wheel):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if sha(Path(wheel).read_bytes()) != manifest["sha256"]:
        raise ValueError("Wheel SHA256 does not match the pinned vendor distribution")
    with zipfile.ZipFile(wheel) as z:
        init = ast.parse(z.read("lightop/__init__.py"))
        inventory, capabilities = [], []
        for node in ast.walk(init):
            if not isinstance(node, ast.ImportFrom) or node.level != 1 or node.module == "version":
                continue
            source = "lightop/" + node.module.replace(".", "/") + ".py"
            raw = z.read(source)
            tree = ast.parse(raw)
            defs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
            for alias in node.names:
                name = alias.asname or alias.name
                definition = defs.get(alias.name)
                record = {"binding": "lightop." + name, "source": source,
                          "line": definition.lineno if definition else None,
                          "source_sha256": sha(raw), "definition": type(definition).__name__ if definition else "value",
                          "classification": "explicit_fusion" if name in FUSIONS else "other_public_export"}
                inventory.append(record)
                if name not in FUSIONS:
                    continue
                if definition is None:
                    raise ValueError("Fusion definition missing: " + name)
                record = dict(record, id=name, family=FUSIONS[name], implementation_status="implemented",
                              implementation_owner="Hygon LightOP", project_registered=True,
                              placed_graph_adapter=False, gpu_measurement="not_yet_measured_in_project",
                              kernel_count="not_certified", conditions="vendor implementation guards; not inferred from the interface name",
                              native_calls=sorted(c for c in calls(definition) if c.startswith("op.")))
                if isinstance(definition, ast.FunctionDef):
                    record["signature"] = ast.unparse(definition.args)
                capabilities.append(record)
        for name, act in OPTIMIZED.items():
            capabilities.append({"id": "op." + name, "binding": "lightop.op." + name,
                "family": "activation_elementwise", "implementation_status": "implemented",
                "implementation_owner": "Hygon LightOP", "project_registered": True,
                "placed_graph_adapter": False,
                "gpu_measurement": "trial02_exploratory" if act == "silu" else "not_yet_measured_in_project",
                "evidence": "data/hygon_silu_trial02_review/report.json" if act == "silu" else None,
                "kernel_count": "one_target_kernel_in_trial02" if act == "silu" else "not_certified",
                "source": "test/test_act_and_mul.py", "source_sha256": sha(z.read("test/test_act_and_mul.py")),
                "line": next(i for i, s in enumerate(z.read("test/test_act_and_mul.py").decode().splitlines(), 1) if "custom_func = op." + name in s),
                "signature": "out, input", "native_calls": ["op." + name],
                "conditions": "packed input; preallocated output; vendor test shapes and dtypes"})
        if {c["id"] for c in capabilities if not c["id"].startswith("op.")} != set(FUSIONS):
            raise ValueError("Public fusion export inventory incomplete")
        jobs = []
        for path in sorted(z.namelist()):
            if not(path.startswith("test/") and path.endswith(".py")):
                continue
            raw = z.read(path)
            tree = ast.parse(raw)
            imports, deps, op_aliases, root_aliases = {}, set(), {"op"}, {"lightop"}
            for n in ast.walk(tree):
                if isinstance(n, ast.ImportFrom):
                    deps.add((n.module or "").split(".")[0])
                    if n.module == "lightop":
                        for a in n.names:
                            imports[a.asname or a.name] = a.name
                            if a.name == "op":
                                op_aliases.add(a.asname or a.name)
                elif isinstance(n, ast.Import):
                    for a in n.names:
                        deps.add(a.name.split(".")[0])
                        if a.name == "lightop":
                            root_aliases.add(a.asname or a.name)
            invoked = calls(tree)
            native = {"op." + c.split(".", 1)[1] for c in invoked if "." in c and c.split(".")[0] in op_aliases}
            refs = []
            for c in capabilities:
                name = c["id"]
                direct = any(imports.get(call) == name for call in invoked) or any(a + "." + name in invoked for a in root_aliases)
                if direct or native.intersection(c["native_calls"]):
                    refs.append(name)
            if path == "test/test_act_and_mul.py":
                refs = ["op." + name for name in OPTIMIZED]
            if not refs:
                continue
            main = any(isinstance(n, ast.If) and "__main__" in ast.unparse(n.test) for n in tree.body)
            variants = [([], refs)]
            if path == "test/test_act_and_mul.py":
                variants = [(["--act", act], ["op." + name]) for name, act in OPTIMIZED.items()]
            elif path == "test/test_fused_trans_bmm_trans.py":
                variants = [(["benchmark"], refs)]
            for args, selected in variants:
                jobs.append({"id": Path(path).stem + ("_" + args[-1] if args else ""), "script": path,
                    "args": args, "source_sha256": sha(raw), "capability_references": sorted(selected),
                    "reference_scope": "static call references; execution and shape coverage require vendor output review",
                    "launch_mode": "script" if main else "no_main_requires_review",
                    "imports": sorted(deps), "method": "unmodified_vendor_script",
                    "timing_helpers": sorted(c for c in invoked if any(x in c for x in ("Event", "do_bench", "time.time", "benchmark.run", "bench_fn"))),
                    "includes_profiler": any("profiler" in c for c in invoked)})
        for c in capabilities:
            c["vendor_jobs"] = [j["id"] for j in jobs if c["id"] in j["capability_references"]]
            c["vendor_test_status"] = "source_references_available" if c["vendor_jobs"] else "no_test_reference_found"
        sources = {p: sha(z.read(p)) for p in sorted(z.namelist()) if p.startswith("test/") and not p.endswith("/")}
        native_source, = [p for p in z.namelist() if p.startswith("lightop/op.") and p.endswith(".so")]
        native_hash = sha(z.read(native_source))
    catalog = {"schema_version": 1, "provider": "lightop", "version": manifest["version"],
        "wheel_sha256": manifest["sha256"], "op_so_sha256": native_hash, "op_so_path": native_source,
        "init_sha256": manifest["wheel_file_sha256"]["lightop/__init__.py"],
        "public_export_count": len(inventory), "fusion_interface_count": len(capabilities),
        "public_export_inventory_complete": True, "native_cpp_inventory_complete": False,
        "count_unit": "callable interface including variants; not unique graph patterns",
        "registration_scope": "provider API catalog; not automatic graph rewriting or Profile admission",
        "public_exports": sorted(inventory, key=lambda c: c["binding"]),
        "capabilities": sorted(capabilities, key=lambda c: c["id"])}
    matrix = {"schema_version": 1, "scope": "vendor_original_tests_not_profile_admission",
        "wheel_sha256": manifest["sha256"], "job_count": len(jobs), "jobs": jobs,
        "test_tree_sha256": sources, "script_bytes_modified": False,
        "timing_contract": "per vendor script; do not pool across different scripts",
        "zero_exit_proves_correctness": False, "one_attempt_per_job": True}
    return catalog, matrix


def main():
    p = argparse.ArgumentParser()
    p.add_argument("wheel", type=Path)
    p.add_argument("--output", type=Path, default=OUTPUT)
    a = p.parse_args()
    catalog, matrix = build(a.wheel)
    save(a.output / "capabilities.json", catalog)
    save(a.output / "vendor_jobs.json", matrix)
    print(json.dumps({"fusion_interfaces": catalog["fusion_interface_count"], "public_exports": catalog["public_export_count"],
                      "vendor_jobs": matrix["job_count"], "without_vendor_test": [c["id"] for c in catalog["capabilities"] if not c["vendor_jobs"]]}))


if __name__ == "__main__":
    main()
