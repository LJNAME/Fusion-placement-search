"""Measured-first Profile overlay that is independent of node names (cross-layer reuse).

A group matches a reviewed record by a condition signature:

- kind (stage / gemm / fused)
- operator types, output shape, dtype and the shapes/dtypes of external inputs
  (GEMM: m, n, k)
- GPU: ordinal plus the normalized software stack (torch build, extension,
  HIP, driver, architecture)
- implementation recipe id and source version (the fused case also checks
  the LightOP native and package hashes)
- execution contract class: Graph Replay, 50 calls per replay, median us/call

Node names are not part of the signature, so one record serves every
Transformer layer with identical conditions. Graph names, structure and
dependencies are untouched. Misses fall through to the base (analytic)
profile with an explicit reason, e.g. add_rmsnorm, which has no measurement.

Implementations are taken from an explicit execution declaration:

- singleton silu/mul: aten-out stage recipe
- singleton matmul: aten.mm.out over an nn.Linear [out,in] weight
- silu_mul group: LightOP split-pack adapter
- activation outputs: contiguous, preallocated

The declared versions must equal the current repository files, so the
deployed code is the measured code. Every source must have passed review;
reference-only cases are excluded.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from cost_model.hardware_profile import GroupCost

ROOT = Path(__file__).resolve().parents[1]
STACK_FIELDS = ("torch_distribution", "torch_git_revision", "torch_build_sha256", "torch_extension_sha256",
                "hip", "driver", "architecture")
SOURCES = {
    "stage_gpu0": "data/native_stage_calibration_20261105_01",
    "fused_gpu0": "data/lightop_profile_calibration_20261104_01",
    "gemm": "data/gemm_projection_20261106_01",
    "gpu1": "data/gpu1_cost_completion_20261106_01",
    "add_rmsnorm_gpu0": "data/add_rmsnorm_calibration_20261106_01",
}
REFERENCE_ONLY = {("gpu1", "gemm/down/gpu0"), ("add_rmsnorm_gpu0", "chain/add_rmsnorm_native/gpu0")}
REPLICATE_SPREAD_LIMIT = .10  # same threshold as the per-record IQR/Median quality rule


def _sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def _read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def normalize_stack(runtime_json, ordinal):
    """Map the three runtime-observation formats in this repo to one stack dict."""
    rt = runtime_json["runtime"]
    if "devices" in runtime_json:  # stage_size / gemm / gpu1 collectors
        arch = runtime_json["devices"][str(ordinal)]["architecture"]
        hip, driver = rt["hip"], rt["driver"]
    elif "accelerator_version" in rt:  # LightOP observe_runtime
        arch, hip, driver = runtime_json["device"]["architecture"], rt["accelerator_version"], rt["driver_version"]
    else:  # native_stage collector
        arch, hip, driver = runtime_json["device"]["architecture"], rt["hip"], rt["driver"]
    return {"torch_distribution": rt["torch_distribution"], "torch_git_revision": rt["torch_git_revision"],
            "torch_build_sha256": rt["torch_build_sha256"], "torch_extension_sha256": rt["torch_extension_sha256"],
            "hip": str(hip), "driver": str(driver), "architecture": str(arch).split(":")[0]}


def contract_class(contract):
    calls = contract.get("calls_per_replay", contract.get("calls_per_round"))
    cls = {"mode": contract.get("mode"), "calls": calls, "statistic": contract.get("statistic"),
           "unit": contract.get("unit")}
    if cls != {"mode": "graph_replay", "calls": 50, "statistic": "median", "unit": "us/call"}:
        raise ValueError("record contract is outside the declared Graph Replay class: " + repr(cls))
    return cls


def execution_declaration():
    from calibration import gemm_projection_calibration as gp
    from calibration import add_rmsnorm_calibration as addnorm
    from calibration import native_stage_calibration as native
    from integration.lightop import add_rmsnorm as add_rmsnorm_adapter
    from integration.lightop import catalog as lightop_catalog
    data = lightop_catalog.catalog()
    return {
        "stage": {**{op: {"id": native.RECIPES[op]["id"], "version": _sha("calibration/native_stage_calibration.py")}
                     for op in ("silu", "mul")},
                  **{op: {"id": addnorm.RECIPES[op]["id"], "version": _sha("calibration/add_rmsnorm_calibration.py")}
                     for op in ("add", "rmsnorm")}},
        "gemm": {"id": gp.RECIPE["id"], "version": _sha("calibration/gemm_projection_calibration.py")},
        "fused": {"silu_mul": {"id": "lightop.silu_mul.split_pack_preallocated.v1",
                               "adapter": _sha("integration/lightop/silu_mul.py"),
                               "native": data["op_so_sha256"], "package": data["init_sha256"]},
                  "add_rmsnorm": {"id": add_rmsnorm_adapter.RECIPE["id"],
                                  "adapter": _sha("integration/lightop/add_rmsnorm.py"),
                                  "native": data["op_so_sha256"], "package": data["init_sha256"]}},
        "layout": "contiguous_preallocated", "contract_class": contract_class(
            {"mode": "graph_replay", "calls_per_replay": 50, "statistic": "median", "unit": "us/call"}),
    }


def _stage_sig(op, ordinal, stack, implementation, out):
    return ("stage", op, ordinal, json.dumps(stack, sort_keys=True), implementation["recipe"]["id"],
            implementation["source_sha256"], tuple(out["shape"]), out["dtype"], tuple(out["strides"]))


def _gemm_sig(ordinal, stack, implementation, case):
    return ("gemm", ordinal, json.dumps(stack, sort_keys=True), implementation["recipe"]["id"],
            implementation["source_sha256"], case["m"], case["n"], case["k"], "bf16")


def _fused_sig(ordinal, stack, key):
    impl = key["implementation"]
    inputs = tuple(sorted((tuple(x["shape"]), x["dtype"]) for x in key["external_inputs"]))
    return ("fused", key["pattern_id"], ordinal, json.dumps(stack, sort_keys=True), impl["id"],
            impl["loaded_source_sha256"]["adapter"], impl["loaded_source_sha256"]["native"],
            impl["loaded_source_sha256"]["package"], inputs)


def _lightop_package_sha():
    from integration.lightop import catalog as lightop_catalog
    return lightop_catalog.catalog()["init_sha256"]


class LayerInvariantStore:
    def __init__(self, sources=SOURCES):
        from calibration.lightop_profile_calibration import review_receipt
        from cost_model.gemm_profile import ReviewedGemmStore
        from cost_model.gpu1_profile import ReviewedGpu1Store
        from cost_model.native_stage_profile import ReviewedStageStore
        self.records, self.stacks = {}, {}

        replicates = {}

        def add(sig, duration_s, provenance, contract, stack, ordinal):
            if self.stacks.setdefault(ordinal, stack) != stack:
                raise ValueError(f"GPU{ordinal} records come from different software stacks")
            contract_class(contract)
            # Same condition measured more than once (e.g. Gate and Up projections share
            # m/n/k): independent replicates of one condition, combined below.
            replicates.setdefault(sig, []).append((duration_s, provenance))

        stage = ReviewedStageStore(ROOT / sources["stage_gpu0"] / "raw")
        runtime = _read(sources["stage_gpu0"] + "/raw/runtime.json")
        for record in stage.records.values():
            ident = record["identity"]
            stack = normalize_stack(runtime, ident["device"]["ordinal"])
            add(_stage_sig(record["stage"], ident["device"]["ordinal"], stack, ident["implementation"],
                           ident["implementation"]["tensors"]["output"]), record["duration_s"],
                f"{sources['stage_gpu0']}:{record['stage']}", ident["contract"], stack, ident["device"]["ordinal"])

        decision, profile = review_receipt(ROOT / sources["fused_gpu0"] / "raw")
        if profile is None:
            raise ValueError("GPU0 LightOP Profile did not pass admission")
        identity = _read(sources["fused_gpu0"] + "/raw/identity.json")
        runtime = _read(sources["fused_gpu0"] + "/raw/runtime.json")
        ordinal = identity["kernel_key"]["hardware"]["physical_index"]
        stack = normalize_stack(runtime, ordinal)
        add(_fused_sig(ordinal, stack, identity["kernel_key"]), decision["latency"]["duration_s"],
            f"{sources['fused_gpu0']}:kernel_key={decision['kernel_key_id']}", identity["measurement_contract"],
            stack, ordinal)

        gemm = ReviewedGemmStore(ROOT / sources["gemm"] / "raw", ROOT / sources["gemm"] / "review/evaluation.json")
        runtime = _read(sources["gemm"] + "/raw/runtime.json")
        for record in gemm.records.values():
            ident, case = record["identity"], record["identity"]["case"]
            stack = normalize_stack(runtime, case["device"])
            add(_gemm_sig(case["device"], stack, ident["implementation"], case), record["duration_s"],
                f"{sources['gemm']}:{record['case_id']}", ident["contract"], stack, case["device"])

        gpu1 = ReviewedGpu1Store(ROOT / sources["gpu1"] / "raw", ROOT / sources["gpu1"] / "review/evaluation.json",
                                 excluded_cases=tuple(c for s, c in REFERENCE_ONLY if s == "gpu1"))
        runtime = _read(sources["gpu1"] + "/raw/runtime.json")
        fused_identity = _read(sources["gpu1"] + "/raw/fused_identity.json")
        fused_runtime = _read(sources["gpu1"] + "/raw/fused_runtime.json")
        for record in gpu1.records.values():
            case, ident = record["case"], record["identity"]
            ordinal = case["device"]
            stack = normalize_stack(runtime, ordinal)
            source = f"{sources['gpu1']}:{record['case_id']}"
            if case["kind"] == "stage":
                add(_stage_sig(case["stage"], ordinal, stack, ident["implementation"],
                               ident["implementation"]["tensors"]["output"]), record["duration_s"], source,
                    ident["contract"], stack, ordinal)
            elif case["kind"] == "gemm":
                add(_gemm_sig(ordinal, stack, ident["implementation"], case), record["duration_s"], source,
                    ident["contract"], stack, ordinal)
            else:
                if normalize_stack(fused_runtime, ordinal) != stack:
                    raise ValueError("GPU1 LightOP runtime differs from the batch runtime")
                add(_fused_sig(ordinal, stack, fused_identity["kernel_key"]), record["duration_s"],
                    source + f":kernel_key={fused_identity['kernel_key_id']}",
                    fused_identity["measurement_contract"], stack, ordinal)
        src = sources["add_rmsnorm_gpu0"]
        review = _read(src + "/review/evaluation.json")
        for name, expected in review["raw_sha256"].items():
            if Path(name).name != name or _sha(f"{src}/raw/{name}") != expected:
                raise ValueError("Reviewed Add+RMSNorm receipt changed: " + name)
        if _read(src + "/raw/identity.json") != review["identity"]:
            raise ValueError("Reviewed Add+RMSNorm identities differ")
        runtime = _read(src + "/raw/runtime.json")
        for case_id, quality in review["cases"].items():
            if quality["status"] != "quality_passed" or ("add_rmsnorm_gpu0", case_id) in REFERENCE_ONLY:
                continue
            ident = review["identity"]["cases"][case_id]["case_identity"]
            case, impl, ordinal = ident["case"], ident["implementation"], ident["case"]["device"]
            stack = normalize_stack(runtime, ordinal)
            duration, source = quality["statistics"]["median_us"] / 1e6, f"{src}:{case_id}"
            if case["kind"] == "stage":
                out = {"shape": case["shape"], "dtype": case["dtype"],
                       "strides": list(LayerInvariantMeasuredProfile._contiguous(case["shape"]))}
                add(_stage_sig(case["name"], ordinal, stack, impl["implementation"], out), duration, source, ident["contract"],
                    stack, ordinal)
            else:
                version = impl["implementation"]
                key = {"pattern_id": "add_rmsnorm", "implementation": {"id": version["id"], "loaded_source_sha256": {
                           "adapter": version["version"].split(":")[1], "native": version["op_so_sha256"],
                           "package": _lightop_package_sha()}},
                       "external_inputs": [{"shape": impl["tensors"][r]["shape"], "dtype": impl["tensors"][r]["dtype"]}
                                           for r in ("x", "residual")]}
                add(_fused_sig(ordinal, stack, key), duration, source, ident["contract"], stack, ordinal)
        for sig, values in replicates.items():
            durations = [d for d, _ in values]
            spread = (max(durations) - min(durations)) / min(durations)
            if spread > REPLICATE_SPREAD_LIMIT:
                raise ValueError(f"replicate measurements of one condition differ by {spread:.1%}: "
                                 + ", ".join(p for _, p in values))
            self.records[sig] = {"duration_s": sum(durations) / len(durations),
                                 "provenance": " + ".join(sorted(p for _, p in values)),
                                 "replicates": [{"duration_us": d * 1e6, "provenance": p} for d, p in values],
                                 "replicate_spread": spread, "combination": "mean of replicate medians"}
        self.sources = dict(sources)


class LayerInvariantMeasuredProfile:
    """HardwareCostProfile adapter: measured record by condition, else base with a reason."""

    def __init__(self, base, store, declaration=None):
        self.base, self.store = base, store
        self.declaration = declaration or execution_declaration()
        self.hits, self.misses = {}, {}

    def __getattr__(self, name):
        return getattr(self.base, name)

    @staticmethod
    def _contiguous(shape):
        strides, stride = [], 1
        for size in reversed(shape):
            strides.insert(0, stride)
            stride *= size
        return tuple(strides)

    def _declared_layout_ok(self, node):
        # Graph metadata may be silent (executor contract applies) but must not contradict it.
        layout, strides = node.extra.get("layout"), node.extra.get("strides")
        shape = tuple(node.extra.get("output_shape") or ())
        return layout in (None, "unknown", "contiguous") and (strides is None or tuple(strides) == self._contiguous(shape))

    def signature(self, graph, group):
        nodes = [graph.nodes[n] for n in group.node_ids]
        device = group.device
        stack = self.store.stacks.get(device)
        if stack is None:
            return None, "no_records_for_gpu"
        stack = json.dumps(stack, sort_keys=True)
        pattern = group.pattern_id if hasattr(group, "pattern_id") else group.extra.get("pattern_id")
        if len(nodes) == 1 and nodes[0].op_type in ("silu", "mul", "add", "rmsnorm"):
            node = nodes[0]
            if node.extra.get("dtype") != "bf16" or not self._declared_layout_ok(node):
                return None, "dtype_or_layout_outside_contract"
            if node.op_type == "mul":
                producers = [graph.nodes[e.src].op_type for e in graph.in_edges(group.node_ids[0])]
                if "silu" not in producers:
                    return None, "mul_left_operand_not_silu"
            if node.op_type in ("add", "rmsnorm"):
                incoming = [graph.nodes[e.src] for e in graph.in_edges(group.node_ids[0])]
                if (len(incoming) != (2 if node.op_type == "add" else 1)
                        or any(tuple(i.extra.get("output_shape") or ()) != tuple(node.extra["output_shape"]) for i in incoming)):
                    return None, node.op_type + "_operands_outside_recipe"
                if node.op_type == "rmsnorm" and node.param_bytes != 2 * node.extra["output_shape"][-1]:
                    return None, "rmsnorm_weight_outside_recipe"
            decl = self.declaration["stage"][node.op_type]
            shape = tuple(node.extra["output_shape"])
            return ("stage", node.op_type, device, stack, decl["id"], decl["version"], shape, "bf16",
                    self._contiguous(shape)), None
        if len(nodes) == 1 and nodes[0].op_type == "matmul":
            shape = nodes[0].extra.get("shape") or {}
            if nodes[0].extra.get("dtype") != "bf16":
                return None, "dtype_outside_contract"
            decl = self.declaration["gemm"]
            return ("gemm", device, stack, decl["id"], decl["version"], shape.get("m"), shape.get("n"),
                    shape.get("k"), "bf16"), None
        if pattern == "silu_mul" and [n.op_type for n in nodes] == ["silu", "mul"]:
            decl = self.declaration["fused"]["silu_mul"]
            members = set(group.node_ids)
            inputs = tuple(sorted((tuple(graph.nodes[e.src].extra["output_shape"]), graph.nodes[e.src].extra["dtype"])
                                  for n in group.node_ids for e in graph.in_edges(n) if e.src not in members))
            return ("fused", "silu_mul", device, stack, decl["id"], decl["adapter"], decl["native"],
                    decl["package"], inputs), None
        if pattern == "add_rmsnorm" and [n.op_type for n in nodes] == ["add", "rmsnorm"]:
            decl = self.declaration["fused"]["add_rmsnorm"]
            add_inputs = tuple(sorted((tuple(graph.nodes[e.src].extra["output_shape"]), graph.nodes[e.src].extra["dtype"])
                                      for e in graph.in_edges(group.node_ids[0])))
            return ("fused", "add_rmsnorm", device, stack, decl["id"], decl["adapter"], decl["native"],
                    decl["package"], add_inputs), None
        if all(n.op_type == "input" for n in nodes):
            return None, "logical_input"
        return None, "no_measured_recipe_for_group"

    def estimate_group(self, graph, group):
        signature, reason = self.signature(graph, group)
        record = self.store.records.get(signature) if signature is not None else None
        if record is not None:
            d = record["duration_s"]
            self.hits[record["provenance"]] = self.hits.get(record["provenance"], 0) + 1
            return GroupCost(d, d, 0., 0., .95, "measured_layer_invariant",
                             ("measurement_context:isolated_case", "profile:" + record["provenance"]))
        if reason is None:
            reason = f"no_record:{signature[0]}:{signature[1]}:gpu{group.device}"
        self.misses[reason] = self.misses.get(reason, 0) + 1
        cost = self.base.estimate_group(graph, group)
        return replace(cost, ood_reasons=tuple(sorted(set(cost.ood_reasons) | {"measured_profile_miss:" + reason})))
