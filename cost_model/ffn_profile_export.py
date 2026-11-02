"""Recorded B4 observations -> existing canonical group keys, never admission.

No local Torch/GPU observation is used to fill an archived environment.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import statistics

from cost_model.fusion_kernel_profile import GroupKernelKeys, key_id
from cost_model.kerv_profile import digest
from cost_model.measurement_labels import MeasurementContractError, MeasurementLabel, MeasurementLabelLookup, measurement_contract_id
from simulator.dataset import graph_from_dict
from simulator.hardware import hardware_from_dict
from simulator.schema import OpGraph, OpNode, TensorEdge, FusionGroup

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "L000.mlp."
USAGES = {"singleton": "singleton_identity_inspection", "fused": "selected_fused_identity_inspection",
          "native_chain": "native_chain_comparison", "transfer": "observed_driver_copy_inspection"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def tensor_schema(meta):
    if meta.get("layout_source") != "runtime Tensor observation":
        raise ValueError("Tensor layout lacks recorded runtime provenance")
    shape, strides = meta["shape"], meta["strides"]
    if not isinstance(shape,list) or not isinstance(strides,list) or not shape or len(shape) != len(strides) or any(type(s) is not int or s <= 0 for s in shape):
        raise ValueError("invalid Tensor shape/strides")
    if any(type(s) is not int or s < 0 for s in strides):
        raise ValueError("unknown Tensor strides cannot become contiguous")
    dtype = {"torch.bfloat16": "bf16", "torch.float16": "fp16", "torch.float32": "fp32"}[meta["dtype"]]
    expected, contiguous = 1, True
    for size, stride in zip(reversed(shape), reversed(strides)):
        contiguous &= size == 1 or stride == expected
        expected *= size
    return {"shape": list(shape), "dtype": dtype, "strides": list(strides),
            "layout": "contiguous" if contiguous else "strided"}


def tensor_pool(metas, device):
    if len(metas) != 50 or any(m["device"] != f"cuda:{device}" for m in metas):
        raise ValueError("Tensor pool size/device differs from measurement")
    if len({m["data_ptr"] for m in metas}) != 50:
        raise ValueError("independent invocation buffers were not observed")
    descriptors = [tensor_schema(m) for m in metas]
    if any(d != descriptors[0] for d in descriptors):
        raise ValueError("Tensor pool has inconsistent metadata")
    return descriptors[0]


def observed_key(record, placed):
    """Reuse fusion-group-kernel.v1 over the actual source graph and metadata."""
    row, identity = record["row_id"], record["identity"]
    graph = graph_from_dict(copy.deepcopy(placed["graph"]))
    hardware = hardware_from_dict(copy.deepcopy(placed["hardware"]))
    implementation = copy.deepcopy(identity["implementation"])
    build = identity["actual_torch_build"]
    if not implementation.get("id") or not implementation.get("version") or not build.get("torch_extension_sha256"):
        raise ValueError("implementation identity incomplete")
    if implementation.get("collector_source_sha256") != build.get("collector_source_sha256"):
        raise ValueError("implementation differs from recorded executing collector")
    for index, observed in identity["devices"].items():
        dev = hardware.devices[int(index)]
        obs, env = observed["observed"], observed["environment"]
        if not obs.get("uuid") or obs["uuid"] == "unavailable" or not obs.get("architecture"):
            raise ValueError("hardware identity incomplete")
        if env["device_uuid"] != obs["uuid"] or env["device_architecture"] != obs["architecture"]:
            raise ValueError("device observation/environment mismatch")
        dev.name, dev.max_mem = obs["name"], obs["total_memory"]
        dev.extra["physical_index"] = int(index)
        dev.extra["source"] = {"kind": "runtime_observation", "uuid": obs["uuid"], "architecture": obs["architecture"]}
    kind = "transfer" if row.startswith("copy/") else "native_chain" if "/native@" in row else "fused" if "/fused@" in row else "singleton"
    device = int(row.split("->")[-1]) if kind == "transfer" else int(row.rsplit("gpu",1)[1])
    env = identity["devices"][str(device)]["environment"]
    runtime = {"framework": "torch", "framework_version": env["torch"], "accelerator": "HIP",
               "accelerator_version": env["hip"], "triton_version": env["triton"], "driver_version": env["driver_version"],
               "torch_git_revision": build["git_revision"],
               "flags": {k:build[k] for k in ("grad_enabled","matmul_fp32_precision","allow_bf16_reduced_precision_reduction")}}
    implementation.pop("weight", None)  # Fixture label, not an implementation.
    implementation.pop("kerv_binding", None)
    implementation["torch_extension_sha256"] = build["torch_extension_sha256"]
    implementation["functional_source_sha256"] = build["native_binding"]["functional_source_sha256"]
    if kind == "fused":
        loaded = identity["loaded_kerv_source"]
        if implementation["version"] != "sha256:"+loaded["module_sha256"]:
            raise ValueError("fused version differs from executed source")
        implementation["loaded_source"] = {k:loaded[k] for k in ("module_sha256","git_revision")}
    elif implementation["version"] != build["torch_module_version"]:
        raise ValueError("Reference version differs from recorded executing build")
    gaps = set()
    def assign(node_id, tensor):
        node = graph.nodes[node_id]
        if node.extra["output_shape"] != tensor["shape"] or node.extra["dtype"] != tensor["dtype"]:
            raise ValueError("recorded Tensor disagrees with source graph")
        node.extra.update(layout=copy.deepcopy(tensor["layout"]), strides=copy.deepcopy(tensor["strides"]))
    output = tensor_pool(identity["output_layouts"], device)
    alias = {"output_alias_input": None, "inplace": False}
    if kind not in ("fused","transfer"):
        recorded_inputs = identity["inputs"]
        metadata = recorded_inputs if isinstance(recorded_inputs,list) else recorded_inputs["gates"]+[recorded_inputs["up"]]
        if "weight" in identity: metadata = metadata+[identity["weight"]]
        if {m["data_ptr"] for m in metadata}.intersection(o["data_ptr"] for o in identity["output_layouts"]):
            raise ValueError("out-of-place output aliases a recorded input")
    if kind == "transfer":
        _, size, direction = row.split("/"); source, target = map(int,direction.split("->"))
        inp = tensor_pool(identity["inputs"], source)
        destination = tensor_pool(identity["destinations"], target)
        if output != destination or any(a["data_ptr"] != b["data_ptr"] for a,b in zip(identity["output_layouts"], identity["destinations"])):
            raise ValueError("copy output does not alias the preallocated destination")
        size_bytes = math.prod(inp["shape"])*{"bf16":2,"fp16":2,"fp32":4}[inp["dtype"]]
        if int(size) != size_bytes or inp != output:
            raise ValueError("copy size or endpoint layout mismatch")
        graph = OpGraph(name="observed-preallocated-copy")
        for name, schema, op in (("source",inp,"input"),("copy",output,"copy")):
            graph.add_node(OpNode(name,op_type=op,out_bytes=size_bytes,extra={"output_shape":schema["shape"],"dtype":schema["dtype"],"layout":schema["layout"],"strides":schema["strides"]}))
        graph.add_edge(TensorEdge("source","copy",bytes=size_bytes))
        nodes, pattern = ["copy"], "preallocated_copy"
        transfer = {"source_device":source,"destination_device":target,
                    "endpoints": {k: {"observed":v["observed"],"environment":v["environment"]} for k,v in identity["devices"].items()},
                    "mechanism":"preallocated Tensor.copy_", "physical_route":"not_certified"}
        alias = {"output_alias_input": "preallocated_destination", "source_readonly":True}
    else:
        name = row.split("@",1)[0]
        if kind in ("native_chain","fused"):
            nodes, pattern = [PREFIX+"silu",PREFIX+"gated_mul"], "silu_mul"
        else:
            nodes, pattern = [PREFIX+name], "unfused"
        if name.endswith("_proj"):
            inp = tensor_pool(identity["inputs"],device)
            edges = graph.in_edges(nodes[0]); assert len(edges) == 1
            assign(edges[0].src,inp)
            weight = tensor_schema(identity["weight"])
            if identity["weight"]["device"] != f"cuda:{device}": raise ValueError("weight on wrong device")
            dims = graph.nodes[nodes[0]].extra["shape"]
            if weight["shape"] != [dims["k"],dims["n"]]: raise ValueError("weight shape differs from graph")
            graph.nodes[nodes[0]].extra["reference_parameters"] = {"weight":weight}
        else:
            gate = tensor_pool(identity["inputs"]["gates"],device)
            assign(PREFIX+"gate_proj",gate)
            if name != "silu":
                up_meta = identity["inputs"]["up"]
                if up_meta["device"] != f"cuda:{device}": raise ValueError("up on wrong device")
                assign(PREFIX+"up_proj",tensor_schema(up_meta))
            if name == "gated_mul":
                # The collector observed gates, but the actual left operand is
                # a separate precomputed SiLU pool. Never relabel its layout.
                graph.nodes[PREFIX+"silu"].extra.update(layout=None,strides=None)
                gaps.add("mul_left.runtime_tensor_metadata_missing")
            elif kind in ("native_chain","fused"):
                graph.nodes[PREFIX+"silu"].extra.update(layout={"materialization":
                    "internal_fused_expression" if kind=="fused" else "native_intermediate_not_observed"},strides=None)
        assign(nodes[-1],output)
        if kind == "fused":
            if any(o["data_ptr"] != g["data_ptr"] for o,g in zip(identity["output_layouts"],identity["inputs"]["gates"])):
                raise ValueError("fused gate alias not observed")
            alias = {"output_alias_input":"gate", "inplace":True}
    keys = GroupKernelKeys(graph,hardware,{pattern:implementation},runtime)
    key, base_gaps = keys.describe(FusionGroup(nodes,device,{"pattern_id":pattern}))
    key["execution_kind"], key["alias_semantics"] = kind, alias
    if kind == "transfer": key["transfer"] = transfer
    gaps.update(base_gaps)
    return key, sorted(gaps), kind


def export_bundle(root, review):
    root = Path(root)
    for rel, sha in review["raw_file_sha256"].items():
        if digest(root/rel) != sha: raise ValueError("source evidence changed: "+rel)
    grid = read(root/"formal/grid.json")
    records = []
    for row in review["cpu_admission_candidates"]:
        source = grid["cells"][row["row_id"]]
        if source["status"] != "quality_passed": raise ValueError("failed row cannot be exported")
        device = int(source["row_id"].split("->")[-1]) if source["row_id"].startswith("copy/") else int(source["row_id"].rsplit("gpu",1)[1])
        placed_path = f"placed_gpu{device}/placed.json"
        key,gaps,kind = observed_key(source,read(root/placed_path))
        relative = "formal/"+source["worker_evidence"]+"/formal/"+source["row_id"].replace("/","_").replace(">","to")
        evidence_names = ["plan.json", "formal/grid.json", placed_path]+[relative+"/"+n for n in ("identity.json","result.json","samples.json","replay_trace.json")]
        values = [s["duration_us"] for s in source["samples"]]
        record = {"row_id":source["row_id"], "kernel_key":key, "kernel_key_id":key_id(key), "identity_gaps":gaps,
                  "kind":kind,"usage":USAGES[kind], "measurement_contract":source["measurement_contract"],
                  "measurement_contract_id":source["measurement_contract_id"],
                  "statistics":{**source["quality"]["statistics"]["body"], "min_us":min(values),"max_us":max(values)},
                  "quality_status":"quality_passed", "admission":"conditional_dev", "cost_use_open":False,
                  "source_record_sha256":digest(root/(relative+"/result.json")),
                  "evidence":{n:review["raw_file_sha256"][n] for n in evidence_names},
                  "admission_blockers":["no_scoped_admission_decision"] + (["direct_P2P_route_not_certified"] if kind=="transfer" else []),
                  "scope_notes": "native internal temporary unobserved; only boundary layouts are identity requirements" if kind=="native_chain" else None}
        if measurement_contract_id(record["measurement_contract"]) != record["measurement_contract_id"]: raise ValueError("contract hash mismatch")
        records.append({**record,"record_id":key_id(record)})
    payload = {"format":"fusion-placement-search.ffn-profile-export.v1", "source_archive_sha256":review["archive_sha256"],
               "records":records, "new_accepted_profiles":0, "source_ref":review["source_ref"],
               "exporter_source_sha256": {name:digest(ROOT/name) for name in
                   ("cost_model/ffn_profile_export.py","cost_model/fusion_kernel_profile.py","cost_model/measurement_labels.py")}}
    return {**payload,"export_id":key_id(payload)}


class FFNExportStore:
    """Exact audit inspection is distinct from the shared strict cost gate."""
    def __init__(self, bundle):
        if key_id({k:v for k,v in bundle.items() if k!="export_id"}) != bundle["export_id"]:
            raise ValueError("export identity mismatch")
        self.records = {}
        for record in bundle["records"]:
            if (record["record_id"] != key_id({k:v for k,v in record.items() if k!="record_id"}) or
                record["kernel_key_id"] != key_id(record["kernel_key"]) or
                record["measurement_contract_id"] != measurement_contract_id(record["measurement_contract"])):
                raise ValueError("record/key/contract identity mismatch")
            if record["admission"] != "conditional_dev" or record["cost_use_open"]:
                raise MeasurementContractError("this export carries no admission authority")
            index = record["kernel_key_id"],record["measurement_contract_id"]
            if index in self.records: raise ValueError("duplicate key/contract")
            self.records[index] = copy.deepcopy(record)

    @classmethod
    def load(cls, path, evidence_root):
        bundle = read(path)
        root = Path(evidence_root).resolve()
        for record in bundle["records"]:
            for rel,sha in record["evidence"].items():
                target = (root/rel).resolve()
                if not target.is_relative_to(root) or digest(target)!=sha: raise ValueError("evidence mismatch")
        # Rebuild from the immutable source rows rather than trust copied keys.
        review = read(ROOT/"data/ffn_b4_run05_review/report.json")
        if key_id(export_bundle(root,review)) != key_id(bundle): raise ValueError("export differs from source")
        return cls(bundle)

    def inspect_exact(self, kernel_key, contract_id, *, usage):
        index = key_id(kernel_key),contract_id
        if index not in self.records: raise KeyError("no exact key/contract record")
        record = self.records[index]
        if usage != record["usage"]: raise MeasurementContractError("record kind cannot substitute another cost scope")
        if record["identity_gaps"]: raise MeasurementContractError("incomplete identity: "+",".join(record["identity_gaps"]))
        return copy.deepcopy(record)

    def query_cost(self, kernel_key, contract_id, *, usage):
        record = self.inspect_exact(kernel_key,contract_id,usage=usage)
        label = MeasurementLabel(record["kernel_key_id"],contract_id,record["statistics"],record["admission"])
        return MeasurementLabelLookup({label.key_id:label}).lookup(label.key_id,contract_id)
