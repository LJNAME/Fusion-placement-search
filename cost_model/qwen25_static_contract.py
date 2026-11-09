"""3B static execution declarations, following the 0.6B condition overlay.

Declarations are target executor requirements, never model observations. Only
evidence provenance is removed when comparing complete Kernel conditions;
shape, strides, implementation, runtime, device and recipe remain mandatory.
"""
import copy
from dataclasses import replace
from collections import Counter

from calibration.kerv_add_rmsnorm_probe import EPS, ROOT, SHAPE, sha
from calibration.kerv_add_rmsnorm_timing import SOURCES
from calibration.native_add_rmsnorm_timing import comparable_contract
from cost_model.fusion_kernel_profile import GroupKernelKeys, KeyedKernelProfile, key_id
from cost_model.hardware_profile import GroupCost
from cost_model.qwen25_add_rmsnorm import GraphPlanningProfile, bind_candidate, direct_pairs
from cost_model.profile_hardware_class import hardware_class_audit,hardware_conditions
from integration.flagscale.kerv_add_rmsnorm import group_io


def conditions(value):
    """Class query conditions; physical ordinals and evidence stay in receipts."""
    if isinstance(value, dict):
        result={k:conditions(v) for k,v in value.items() if k!="layout_provenance"}
        return hardware_conditions(result) if "key_version" in result and "hardware" in result else result
    if isinstance(value, list): return [conditions(v) for v in value]
    return value


def execution_declaration(adapter, fused, native, hardware):
    return {"id":"qwen25.add_rmsnorm.hardware_class_graph_replay.v2",
            "basis":"same explicit target-executor declaration approach as 0.6B LayerInvariantMeasuredProfile",
            "model_runtime_observed":False,
            "tensor_allocation":"fresh contiguous activations; no views aliasing the disposable matmul output",
            "matmul_recipe":"aten.mm.out.linear_weight_out_in.preallocated.v1",
            "matmul_source_sha256":sha(ROOT/"calibration/gemm_projection_calibration.py"),
            "shape":[1,256,2048],"layout":"contiguous","strides":[524288,2048,1],"dtype":"bf16",
            "weight":{"shape":[2048],"strides":[1],"dtype":"bf16","aliases_activation":False},
            "eps":EPS,"mode":"graph_replay","grad_enabled":False,"autocast_enabled":False,
            "fused_implementation":adapter.implementation,"native_implementation":native["implementation"],
            "target_runtime":fused["kernel_key"]["runtime"],"target_hardware":fused["kernel_key"]["hardware"],
            "hardware_class_audit":hardware_class_audit(hardware,fused),
            "fused_measurement_contract":fused["measurement_contract"],
            "native_measurement_contract":native["measurement_contract"],
            "fused_outputs":{"sum":"aliases consumed x; published at original Add node",
                             "normalized":"fresh output; published at original Norm node"},
            "native_outputs":"out-of-place Add then F.rms_norm; both outputs preserved",
            "enforcement":"existing Adapter rejects incompatible shape/dtype/stride/device/grad/live aliases before dispatch; no implicit copy",
            "scope":"isolated Graph Kernel costs, not full-model timing or a claim about current model tensors"}


def _tensor(shape, declaration_id):
    stride=1; strides=[]
    for n in reversed(shape): strides.insert(0,stride);stride*=n
    return {"shape":list(shape),"dtype":"bf16","layout":"contiguous","strides":strides,
            "layout_provenance":{"kind":"graph_declaration","source_id":declaration_id}}


def declared_keys(bound, declaration, implementation, pattern):
    graph=copy.deepcopy(bound.placed.graph);hardware=copy.deepcopy(bound.placed.hardware)
    for node in (bound.x_node,bound.residual_node,*bound.node_ids):
        spec=_tensor(SHAPE,declaration["id"])
        graph.nodes[node].extra.update({k:spec[k] for k in ("layout","strides","layout_provenance")})
    device=bound.placed.plan.groups[bound.index].device
    target=declaration["hardware_class_audit"]["devices"][str(device)]
    hardware.devices[device].name=target["fingerprint"]["name"]
    hardware.devices[device].extra.update({"class":target["fingerprint"]["class"],"physical_index":device,
                                         "source":copy.deepcopy(target["source"])})
    binding=copy.deepcopy(implementation)
    if pattern=="add_rmsnorm":
        binding.update(weight=_tensor((2048,),declaration["id"]),
            output_aliases={"sum":"consumed_x","normalized":"independent"},
            execution_sources={p:sha(ROOT/p) for p in SOURCES[1:4]})
    keys=GroupKernelKeys(graph,hardware,{pattern:binding},declaration["target_runtime"])
    group=copy.deepcopy(bound.placed.plan.groups[bound.index]);group.extra["pattern_id"]=pattern
    return keys,group


def qualify_static(graph,hardware,adapter,pair,device,declaration,fused,native):
    inspected=(*pair,*graph.predecessors(pair[0]))
    if any(graph.nodes[n].extra.get(k) is None for n in inspected for k in ("output_shape","dtype","phase","layer")):
        return {"capability":"insufficient_info","profile":"missing","reasons":["missing_static_metadata"]},None
    try:
        x,residual=group_io(graph,pair)
    except ValueError as error:
        return {"capability":"not_fusible","profile":"unavailable","reasons":[str(error)]},None
    # Node names alone are insufficient to prove a disposable fresh result.
    producer=graph.nodes[x]
    if producer.op_type!="matmul" or producer.extra.get("shape",{}).get("n")!=2048:
        return {"capability":"insufficient_info","profile":"missing","reasons":["producer_outside_declared_fresh_matmul_recipe"]},None
    if any(graph.nodes[n].extra.get("alias_of") is not None or graph.nodes[n].extra.get("view_of") is not None
           or graph.nodes[n].extra.get("storage_aliases") for n in inspected):
        return {"capability":"not_fusible","profile":"unavailable","reasons":["declared_storage_alias_conflicts_with_consume_x"]},None
    base={"capability":"fusible","profile":"missing","reasons":[],
          "evidence_kind":"static_execution_declaration","model_runtime_observed":False,
          "declaration_id":declaration["id"],
          "static_facts":{"same_shape_dtype_layer":True,"x_only_consumer":pair[0],
              "x_producer":x,"residual":residual,"double_outputs_preserved":True},
          "runtime_requirements":["contiguous declared strides","no live storage alias of consumed x","HIP no-grad Graph target"]}
    hardware_evidence=declaration["hardware_class_audit"]["devices"].get(str(device),{})
    if not hardware_evidence.get("class_and_stack_match"):
        return {**base,"reasons":["hardware_class_or_stack_mismatch"]},None
    if (declaration!=execution_declaration(adapter,fused,native,hardware)
        or declaration["fused_implementation"]!=adapter.implementation
        or declaration["target_runtime"]!=fused["kernel_key"]["runtime"]
        or declaration["target_runtime"]!=native["kernel_key"]["runtime"]
        or declaration["fused_measurement_contract"]!=fused["measurement_contract"]
        or declaration["native_measurement_contract"]!=native["measurement_contract"]
        or comparable_contract(fused["measurement_contract"])!=comparable_contract(native["measurement_contract"])):
        return {**base,"reasons":["implementation_runtime_or_contract_mismatch"]},None
    bound=bind_candidate(graph,hardware,adapter,pair,device)
    fk,fg=declared_keys(bound,declaration,adapter.implementation,"add_rmsnorm")
    nk,ng=declared_keys(bound,declaration,declaration["native_implementation"],"native_add_rmsnorm")
    fkey,fgaps=fk.describe(fg);nkey,ngaps=nk.describe(ng)
    if fgaps or ngaps or conditions(fkey)!=conditions(fused["kernel_key"]) or conditions(nkey)!=conditions(native["kernel_key"]):
        return {**base,"reasons":["complete_kernel_conditions_mismatch"]},None
    return {**base,"profile":"static_contract_equivalent" if device==0 else "hardware_class_transfer",
            "measured_on":0,"reuse_on":device,
            "reuse_kind":"same_device" if device==0 else "homogeneous_hardware_class_transfer",
            "hardware_class_key":hardware_evidence["hardware_class_key"],
            "target_device_independently_measured":device==0,
            "kernel_key_id":key_id(fkey),
            "measured_kernel_key_id":fused["kernel_key_id"],"native_measured_key_id":native["kernel_key_id"],
            "measurement_contract_id":fused["measurement_contract_id"],
            "condition_id":key_id(conditions(fkey)),
            "reuse_authorization":"complete class conditions equal; target declaration enforced by existing Adapter"},(fk,fg)


class StaticContractProfile(GraphPlanningProfile):
    def __init__(self,graph,hardware,adapter,measured,native_record,*,declaration=None):
        super().__init__(graph,hardware,adapter,measured,conditional=False)
        self.native_record=copy.deepcopy(native_record);self.require_measured_pair=True
        self.declaration=copy.deepcopy(declaration or execution_declaration(adapter,self.accepted,native_record,self.hardware))
        self.assumption_id=key_id(self.declaration)
        self.qualifications={};self.reuse={};self.native_reuse={};self.fused_query_counts=Counter()
        for pair in direct_pairs(graph):
            for device in range(self.n_devs):
                q,request=qualify_static(graph,self.hardware,adapter,pair,device,self.declaration,self.accepted,native_record)
                self.qualifications[(pair,device)]=q
                if request is None: continue
                keys,group=request
                latency=measured.latencies[self.accepted["kernel_key_id"]]
                self.reuse[(pair,device)]=(KeyedKernelProfile(keys,{q["kernel_key_id"]:latency},self,strict=True,
                    measurement_contract_id=self.accepted["measurement_contract_id"],allow_unbound=False),group)
                d=native_record["duration_s"]
                source="measured_native_chain_static_contract" if device==0 else "transferred_native_chain_hardware_class"
                reasons=("measurement_context:isolated_case","static_execution_declaration")
                if device!=0: reasons+= ("hardware_class_transfer_not_target_measurement",)
                self.native_reuse[(pair,device)]=GroupCost(d,d,0.,0.,.95,source,reasons)

    def estimate_group(self,graph,group):
        cost=super().estimate_group(graph,group)
        if (tuple(group.node_ids),group.device) in self.reuse:
            self.fused_query_counts[group.device]+=1
            reasons=("measurement_context:isolated_case","static_execution_declaration")
            if group.device!=0: reasons+=("hardware_class_transfer_not_target_measurement",)
            return replace(cost,source="measured_static_contract_equivalent" if group.device==0 else "transferred_hardware_class",
                           ood_reasons=reasons)
        return cost
