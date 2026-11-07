"""3B AddRMS capability/cost audit and conditional CPU search integration.

Unobserved model layouts never inherit benchmark observations. A measured cost
is reusable only with an explicit per-group runtime context and exact key and
contract equality. The default planning experiment uses analytic Graph proxies.
"""
import copy
from collections import Counter
from dataclasses import asdict, replace
import json

from calibration.kerv_add_rmsnorm_probe import ROOT, EPS, SHAPE
from calibration.kerv_add_rmsnorm_timing import kernel_identity
from cost_model.fusion_kernel_profile import candidate_structure, key_id, KeyedKernelProfile
from cost_model.hardware_profile import AnalyticHardwareProfile, GroupCost
from cost_model.kerv_add_rmsnorm_profile import load_profile
from cost_model.measurement_labels import measurement_contract_id
from cost_model.structural import StructuralCostModel
from cost_model.search_evaluation import _plan_key, INFEASIBLE_PENALTY_S
from integration.flagscale.kerv_add_rmsnorm import KervAddRmsnormAdapter, group_io
from simulator import FusionGroup, OpGraph, Plan
from simulator.fusion_rules import FusionPattern, FusionPatternRegistry, FusionRuleError, enumerate_fusion_candidates
from simulator.placement_artifacts import PlacedOpGraph, graph_fingerprint
from tools.evaluate_search_quality import BudgetScorer, EvaluationBudgetReached

PROFILE_DIRECTORY = ROOT / "data/kerv_add_rmsnorm_timing_01"
PROXY_CONFIG = ROOT / "configs/ffn_profile_search.json"
SCHEDULING_POLICY = "dependency_level_node_id_v1"


def direct_pairs(graph):
    return sorted({(e.src, e.dst) for e in graph.edges
                   if graph.nodes[e.src].op_type == "add" and graph.nodes[e.dst].op_type == "rmsnorm"})


def reference_plan(graph, registry, *, fused=False, device=0):
    selected = {c.node_ids[-1]:c for c in enumerate_fusion_candidates(graph, registry, require_executable=True)
                if fused and len(c.node_ids) > 1}
    skipped = {c.node_ids[0] for c in selected.values()}
    groups = []
    for node in graph.topological_order():
        if node in skipped:
            continue
        groups.append(selected[node].to_group(device) if node in selected else
                      FusionGroup([node], device, {"pattern_id":"unfused"}))
    return Plan(groups=groups)


def bind_candidate(graph, hardware, adapter, pair, device):
    groups = []
    for node in graph.topological_order():
        if node == pair[0]:
            continue
        groups.append(FusionGroup(list(pair), device, {"pattern_id":"add_rmsnorm", "implementation":adapter.implementation})
                      if node == pair[1] else FusionGroup([node], device, {"pattern_id":"unfused"}))
    placed = PlacedOpGraph(graph, hardware, Plan(groups=groups))
    index = next(i for i,g in enumerate(groups) if tuple(g.node_ids) == tuple(pair))
    return adapter.bind(placed, index, {device:f"cuda:{device}"})


def qualify_context(graph, hardware, adapter, pair, device, context, accepted):
    """No context is inferred from the Layer0 benchmark or another layer."""
    inspected = [*pair, *graph.predecessors(pair[0])]
    missing = [f"{n}:{field}" for n in inspected for field in ("output_shape","dtype","phase")
               if graph.nodes[n].extra.get(field) is None]
    if graph.nodes[pair[1]].extra.get("local_name") == "pre_mlp_norm":
        missing.extend(n+":layer" for n in inspected if type(graph.nodes[n].extra.get("layer")) is not int)
    if missing:
        return {"capability":"insufficient_info", "profile":"missing", "reasons":["missing_graph_metadata",*missing]}
    try:
        x, residual = group_io(graph, pair)
    except ValueError as error:
        return {"capability":"not_fusible", "profile":"unavailable", "reasons":[str(error)]}
    if context is None:
        return {"capability":"insufficient_info", "profile":"missing",
                "reasons":["model_tensor_layout_unknown", "model_runtime_aliases_unknown", "execution_contract_unspecified"]}
    if (context.get("evidence_kind") != "model_runtime" or not context.get("source_id")
            or context.get("graph_fingerprint") != graph_fingerprint(graph)
            or context.get("node_ids") != list(pair) or context.get("device_index") != device):
        return {"capability":"insufficient_info", "profile":"missing", "reasons":["missing_or_stale_group_runtime_evidence"]}
    required = ("observations", "runtime", "device", "measurement_contract", "implementation", "storage_aliases")
    if any(k not in context for k in required):
        return {"capability":"insufficient_info", "profile":"missing", "reasons":["incomplete_runtime_context"]}
    if context["implementation"] != adapter.implementation:
        return {"capability":"not_fusible", "profile":"unavailable", "reasons":["implementation_mismatch"]}
    aliases = context["storage_aliases"]
    if any(k not in aliases for k in ("x_residual", "x_weight", "x_other_live")):
        return {"capability":"insufficient_info", "profile":"missing", "reasons":["alias_evidence_missing"]}
    if any(aliases[k] is not False for k in ("x_residual", "x_weight", "x_other_live")):
        return {"capability":"not_fusible", "profile":"unavailable", "reasons":["consumed_x_has_live_alias"]}
    observations = context["observations"]
    for role in ("x", "residual", "sum", "normalized", "weight"):
        tensor = observations.get(role, {})
        if tensor.get("requires_grad") is True:
            return {"capability":"not_fusible", "profile":"unavailable", "reasons":["inference_tensor_required:"+role]}
        shape, strides = ([2048], [1]) if role == "weight" else (list(SHAPE), [524288,2048,1])
        if tensor.get("layout") in (None, "unknown") or not tensor.get("strides"):
            return {"capability":"insufficient_info", "profile":"missing", "reasons":["runtime_layout_unknown:"+role]}
        if (tensor.get("shape") != shape or tensor.get("dtype") != "bf16" or tensor["layout"] != "contiguous"
                or tensor["strides"] != strides):
            return {"capability":"not_fusible", "profile":"unavailable", "reasons":["tensor_contract_mismatch:"+role]}
        provenance = tensor.get("layout_provenance", {})
        if provenance.get("kind") != "runtime_tensor" or provenance.get("device") != f"cuda:{device}":
            return {"capability":"insufficient_info", "profile":"missing", "reasons":["model_layout_not_observed:"+role]}
    runtime, contract = context["runtime"], context["measurement_contract"]
    if any(k not in runtime for k in ("accelerator","grad_enabled","autocast_enabled")) or not contract.get("mode"):
        return {"capability":"insufficient_info", "profile":"missing", "reasons":["runtime_or_contract_incomplete"]}
    if runtime.get("accelerator") != "HIP" or runtime.get("grad_enabled") is not False or runtime.get("autocast_enabled") is not False:
        return {"capability":"not_fusible", "profile":"unavailable", "reasons":["HIP_inference_context_required"]}
    if contract["mode"] != "graph_replay":
        return {"capability":"not_fusible", "profile":"unavailable", "reasons":["target_Graph_contract_mismatch"]}
    result = {"capability":"fusible", "profile":"missing", "reasons":[], "evidence_source":context["source_id"]}
    if device != 0:
        return {**result, "reasons":["no_device_specific_accepted_profile"]}
    if contract != accepted["measurement_contract"]:
        return {**result, "reasons":["measurement_contract_mismatch"]}
    bound = bind_candidate(graph, hardware, adapter, pair, device)
    try:
        identity = kernel_identity(bound, observations, runtime, context["device"])
    except ValueError as error:
        return {**result, "reasons":["canonical_identity_rejected:"+str(error)]}
    if identity["kernel_key"] != accepted["kernel_key"]:
        return {**result, "reasons":["canonical_key_mismatch"], "requested_key_id":identity["kernel_key_id"]}
    return {**result, "profile":"exact_equivalent", "kernel_key_id":identity["kernel_key_id"],
            "measurement_contract_id":measurement_contract_id(contract),
            "reuse_authorization":"exact_key_and_contract_with_explicit_group_runtime_evidence"}


class GraphPlanningProfile(AnalyticHardwareProfile):
    """Same Graph analytic assumptions on both arms; no implicit profile reuse."""
    def __init__(self, graph, hardware, adapter, accepted_profile, contexts=None, *, boundaries=(), conditional=True, native_record=None):
        hardware = copy.deepcopy(hardware)
        original_launch_us = hardware.launch_overhead * 1e6
        config = json.loads(PROXY_CONFIG.read_text(encoding="utf-8"))
        hardware.launch_overhead = config["projection_graph_overhead_us"] / 1e6
        hardware.fusion_overhead = 0.
        super().__init__(hardware)
        self.graph, self.adapter, self.boundaries = graph, adapter, set(boundaries)
        self.conditional = conditional
        self.native_record = copy.deepcopy(native_record)
        self.require_measured_pair = native_record is not None and not conditional
        self.accepted = json.loads((PROFILE_DIRECTORY/"review/profile.json").read_text(encoding="utf-8"))
        self.measured = accepted_profile
        self.qualifications, self.reuse, self.native_reuse = {}, {}, {}
        self.declaration = {"kind":"conditional_graph_planning", "layout":"contiguous required at runtime, not observed",
            "mode":"graph_replay", "calls_per_replay":50, "independent_x_buffers":50,
            "restoration_and_validation":"outside Events", "allocation":"captured private pool",
            "native_recipe":"Add then aten RMSNorm", "fused_recipe":"KERV consume-x dual-output",
            "eps":EPS, "eps_source":"explicit adapter/Reference contract, not graph observation",
            "native_dispatch_verified":False, "stage_overhead_scope":"logical op/group proxy, not a verified native kernel count",
            "graph_stage_overhead_us":config["projection_graph_overhead_us"], "graph_overhead_is_calibrated":False,
            "excluded_original_eager_launch_us":original_launch_us,
            "graph_overhead_source":"configs/ffn_profile_search.json", "kernel_cost_formula":"existing AnalyticHardwareProfile unchanged"}
        self.assumption_id = key_id(self.declaration)
        for pair in direct_pairs(graph):
            for device in range(self.n_devs):
                context = (contexts or {}).get(pair[-1]+f"@gpu{device}")
                qualification = qualify_context(graph, hardware, adapter, pair, device, context, self.accepted)
                self.qualifications[(pair,device)] = qualification
                if qualification["profile"] == "exact_equivalent":
                    bound = bind_candidate(graph, hardware, adapter, pair, device)
                    keys = kernel_identity(bound, context["observations"], context["runtime"], context["device"], return_keys=True)
                    self.reuse[(pair,device)] = (KeyedKernelProfile(keys, accepted_profile.latencies, self, strict=True,
                        measurement_contract_id=self.accepted["measurement_contract_id"], allow_unbound=False), bound.placed.plan.groups[bound.index])
                    if native_record is not None:
                        from cost_model.native_add_rmsnorm_profile import qualified_native_cost
                        native = qualified_native_cost(graph,hardware,adapter,pair,device,context,self.accepted,native_record)
                        if native is not None:
                            self.native_reuse[(pair,device)] = native

    def estimate_group(self, graph, group):
        if len(group.node_ids) == 1 and group.node_ids[0] in self.boundaries:
            return GroupCost(0.,0.,0.,0.,1.,"resident_slice_boundary")
        if len(group.node_ids)>1:
            qualification = self.qualifications[(tuple(group.node_ids),group.device)]
            if qualification["capability"] == "not_fusible" or (not self.conditional and qualification["capability"] != "fusible"):
                raise ValueError("fusion capability/contract unresolved or violated")
            if self.require_measured_pair and (tuple(group.node_ids),group.device) not in self.native_reuse:
                raise ValueError("same-contract Native/Fused measured pair required")
        selected = self.reuse.get((tuple(group.node_ids), group.device))
        if selected is not None:
            measured, target = selected
            cost = measured.estimate_group(measured.keys.graph, target)
            return replace(cost, source="measured_graph_key_equivalent")
        base = super().estimate_group(graph,group)
        if base.source == "logical_input":
            return base
        return replace(base, source="analytic_graph_proxy", confidence=min(base.confidence,.1),
                       ood_reasons=(*base.ood_reasons,"Graph_overhead_explicit_assumption","no_matching_measured_Graph_cost"))

    def estimate_transfer(self, source, target, nbytes):
        return replace(super().estimate_transfer(source,target,nbytes),source="analytic_link_proxy",
                       confidence=.1,ood_reasons=("not_a_measured_Graph_link",))

    def plan_blockers(self, plan):
        reasons = []
        for group in plan.groups:
            if any(n in self.boundaries for n in group.node_ids) and group.device != 0:
                reasons.append("slice_boundary_residency")
            if len(group.node_ids)>1:
                status = self.qualifications.get((tuple(group.node_ids),group.device),{}).get("capability")
                if status is None or status == "not_fusible" or (not self.conditional and status != "fusible"):
                    reasons.append("fusion_capability:"+str(group.node_ids))
                if self.require_measured_pair and (tuple(group.node_ids),group.device) not in self.native_reuse:
                    reasons.append("measured_pair_missing:"+str(group.node_ids))
        return reasons


def audit(graph, profile):
    all_candidates = enumerate_fusion_candidates(graph,profile.adapter.registry,require_executable=True)
    candidates = {tuple(c.node_ids):c for c in all_candidates if len(c.node_ids)>1}
    singletons = {c.node_ids[0]:c for c in all_candidates if len(c.node_ids)==1}
    rows = []
    for pair in direct_pairs(graph):
        device_rows = []
        for device in range(profile.n_devs):
            qualification = profile.qualifications[(pair,device)]
            native = []
            for node in pair:
                singleton = singletons[node]
                cost = profile.estimate_group(graph,candidate_structure(graph,singleton,device))
                native.append({"node":node,"source":cost.source,"duration_us":cost.duration*1e6})
            fused = None
            if (pair in candidates and qualification["capability"] != "not_fusible"
                and (profile.conditional or qualification["capability"]=="fusible")
                and (not profile.require_measured_pair or (pair,device) in profile.native_reuse)):
                cost = profile.estimate_group(graph,candidate_structure(graph,candidates[pair],device))
                fused = {"source":cost.source,"duration_us":cost.duration*1e6}
            chain = profile.native_reuse.get((pair,device))
            device_rows.append({"device":device,**qualification,"native":native,"fused":fused,
                                "native_chain":None if chain is None else {"source":chain.source,"duration_us":chain.duration*1e6,
                                    "kernel_key_id":profile.native_record["kernel_key_id"],
                                    "measurement_contract_id":profile.native_record.get("measurement_contract_id"),
                                    "requires":"both Native nodes colocated and reserved as one ordered execution unit"},
                                "native_measured_same_contract":chain is not None,
                                "reliable_measured_pair":chain is not None and (pair,device) in profile.reuse})
        statuses = {d["capability"] for d in device_rows}
        status = "fusible" if "fusible" in statuses else "insufficient_info" if "insufficient_info" in statuses else "not_fusible"
        nodes = (*graph.predecessors(pair[0]),*pair)
        rows.append({"nodes":list(pair),"status":status,"static_adapter_match":pair in candidates,
                     "graph_tensor_metadata":{n:{k:graph.nodes[n].extra.get(k,"unknown") for k in
                         ("output_shape","dtype","layout","strides","phase","layer")} for n in nodes},
                     "devices":device_rows})
    return {"scope":"direct two-node Add-to-RMSNorm pattern; no bypass of logical input nodes",
            "graph_fingerprint":graph_fingerprint(graph),"nodes":len(graph.nodes),"edges":len(graph.edges),
            "direct_positions":len(rows),"static_adapter_matches":len(candidates),
            "capability_counts":{s:sum(r["status"]==s for r in rows) for s in ("fusible","not_fusible","insufficient_info")},
            "exact_equivalent_profile_uses":len(profile.reuse), "available_benchmark_profiles":1,
            "matching_measured_native_chains":len(profile.native_reuse),
            "reliable_measured_pairs":sum(d["reliable_measured_pair"] for r in rows for d in r["devices"]),
            "native_query_sources":dict(Counter(n["source"] for r in rows for d in r["devices"] for n in d["native"])),
            "fused_query_sources":dict(Counter(d["fused"]["source"] if d["fused"] else "unavailable" for r in rows for d in r["devices"])),
            "profile_cross_layer_copy_performed":False,"rows":rows}


class NativeChainCostModel(StructuralCostModel):
    """Reserve a measured colocated Native chain intact, never divide its cost.

    Search/export plans remain unfused singletons. Only the scheduling view
    contracts a qualified Native pair. Both outputs become available at chain
    completion, so no independent stage timings or interleaving are invented.
    """
    def predict(self, graph, plan, profile):
        if not getattr(profile,"native_reuse",{}):
            return super().predict(graph,plan,profile)
        self.prepare(graph,plan,profile.n_devs)  # Validate the original plan first.
        by_node = {n:g for g in plan.groups for n in g.node_ids}
        selected = {pair:cost for (pair,device),cost in profile.native_reuse.items()
                    if all(len(by_node[n].node_ids)==1 and by_node[n].device==device for n in pair)}
        if not selected:
            return super().predict(graph,plan,profile)
        view = copy.deepcopy(plan)
        skipped = {pair[0] for pair in selected}
        ends = {pair[-1]:pair for pair in selected}
        view.groups = [FusionGroup(list(ends[g.node_ids[0]]),g.device,{"pattern_id":"native_add_rmsnorm"})
                       if len(g.node_ids)==1 and g.node_ids[0] in ends else g
                       for g in view.groups if not (len(g.node_ids)==1 and g.node_ids[0] in skipped)]
        parent = profile.adapter.registry.get("add_rmsnorm")
        class NativePattern(FusionPattern):
            def validate(self,g,ids): parent.validate(g,ids)
        registry = FusionPatternRegistry([*self.registry.patterns(), NativePattern(**{
            **asdict(parent),"pattern_id":"native_add_rmsnorm","backend":"torch_native",
            "implementation_status":"reference"})])
        class ChainProfile:
            def __getattr__(self,name): return getattr(profile,name)
            def estimate_group(self,g,group):
                if group.pattern_id=="native_add_rmsnorm": return selected[tuple(group.node_ids)]
                return profile.estimate_group(g,group)
        model = StructuralCostModel(registry=registry,scheduling_policy=self.scheduling_policy)
        estimate = model.predict(graph,view,ChainProfile())
        return replace(estimate,details={**estimate.details,"native_chain_execution_units":[list(p) for p in selected],
            "native_chain_output_release":"both original outputs at chain completion; no interleaving",
            "search_plan_rewritten":False})


def search_registry(profile):
    if profile.conditional:
        return profile.adapter.registry
    parent = profile.adapter.registry.get("add_rmsnorm")
    allowed = {pair for (pair,device),q in profile.qualifications.items() if q["capability"]=="fusible"
               and (not profile.require_measured_pair or (pair,device) in profile.native_reuse)}
    class QualifiedPattern(FusionPattern):
        def validate(self,graph,node_ids):
            parent.validate(graph,node_ids)
            if tuple(node_ids) not in allowed:
                raise FusionRuleError("runtime capability/contract evidence is insufficient")
    return FusionPatternRegistry([*[p for p in profile.adapter.registry.patterns() if p.max_nodes==1],
                                  QualifiedPattern(**asdict(parent))])


class IntegrationScorer(BudgetScorer):
    """Reject capability/residency violations without inventing a kernel cost."""
    def start_run(self):
        super().start_run()
        self.rejected_requests = 0

    def __call__(self, plan):
        if self.run_model_evaluations >= self.budget:
            raise EvaluationBudgetReached
        if self.profile.plan_blockers(plan):
            self.run_requests += 1
            self.run_keys.add(_plan_key(plan))
            self.rejected_requests += 1
            self.last_feasible, self.last_makespan = False, INFEASIBLE_PENALTY_S
            return 2*INFEASIBLE_PENALTY_S
        return super().__call__(plan)


def annotate(plan, profile):
    plan = copy.deepcopy(plan)
    plan.extra.update(execution_declaration=profile.declaration, deployable=False,
                      model_layouts_observed=False, assumption_id=profile.assumption_id)
    by_node = {n:g for g in plan.groups for n in g.node_ids}
    units = [{"node_ids":list(pair),"device":device,
              "kernel_key_id":profile.native_record["kernel_key_id"],
              "measurement_contract_id":profile.native_record.get("measurement_contract_id"),
              "required_execution":"consecutive Native Add then RMSNorm; both outputs released at end"}
             for pair,device in profile.native_reuse
             if all(len(by_node[n].node_ids)==1 and by_node[n].device==device for n in pair)]
    if units:
        plan.extra["native_chain_execution_units"] = units
    for group in plan.groups:
        if len(group.node_ids)>1:
            group.extra["implementation"] = copy.deepcopy(profile.adapter.implementation)
    return plan


def activation_slice(graph):
    names = {"L000.self_attention.out_proj","L000.input","L000.residual_add_1","L000.pre_mlp_norm",
             "L000.mlp.down_proj","L000.residual_add_2"}
    result = OpGraph(name="qwen25-3b-existing-layer0-add-rmsnorm-slice")
    result.nodes = {n:copy.deepcopy(graph.nodes[n]) for n in graph.topological_order() if n in names}
    result.edges = [copy.deepcopy(e) for e in graph.edges if e.src in names and e.dst in names]
    boundaries = names - {"L000.residual_add_1","L000.pre_mlp_norm"}
    return result,boundaries
