"""Exact LightOP adapter-call identities and the existing strict cost interface.

No admission decision or real latency is created here. UUID is run provenance;
ordinal, architecture, build, source, layout and alias semantics remain in keys.
"""
import copy

from .fusion_kernel_profile import GroupKernelKeys, KeyedKernelProfile
from .hardware_profile import AnalyticHardwareProfile
from .measurement_labels import MeasurementContractError, measurement_contract_id
from integration.lightop.silu_mul import RECIPE, SHAPE, declared_binding, group_inputs
from integration.lightop.catalog import catalog


EXECUTOR = "placed_lightop_split_pack_graph_replay"


def measurement_contract(environment, *, purpose="smoke"):
    if purpose not in ("smoke", "calibration"):
        raise ValueError("unknown LightOP measurement purpose")
    return {"executor": EXECUTOR, "purpose": purpose,
            "submission_included": False, "allocation_included": False,
            "granularity": "graph_batch_amortized_complete_adapter_call", "unit": "us/call",
            "mode": "graph_replay", "calls_per_round": 50,
            "rounds": 3 if purpose == "smoke" else 7, "extra_attempts": 0,
            "eager_warmup_calls": 50, "warmup_replays": 10,
            "statistic": "diagnostic_only" if purpose == "smoke" else "median",
            "timing_boundary": "start Event -> one Graph replay -> end Event; same device and stream",
            "synchronization": "end Event and device synchronized outside Event interval",
            "input_reuse": "shared read-only gate/up; 50 distinct packed/output buffer pairs",
            "buffer_preparation": "fill packed/output with NaN before checked replay; outside Event interval on same stream",
            "recipe": copy.deepcopy(RECIPE),
            "correctness": {"rtol": .01, "atol": .001, "check_all_outputs": True,
                            "input_check": "bytewise unchanged", "output_alias_input": False},
            "environment": copy.deepcopy(environment), "automatic_admission": False}


def _tensor_spec(value, shape, device):
    if (value.get("shape") != list(shape) or value.get("dtype") != "bf16"
            or value.get("layout") != "contiguous"):
        raise MeasurementContractError("LightOP identity requires observed BF16 contiguous tensors of the exact shape")
    strides, expected = value.get("strides"), 1
    if not isinstance(strides, list) or len(strides) != len(shape):
        raise MeasurementContractError("observed strides are missing")
    for size, stride in zip(reversed(shape), reversed(strides)):
        if type(stride) is not int or stride < 0 or (size != 1 and stride != expected):
            raise MeasurementContractError("observed strides disagree with contiguous layout")
        expected *= size
    provenance = value.get("layout_provenance") or {}
    if (provenance.get("kind") != "runtime_tensor" or provenance.get("device") != device
            or not provenance.get("observer")):
        raise MeasurementContractError("layout must come from a Tensor on the observed runtime GPU")


class LightopKernelKeys(GroupKernelKeys):
    """Only one observed group/device; no fabricated layouts for other nodes."""
    def __init__(self, placed, group_index, prepared_observation, observed_runtime):
        graph, hardware = copy.deepcopy(placed.graph), copy.deepcopy(placed.hardware)
        group = copy.deepcopy(placed.plan.groups[group_index])
        gate, up = group_inputs(graph, group.node_ids)
        declared = declared_binding()
        if (group.extra.get("implementation") != declared or prepared_observation.get("implementation") != declared
                or observed_runtime.get("implementation") != declared):
            raise MeasurementContractError("plan, preparation and loaded implementation must agree")
        runtime, device = copy.deepcopy(observed_runtime["runtime"]), observed_runtime["device"]
        target = runtime.get("execution_device")
        if (runtime.get("accelerator") != "HIP" or type(device.get("physical_index")) is not int
                or device["physical_index"] < 0 or target != f"cuda:{device['physical_index']}"):
            raise MeasurementContractError("identity must describe the actual HIP runtime device")
        source = observed_runtime["loaded_source"]
        hashes = source.get("source_sha256", {})
        if (hashes.get("native") != declared["op_so_sha256"]
                or "sha256:" + str(hashes.get("adapter")) != declared["version"]
                or source.get("installed", {}).get("version") != declared["provider_version"]
                or (hashes.get("package") and hashes["package"] != catalog()["init_sha256"])):
            raise MeasurementContractError("actual loaded source differs from the declared implementation")
        if prepared_observation.get("recipe") != RECIPE:
            raise MeasurementContractError("preparation recipe is not the complete split-input adapter call")
        tensors = prepared_observation["tensors"]
        for name in ("gate", "up", "output"):
            _tensor_spec(tensors[name], SHAPE, target)
        _tensor_spec(tensors["packed"], (256, 6144), target)
        alias = prepared_observation.get("storage_overlap", {})
        if (type(alias.get("gate_up")) is not bool or alias.get("output_gate") is not False
                or alias.get("output_up") is not False):
            raise MeasurementContractError("observed input/output alias semantics are incomplete or invalid")
        for name, node_id in (("gate", gate), ("up", up), ("output", group.node_ids[-1])):
            for field in ("layout", "strides", "layout_provenance"):
                graph.nodes[node_id].extra[field] = copy.deepcopy(tensors[name][field])
        internal = graph.nodes[group.node_ids[0]]
        internal.extra.update(layout={"materialization": "internal_fused_expression"}, strides=None)
        # Normalize commutative Mul edge order without dropping input roles or ports.
        graph.edges.sort(key=lambda edge: (edge.dst, edge.src))
        physical = hardware.devices[group.device]
        physical.name = device["name"]
        physical.max_mem = device["total_memory"]
        physical.extra.update(physical_index=device["physical_index"],
                              source={"kind": "runtime_observation", "architecture": device.get("architecture")})
        binding = copy.deepcopy(declared)
        binding.update(loaded_source_sha256=copy.deepcopy(hashes),
                       packed_tensor={k: copy.deepcopy(tensors["packed"][k]) for k in ("shape", "dtype", "layout", "strides")},
                       storage_overlap=copy.deepcopy(alias))
        self._selected = (tuple(group.node_ids), group.device)
        self._gaps = []
        for field in ("torch_distribution", "torch_git_revision", "torch_extension_sha256", "torch_build_sha256", "driver_version"):
            if not runtime.get(field):
                self._gaps.append("runtime." + field)
        if not device.get("architecture"):
            self._gaps.append("device.architecture")
        if not hashes.get("package"):
            self._gaps.append("implementation.package_source")
        flags = runtime.get("flags", {})
        if any(type(flags.get(field)) is not bool for field in
               ("grad_enabled", "cuda_autocast_enabled", "deterministic_algorithms")):
            self._gaps.append("runtime.flags")
        if flags.get("grad_enabled") is True:
            raise MeasurementContractError("runtime observation must match the adapter no_grad context")
        self.environment = copy.deepcopy(observed_runtime["environment"])
        for field, expected in (("torch", runtime.get("torch_distribution")),
                                ("hip", runtime.get("accelerator_version")), ("runtime_device", target),
                                ("device_model", device["name"]), ("device_architecture", device.get("architecture")),
                                ("driver_version", runtime.get("driver_version")),
                                ("torch_build_sha256", runtime.get("torch_build_sha256"))):
            if self.environment.get(field) != expected:
                raise MeasurementContractError("contract environment differs from runtime identity: " + field)
        # Paths and UUID are retained in the raw observation, never the key.
        super().__init__(graph, hardware, {"silu_mul": binding}, runtime)

    def describe(self, group):
        if (tuple(group.node_ids), group.device) != self._selected:
            raise MeasurementContractError("LightOP identity covers only the observed group and device")
        key, gaps = super().describe(group)
        return key, tuple(sorted(set(gaps) | set(self._gaps)))


def selected_group_profile(keys, contract, latencies):
    """Consume already-reviewed labels through the shared gate; never promote.

    Missing, conditional, unbound, wrong-contract or incomplete-key data fail.
    The base supplies hardware parameters only: no selected cost fallback.
    """
    if (contract.get("purpose") != "calibration" or contract.get("executor") != EXECUTOR
            or contract.get("recipe") != RECIPE or contract.get("mode") != "graph_replay"):
        raise MeasurementContractError("Smoke/eager/kernel-only data cannot price this adapter call")
    if contract != measurement_contract(contract["environment"], purpose="calibration"):
        raise MeasurementContractError("unsupported LightOP measurement contract")
    if contract["environment"] != keys.environment:
        raise MeasurementContractError("measurement contract belongs to a different execution environment")
    if any(entry.kind != "measured" for entry in latencies.values()):
        raise MeasurementContractError("synthetic costs are not LightOP measurements")
    return KeyedKernelProfile(keys, latencies, AnalyticHardwareProfile(keys.hardware), strict=True,
                              measurement_contract_id=measurement_contract_id(contract), allow_unbound=False)
