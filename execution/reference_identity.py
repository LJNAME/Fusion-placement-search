"""Reference singleton identities observed in the executing Torch environment.

Identity eligibility is separate from measurement admission. No latency labels
or GPU evidence are created here, and unknown layouts remain incomplete.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import inspect
from pathlib import Path
import platform

import torch
import torch.nn.functional as F

from cost_model.fusion_kernel_profile import GroupKernelKeys, canonical, key_id
from cost_model.measurement_labels import MeasurementContractError
from execution.reference import _ReferenceExecutor

OPERATORS = {"matmul": "torch.matmul", "silu": "torch.nn.functional.silu", "mul": "torch.Tensor.__mul__"}
DTYPES = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observe_reference_runtime(device="cpu"):
    """Observe loaded code/build/flags; GPU metadata queries do not time kernels."""
    if (torch.matmul is not torch._C._VariableFunctions.matmul
            or torch.Tensor.__mul__ is not torch._C.TensorBase.__mul__
            or F.silu.__module__ != "torch.nn.functional" or F.silu.__name__ != "silu"):
        raise MeasurementContractError("Reference entrypoint has been replaced by another implementation")
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("only CPU and CUDA/HIP Reference runtimes are supported")
    if device.type == "cuda":
        if device.index is None:
            raise ValueError("Reference GPU identity requires an explicit physical ordinal")
        props = torch.cuda.get_device_properties(device)
        architecture = getattr(props, "gcnArchName", None)
        if not architecture:
            architecture = "sm_" + "".join(map(str, torch.cuda.get_device_capability(device)))
        identity = {"kind": "runtime_observation", "execution_device": str(device),
                    "name": props.name, "uuid": str(getattr(props, "uuid", "unavailable")),
                    "architecture": architecture}
        if identity["uuid"] == "unavailable":
            raise MeasurementContractError("Reference GPU identity requires an observed UUID")
    else:
        identity = {"kind": "runtime_observation", "execution_device": "cpu",
                    "name": platform.processor() or platform.machine(),
                    "architecture": platform.machine(), "platform": platform.system()}
    runtime = {"framework": "torch", "framework_version": str(torch.__version__),
               "distribution_version": importlib.metadata.version("torch"),
               "torch_git_revision": torch.version.git_version,
               "accelerator": ("HIP" if torch.version.hip else "CUDA") if device.type == "cuda" else "CPU",
               "accelerator_version": (torch.version.hip or torch.version.cuda) if device.type == "cuda" else "not_applicable",
               "execution_device_type": device.type, "execution_device": str(device),
               "build_config_sha256": hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
               "flags": {"grad_enabled": torch.is_grad_enabled(), "num_threads": torch.get_num_threads(),
                         "num_interop_threads": torch.get_num_interop_threads(),
                         "matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision,
                         "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                         "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}}
    files = {"reference_executor": Path(inspect.getsourcefile(_ReferenceExecutor)).resolve(),
             "torch_extension": Path(torch._C.__file__).resolve(),
             "torch_functional": Path(inspect.getsourcefile(F.silu)).resolve()}
    hashes = {name: _sha(path) for name, path in files.items()}
    bindings = {"unfused:" + op: {
        "id": "reference." + entry, "version": str(torch.__version__),
        "entrypoint": entry, "backend": "torch_native_reference",
        "source_sha256": hashes, "torch_git_revision": torch.version.git_version,
        "runtime_flags": runtime["flags"],
        "semantics": "flatten_input_mm_resident_weight_reshape" if op == "matmul" else "out_of_place",
    } for op, entry in OPERATORS.items()}
    return {"runtime": runtime, "device": identity, "bindings": bindings,
            "source_paths": {name: str(path) for name, path in files.items()}}


def tensor_observation(tensor):
    """Read a real Tensor; no layout is inferred from its shape."""
    if tensor.dtype not in DTYPES or tensor.layout != torch.strided:
        raise ValueError("unsupported Reference Tensor dtype/layout")
    return {"shape": list(tensor.shape), "dtype": DTYPES[tensor.dtype],
            "layout": "contiguous" if tensor.is_contiguous() else "strided",
            "strides": list(tensor.stride()),
            "layout_provenance": {"kind": "runtime_tensor", "device": str(tensor.device),
                                  "observer": "Tensor.shape/dtype/stride/is_contiguous"}}


def with_tensor_layouts(graph, observations, *, parameters=None):
    """Copy a graph and apply explicit runtime observations or declarations.

    A graph declaration needs a source_id identifying the declaring artifact.
    Missing entries are deliberately left unknown. Parameters are not data edges.
    """
    graph = copy.deepcopy(graph)
    for node_id, observed in observations.items():
        node = graph.nodes[node_id]
        if observed["shape"] != list(node.extra["output_shape"]) or observed["dtype"] != node.extra["dtype"]:
            raise MeasurementContractError("Reference observation disagrees with graph shape/dtype: " + node_id)
        for field in ("layout", "strides", "layout_provenance"):
            node.extra[field] = copy.deepcopy(observed.get(field))
    for node_id, tensors in (parameters or {}).items():
        graph.nodes[node_id].extra["reference_parameters"] = copy.deepcopy(tensors)
    return graph


def _layout_gaps(tensor, runtime):
    gaps = set()
    layout, strides, shape = tensor.get("layout"), tensor.get("strides"), tensor.get("shape")
    provenance = tensor.get("layout_provenance") or {}
    if (layout not in ("contiguous", "strided") or not shape or not isinstance(strides, list)
            or len(strides) != len(shape) or any(type(s) is not int or s < 0 for s in strides)):
        gaps.add("tensor.layout_or_strides")
    elif layout == "contiguous":
        stride = 1
        for size, actual in zip(reversed(shape), reversed(strides)):
            if size != 1 and actual != stride:
                gaps.add("tensor.layout_stride_mismatch")
            stride *= size
    if provenance.get("kind") == "runtime_tensor":
        if not provenance.get("observer") or provenance.get("device") != runtime.get("execution_device"):
            gaps.add("tensor.layout_provenance")
    elif provenance.get("kind") == "graph_declaration":
        if not provenance.get("source_id"):
            gaps.add("tensor.layout_provenance")
    else:
        gaps.add("tensor.layout_provenance")
    return gaps


class ReferenceKernelKeys(GroupKernelKeys):
    """Scoped Reference identity gate over the existing Canonical Key factory."""
    def __init__(self, graph, hardware, observation):
        self.observation = copy.deepcopy(observation)
        super().__init__(graph, hardware, observation["bindings"], observation["runtime"])

    def verify_runtime(self):
        current = observe_reference_runtime(self.observation["device"]["execution_device"])
        if (current["runtime"] != self.runtime or current["bindings"] != self.bindings
                or current["device"] != self.observation["device"]):
            raise MeasurementContractError("Reference binding differs from the actual loaded runtime")

    def describe(self, group):
        self.verify_runtime()
        if len(group.node_ids) != 1 or self.graph.nodes[group.node_ids[0]].op_type not in OPERATORS:
            raise MeasurementContractError("Reference identity scope only admits matmul/silu/mul singletons")
        node_id = group.node_ids[0]
        node = self.graph.nodes[node_id]
        key, original_gaps = super().describe(group)
        gaps = set(original_gaps)
        if key["hardware"]["source"] != self.observation["device"]:
            gaps.add("hardware.runtime_observation")
        sources, seen = [], set()
        for edge in self.incoming[node_id]:
            alias = (edge.src, canonical(edge.extra.get("src_port")))
            if alias not in seen:
                seen.add(alias)
                sources.append(self.graph.nodes[edge.src])
        for tensor, source in zip(key["external_inputs"], sources):
            tensor["layout_provenance"] = source.extra.get("layout_provenance")
            gaps.update(_layout_gaps(tensor, self.runtime))
        output = key["operators"][0]["output"]
        output["layout_provenance"] = node.extra.get("layout_provenance")
        gaps.update(_layout_gaps(output, self.runtime))
        parameters = node.extra.get("reference_parameters", {})
        if node.op_type == "matmul":
            weight = parameters.get("weight", {})
            gaps.update(_layout_gaps(weight, self.runtime))
            shape = node.extra["shape"]
            if weight.get("shape") != [shape["k"], shape["n"]] or weight.get("dtype") != node.extra["dtype"]:
                gaps.add("parameter.weight_shape_dtype")
        key["reference_identity_scope"] = "singleton_identity_only_not_measurement_admission"
        return key, tuple(sorted(gaps))

    def require_complete(self, group):
        key, gaps = self.describe(group)
        if gaps:
            raise MeasurementContractError("incomplete Reference identity: " + ",".join(gaps))
        return key, key_id(key)
