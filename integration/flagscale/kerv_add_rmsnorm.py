"""Opt-in, fixed-shape KERV Add+RMSNorm graph adapter (forward only).

Consumes the disposable attention output; exports both original graph outputs.
The caller owns streams, dependencies and tensor lifetime. No Profile or timing
is supplied by this adapter. CPU tests mock dispatch, not accelerator evidence.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch

from calibration.kerv_add_rmsnorm_probe import (
    EPS, SHAPE, SOURCE_SHA256, dispatch_once, load_module, source_path,
)
from calibration.kerv_silu_mul import DEFAULT_FLAGSCALE
from execution.reference_identity import tensor_observation
from simulator.fusion_rules import FusionPattern, FusionPatternRegistry, FusionRuleError, default_fusion_registry
from simulator.placement_artifacts import validate_placed_op_graph

STRIDES = (524288, 2048, 1)


def declared_binding():
    return {"id": "kerv.add_rmsnorm.adaptive.consume_x.dual_output.v1",
            "version": "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "entrypoint": "adaptive_rms_norm.add_rms_norm_inference", "backend": "triton",
            "kernel_source_sha256": SOURCE_SHA256,
            "semantics": {"shape": list(SHAPE), "dtype": "bf16", "eps": EPS,
                          "eps_source": "explicit contract matching project Reference; not inferred from graph",
                          "input": "consume attention output; residual and weight read-only",
                          "outputs": "Add aliases consumed input; Norm is independently allocated",
                          "rounding": "BF16 add; FP32 reduction; BF16 normalized value before weight multiply",
                          "allocation": "vendor wrapper allocates normalized output",
                          "capture": "unverified", "profile": "none"}}


def _check_layout(node, strides=None):
    layout = node.extra.get("layout")
    if isinstance(layout, dict):
        if layout.get("order") not in (None, "row_major"):
            raise FusionRuleError("KERV requires row-major input layout")
        layout = layout.get("contiguity")
    if layout not in (None, "unknown", "contiguous"):
        raise FusionRuleError("KERV requires contiguous runtime input layout")
    declared = node.extra.get("strides")
    if declared is not None and tuple(declared) != STRIDES:
        raise FusionRuleError("declared strides are outside the evidenced KERV case")
    if strides is not None and tuple(strides) != STRIDES:
        raise ValueError("runtime strides are outside the evidenced KERV case")


def group_io(graph, node_ids):
    """Resolve input roles from graph metadata, independently of edge order."""
    ids = tuple(node_ids)
    if len(ids) != 2 or len(set(ids)) != 2 or any(n not in graph.nodes for n in ids):
        raise FusionRuleError("KERV AddRMS requires two existing distinct nodes")
    add, norm = ids
    if tuple(graph.nodes[n].op_type for n in ids) != ("add", "rmsnorm"):
        raise FusionRuleError("KERV AddRMS requires (add, rmsnorm)")
    inputs = graph.predecessors(add)
    if len(inputs) != 2 or len(set(inputs)) != 2 or graph.predecessors(norm) != [add]:
        raise FusionRuleError("KERV AddRMS requires two Add inputs and one Add-to-Norm edge")
    attention = [n for n in inputs if graph.nodes[n].extra.get("local_name") == "self_attention.out_proj"]
    if len(attention) != 1:
        raise FusionRuleError("KERV requires an explicit disposable attention output")
    x = attention[0]
    residual = next(n for n in inputs if n != x)
    if graph.nodes[residual].extra.get("local_name") != "input":
        raise FusionRuleError("KERV requires the same-layer residual input")
    nodes = [graph.nodes[n] for n in (x, residual, add, norm)]
    if any(tuple(n.extra.get("output_shape", ())) != SHAPE or n.extra.get("dtype") != "bf16" for n in nodes):
        raise FusionRuleError("KERV AddRMS v1 requires BF16 [1,256,2048]")
    layers = [n.extra.get("layer") for n in nodes]
    if any(type(layer) is not int for layer in layers) or len(set(layers)) != 1:
        raise FusionRuleError("KERV AddRMS requires explicit same-layer metadata")
    if any(n.extra.get("phase") != "forward" for n in nodes):
        raise FusionRuleError("KERV AddRMS supports forward inference only")
    if graph.nodes[norm].param_bytes != SHAPE[-1] * 2:
        raise FusionRuleError("KERV requires a BF16 weight vector of length 2048")
    if graph.nodes[norm].extra.get("eps", EPS) != EPS:
        raise FusionRuleError("graph epsilon differs from the explicit adapter contract")
    exported = set(graph.extra.get("output_nodes", []))
    if graph.extra.get("output_node"):
        exported.add(graph.extra["output_node"])
    if set(graph.successors(x)) != {add} or x in exported:
        raise FusionRuleError("attention input has external consumers or is exported")
    for n in nodes:
        _check_layout(n)
    return x, residual


class _AddRmsnormPattern(FusionPattern):
    def __init__(self):
        super().__init__("add_rmsnorm", "template", "implemented", op_types=("add", "rmsnorm"),
                         supported_dtypes=("bf16",), min_nodes=2, max_nodes=2,
                         require_same_shape=True, require_linear_chain=False, priority=30)

    def validate(self, graph, node_ids):
        super().validate(graph, node_ids)
        group_io(graph, node_ids)


def adapter_registry():
    """Explicit opt-in only; existing LightOP/default search registries unchanged."""
    return FusionPatternRegistry([
        *[p for p in default_fusion_registry().patterns() if p.max_nodes == 1], _AddRmsnormPattern(),
    ])


def _check_device(tensors, target):
    if not torch.version.hip or any(not t.is_cuda for t in tensors):
        raise RuntimeError("KERV AddRMS requires HIP GPU tensors; no CPU/native fallback")
    if any(t.device != target for t in tensors):
        raise ValueError("all tensors must already be on the placed runtime device")


@dataclass(frozen=True)
class AddRmsnormCall:
    add_output: torch.Tensor
    normalized_output: torch.Tensor
    consumed_node: str
    observed_jit_launches: int | None
    observations: dict


class KervAddRmsnormAdapter:
    def __init__(self, flagscale_root=DEFAULT_FLAGSCALE):
        self.flagscale_root = Path(flagscale_root)
        source_path(self.flagscale_root)
        self.implementation = declared_binding()
        self.registry = adapter_registry()
        self._module = None

    def bind(self, placed, group_index, device_map):
        validate_placed_op_graph(placed, registry=self.registry)
        if type(group_index) is not int or not 0 <= group_index < len(placed.plan.groups):
            raise ValueError("invalid group index")
        group = placed.plan.groups[group_index]
        if group.extra.get("pattern_id") != "add_rmsnorm":
            raise ValueError("adapter only executes add_rmsnorm")
        if group.extra.get("implementation") != self.implementation:
            raise ValueError("group must explicitly name this KERV implementation")
        if group.device not in device_map:
            raise ValueError("missing runtime device mapping")
        target = torch.device(device_map[group.device])
        if target.type != "cuda" or target.index is None:
            raise ValueError("runtime device must be explicit cuda:N (including HIP)")
        return BoundAddRmsnorm(self, placed, group_index, target)

    def _invoke(self, x, residual, weight, observe_dispatch):
        if self._module is None:
            self._module = load_module(self.flagscale_root)
        with torch.cuda.device(x.device):
            if observe_dispatch:
                outputs, _ = dispatch_once(self._module, x, residual, weight)
                return outputs
            return self._module.add_rms_norm_inference(x, residual, (2048,), weight, EPS)


class BoundAddRmsnorm:
    def __init__(self, adapter, placed, index, target):
        self.adapter, self.placed, self.index, self.target = adapter, placed, index, target
        group = placed.plan.groups[index]
        self.node_ids, self.device = tuple(group.node_ids), group.device
        self.binding = copy.deepcopy(adapter.implementation)
        self.x_node, self.residual_node = group_io(placed.graph, self.node_ids)

    def __call__(self, values, weight, *, observe_dispatch=False):
        group = self.placed.plan.groups[self.index]
        if (tuple(group.node_ids) != self.node_ids or group.device != self.device
                or group.extra.get("pattern_id") != "add_rmsnorm"
                or group.extra.get("implementation") != self.binding
                or self.adapter.implementation != self.binding):
            raise ValueError("bound group/placement/implementation changed; bind again")
        if group_io(self.placed.graph, self.node_ids) != (self.x_node, self.residual_node):
            raise ValueError("bound input roles changed; bind again")
        if any(n in values for n in self.node_ids):
            raise ValueError("group outputs must not already exist in the live tensor map")
        if any(n not in values for n in (self.x_node, self.residual_node)):
            raise ValueError("missing boundary tensors")
        x, residual = values[self.x_node], values[self.residual_node]
        for tensor, shape in ((x, SHAPE), (residual, SHAPE), (weight, (2048,))):
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape or tensor.dtype != torch.bfloat16:
                raise ValueError("runtime tensor shape/dtype differs from the fixed KERV contract")
            if tensor.requires_grad or not tensor.is_contiguous():
                raise ValueError("inference contiguous tensors required; no implicit copy")
        for node, tensor in ((self.x_node, x), (self.residual_node, residual)):
            _check_layout(self.placed.graph.nodes[node], tensor.stride())
        _check_device((x, residual, weight), self.target)
        if torch._C._overlaps(x, weight):
            raise ValueError("consumed attention storage aliases the norm weight")
        for name, tensor in values.items():
            if name != self.x_node and isinstance(tensor, torch.Tensor) and torch._C._overlaps(x, tensor):
                raise ValueError("consumed attention storage aliases a live tensor: " + name)
        inputs = {"x": tensor_observation(x), "residual": tensor_observation(residual),
                  "weight": tensor_observation(weight)}
        # Invalidate ownership before an in-place dispatch, including partial failure.
        del values[self.x_node]
        with torch.no_grad():
            normalized, summed = self.adapter._invoke(x, residual, weight, observe_dispatch)
        for tensor in (normalized, summed):
            if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != SHAPE
                    or tensor.dtype != x.dtype or tensor.device != x.device or not tensor.is_contiguous()):
                raise RuntimeError("KERV output violates the tensor contract")
        if summed.data_ptr() != x.data_ptr() or summed.stride() != x.stride():
            raise RuntimeError("KERV Add output must alias the consumed attention buffer")
        live_tensors = [x, weight, *[t for t in values.values() if isinstance(t, torch.Tensor)]]
        if any(torch._C._overlaps(normalized, t) for t in live_tensors):
            raise RuntimeError("KERV normalized output must have independent storage")
        add, norm = self.node_ids
        values[add], values[norm] = summed, normalized
        observations = {"inputs": inputs, "outputs": {add: tensor_observation(summed), norm: tensor_observation(normalized)}}
        return AddRmsnormCall(summed, normalized, self.x_node, 1 if observe_dispatch else None, observations)
