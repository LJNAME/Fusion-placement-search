"""Execute one placed SiLU-Mul group with the source-pinned KERV operator.

The caller supplies boundary tensors already on the selected device. This is
an inference-only group adapter; it does not schedule or execute other groups.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import MutableMapping

import torch

from calibration.kerv_silu_mul import (
    DEFAULT_FLAGSCALE, candidate_applicability, check_capability,
)
from calibration.probe_kerv_silu_mul import (
    checked_fused_call, checked_launch_probe, load_kernel_module,
)
from simulator.fusion_rules import (
    FusionPattern, FusionPatternRegistry, FusionRuleError, default_fusion_registry,
)
from simulator.placement_artifacts import validate_placed_op_graph


class _ImplementedSiluMul(FusionPattern):
    def __init__(self):
        # Keep the v1 artifact backend spelling. The concrete implementation is
        # supplied by the verified capability, not by this historical field.
        super().__init__("silu_mul", "template", "implemented", op_types=("silu", "mul"),
                         min_nodes=2, max_nodes=2, require_same_shape=True, priority=30)

    def validate(self, graph, node_ids):
        super().validate(graph, node_ids)
        result = candidate_applicability(
            graph, node_ids, layout_policy="require_contiguous_at_runtime")
        if not result["eligible"]:
            raise FusionRuleError("KERV guard: " + ",".join(result["reasons"]))
        layers = [graph.nodes[n].extra.get("layer") for n in node_ids]
        if any(type(layer) is not int for layer in layers) or len(set(layers)) != 1:
            raise FusionRuleError("KERV requires explicit same-layer metadata")
        exported = set(graph.extra.get("output_nodes", []))
        if graph.extra.get("output_node"):
            exported.add(graph.extra["output_node"])
        if node_ids[0] in exported:
            raise FusionRuleError("KERV cannot remove an exported SiLU intermediate")
        if result["gate_node"] == result["up_node"]:
            raise FusionRuleError("KERV requires distinct gate and up inputs")


def _registry():
    return FusionPatternRegistry([
        *[p for p in default_fusion_registry().patterns() if p.max_nodes == 1],
        _ImplementedSiluMul(),
    ])


def load_kerv_registry(capability_path, flagscale_root=DEFAULT_FLAGSCALE):
    """Opt in to the inspected implementation; runtime requirements still apply.

    'implemented' describes this local adapter. It does not promote the old
    capability's GPU evidence, measured labels, or downstream deployment status.
    """
    manifest = check_capability(capability_path, flagscale_root)
    return _registry(), manifest


@contextmanager
def _triton_backend(module):
    # KERV configuration is process-global. Use this adapter sequentially.
    saved = (module._BACKEND, module._ENABLED_OPERATORS, module._RECORD_PATH,
             module._RECORD_ONCE, module._RECORDED.copy())
    try:
        module.configure_backend("triton")
        module.configure_enabled_operators({"kerv_silu_mul"})
        module.configure_recording(None, False)
        yield
    finally:
        module.configure_backend(saved[0])
        module.configure_enabled_operators(saved[1])
        module._RECORD_PATH, module._RECORD_ONCE, module._RECORDED = saved[2:]


def _check_device(gate, up, target):
    if not gate.is_cuda or not up.is_cuda:
        raise RuntimeError("KERV execution requires GPU tensors; CPU fallback is forbidden")
    if gate.device != target or up.device != target:
        raise ValueError("boundary tensors must already be on the placed runtime device")


@dataclass(frozen=True)
class KervGroupCall:
    """Result aliases the consumed gate; only the Mul node is materialized."""

    output_node: str
    consumed_node: str
    output: torch.Tensor
    observed_jit_launches: int | None


class KervSiluMulAdapter:
    def __init__(self, capability_path, flagscale_root=DEFAULT_FLAGSCALE):
        self.module, self.manifest = load_kernel_module(flagscale_root, capability_path)
        self.registry = _registry()

    def bind(self, placed, group_index, device_map):
        """Bind a selected group in an otherwise unexecuted PlacedOpGraph."""
        validate_placed_op_graph(placed, registry=self.registry)
        if type(group_index) is not int or not 0 <= group_index < len(placed.plan.groups):
            raise ValueError("group index is outside the placed plan")
        group = placed.plan.groups[group_index]
        if group.extra.get("pattern_id") != "silu_mul":
            raise ValueError("the KERV adapter only executes a silu_mul group")
        binding = group.extra.get("implementation")
        if binding is not None and binding != self.manifest["binding"]:
            raise ValueError("group implementation differs from the verified capability")
        if group.device not in device_map:
            raise ValueError("no runtime device mapping for the placed group")
        target = torch.device(device_map[group.device])
        if target.type != "cuda" or target.index is None:
            raise ValueError("runtime device must be an explicit cuda:N device (including HIP)")
        info = candidate_applicability(placed.graph, group.node_ids,
                                      layout_policy="require_contiguous_at_runtime")
        return BoundKervGroup(self, placed, group_index, target, info)


class BoundKervGroup:
    def __init__(self, adapter, placed, group_index, target, info):
        self.adapter = adapter
        self.placed = placed
        self.group_index = group_index
        self.target = target
        self.logical_device = placed.plan.groups[group_index].device
        self.gate_node = info["gate_node"]
        self.up_node = info["up_node"]
        self.node_ids = tuple(placed.plan.groups[group_index].node_ids)

    def _validate_tensors(self, values):
        # Recheck graph guards because the caller owns the mutable graph/plan.
        group = self.placed.plan.groups[self.group_index]
        self.adapter.registry.get("silu_mul").validate(self.placed.graph, self.node_ids)
        if tuple(group.node_ids) != self.node_ids or group.extra.get("pattern_id") != "silu_mul":
            raise ValueError("bound group changed; bind the placed graph again")
        expected_binding = self.adapter.manifest["binding"]
        if (group.device != self.logical_device
                or group.extra.get("implementation", expected_binding) != expected_binding):
            raise ValueError("bound placement or implementation changed; bind again")
        info = candidate_applicability(self.placed.graph, self.node_ids,
                                      layout_policy="require_contiguous_at_runtime")
        if (info["gate_node"], info["up_node"]) != (self.gate_node, self.up_node):
            raise ValueError("bound input edges changed; bind the placed graph again")
        if any(node in values for node in self.node_ids):
            raise ValueError("group outputs must not already exist in the live tensor map")
        if self.gate_node not in values or self.up_node not in values:
            raise ValueError("missing gate or up boundary tensor")
        gate, up = values[self.gate_node], values[self.up_node]
        for name, tensor in ((self.gate_node, gate), (self.up_node, up)):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("boundary inputs must be Torch tensors")
            node = self.placed.graph.nodes[name]
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[node.extra["dtype"]]
            if tuple(tensor.shape) != tuple(node.extra["output_shape"]) or tensor.dtype != dtype:
                raise ValueError("runtime tensor shape/dtype differs from graph metadata")
            if tensor.requires_grad:
                raise ValueError("KERV group adapter supports inference tensors only")
            if not tensor.is_contiguous():
                raise ValueError("KERV boundary tensors must be contiguous; no implicit copy")
            declared = node.extra.get("strides")
            if declared is not None and tuple(declared) != tuple(tensor.stride()):
                raise ValueError("runtime tensor strides differ from graph metadata")
        _check_device(gate, up, self.target)
        # Different node IDs can still name views of the same storage. Reject
        # these aliases before the mutating call, including aliases to live up.
        for name, tensor in values.items():
            if name != self.gate_node and isinstance(tensor, torch.Tensor):
                if torch._C._overlaps(gate, tensor):
                    raise ValueError("gate storage aliases another live boundary tensor: " + name)
        return gate, up

    def __call__(self, values: MutableMapping[str, torch.Tensor], *, observe_dispatch=False):
        gate, up = self._validate_tensors(values)
        with _triton_backend(self.adapter.module), torch.no_grad():
            call = checked_launch_probe if observe_dispatch else checked_fused_call
            # The pinned wrapper calls torch.ops.flagos_embodied.kerv_silu_mul.
            # Never replace this with SiLU followed by Mul.
            result = call(self.adapter.module, gate, up)
        if result.data_ptr() != gate.data_ptr() or result.stride() != gate.stride():
            raise RuntimeError("KERV output no longer aliases gate as declared")
        if result.shape != gate.shape or result.dtype != gate.dtype or result.device != gate.device:
            raise RuntimeError("KERV output violates its tensor contract")
        del values[self.gate_node]
        values[self.node_ids[-1]] = result
        return KervGroupCall(self.node_ids[-1], self.gate_node, result,
                             1 if observe_dispatch else None)
