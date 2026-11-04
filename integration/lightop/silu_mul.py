"""One inference-only placed SiLU-Mul group, with explicit packing cost.

prepare() validates and allocates outside capture/timing. The resulting call
always packs split inputs before invoking the real, source-pinned LightOP API.
This module does not time kernels, admit Profiles or execute a whole graph.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

import torch

from .catalog import catalog, check_installed, resolve
from execution.reference_identity import tensor_observation
from simulator.fusion_rules import (
    FusionPattern, FusionPatternRegistry, FusionRuleError, default_fusion_registry,
)
from simulator.placement_artifacts import validate_placed_op_graph


CAPABILITY = "op.silu_and_mul_opt"
SHAPE = (1, 256, 3072)
RECIPE = {
    "id": "lightop.silu_mul.split_pack_preallocated.v1",
    "outside_timing": ["validate graph and tensors", "allocate packed input and output", "create reshape views"],
    "timed_body": ["torch.cat((gate_2d, up_2d), dim=-1, out=packed)",
                   "lightop.op.silu_and_mul_opt(output_2d, packed)"],
    "input_semantics": "gate and up read-only; no restoration between replays",
    "output_semantics": "independent preallocated output; not an alias of either input",
    "buffer_ownership": "one packed input and one output per prepared call; retain through GPU completion",
    "scheduling": "caller owns stream, dependencies, synchronization and timing events",
    "capture_support": "unverified_on_GPU",
}


def declared_binding():
    """CPU declaration for plan annotation, not observed runtime/Profile identity."""
    data = catalog()
    return {
        "id": RECIPE["id"],
        "version": "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "entrypoint": "lightop.op.silu_and_mul_opt",
        "provider_version": data["version"], "op_so_sha256": data["op_so_sha256"],
        "semantics": copy.deepcopy(RECIPE),
    }


def group_inputs(graph, node_ids):
    """Only the current evidenced case; unknown physical layout stays unknown."""
    default_fusion_registry().get("silu_mul").validate(graph, node_ids)
    silu, mul = node_ids
    incoming, operands = graph.in_edges(silu), graph.in_edges(mul)
    other = [edge for edge in operands if edge.src != silu]
    if len(incoming) != 1 or len(operands) != 2 or len(other) != 1:
        raise FusionRuleError("LightOP requires one gate and one up input without broadcasting")
    gate, up = incoming[0].src, other[0].src
    if gate == up:
        raise FusionRuleError("this LightOP adapter requires distinct gate/up producer nodes")
    nodes = [graph.nodes[n] for n in (gate, up, silu, mul)]
    if any(tuple(n.extra.get("output_shape") or ()) != SHAPE or n.extra.get("dtype") != "bf16" for n in nodes):
        raise FusionRuleError("LightOP v1 is restricted to BF16 [1,256,3072]")
    layers = [n.extra.get("layer") for n in nodes]
    if any(type(n) is not int for n in layers) or len(set(layers)) != 1:
        raise FusionRuleError("LightOP requires explicit same-layer metadata for inputs and group")
    if any(n.extra.get("phase") != "forward" for n in nodes):
        raise FusionRuleError("LightOP adapter supports forward inference only")
    exported = set(graph.extra.get("output_nodes", []))
    if graph.extra.get("output_node"):
        exported.add(graph.extra["output_node"])
    if silu in exported:
        raise FusionRuleError("cannot remove an exported SiLU intermediate")
    return gate, up


class _LightopPattern(FusionPattern):
    def __init__(self):
        # Keep artifact schema/backend compatibility; concrete identity is an
        # explicit implementation annotation, not the historical template name.
        super().__init__("silu_mul", "template", "implemented", op_types=("silu", "mul"),
                         min_nodes=2, max_nodes=2, require_same_shape=True, priority=30)

    def validate(self, graph, node_ids):
        group_inputs(graph, node_ids)


def adapter_registry():
    """Opt-in local execution rule; default/KERV registries remain unchanged."""
    return FusionPatternRegistry([
        *[p for p in default_fusion_registry().patterns() if p.max_nodes == 1],
        _LightopPattern(),
    ])


def _check_device(gate, up, target):
    if not gate.is_cuda or not up.is_cuda or not torch.version.hip:
        raise RuntimeError("LightOP requires HIP GPU tensors; no CPU/native fallback")
    if gate.device != target or up.device != target:
        raise ValueError("boundary tensors must already be on the placed runtime device")


def _check_declared_layout(node, tensor):
    declared = node.extra.get("strides")
    if declared is not None and tuple(declared) != tuple(tensor.stride()):
        raise ValueError("runtime strides differ from graph metadata")
    layout = node.extra.get("layout")
    if isinstance(layout, dict):
        if layout.get("order") not in (None, "row_major"):
            raise ValueError("graph layout is outside this row-major adapter contract")
        layout = layout.get("contiguity")
    if layout not in (None, "unknown", "contiguous"):
        raise ValueError("graph layout disagrees with the contiguous runtime tensor")


class LightopSiluMulAdapter:
    def __init__(self):
        # resolve also verifies the pinned package. No native implementation is
        # substituted if loading or execution fails.
        self.installed = check_installed()
        self.operator = resolve(CAPABILITY)
        self.implementation = declared_binding()
        self.registry = adapter_registry()

    def bind(self, placed, group_index, device_map):
        validate_placed_op_graph(placed, registry=self.registry)
        if type(group_index) is not int or not 0 <= group_index < len(placed.plan.groups):
            raise ValueError("group index is outside the placed plan")
        group = placed.plan.groups[group_index]
        if group.extra.get("pattern_id") != "silu_mul":
            raise ValueError("LightOP adapter only executes a silu_mul group")
        if group.extra.get("implementation") != self.implementation:
            raise ValueError("group must explicitly name this LightOP adapter implementation")
        if group.device not in device_map:
            raise ValueError("missing runtime device mapping for the placed group")
        target = torch.device(device_map[group.device])
        if target.type != "cuda" or target.index is None:
            raise ValueError("runtime device must be explicit cuda:N (including HIP)")
        return BoundLightopGroup(self, placed, group_index, target)


class BoundLightopGroup:
    def __init__(self, adapter, placed, group_index, target):
        self.adapter, self.placed, self.group_index, self.target = adapter, placed, group_index, target
        group = placed.plan.groups[group_index]
        self.node_ids, self.logical_device = tuple(group.node_ids), group.device
        self.gate_node, self.up_node = group_inputs(placed.graph, self.node_ids)
        self.output_node = self.node_ids[-1]

    def prepare(self, values):
        """Bind stable addresses; caller keeps plan/metadata fixed through replay.

        This does not publish an output or mutate the graph/input map. Use
        values[call.output_node] = call() when materializing the Mul result.
        """
        group = self.placed.plan.groups[self.group_index]
        if (tuple(group.node_ids) != self.node_ids or group.device != self.logical_device
                or group.extra.get("pattern_id") != "silu_mul"
                or group.extra.get("implementation") != self.adapter.implementation):
            raise ValueError("bound group/placement/implementation changed; bind again")
        if group_inputs(self.placed.graph, self.node_ids) != (self.gate_node, self.up_node):
            raise ValueError("bound input edges changed; bind again")
        if any(node in values for node in self.node_ids):
            raise ValueError("group outputs must not already exist in the tensor map")
        if any(node not in values for node in (self.gate_node, self.up_node)):
            raise ValueError("missing gate or up boundary tensor")
        gate, up = (values[node] for node in (self.gate_node, self.up_node))
        for node_id, tensor in ((self.gate_node, gate), (self.up_node, up)):
            if type(tensor) is not torch.Tensor:
                raise TypeError("boundary inputs must be ordinary Torch tensors")
            if tuple(tensor.shape) != SHAPE or tensor.dtype != torch.bfloat16:
                raise ValueError("runtime tensor shape/dtype differs from the graph contract")
            if tensor.requires_grad or not tensor.is_contiguous():
                raise ValueError("inference contiguous tensors required; no implicit layout conversion")
            _check_declared_layout(self.placed.graph.nodes[node_id], tensor)
        _check_device(gate, up, self.target)
        return PreparedLightopCall(self, gate, up)


class PreparedLightopCall:
    """Packing is timed; adapter allocation and validation happen in prepare().

    Inputs, buffers and plan must stay alive at fixed addresses. A prepared call
    owns a reusable output; use separate calls for simultaneously live outputs
    or concurrent streams. Caller synchronizes before reading/publishing data.
    """
    def __init__(self, bound, gate, up):
        self.output_node, self.target = bound.output_node, bound.target
        self._operator = bound.adapter.operator
        self._gate, self._up = gate.view(-1, SHAPE[-1]), up.view(-1, SHAPE[-1])
        self._packed = torch.empty((math.prod(SHAPE[:-1]), 2 * SHAPE[-1]),
                                   dtype=gate.dtype, device=gate.device)
        self.output = torch.empty(SHAPE, dtype=gate.dtype, device=gate.device)
        _check_declared_layout(bound.placed.graph.nodes[self.output_node], self.output)
        self._output_2d = self.output.view(-1, SHAPE[-1])
        # All observations happen before capture/timing. Device values come from
        # tensors; a CPU test fixture can never turn these into GPU evidence.
        self.observation = {
            "scope": "execution_preparation_not_profile_admission",
            "implementation": copy.deepcopy(bound.adapter.implementation),
            "recipe": copy.deepcopy(RECIPE),
            "tensors": {"gate": tensor_observation(gate), "up": tensor_observation(up),
                        "packed": tensor_observation(self._packed), "output": tensor_observation(self.output)},
            "storage_overlap": {"gate_up": torch._C._overlaps(gate, up),
                                "output_gate": torch._C._overlaps(self.output, gate),
                                "output_up": torch._C._overlaps(self.output, up)},
            "explicit_buffer_bytes": self._packed.numel() * self._packed.element_size()
                                     + self.output.numel() * self.output.element_size(),
            "gpu_execution_verified": False, "profile_accepted": False,
        }

    def __call__(self):
        # No prepacked-only branch: the graph's split-input cost always includes
        # the necessary cat. Device guard creates no stream or synchronization.
        with torch.cuda.device(self.target), torch.no_grad():
            torch.cat((self._gate, self._up), dim=-1, out=self._packed)
            self._operator(self._output_2d, self._packed)
        return self.output
