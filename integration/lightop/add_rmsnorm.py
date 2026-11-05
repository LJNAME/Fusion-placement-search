"""Graph-level adapter for Hygon LightOP fused Add+RMSNorm (lightop.fused_add_rms_norm).

Vendor semantics, from lightop/rmsnorm.py (hash pinned in the catalog): the
op is in place.

    op.fused_add_rms_norm_opt(x, residual, weight, eps)
    # afterwards: residual <- x + residual;  x <- RMSNorm(residual) * weight

The adapter keeps graph inputs read-only. Each prepared call copies x and
residual into its own preallocated buffers, then invokes the in-place op;
the copies are part of the call. A group exports two tensors: the Add output
(residual stream, which may have other consumers) and the RMSNorm output.

Scope is the evidenced case only: BF16, contiguous, [1,256,1024], one layer,
forward. eps is a declared model contract (Qwen3 rms_norm_eps), because the
graph does not record it. No timing, Profile admission or whole-graph
execution happens here.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import torch

from .catalog import catalog, check_installed
from .silu_mul import _LightopPattern, _check_declared_layout
from execution.reference_identity import tensor_observation
from simulator.fusion_rules import FusionPattern, FusionPatternRegistry, FusionRuleError, default_fusion_registry

ENTRYPOINT = "lightop.fused_add_rms_norm"
SHAPE = (1, 256, 1024)
EPS = 1e-6
RECIPE = {
    "id": "lightop.add_rmsnorm.copy_inplace_preallocated.v1",
    "timed_body": ["x_buf.copy_(x)", "res_buf.copy_(residual)",
                   "lightop.op.fused_add_rms_norm_opt(x_buf, res_buf, weight, eps)"],
    "outputs": {"add": "res_buf (x + residual)", "rmsnorm": "x_buf (RMSNorm(x + residual) * weight)"},
    "input_semantics": "graph inputs read-only; vendor in-place writes go to private buffers",
    "eps": {"value": EPS, "source": "declared Qwen3 rms_norm_eps; not recorded in the graph"},
    "capture_support": "unverified_on_GPU",
}


def declared_binding():
    data = catalog()
    return {"id": RECIPE["id"], "version": "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "entrypoint": ENTRYPOINT, "provider_version": data["version"], "op_so_sha256": data["op_so_sha256"],
            "semantics": copy.deepcopy(RECIPE)}


def group_io(graph, node_ids):
    """Return (x_node, residual_node, weight_owner) for a legal (add, rmsnorm) group."""
    if len(node_ids) != 2:
        raise FusionRuleError("add_rmsnorm groups exactly one add and its rmsnorm consumer")
    add, norm = node_ids
    if (graph.nodes[add].op_type, graph.nodes[norm].op_type) != ("add", "rmsnorm"):
        raise FusionRuleError("add_rmsnorm expects ops (add, rmsnorm)")
    norm_inputs = [e.src for e in graph.in_edges(norm)]
    if norm_inputs != [add]:
        raise FusionRuleError("rmsnorm must read only the add output")
    add_inputs = [e.src for e in graph.in_edges(add)]
    if len(add_inputs) != 2 or add_inputs[0] == add_inputs[1]:
        raise FusionRuleError("add must have two distinct producer inputs (no broadcasting)")
    nodes = [graph.nodes[n] for n in (*add_inputs, add, norm)]
    if any(tuple(n.extra.get("output_shape") or ()) != SHAPE or n.extra.get("dtype") != "bf16" for n in nodes):
        raise FusionRuleError("LightOP add_rmsnorm v1 is restricted to BF16 [1,256,1024]")
    layers = {graph.nodes[n].extra.get("layer") for n in (add, norm)}
    if len(layers) != 1 or type(next(iter(layers))) is not int:
        raise FusionRuleError("add_rmsnorm requires explicit same-layer metadata")
    if any(graph.nodes[n].extra.get("phase") != "forward" for n in (add, norm)):
        raise FusionRuleError("add_rmsnorm adapter supports forward inference only")
    if graph.nodes[norm].param_bytes != SHAPE[-1] * 2:
        raise FusionRuleError("rmsnorm weight must be one BF16 vector of the hidden size")
    # The add output is exported by the kernel, so external consumers are allowed.
    # Convention: the first graph input is x, the second is residual; addition commutes.
    return add_inputs[0], add_inputs[1], norm


class _LightopAddRmsnormPattern(FusionPattern):
    def __init__(self):
        super().__init__("add_rmsnorm", "template", "implemented", op_types=("add", "rmsnorm"),
                         min_nodes=2, max_nodes=2, require_same_shape=True, require_linear_chain=False,
                         priority=30)

    def validate(self, graph, node_ids):
        super().validate(graph, node_ids)
        group_io(graph, node_ids)


def executable_registry():
    """Singletons + LightOP silu_mul + LightOP add_rmsnorm; default registry unchanged."""
    return FusionPatternRegistry([
        *[p for p in default_fusion_registry().patterns() if p.max_nodes == 1],
        _LightopPattern(), _LightopAddRmsnormPattern(),
    ])


class PreparedAddRmsnormCall:
    def __init__(self, x, residual, weight, eps=EPS):
        for t in (x, residual):
            if tuple(t.shape) != SHAPE or t.dtype != torch.bfloat16 or not t.is_contiguous() or t.requires_grad:
                raise ValueError("x/residual must be contiguous BF16 [1,256,1024] inference tensors")
        if tuple(weight.shape) != (SHAPE[-1],) or weight.dtype != torch.bfloat16 or not weight.is_contiguous():
            raise ValueError("weight must be a contiguous BF16 hidden-size vector")
        if len({x.device, residual.device, weight.device}) != 1 or not x.is_cuda or not torch.version.hip:
            raise RuntimeError("LightOP requires all tensors on one HIP GPU")
        from lightop import op  # resolved only on GPU; no native fallback
        self._op, self.eps = op, float(eps)
        self.x, self.residual, self.weight = x, residual, weight
        self.x_buf, self.res_buf = torch.empty_like(x), torch.empty_like(residual)
        self.observation = {"implementation": declared_binding(), "recipe": copy.deepcopy(RECIPE),
            "tensors": {k: tensor_observation(v) for k, v in (("x", x), ("residual", residual), ("weight", weight),
                        ("add_output", self.res_buf), ("rmsnorm_output", self.x_buf))}}

    def __call__(self):
        with torch.cuda.device(self.x.device), torch.no_grad():
            self.x_buf.copy_(self.x)
            self.res_buf.copy_(self.residual)
            self._op.fused_add_rms_norm_opt(self.x_buf.view(-1, SHAPE[-1]), self.res_buf.view(-1, SHAPE[-1]),
                                            self.weight, self.eps)
        return self.res_buf, self.x_buf


def bind(graph, node_ids, values, weight, eps=EPS):
    """values maps graph node ids to tensors; returns a prepared call for this group."""
    check_installed()
    x_node, residual_node, _ = group_io(graph, node_ids)
    for node_id in (x_node, residual_node):
        _check_declared_layout(graph.nodes[node_id], values[node_id])
    return PreparedAddRmsnormCall(values[x_node], values[residual_node], weight, eps)
