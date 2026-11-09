"""P1-A training-memory model: static training state, M_base and Evaluation.

Method basis: LLMem (Kim et al., IJCAI 2024, doi:10.24963/ijcai.2024/699).
LLMem's peak is M_base + M_p + M_os + M_out + M_lm. This module supplies the
static terms (M_base, M_p, M_os) per device for operator-level placement:

    M_state[d] = bytes_per_param * P_d

where P_d counts unique trainable parameters resident on device d. The default
config is FlagScale/Megatron BF16 full fine-tuning with AdamW
(``--bf16`` without ``--grad-reduce-in-bf16`` or the precision-aware
optimizer), i.e. 2 + 4 + 4 + 4 + 4 = 18 B/param.

P1-B adds ``estimate_training_lifetime``: it reuses the structural forward
schedule (fusion groups, cross-device transfers, workspace) and extends the
tensor lifetimes through a backward pass. LLMem's M_out is not added on top;
it emerges from saved-activation lifetimes, so fused-internal tensors and
activations are counted exactly once. The scan holds no weights; those are in
the static state only.

LLMem's ZeRO-3 / TP formulas are intentionally not used: placement here is
per operator, so each device's state follows from the parameters placed on it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

from simulator.schema import OpGraph, Plan

_DTYPE_BYTES = {"fp32": 4, "bf16": 2, "fp16": 2, "fp8": 1}
_ADAM_OPTIMIZERS = {"adam", "adamw"}

# Semantics label for the activation part of the peak, so reports cannot
# mistake the P1-A result for a full training peak.
ACTIVATION_SEMANTICS = "training_lifetime_v1"


def _dtype_bytes(dtype: str) -> int:
    key = str(dtype).lower()
    if key not in _DTYPE_BYTES:
        raise ValueError(f"unsupported dtype: {dtype}")
    return _DTYPE_BYTES[key]


@dataclass(frozen=True)
class TrainingMemoryConfig:
    """Per-parameter training-state precision; change config, not formulas."""

    weight_dtype: str = "bf16"
    main_grad_dtype: str = "fp32"
    main_param_dtype: Optional[str] = "fp32"   # None: no separate master copy
    exp_avg_dtype: str = "fp32"
    exp_avg_sq_dtype: str = "fp32"
    optimizer: str = "adamw"
    source: str = "flagscale_bf16_adamw_default"

    def __post_init__(self) -> None:
        for name in ("weight_dtype", "main_grad_dtype", "exp_avg_dtype", "exp_avg_sq_dtype"):
            _dtype_bytes(getattr(self, name))
        if self.main_param_dtype is not None:
            _dtype_bytes(self.main_param_dtype)
        if self.optimizer.lower() not in _ADAM_OPTIMIZERS:
            raise NotImplementedError(f"only Adam/AdamW is modeled, got {self.optimizer}")

    @property
    def components(self) -> dict[str, int]:
        """Bytes per parameter for each training-state component."""

        return {
            "weight": _dtype_bytes(self.weight_dtype),
            "main_grad": _dtype_bytes(self.main_grad_dtype),
            "main_param": 0 if self.main_param_dtype is None else _dtype_bytes(self.main_param_dtype),
            "exp_avg": _dtype_bytes(self.exp_avg_dtype),
            "exp_avg_sq": _dtype_bytes(self.exp_avg_sq_dtype),
        }

    @property
    def bytes_per_param(self) -> int:
        return sum(self.components.values())

    @classmethod
    def from_flagscale_args(
        cls,
        *,
        bf16: bool = True,
        optimizer: str = "adam",
        grad_reduce_in_bf16: bool = False,
        use_precision_aware_optimizer: bool = False,
        main_grads_dtype: str = "fp32",
        main_params_dtype: str = "fp32",
        exp_avg_dtype: str = "fp32",
        exp_avg_sq_dtype: str = "fp32",
        use_distributed_optimizer: bool = False,
    ) -> "TrainingMemoryConfig":
        """Mirror flagscale/train/megatron/training/arguments.py precision rules.

        With --bf16, Megatron forces FP32 grad accumulation unless
        --grad-reduce-in-bf16 is set; the --main-*/--exp-avg* dtypes only take
        effect with --use-precision-aware-optimizer.
        """

        if not bf16:
            raise NotImplementedError("P1 models BF16 training only")
        if use_distributed_optimizer:
            raise NotImplementedError(
                "distributed optimizer shards state across DP ranks; operator-level "
                "placement has no DP group to shard over"
            )
        if use_precision_aware_optimizer:
            grad = "bf16" if grad_reduce_in_bf16 else main_grads_dtype
            main, m, v = main_params_dtype, exp_avg_dtype, exp_avg_sq_dtype
        else:
            grad = "bf16" if grad_reduce_in_bf16 else "fp32"
            main, m, v = "fp32", "fp32", "fp32"
        return cls(
            weight_dtype="bf16",
            main_grad_dtype=grad,
            main_param_dtype=main,
            exp_avg_dtype=m,
            exp_avg_sq_dtype=v,
            optimizer=optimizer,
            source="flagscale_args",
        )


@dataclass(frozen=True)
class MemoryBaseline:
    """M_base per device: bytes charged against capacity outside the model.

    ``per_device`` is what the evaluator adds. Built with ``from_components``
    it is runtime + persistent workspace + safety margin:

    - ``runtime_baseline_bytes``: device memory used outside the caching
      allocator after context init (HIP runtime, kernels, handles).
    - ``persistent_workspace_bytes``: allocator memory that stays allocated
      for the whole step once warmed up (BLAS workspaces), live during
      forward, backward and the optimizer step alike.
    - ``allocator_reserved_slack_bytes``: reserved minus allocated at peak.
      Recorded only, never charged: the caching allocator can reuse it.
    - ``capacity_safety_margin_bytes``: an explicit policy margin (default 0).

    Until measured on the target device it must stay ``calibrated=False``.
    """

    per_device: tuple[float, ...]
    calibrated: bool = False
    source: str = "uncalibrated_zero"
    runtime_baseline_bytes: Optional[tuple[float, ...]] = None
    persistent_workspace_bytes: Optional[tuple[float, ...]] = None
    allocator_reserved_slack_bytes: Optional[tuple[float, ...]] = None
    capacity_safety_margin_bytes: Optional[tuple[float, ...]] = None

    def __post_init__(self) -> None:
        n = len(self.per_device)
        for name in ("per_device", "runtime_baseline_bytes", "persistent_workspace_bytes",
                     "allocator_reserved_slack_bytes", "capacity_safety_margin_bytes"):
            values = getattr(self, name)
            if values is None:
                continue
            if len(values) != n:
                raise ValueError(f"{name} needs one value per device")
            if any(not math.isfinite(float(x)) or float(x) < 0.0 for x in values):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.calibrated and self.source.startswith("uncalibrated"):
            raise ValueError("a calibrated M_base needs a measurement source")
        charged = self.charged_components
        if charged:
            total = [sum(v[d] for v in charged.values()) for d in range(n)]
            if any(abs(t - float(p)) > 1e-6 * max(1.0, t) for t, p in zip(total, self.per_device)):
                raise ValueError("per_device must equal runtime + persistent workspace + margin")

    @property
    def charged_components(self) -> dict[str, tuple[float, ...]]:
        parts = {
            "runtime_baseline_bytes": self.runtime_baseline_bytes,
            "persistent_workspace_bytes": self.persistent_workspace_bytes,
            "capacity_safety_margin_bytes": self.capacity_safety_margin_bytes,
        }
        return {k: tuple(float(x) for x in v) for k, v in parts.items() if v is not None}

    @classmethod
    def from_components(
        cls,
        runtime_baseline_bytes: Sequence[float],
        persistent_workspace_bytes: Sequence[float],
        *,
        allocator_reserved_slack_bytes: Optional[Sequence[float]] = None,
        capacity_safety_margin_bytes: Optional[Sequence[float]] = None,
        calibrated: bool = True,
        source: str,
    ) -> "MemoryBaseline":
        runtime = tuple(float(x) for x in runtime_baseline_bytes)
        persistent = tuple(float(x) for x in persistent_workspace_bytes)
        margin = tuple(float(x) for x in (capacity_safety_margin_bytes or [0.0] * len(runtime)))
        slack = (None if allocator_reserved_slack_bytes is None
                 else tuple(float(x) for x in allocator_reserved_slack_bytes))
        return cls(
            tuple(r + p + m for r, p, m in zip(runtime, persistent, margin)),
            calibrated=calibrated, source=source,
            runtime_baseline_bytes=runtime, persistent_workspace_bytes=persistent,
            allocator_reserved_slack_bytes=slack, capacity_safety_margin_bytes=margin,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"per_device": list(self.per_device),
                               "calibrated": self.calibrated, "source": self.source}
        for name in ("runtime_baseline_bytes", "persistent_workspace_bytes",
                     "allocator_reserved_slack_bytes", "capacity_safety_margin_bytes"):
            value = getattr(self, name)
            if value is not None:
                out[name] = list(value)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryBaseline":
        kw = {k: tuple(data[k]) for k in ("runtime_baseline_bytes", "persistent_workspace_bytes",
                                          "allocator_reserved_slack_bytes",
                                          "capacity_safety_margin_bytes") if k in data}
        return cls(tuple(data["per_device"]), calibrated=bool(data.get("calibrated", False)),
                   source=str(data.get("source", "uncalibrated_zero")), **kw)

    @classmethod
    def uncalibrated(cls, n_devs: int) -> "MemoryBaseline":
        return cls(tuple(0.0 for _ in range(int(n_devs))))


def parameter_id(node) -> str:
    """Shared weights (tied embedding) map to the owning node's parameter ID."""

    extra = node.extra or {}
    return str(extra.get("param_id") or extra.get("shared_param_with") or node.id)


@dataclass(frozen=True)
class TrainingStateEstimate:
    params_per_device: tuple[float, ...]
    state_bytes: tuple[float, ...]
    raw_param_bytes: tuple[float, ...]   # weight bytes as the lifetime scan counts them
    violations: tuple[str, ...]


def estimate_training_state(
    graph: OpGraph, plan: Plan, n_devs: int, config: TrainingMemoryConfig
) -> TrainingStateEstimate:
    """Count unique trainable parameters per device and their training state.

    First version requires every user of a shared parameter to sit on the same
    device; a split tied embedding needs a replica plus grad sync, which is not
    modeled and is reported as a violation.
    """

    weight_bytes = _dtype_bytes(config.weight_dtype)
    graph_dtype = (graph.extra or {}).get("dtype")
    if graph_dtype is not None and _dtype_bytes(graph_dtype) != weight_bytes:
        raise ValueError(
            f"graph param_bytes use {graph_dtype}, config weight dtype is {config.weight_dtype}"
        )

    node_device = {nid: int(g.device) for g in plan.groups for nid in g.node_ids}
    param_bytes: dict[str, float] = {}
    param_devices: dict[str, set[int]] = {}
    raw = [0.0] * n_devs
    for nid, node in graph.nodes.items():
        device = node_device[nid]
        raw[device] += float(node.param_bytes)
        pid = parameter_id(node)
        shared = pid != nid
        if node.param_bytes <= 0.0 and not shared:
            continue
        if not (node.extra or {}).get("trainable", True):
            continue
        param_bytes[pid] = max(param_bytes.get(pid, 0.0), float(node.param_bytes))
        param_devices.setdefault(pid, set()).add(device)

    violations = []
    params = [0.0] * n_devs
    for pid in sorted(param_bytes):
        devices = param_devices[pid]
        if len(devices) > 1:
            violations.append(f"shared_param_split:{pid}:{sorted(devices)}")
        owner = min(devices)  # one copy only; split placements are already rejected
        params[owner] += param_bytes[pid] / weight_bytes

    bpp = config.bytes_per_param
    return TrainingStateEstimate(
        params_per_device=tuple(params),
        state_bytes=tuple(p * bpp for p in params),
        raw_param_bytes=tuple(raw),
        violations=tuple(violations),
    )


@dataclass(frozen=True)
class Evaluation:
    """Unified candidate result: feasibility is a state, not a score penalty.

    Infeasible candidates carry ``latency=None`` and must not enter Q averages
    or best-plan selection.
    """

    feasible: bool
    peak_per_device: dict[int, float]
    overflow_devices: tuple[int, ...]
    latency: Optional[float]
    reason: Optional[str]
    breakdown: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.feasible and (self.latency is None or self.reason is not None):
            raise ValueError("a feasible evaluation needs a latency and no reason")
        if not self.feasible and (self.latency is not None or not self.reason):
            raise ValueError("an infeasible evaluation needs a reason and no latency")


BACKWARD_SAVE = "save"
BACKWARD_RECOMPUTE = "recompute"


def _backward_policies(verified: Sequence[str] | Mapping[str, str]) -> dict[str, str]:
    """Pattern -> how its fused backward treats internal tensors.

    A plain sequence means "save" (the conservative choice): every internal
    tensor is kept until the group's backward. "recompute" keeps none and
    rebuilds them transiently during the group's backward.
    """

    if isinstance(verified, Mapping):
        policies = {str(k): str(v) for k, v in verified.items()}
    else:
        policies = {str(k): BACKWARD_SAVE for k in verified}
    bad = {k: v for k, v in policies.items() if v not in (BACKWARD_SAVE, BACKWARD_RECOMPUTE)}
    if bad:
        raise ValueError(f"unknown fused backward policy: {bad}")
    return policies


def unverified_training_fusions(
    plan: Plan, verified_patterns: Sequence[str] | Mapping[str, str] = ()
) -> tuple[str, ...]:
    """Multi-node groups whose backward is not source-verified are not training candidates."""

    allowed = set(_backward_policies(verified_patterns))
    found = set()
    for group in plan.groups:
        if len(group.node_ids) > 1:
            pattern = str(group.extra.get("pattern_id", "legacy"))
            if pattern not in allowed:
                found.add(pattern)
    return tuple(sorted(found))


@dataclass(frozen=True)
class OpSaveRule:
    """What one op's autograd node keeps for backward, per implementation.

    ``saves_inputs``: the op keeps its graph inputs (on its own device).
    ``internal``: tensors the op creates and keeps that are not graph edges,
    as (basis, dtype) pairs. basis "output" = one element per output element,
    "row" = one per output row (all dims but the last). dtype is a dtype name,
    "output" (the node's dtype) or "compute" (the node's compute_dtype).
    ``saves_output``: the op keeps its own output (the old output_saving_ops).
    """

    saves_inputs: bool = True
    internal: tuple[tuple[str, str], ...] = ()
    saves_output: bool = False


@dataclass(frozen=True)
class SavedTensorRules:
    """Implementation-specific saved-tensor semantics for unfused ops.

    Only applied to ops in single-node groups; fused groups keep their own
    backward policy (save / recompute). ``gqa_expanded_inputs`` lists
    (consumer local_name, producer local_name) edges that the consumer saves
    after repeat_kv expansion, scaled by the layer's ``gqa_repeat_factor``
    (taken from whichever node of that layer records it).
    Ops not in ``ops`` fall back to ``default``.
    """

    name: str
    ops: Mapping[str, OpSaveRule]
    default: OpSaveRule = OpSaveRule()
    gqa_expanded_inputs: frozenset[tuple[str, str]] = frozenset()
    # Whether the loss keeps the graph output (logits) itself for backward.
    # HF ForCausalLMLoss upcasts first, so only the FP32 copy (M_lm) is kept.
    loss_saves_logits: bool = True
    # FP32 logits-sized gradients live at once inside the loss backward, on
    # top of the saved FP32 copy (M_lm). HF: NllLossBackward and
    # LogSoftmaxBackward each allocate one before either is freed.
    loss_backward_fp32_grads: float = 0.0
    source: str = ""

    def rule(self, op_type: str) -> OpSaveRule:
        return self.ops.get(op_type, self.default)


# Audited 2026-11-08 on BW1000 (data/training_memory_p1d_v1_audit/
# saved_tensors_seq256.json, transformers 4.57.6, torch 2.9.0): one Qwen2
# eager decoder layer at seq 256 saves 41,420,800 B, which these rules
# reproduce exactly.
#   rmsnorm: x.to(fp32) kept by pow/mul, rsqrt kept, BF16 normalized kept by
#            the weight mul; the BF16 input itself is not kept.
#   softmax: FP32 result kept by SoftmaxBackward; the BF16 .to() copy is a
#            graph edge kept by the av bmm.
#   rope:    keeps only cos/sin, shared by all layers (not counted).
#   add / input passthrough / embedding (indices): keep no activation.
#   qk / av bmm keep K / V after repeat_kv (GQA-expanded).
HF_QWEN2_EAGER_RULES = SavedTensorRules(
    name="hf_qwen2_eager_saved_v1",
    ops={
        "matmul": OpSaveRule(saves_inputs=True),
        "silu": OpSaveRule(saves_inputs=True),
        "mul": OpSaveRule(saves_inputs=True),
        "add": OpSaveRule(saves_inputs=False),
        "input": OpSaveRule(saves_inputs=False),
        "embedding": OpSaveRule(saves_inputs=False),
        "rope": OpSaveRule(saves_inputs=False),
        "rmsnorm": OpSaveRule(
            saves_inputs=False,
            internal=(("output", "fp32"), ("output", "output"), ("row", "fp32")),
        ),
        "softmax": OpSaveRule(saves_inputs=False, internal=(("output", "compute"),)),
    },
    gqa_expanded_inputs=frozenset({
        ("self_attention.qk_matmul", "self_attention.k_rope"),
        ("self_attention.av_matmul", "self_attention.v_proj"),
    }),
    loss_saves_logits=False,
    loss_backward_fp32_grads=2.0,
    source="transformers==4.57.6 modeling_qwen2 eager; BW1000 saved_tensors_hooks audit",
)


def _internal_saved_bytes(node, rule: OpSaveRule) -> float:
    if not rule.internal:
        return 0.0
    extra = node.extra or {}
    out_elems = float(node.out_bytes) / _element_bytes(node)
    shape = extra.get("output_shape")
    rows = out_elems / float(shape[-1]) if shape else out_elems
    total = 0.0
    for basis, dtype in rule.internal:
        if dtype == "output":
            nbytes = _element_bytes(node)
        elif dtype == "compute":
            nbytes = _dtype_bytes(extra.get("compute_dtype") or extra.get("dtype") or "bf16")
        else:
            nbytes = _dtype_bytes(dtype)
        count = {"output": out_elems, "row": rows}.get(basis)
        if count is None:
            raise ValueError(f"unknown internal saved basis: {basis}")
        total += count * nbytes
    return total


@dataclass(frozen=True)
class TrainingLifetimeConfig:
    """P1-B knobs for activation, gradient and LM-head memory.

    ``loss_fp32_logits_factor``: extra FP32 bytes per logit held by the loss
    (FP32 logits / softmax kept for backward). Default 1.0 = one FP32 copy;
    not yet checked against the installed Megatron cross-entropy source.
    ``output_saving_ops``: ops whose backward needs their own output, so the
    producer device keeps it even when every consumer is remote.
    """

    loss_fp32_logits_factor: float = 1.0
    loss_semantics_verified: bool = False
    gradient_accumulation_fusion: bool = True   # Megatron default: wgrad -> main_grad
    output_saving_ops: frozenset[str] = frozenset({"softmax"})
    # None = training_lifetime_v1 semantics (every consumer keeps its inputs,
    # output_saving_ops keep their output); set to use per-op audited rules.
    saved_tensor_rules: Optional[SavedTensorRules] = None

    @property
    def activation_semantics(self) -> str:
        if self.saved_tensor_rules is None:
            return ACTIVATION_SEMANTICS
        return f"{ACTIVATION_SEMANTICS}+{self.saved_tensor_rules.name}"

    def __post_init__(self) -> None:
        if not math.isfinite(self.loss_fp32_logits_factor) or self.loss_fp32_logits_factor < 0:
            raise ValueError("loss_fp32_logits_factor must be finite and non-negative")


@dataclass(frozen=True)
class TrainingLifetimeEstimate:
    peak_per_device: tuple[float, ...]       # excludes weights and static state
    component_at_peak: tuple[dict[str, float], ...]
    backward_cross_device_bytes: float
    m_lm_bytes: tuple[float, ...]
    # Bytes still held when backward starts: saved activations, M_lm, send buffers.
    retained_at_forward_end: tuple[float, ...] = ()


def _element_bytes(node) -> int:
    dtype = (node.extra or {}).get("dtype")
    return _dtype_bytes(dtype) if dtype else 2


def estimate_training_lifetime(
    graph: OpGraph,
    plan: Plan,
    structure,
    forward_times: Sequence[tuple[float, float]],
    transfer_end: Mapping[int, float],
    makespan: float,
    *,
    config: TrainingLifetimeConfig = TrainingLifetimeConfig(),
    backward_policies: Mapping[str, str] | None = None,
) -> TrainingLifetimeEstimate:
    """training_lifetime_v1: one forward + one backward pass, no pipelining.

    Forward times come from the structural schedule. Backward is not a DAG
    yet: one loss-backward slot, then groups in reverse forward-completion
    order on logical unit slots after the forward makespan, so only the event
    order (and hence the peak) is meaningful, not backward durations.

    Saved activations: a tensor is kept on every device that runs one of its
    consumers (consumers need their input for backward) and on the producer
    device when the producer op needs its own output. It is freed after the
    last backward that needs it on that device. Activation gradients are one
    buffer per tensor on the producer device; remote consumers hold their
    computed gradient until their backward ends (the backward transfer).
    """

    backward_policies = dict(backward_policies or {})
    rules = config.saved_tensor_rules
    n_devs = structure.n_devs
    groups = structure.groups
    node_group = dict(structure.node_to_group)
    group_device = [g.device for g in groups]

    order = sorted(range(len(groups)), key=lambda i: (forward_times[i][1], i), reverse=True)
    bwd = [(0.0, 0.0)] * len(groups)
    # Slot [makespan, makespan + 1) is the loss backward; groups follow.
    loss_done = makespan + 0.5
    for k, gi in enumerate(order):
        bwd[gi] = (makespan + 1 + k, makespan + 2 + k)

    events: list[list[tuple[float, float, str]]] = [[] for _ in range(n_devs)]
    retained = [0.0] * n_devs

    def add(device, start, end, size, kind):
        if size > 0.0 and end > start:
            events[device].append((start, size, kind))
            events[device].append((end, -size, kind))
            if start <= makespan < end and kind in ("saved_activation", "m_lm", "send_buffer"):
                retained[device] += size

    deps_by_edge: dict[tuple[str, str], list] = {}
    for dep in structure.dependencies:
        deps_by_edge.setdefault((dep.src_node, dep.dst_node), []).append(dep)
    consumers: dict[str, list] = {nid: [] for nid in graph.nodes}
    for edge in graph.edges:
        consumers[edge.src].append(edge.dst)

    param_size: dict[str, float] = {}
    for nid, node in graph.nodes.items():
        pid = parameter_id(node)
        param_size[pid] = max(param_size.get(pid, 0.0), float(node.param_bytes))

    # repeat_kv factor per layer (recorded on the qk matmul only).
    gqa_factor: dict[Any, float] = {}
    for node in graph.nodes.values():
        factor = (node.extra or {}).get("gqa_repeat_factor")
        if factor:
            gqa_factor[(node.extra or {}).get("layer")] = float(factor)

    m_lm = [0.0] * n_devs
    backward_bytes = 0.0
    for nid, node in graph.nodes.items():
        size = float((node.extra or {}).get("saved_activation_bytes", node.out_bytes))
        if node.out_bytes <= 0.0:
            continue
        gp = node_group[nid]
        dp = group_device[gp]
        policy = backward_policies.get(groups[gp].pattern_id, BACKWARD_SAVE)
        internal = [c for c in consumers[nid] if node_group[c] == gp]
        external = [c for c in consumers[nid] if node_group[c] != gp]

        if not consumers[nid]:
            # Graph output (logits) and the loss's FP32 copy are needed only
            # by the loss backward, which frees them before the producer's
            # backward (the producer's backward needs its inputs, not logits).
            if rules is None or rules.loss_saves_logits:
                add(dp, forward_times[gp][1], loss_done, size, "saved_activation")
            elements = float(node.out_bytes) / _element_bytes(node)
            m_lm[dp] += elements * 4.0 * config.loss_fp32_logits_factor
            add(dp, forward_times[gp][1], loss_done,
                elements * 4.0 * config.loss_fp32_logits_factor, "m_lm")
            if rules is not None and rules.loss_backward_fp32_grads > 0.0:
                add(dp, makespan, loss_done, elements * 4.0 * rules.loss_backward_fp32_grads,
                    "loss_backward_transient")
                # The BF16 d(logits) is cast after the FP32 grads are freed.
                add(dp, loss_done + 0.25, bwd[gp][1], float(node.out_bytes), "activation_grad")
            else:
                add(dp, makespan, bwd[gp][1], float(node.out_bytes), "activation_grad")
            continue

        if internal and not external:
            if policy == BACKWARD_SAVE:
                add(dp, forward_times[gp][1], bwd[gp][1], size, "saved_activation")
            else:
                add(dp, bwd[gp][0], bwd[gp][1], float(node.out_bytes), "recompute")
            continue

        # Per device: (alloc time, release time, bytes) for the saved copy.
        saved: dict[int, list[float]] = {}
        unfused = len(groups[gp].node_ids) == 1
        if rules is not None and unfused:
            rule = rules.rule(node.op_type)
            needs_self = rule.saves_output
            add(dp, forward_times[gp][1], bwd[gp][1], _internal_saved_bytes(node, rule),
                "saved_activation")
        else:
            needs_self = node.op_type in config.output_saving_ops
        needs_self = needs_self or bool(internal and policy == BACKWARD_SAVE)
        if needs_self:
            saved[dp] = [forward_times[gp][1], bwd[gp][1], size]
        elif internal:
            add(dp, bwd[gp][0], bwd[gp][1], float(node.out_bytes), "recompute")
        send_release = forward_times[gp][1]
        grad_arrival = []
        for c in external:
            gc = node_group[c]
            dc = group_device[gc]
            for dep in deps_by_edge[(nid, c)]:
                if dep.cross_device:
                    arrive = transfer_end[dep.index]
                    send_release = max(send_release, arrive)
                    nbytes = dep.bytes
                    # Remote consumer computes d(input) and sends it back.
                    add(dc, bwd[gc][0], bwd[gc][1], dep.bytes, "activation_grad")
                    backward_bytes += dep.bytes
                    grad_arrival.append(bwd[gc][1])
                else:
                    arrive = forward_times[gp][1]
                    nbytes = size
                    grad_arrival.append(bwd[gc][0])
                if rules is not None and len(groups[gc].node_ids) == 1:
                    cnode = graph.nodes[c]
                    if not rules.rule(cnode.op_type).saves_inputs:
                        continue
                    pair = ((cnode.extra or {}).get("local_name"),
                            (node.extra or {}).get("local_name"))
                    if pair in rules.gqa_expanded_inputs:
                        nbytes *= gqa_factor.get((cnode.extra or {}).get("layer"), 1.0)
                slot = saved.setdefault(dc, [arrive, bwd[gc][1], nbytes])
                slot[0] = min(slot[0], arrive)
                slot[1] = max(slot[1], bwd[gc][1])
                slot[2] = max(slot[2], nbytes)
        for device, (start, end, nbytes) in saved.items():
            add(device, start, end, nbytes, "saved_activation")
        if dp not in saved:
            add(dp, forward_times[gp][1], send_release, float(node.out_bytes), "send_buffer")
        if grad_arrival:
            # d(output) accumulates on the producer from the first consumer
            # backward until the producer's own backward consumes it.
            add(dp, min(grad_arrival), bwd[gp][1], float(node.out_bytes), "activation_grad")

    for group in groups:
        fwd_ws = float(group.workspace_bytes)
        start, end = forward_times[group.index]
        add(group.device, start, end, fwd_ws, "workspace")
        bwd_ws = float(plan.groups[group.index].extra.get("backward_workspace_bytes", fwd_ws))
        add(group.device, bwd[group.index][0], bwd[group.index][1], bwd_ws, "workspace")
    if not config.gradient_accumulation_fusion:
        # BF16 wgrad buffers (autograd .grad path). A weight used by one group
        # holds its wgrad during that group's backward. A weight used by
        # several groups (tied lm_head / embedding) follows autograd's
        # InputBuffer: the first wgrad is held until the last user's backward,
        # and each later user adds its own wgrad plus an out-of-place sum, so
        # 3x the weight is live inside the last backward (BW1000 trace,
        # data/training_memory_p1d_v1_audit/backward_trace).
        users: dict[str, list[int]] = {}
        for group in groups:
            for pid in {parameter_id(graph.nodes[n]) for n in group.node_ids}:
                if param_size.get(pid, 0.0) > 0.0:
                    users.setdefault(pid, []).append(group.index)
        for pid, gis in users.items():
            size = param_size[pid]
            if len(gis) == 1:
                g = gis[0]
                add(group_device[g], bwd[g][0], bwd[g][1], size, "weight_grad_transient")
                continue
            if len({group_device[g] for g in gis}) > 1:
                # Split shared weights are rejected by estimate_training_state.
                for g in gis:
                    add(group_device[g], bwd[g][0], bwd[g][1], size, "weight_grad_transient")
                continue
            gis = sorted(gis, key=lambda g: bwd[g][0])
            device = group_device[gis[0]]
            add(device, bwd[gis[0]][0], bwd[gis[-1]][1], size, "weight_grad_transient")
            for g in gis[1:]:
                add(device, bwd[g][0], bwd[g][1], 2.0 * size, "weight_grad_transient")

    peaks, at_peak = [], []
    for device in range(n_devs):
        current, peak = 0.0, 0.0
        parts: dict[str, float] = {}
        best: dict[str, float] = {}
        for _t, delta, kind in sorted(events[device], key=lambda e: (e[0], -e[1])):
            current += delta
            parts[kind] = parts.get(kind, 0.0) + delta
            if current > peak:
                peak, best = current, {k: v for k, v in parts.items() if v > 0.0}
        peaks.append(float(peak))
        at_peak.append(best)
    return TrainingLifetimeEstimate(
        tuple(peaks), tuple(at_peak), float(backward_bytes), tuple(m_lm), tuple(retained)
    )


def evaluate_training_candidate(
    graph: OpGraph,
    plan: Plan,
    profile,
    capacity: Sequence[float],
    *,
    config: TrainingMemoryConfig = TrainingMemoryConfig(),
    baseline: Optional[MemoryBaseline] = None,
    lifetime_model=None,
    latency_fn: Optional[Callable[[OpGraph, Plan], float]] = None,
    verified_backward_patterns: Sequence[str] | Mapping[str, str] = (),
    lifetime_config: TrainingLifetimeConfig = TrainingLifetimeConfig(),
) -> Evaluation:
    """Memory first, per device; latency only for feasible candidates.

    1. structural checks (tied-param split, unverified fused backward);
    2. static M_base + training state vs capacity (no scheduling needed);
    3. full peak = M_base + state + lifetime peak without weights;
    4. latency from ``latency_fn`` or the lifetime model's makespan.
    """

    from .structural import StructuralCostModel

    n_devs = int(profile.n_devs)
    capacity = tuple(float(x) for x in capacity)
    if len(capacity) != n_devs or any(not math.isfinite(x) or x <= 0.0 for x in capacity):
        raise ValueError("capacity must hold one finite positive byte limit per device")
    baseline = baseline or MemoryBaseline.uncalibrated(n_devs)
    if len(baseline.per_device) != n_devs:
        raise ValueError("M_base and device counts differ")

    state = estimate_training_state(graph, plan, n_devs, config)
    breakdown: dict[str, Any] = {
        "bytes_per_param": config.bytes_per_param,
        "config_source": config.source,
        "params_per_device": list(state.params_per_device),
        "training_state_bytes": list(state.state_bytes),
        "m_base_bytes": list(baseline.per_device),
        "m_base_calibrated": baseline.calibrated,
        "m_base_components": baseline.to_dict(),
        "activation_semantics": lifetime_config.activation_semantics,
    }

    def infeasible(peaks, reason, overflow=()):
        return Evaluation(False, dict(enumerate(peaks)), tuple(overflow), None, reason, breakdown)

    static = [b + s for b, s in zip(baseline.per_device, state.state_bytes)]
    policies = _backward_policies(verified_backward_patterns)
    unverified = unverified_training_fusions(plan, policies)
    if state.violations or unverified:
        reasons = list(state.violations) + [f"fusion_backward_unverified:{p}" for p in unverified]
        return infeasible(static, ";".join(reasons))

    overflow = [d for d in range(n_devs) if static[d] > capacity[d]]
    if overflow:
        return infeasible(static, "static_memory_overflow", overflow)

    model = lifetime_model or StructuralCostModel()
    prepared = model.prepare(graph, plan, n_devs)
    estimate = model.predict_prepared(prepared, profile)
    details = estimate.details
    forward_times = [(float(g["start"]), float(g["end"])) for g in details["groups"]]
    transfer_end = {int(t["dependency_index"]): float(t["end"]) for t in details["transfers"]}
    lifetime = estimate_training_lifetime(
        graph, plan, prepared.structure, forward_times, transfer_end, float(estimate.makespan),
        config=lifetime_config, backward_policies=policies,
    )
    # The training scan never holds weights, so static state is added once.
    dynamic = list(lifetime.peak_per_device)
    peaks = [s + a for s, a in zip(static, dynamic)]
    breakdown.update(
        lifetime_bytes_excluding_weights=dynamic,
        lifetime_components_at_peak=[dict(c) for c in lifetime.component_at_peak],
        m_lm_bytes=list(lifetime.m_lm_bytes),
        retained_at_forward_end=list(lifetime.retained_at_forward_end),
        loss_semantics_verified=lifetime_config.loss_semantics_verified,
        backward_cross_device_bytes=lifetime.backward_cross_device_bytes,
        forward_cross_device_bytes=float(estimate.cross_device_bytes),
    )
    overflow = [d for d in range(n_devs) if peaks[d] > capacity[d]]
    if overflow:
        return infeasible(peaks, "peak_memory_overflow", overflow)

    latency = float(latency_fn(graph, plan)) if latency_fn is not None else float(estimate.makespan)
    if not math.isfinite(latency) or latency < 0.0:
        raise ValueError("latency must be finite and non-negative")
    return Evaluation(True, dict(enumerate(peaks)), (), latency, None, breakdown)


def make_training_evaluator(
    graph: OpGraph,
    profile,
    capacity: Sequence[float],
    **kwargs,
) -> Callable[[Plan], Evaluation]:
    """Cached plan -> Evaluation for ``legal_mcts_search(evaluator=...)``."""

    cache: dict[tuple, Evaluation] = {}

    def evaluator(plan: Plan) -> Evaluation:
        key = tuple(
            (tuple(sorted(g.node_ids)), int(g.device), str(g.extra.get("pattern_id", "")))
            for g in plan.groups
        )
        if key not in cache:
            cache[key] = evaluate_training_candidate(graph, plan, profile, capacity, **kwargs)
        return cache[key]

    evaluator.cache = cache  # type: ignore[attr-defined]
    return evaluator


def make_static_memory_prune(
    graph: OpGraph,
    capacity: Sequence[float],
    *,
    config: TrainingMemoryConfig = TrainingMemoryConfig(),
    baseline: Optional[MemoryBaseline] = None,
    verified_backward_patterns: Sequence[str] | Mapping[str, str] = (),
) -> Callable:
    """Sound prune for partial MCTS states.

    Uses only facts that completion cannot undo: training state of parameters
    already placed (a lower bound, since activations only add memory), a
    shared parameter already split across devices, and an unverified fused
    backward already chosen.
    """

    capacity = tuple(float(x) for x in capacity)
    baseline = baseline or MemoryBaseline.uncalibrated(len(capacity))
    allowed = set(_backward_policies(verified_backward_patterns))
    weight_bytes = _dtype_bytes(config.weight_dtype)
    bpp = config.bytes_per_param
    pid_of = {nid: parameter_id(n) for nid, n in graph.nodes.items()}
    pbytes: dict[str, float] = {}
    for nid, node in graph.nodes.items():
        if (node.extra or {}).get("trainable", True):
            pid = pid_of[nid]
            pbytes[pid] = max(pbytes.get(pid, 0.0), float(node.param_bytes))

    def prune(space, state) -> bool:
        placed: dict[str, int] = {}
        for choice in state.choices:
            nodes = choice.candidate.node_ids
            if len(nodes) > 1 and choice.candidate.pattern_id not in allowed:
                return True
            for nid in nodes:
                pid = pid_of[nid]
                if pbytes.get(pid, 0.0) <= 0.0 and pid == nid:
                    continue
                if placed.setdefault(pid, choice.device) != choice.device:
                    return True
        used = list(baseline.per_device)
        for pid, device in placed.items():
            used[device] += pbytes.get(pid, 0.0) / weight_bytes * bpp
        return any(u > c for u, c in zip(used, capacity))

    return prune


__all__ = [
    "ACTIVATION_SEMANTICS",
    "Evaluation",
    "MemoryBaseline",
    "TrainingMemoryConfig",
    "BACKWARD_RECOMPUTE",
    "BACKWARD_SAVE",
    "HF_QWEN2_EAGER_RULES",
    "OpSaveRule",
    "SavedTensorRules",
    "TrainingLifetimeConfig",
    "TrainingLifetimeEstimate",
    "TrainingStateEstimate",
    "estimate_training_lifetime",
    "estimate_training_state",
    "evaluate_training_candidate",
    "make_static_memory_prune",
    "make_training_evaluator",
    "parameter_id",
    "unverified_training_fusions",
]
