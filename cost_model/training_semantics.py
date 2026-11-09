"""Execution semantics: the framework-specific half of the training-memory model.

``cost_model.training_memory`` owns the generic mechanism: tensor lifetimes
over a forward schedule plus one backward pass, per-device peaks, M_base and
static training state. Everything that depends on *which* training backend
runs the graph lives here, in an ``ExecutionSemantics`` profile:

- per-op saved-for-backward rules (``OpSaveRule``);
- inputs a consumer saves in expanded form (e.g. GQA repeat_kv);
- loss semantics (what the loss keeps, what its backward allocates, when
  logits are released);
- weight-gradient semantics (BF16 wgrad buffers or direct accumulation into
  main_grad, and how a shared/tied weight's wgrads are accumulated).

The generic estimator never names a framework. A backend profile is data; a
new backend (FlagScale/Megatron, another runtime) is a new profile, audited
and measured on its own, not an edit to the estimator. Profiles for concrete
backends live in ``cost_model.backend_semantics``.

``STRUCTURAL_V1`` is the backend-agnostic default the estimator used before
any backend audit (training_lifetime_v1): every consumer keeps its inputs,
softmax keeps its output, one FP32 logits copy held until the producer's
backward, wgrad fused into main_grad. It is unverified for every backend.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

LOSS_RELEASE_AT_PRODUCER_BACKWARD = "producer_backward"
LOSS_RELEASE_AT_LOSS_BACKWARD = "loss_backward"
_LOSS_RELEASES = (LOSS_RELEASE_AT_PRODUCER_BACKWARD, LOSS_RELEASE_AT_LOSS_BACKWARD)
_INTERNAL_BASES = ("output", "row")


@dataclass(frozen=True)
class OpSaveRule:
    """What one op's backward keeps, for one backend implementation.

    ``saves_inputs``: the op keeps its graph inputs (on its own device).
    ``internal``: tensors the op creates and keeps that are not graph edges,
    as (basis, dtype) pairs. basis "output" = one element per output element,
    "row" = one per output row (all dims but the last). dtype is a dtype name,
    "output" (the node's dtype) or "compute" (the node's compute_dtype).
    ``saves_output``: the op keeps its own output.
    """

    saves_inputs: bool = True
    internal: tuple[tuple[str, str], ...] = ()
    saves_output: bool = False

    def __post_init__(self) -> None:
        for basis, _dtype in self.internal:
            if basis not in _INTERNAL_BASES:
                raise ValueError(f"unknown internal saved basis: {basis}")


@dataclass(frozen=True)
class LossSemantics:
    """Memory behaviour of the loss attached to the graph output (logits).

    ``saves_logits``: the loss keeps the logits tensor itself.
    ``fp32_copy_factor``: FP32 logits-sized copies the loss keeps (M_lm).
    ``backward_fp32_grads``: FP32 logits-sized gradients live at once inside
    the loss backward, on top of the kept copy.
    ``release``: when logits / the FP32 copy are freed: at the end of the loss
    backward, or only at the producer group's backward.
    """

    saves_logits: bool = True
    fp32_copy_factor: float = 1.0
    backward_fp32_grads: float = 0.0
    release: str = LOSS_RELEASE_AT_PRODUCER_BACKWARD
    verified: bool = False

    def __post_init__(self) -> None:
        if self.release not in _LOSS_RELEASES:
            raise ValueError(f"unknown loss release point: {self.release}")
        for name in ("fp32_copy_factor", "backward_fp32_grads"):
            value = float(getattr(self, name))
            if value != value or value < 0.0 or value == float("inf"):
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class WeightGradSemantics:
    """How weight gradients reach main_grad.

    ``accumulate_into_main_grad``: wgrad is written straight into the FP32
    main_grad buffer (Megatron gradient_accumulation_fusion); no transient.
    Otherwise every weight gets a weight-sized wgrad buffer during the
    backward of each group that uses it.
    ``shared_held_across_users``: for a weight used by several groups on one
    device (tied embedding / lm_head), the first wgrad stays live until the
    last user's backward instead of being reduced right away.
    ``shared_extra_copies_per_later_user``: weight-sized buffers each later
    user adds during its own backward (its wgrad plus an out-of-place sum = 2).
    """

    accumulate_into_main_grad: bool = True
    shared_held_across_users: bool = False
    shared_extra_copies_per_later_user: float = 1.0


@dataclass(frozen=True)
class ExecutionSemantics:
    """A backend profile consumed by ``estimate_training_lifetime``.

    ``ops`` applies to ops in single-node (unfused) groups; ``saves_output``
    also applies inside fused groups, whose other tensors follow the group's
    verified backward policy. ``expanded_inputs`` maps (consumer local_name,
    producer local_name) to a per-layer node attribute holding the expansion
    factor the consumer saves the input with.
    """

    name: str
    ops: Mapping[str, OpSaveRule] = field(default_factory=dict)
    default: OpSaveRule = OpSaveRule()
    expanded_inputs: Mapping[tuple[str, str], str] = field(default_factory=dict)
    loss: LossSemantics = LossSemantics()
    weight_grad: WeightGradSemantics = WeightGradSemantics()
    verified: bool = False
    source: str = ""

    def rule(self, op_type: str) -> OpSaveRule:
        return self.ops.get(op_type, self.default)

    def with_loss(self, **changes) -> "ExecutionSemantics":
        return replace(self, loss=replace(self.loss, **changes))

    def with_weight_grad(self, **changes) -> "ExecutionSemantics":
        return replace(self, weight_grad=replace(self.weight_grad, **changes))

    def to_dict(self) -> dict[str, Any]:
        """Full, JSON-ready description for frozen plans and reports."""

        return {
            "name": self.name,
            "verified": self.verified,
            "source": self.source,
            "ops": {k: asdict(v) for k, v in sorted(self.ops.items())},
            "default": asdict(self.default),
            "expanded_inputs": [[c, p, attr] for (c, p), attr in sorted(self.expanded_inputs.items())],
            "loss": asdict(self.loss),
            "weight_grad": asdict(self.weight_grad),
        }


STRUCTURAL_V1 = ExecutionSemantics(
    name="structural_v1",
    ops={"softmax": OpSaveRule(saves_output=True)},
    loss=LossSemantics(),
    weight_grad=WeightGradSemantics(),
    verified=False,
    source="backend-agnostic training_lifetime_v1 assumptions; not audited for any backend",
)


__all__ = [
    "ExecutionSemantics",
    "LOSS_RELEASE_AT_LOSS_BACKWARD",
    "LOSS_RELEASE_AT_PRODUCER_BACKWARD",
    "LossSemantics",
    "OpSaveRule",
    "STRUCTURAL_V1",
    "WeightGradSemantics",
]
