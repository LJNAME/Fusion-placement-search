"""Concrete backend execution-semantics profiles.

Each module audits one backend and exports ``ExecutionSemantics`` objects.
Profiles are registered by name so frozen plans can record and reload the
exact profile they were predicted with. Nothing in ``cost_model.training_memory``
imports this package.
"""

from __future__ import annotations

from ..training_semantics import STRUCTURAL_V1, ExecutionSemantics
from .hf_eager import HF_QWEN2_EAGER

_REGISTRY: dict[str, ExecutionSemantics] = {
    s.name: s for s in (STRUCTURAL_V1, HF_QWEN2_EAGER)
}


def get_semantics(name: str) -> ExecutionSemantics:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown execution semantics {name!r}; known: {sorted(_REGISTRY)}") from None


def available_semantics() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


__all__ = ["HF_QWEN2_EAGER", "available_semantics", "get_semantics"]
