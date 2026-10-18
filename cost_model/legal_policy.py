"""Variable-action policy/value interface for legal fusion search.

The legacy policy head uses a fixed ``merge + device`` action vector tied to a
linear node order.  Legal search instead chooses one of the currently valid
``FusionCandidate x device`` pairs.  This module gives those variable action
sets stable identities, structural features, and a pointer-style Torch head.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .legal_search import LegalChoice, LegalFusionSearchSpace, LegalPartialState


ACTION_NUMERIC_DIM = 10
STATE_NUMERIC_DIM = 6


def legal_action_id(choice: LegalChoice) -> str:
    nodes = ",".join(choice.candidate.node_ids)
    return f"{choice.candidate.pattern_id}|{nodes}|device:{choice.device}"


@dataclass(frozen=True)
class LegalPolicyEncoding:
    state_features: torch.Tensor
    action_features: torch.Tensor
    action_ids: Tuple[str, ...]


def _edge_nbytes(space: LegalFusionSearchSpace, edge) -> float:
    if edge.bytes is not None:
        return float(edge.bytes)
    return float(space.graph.nodes[edge.src].out_bytes)


def _ratio(value: float, total: float) -> float:
    return float(value) / max(float(total), 1e-12)


def encode_legal_policy_state(
    space: LegalFusionSearchSpace,
    state: LegalPartialState,
    actions: Sequence[LegalChoice] | None = None,
) -> LegalPolicyEncoding:
    """Encode one state and its current, variable-length legal action set."""

    legal = tuple(space.actions(state) if actions is None else actions)
    if not legal and not space.is_terminal(state):
        raise ValueError("non-terminal legal state has no actions")
    expected = tuple(space.actions(state))
    if legal != expected:
        raise ValueError("actions must equal space.actions(state) in deterministic order")

    graph = space.graph
    patterns = tuple(pattern.pattern_id for pattern in space.registry.patterns())
    pattern_index = {pattern_id: index for index, pattern_id in enumerate(patterns)}
    total_nodes = max(len(graph.nodes), 1)
    total_flops = sum(float(node.flops) for node in graph.nodes.values())
    total_out = sum(float(node.out_bytes) for node in graph.nodes.values())
    total_param = sum(float(node.param_bytes) for node in graph.nodes.values())
    total_edge = sum(_edge_nbytes(space, edge) for edge in graph.edges)
    assigned_nodes = [graph.nodes[node_id] for node_id in state.assigned]
    last_device = state.choices[-1].device if state.choices else -1

    state_values = [
        len(state.assigned) / total_nodes,
        len(state.choices) / total_nodes,
        (total_nodes - len(state.assigned)) / total_nodes,
        _ratio(sum(float(node.flops) for node in assigned_nodes), total_flops),
        _ratio(sum(float(node.out_bytes) for node in assigned_nodes), total_out),
        _ratio(sum(float(node.param_bytes) for node in assigned_nodes), total_param),
    ]
    state_values.extend(
        1.0 if device == last_device else 0.0 for device in range(space.n_devs)
    )

    action_rows = []
    for choice in legal:
        node_ids = set(choice.candidate.node_ids)
        nodes = [graph.nodes[node_id] for node_id in choice.candidate.node_ids]
        internal = 0.0
        incoming = 0.0
        outgoing = 0.0
        for edge in graph.edges:
            nbytes = _edge_nbytes(space, edge)
            if edge.src in node_ids and edge.dst in node_ids:
                internal += nbytes
            elif edge.dst in node_ids and edge.src not in node_ids:
                incoming += nbytes
            elif edge.src in node_ids and edge.dst not in node_ids:
                outgoing += nbytes

        row = [0.0] * (len(patterns) + space.n_devs)
        row[pattern_index[choice.candidate.pattern_id]] = 1.0
        row[len(patterns) + choice.device] = 1.0
        row.extend(
            [
                len(nodes) / total_nodes,
                math.log1p(len(nodes)) / math.log1p(total_nodes),
                _ratio(sum(float(node.flops) for node in nodes), total_flops),
                _ratio(sum(float(node.out_bytes) for node in nodes), total_out),
                _ratio(sum(float(node.param_bytes) for node in nodes), total_param),
                _ratio(internal, total_edge),
                _ratio(incoming, total_edge),
                _ratio(outgoing, total_edge),
                1.0 if choice.device == last_device else 0.0,
                1.0 if len(nodes) > 1 else 0.0,
            ]
        )
        action_rows.append(row)

    action_dim = len(patterns) + space.n_devs + ACTION_NUMERIC_DIM
    action_features = torch.tensor(action_rows, dtype=torch.float32)
    if not action_rows:
        action_features = torch.empty((0, action_dim), dtype=torch.float32)
    return LegalPolicyEncoding(
        state_features=torch.tensor(state_values, dtype=torch.float32),
        action_features=action_features,
        action_ids=tuple(legal_action_id(choice) for choice in legal),
    )


class LegalPolicyValueNet(nn.Module):
    """Pointer-style policy plus scalar value for variable legal action sets."""

    def __init__(
        self,
        n_devs: int,
        pattern_vocab: Sequence[str],
        hidden: int = 128,
    ) -> None:
        super().__init__()
        if n_devs <= 0:
            raise ValueError("n_devs must be positive")
        if not pattern_vocab:
            raise ValueError("pattern_vocab must not be empty")
        self.n_devs = int(n_devs)
        self.pattern_vocab = tuple(pattern_vocab)
        state_dim = STATE_NUMERIC_DIM + self.n_devs
        action_dim = len(self.pattern_vocab) + self.n_devs + ACTION_NUMERIC_DIM
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.policy_scorer = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )

    def forward(
        self,
        state_features: torch.Tensor,
        action_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state_features.ndim != 1 or action_features.ndim != 2:
            raise ValueError("expected state (S,) and actions (A,F)")
        state = self.state_encoder(state_features)
        value = self.value_head(state).squeeze(-1)
        actions = self.action_encoder(action_features)
        pairs = torch.cat(
            [state.unsqueeze(0).expand(actions.size(0), -1), actions], dim=-1
        )
        logits = self.policy_scorer(pairs).squeeze(-1)
        return value, logits


class TorchLegalPolicyValueEvaluator:
    """Adapter from ``LegalPolicyValueNet`` to ``legal_mcts_search``."""

    def __init__(self, net: LegalPolicyValueNet, device: str = "cpu") -> None:
        self.net = net.to(device)
        self.device = device

    def __call__(
        self,
        space: LegalFusionSearchSpace,
        state: LegalPartialState,
        actions: Sequence[LegalChoice],
    ) -> tuple[float, Tuple[float, ...]]:
        expected_patterns = tuple(
            pattern.pattern_id for pattern in space.registry.patterns()
        )
        if space.n_devs != self.net.n_devs:
            raise ValueError(
                f"network n_devs={self.net.n_devs}, search n_devs={space.n_devs}"
            )
        if expected_patterns != self.net.pattern_vocab:
            raise ValueError("network pattern_vocab does not match search registry")
        encoded = encode_legal_policy_state(space, state, actions)
        with torch.no_grad():
            value, logits = self.net(
                encoded.state_features.to(self.device),
                encoded.action_features.to(self.device),
            )
            priors = torch.softmax(logits, dim=0)
        return float(value.cpu()), tuple(float(item) for item in priors.cpu())


def legal_policy_value_loss(
    value: torch.Tensor,
    logits: torch.Tensor,
    value_target: torch.Tensor,
    policy_target: torch.Tensor,
    value_weight: float = 1.0,
    policy_weight: float = 1.0,
) -> torch.Tensor:
    if logits.shape != policy_target.shape:
        raise ValueError("policy target must match the variable action logits")
    target_sum = policy_target.sum()
    if float(target_sum.detach()) <= 0.0:
        raise ValueError("policy target must have positive mass")
    target = policy_target / target_sum
    value_term = F.smooth_l1_loss(value, value_target)
    policy_term = -(target * F.log_softmax(logits, dim=0)).sum()
    return value_weight * value_term + policy_weight * policy_term


__all__ = [
    "ACTION_NUMERIC_DIM",
    "STATE_NUMERIC_DIM",
    "LegalPolicyEncoding",
    "LegalPolicyValueNet",
    "TorchLegalPolicyValueEvaluator",
    "encode_legal_policy_state",
    "legal_action_id",
    "legal_policy_value_loss",
]
