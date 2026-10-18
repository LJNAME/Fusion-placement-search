"""CPU verification for the variable legal-action policy/value interface."""

from __future__ import annotations

import torch

from cost_model import (
    LegalFusionSearchSpace,
    LegalPolicyValueNet,
    TorchLegalPolicyValueEvaluator,
    encode_legal_policy_state,
    legal_mcts_search,
)
from cost_model.verify_exact_search import _eight_node_graph


def verify_legal_policy() -> dict[str, object]:
    graph = _eight_node_graph()
    space = LegalFusionSearchSpace(graph, n_devs=2)
    root = space.initial_state()
    root_actions = space.actions(root)
    root_encoding = encode_legal_policy_state(space, root, root_actions)
    next_state = space.apply(root, root_actions[0])
    next_actions = space.actions(next_state)
    next_encoding = encode_legal_policy_state(space, next_state, next_actions)

    torch.manual_seed(17)
    patterns = tuple(pattern.pattern_id for pattern in space.registry.patterns())
    net = LegalPolicyValueNet(2, patterns, hidden=32)
    evaluator = TorchLegalPolicyValueEvaluator(net)
    value, priors = evaluator(space, next_state, next_actions)
    plans = legal_mcts_search(
        graph,
        lambda plan: float(len(plan.groups) + sum(group.device for group in plan.groups)),
        n_devs=2,
        iterations=32,
        rollouts=1,
        seed=17,
        n_best=4,
        policy_value_evaluator=evaluator,
        value_mix=0.0,
    )
    if len(priors) != len(next_actions) or abs(sum(priors) - 1.0) > 1e-6:
        raise AssertionError("variable policy prior contract failed")
    return {
        "root_actions": len(root_actions),
        "next_actions": len(next_actions),
        "state_features": root_encoding.state_features.numel(),
        "root_action_features": tuple(root_encoding.action_features.shape),
        "next_action_features": tuple(next_encoding.action_features.shape),
        "network_value": f"{value:.6f}",
        "prior_sum": f"{sum(priors):.6f}",
        "returned_plans": len(plans),
    }


def main() -> None:
    for key, value in verify_legal_policy().items():
        print(f"{key}: {value}")
    print("Fusion M4b variable-action policy/value verification: OK")


if __name__ == "__main__":
    main()
