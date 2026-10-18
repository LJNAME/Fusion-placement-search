"""Beam and MCTS over executable fusion-pattern candidates.

Unlike the legacy search, a state selects disjoint graph-local candidates.  A
candidate may cover nodes that are not adjacent in one arbitrary linear
topological order (for example Qwen3 ``silu -> mul`` with a second ``up -> mul``
input).  Complete selections are contracted into a group DAG and only then
topologically ordered into a :class:`simulator.schema.Plan`.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from simulator import (
    FusionCandidate,
    FusionPatternRegistry,
    OpGraph,
    Plan,
    default_fusion_registry,
    enumerate_fusion_candidates,
    validate_fusion_plan,
)


@dataclass(frozen=True)
class LegalChoice:
    candidate: FusionCandidate
    device: int


@dataclass(frozen=True)
class LegalPartialState:
    choices: Tuple[LegalChoice, ...] = ()
    assigned: frozenset[str] = frozenset()

    def key(self) -> tuple:
        return tuple(
            (choice.candidate.pattern_id, choice.candidate.node_ids, choice.device)
            for choice in self.choices
        )


class LegalFusionSearchSpace:
    """Candidate-cover search space with deterministic contraction semantics."""

    def __init__(
        self,
        graph: OpGraph,
        n_devs: int,
        registry: Optional[FusionPatternRegistry] = None,
    ) -> None:
        if n_devs <= 0:
            raise ValueError("n_devs must be positive")
        self.graph = graph
        self.n_devs = int(n_devs)
        self.registry = registry or default_fusion_registry()
        self.order = tuple(graph.topological_order())
        self.position = {node_id: index for index, node_id in enumerate(self.order)}
        self.candidates = tuple(enumerate_fusion_candidates(graph, self.registry))
        by_node: Dict[str, List[FusionCandidate]] = {node_id: [] for node_id in self.order}
        for candidate in self.candidates:
            for node_id in candidate.node_ids:
                by_node[node_id].append(candidate)
        for node_id, node_candidates in by_node.items():
            node_candidates.sort(
                key=lambda candidate: (
                    -len(candidate.node_ids),
                    candidate.pattern_id,
                    candidate.node_ids,
                )
            )
            if not any(len(candidate.node_ids) == 1 for candidate in node_candidates):
                raise ValueError(f"node {node_id} has no singleton fusion candidate")
        self.by_node = by_node

    def initial_state(self) -> LegalPartialState:
        return LegalPartialState()

    def is_terminal(self, state: LegalPartialState) -> bool:
        return len(state.assigned) == len(self.order)

    def anchor(self, state: LegalPartialState) -> Optional[str]:
        return next((node_id for node_id in self.order if node_id not in state.assigned), None)

    def actions(self, state: LegalPartialState) -> List[LegalChoice]:
        anchor = self.anchor(state)
        if anchor is None:
            return []
        result: List[LegalChoice] = []
        for candidate in self.by_node[anchor]:
            if any(node_id in state.assigned for node_id in candidate.node_ids):
                continue
            for device in range(self.n_devs):
                result.append(LegalChoice(candidate, device))
        return result

    def apply(self, state: LegalPartialState, choice: LegalChoice) -> LegalPartialState:
        if choice.device < 0 or choice.device >= self.n_devs:
            raise ValueError(f"device out of range: {choice.device}")
        legal = self.actions(state)
        if choice not in legal:
            raise ValueError(
                f"choice is not legal at anchor {self.anchor(state)}: "
                f"{choice.candidate.pattern_id}{choice.candidate.node_ids}"
            )
        return LegalPartialState(
            choices=state.choices + (choice,),
            assigned=state.assigned.union(choice.candidate.node_ids),
        )

    def to_plan(
        self,
        state: LegalPartialState,
        name: str = "legal-fusion",
        validate: bool = True,
    ) -> Plan:
        if not self.is_terminal(state):
            raise ValueError(
                f"cannot build plan from partial state: "
                f"{len(state.assigned)}/{len(self.order)} nodes assigned"
            )

        node_to_group: Dict[str, int] = {}
        for group_index, choice in enumerate(state.choices):
            for node_id in choice.candidate.node_ids:
                if node_id in node_to_group:
                    raise ValueError(f"node assigned to multiple groups: {node_id}")
                node_to_group[node_id] = group_index

        successors: List[set[int]] = [set() for _ in state.choices]
        indegree = [0] * len(state.choices)
        for edge in self.graph.edges:
            src_group = node_to_group[edge.src]
            dst_group = node_to_group[edge.dst]
            if src_group == dst_group or dst_group in successors[src_group]:
                continue
            successors[src_group].add(dst_group)
            indegree[dst_group] += 1

        def group_key(group_index: int) -> tuple:
            choice = state.choices[group_index]
            return (
                min(self.position[node_id] for node_id in choice.candidate.node_ids),
                choice.candidate.pattern_id,
                choice.device,
                group_index,
            )

        ready = [group_key(index) for index, degree in enumerate(indegree) if degree == 0]
        heapq.heapify(ready)
        group_order: List[int] = []
        while ready:
            *_, group_index = heapq.heappop(ready)
            group_order.append(group_index)
            for successor in sorted(successors[group_index], key=group_key):
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    heapq.heappush(ready, group_key(successor))
        if len(group_order) != len(state.choices):
            raise ValueError("fusion candidate contraction produced a cyclic group graph")

        plan = Plan(
            groups=[
                state.choices[group_index].candidate.to_group(
                    state.choices[group_index].device
                )
                for group_index in group_order
            ],
            name=name,
            extra={"search_space": "legal_fusion_v1"},
        )
        if validate:
            validate_fusion_plan(
                self.graph,
                plan,
                self.n_devs,
                registry=self.registry,
                require_executable=False,
            )
        return plan


def _choose_rollout_action(
    actions: Sequence[LegalChoice],
    rng: random.Random,
    fusion_probability: float,
) -> LegalChoice:
    fused = [choice for choice in actions if len(choice.candidate.node_ids) > 1]
    singleton = [choice for choice in actions if len(choice.candidate.node_ids) == 1]
    if fused and singleton:
        pool = fused if rng.random() < fusion_probability else singleton
    else:
        pool = fused or singleton
    return pool[rng.randrange(len(pool))]


def rollout_legal_plan(
    space: LegalFusionSearchSpace,
    state: LegalPartialState,
    rng: random.Random,
    fusion_probability: float = 0.5,
    name: str = "legal-rollout",
    validate: bool = True,
) -> Plan:
    current = state
    while not space.is_terminal(current):
        actions = space.actions(current)
        if not actions:
            raise RuntimeError(f"no legal action at anchor {space.anchor(current)}")
        current = space.apply(
            current,
            _choose_rollout_action(actions, rng, fusion_probability),
        )
    return space.to_plan(current, name=name, validate=validate)


def _guided_rollout_legal_plan(
    space: LegalFusionSearchSpace,
    state: LegalPartialState,
    name: str = "legal-guided-rollout",
) -> Plan:
    """Complete a state with a deterministic fusion/affinity baseline.

    The completion prefers the largest executable candidate at the current
    anchor and keeps groups on the most recently selected device.  It is not a
    hardware oracle; it merely preserves the two cheap baselines that random
    rollouts often miss: fuse where legal and avoid gratuitous transfers.
    """

    current = state
    preferred_device = current.choices[-1].device if current.choices else 0
    while not space.is_terminal(current):
        actions = space.actions(current)
        if not actions:
            raise RuntimeError(f"no legal action at anchor {space.anchor(current)}")
        action = min(
            actions,
            key=lambda choice: (
                -len(choice.candidate.node_ids),
                choice.device != preferred_device,
                choice.candidate.pattern_id,
                choice.candidate.node_ids,
                choice.device,
            ),
        )
        current = space.apply(current, action)
        preferred_device = action.device
    return space.to_plan(current, name=name, validate=False)


def _plan_key(plan: Plan) -> tuple:
    return tuple(
        (
            group.extra.get("pattern_id"),
            tuple(group.node_ids),
            group.device,
        )
        for group in plan.groups
    )


def legal_beam_search(
    graph: OpGraph,
    scorer: Callable[[Plan], float],
    n_devs: int,
    beam: int = 8,
    rollouts: int = 3,
    fusion_probability: float = 0.5,
    seed: int = 0,
    registry: Optional[FusionPatternRegistry] = None,
) -> List[Plan]:
    """Beam search over legal candidate covers; lower scorer values are better."""

    if beam <= 0 or rollouts <= 0:
        raise ValueError("beam and rollouts must be positive")
    if not 0.0 <= fusion_probability <= 1.0:
        raise ValueError("fusion_probability must be in [0, 1]")

    rng = random.Random(seed)
    space = LegalFusionSearchSpace(graph, n_devs, registry)
    states = [space.initial_state()]
    completed: Dict[tuple, Tuple[float, Plan]] = {}

    for _ in range(len(space.order)):
        ranked_states: List[Tuple[float, tuple, LegalPartialState]] = []
        for state in states:
            for action in space.actions(state):
                next_state = space.apply(state, action)
                if space.is_terminal(next_state):
                    plan = space.to_plan(next_state, name="legal-beam")
                    score = float(scorer(plan))
                    key = _plan_key(plan)
                    if key not in completed or score < completed[key][0]:
                        completed[key] = (score, plan)
                    continue

                score = 0.0
                for _rollout in range(rollouts):
                    plan = rollout_legal_plan(
                        space,
                        next_state,
                        rng,
                        fusion_probability=fusion_probability,
                        validate=False,
                    )
                    score += float(scorer(plan))
                score /= rollouts
                ranked_states.append((score, next_state.key(), next_state))

        ranked_states.sort(key=lambda item: (item[0], item[1]))
        states = [state for _, _, state in ranked_states[:beam]]
        if not states:
            break

    if not completed:
        # Singleton candidates guarantee a completion; this is a defensive
        # fallback for custom registries or extremely unusual scorer failures.
        plan = rollout_legal_plan(
            space,
            space.initial_state(),
            rng,
            fusion_probability=fusion_probability,
            name="legal-beam-fallback",
            validate=False,
        )
        completed[_plan_key(plan)] = (float(scorer(plan)), plan)
    result = [plan for _, plan in sorted(completed.values(), key=lambda item: item[0])[:beam]]
    for plan in result:
        validate_fusion_plan(graph, plan, n_devs, registry=space.registry)
    return result


@dataclass
class _LegalMCTSNode:
    state: LegalPartialState
    parent: Optional["_LegalMCTSNode"] = None
    children: List["_LegalMCTSNode"] = field(default_factory=list)
    n: int = 0
    q: float = 0.0
    expanded: bool = False
    terminal: bool = False
    prior: float = 1.0

    def update(self, reward: float) -> None:
        self.n += 1
        # This tree is used for deterministic minimisation, not for estimating
        # an environment's expected return.  Back up the best completion seen
        # below the node so selection targets the best reachable plan.
        self.q = reward if self.n == 1 else max(self.q, reward)

    def expand(
        self,
        space: LegalFusionSearchSpace,
        policy_value_evaluator: Optional[Callable] = None,
    ) -> None:
        if self.expanded:
            return
        self.expanded = True
        self.terminal = space.is_terminal(self.state)
        if self.terminal:
            return
        actions = space.actions(self.state)
        if policy_value_evaluator is None:
            priors = [1.0 / len(actions)] * len(actions)
        else:
            _value, raw_priors = policy_value_evaluator(space, self.state, actions)
            priors = [float(prior) for prior in raw_priors]
            if len(priors) != len(actions):
                raise ValueError("policy prior count does not match legal action count")
            if any(not math.isfinite(prior) or prior < 0.0 for prior in priors):
                raise ValueError("policy priors must be finite and non-negative")
            total = sum(priors)
            if total <= 0.0:
                raise ValueError("policy priors must have positive mass")
            priors = [prior / total for prior in priors]
        self.children = [
            _LegalMCTSNode(
                state=space.apply(self.state, action),
                parent=self,
                terminal=False,
                prior=prior,
            )
            for action, prior in zip(actions, priors)
        ]

    def select_child(self, exploration: float) -> Optional["_LegalMCTSNode"]:
        unvisited = [child for child in self.children if child.n == 0]
        if unvisited:
            return max(unvisited, key=lambda child: child.prior)
        if not self.children:
            return None
        log_parent = math.log(self.n + 1.0)
        return max(
            self.children,
            key=lambda child: child.q
            + exploration
            * child.prior
            * math.sqrt(log_parent + 1.0)
            / (1.0 + child.n),
        )


def legal_mcts_search(
    graph: OpGraph,
    scorer: Callable[[Plan], float],
    n_devs: int,
    iterations: int = 500,
    rollouts: int = 3,
    fusion_probability: float = 0.5,
    seed: int = 0,
    exploration: float = 1.5,
    n_best: int = 8,
    registry: Optional[FusionPatternRegistry] = None,
    guided_rollout: bool = True,
    policy_value_evaluator: Optional[Callable] = None,
    value_mix: float = 0.0,
) -> List[Plan]:
    """MCTS over legal fusion candidates and device choices.

    With ``guided_rollout=True``, the first completion at every visited leaf is
    a deterministic fuse-and-colocate baseline; remaining completions are
    stochastic.  The best completion is backed up because search minimises the
    best plan in a subtree rather than the mean random completion cost.
    """

    if iterations <= 0 or rollouts <= 0 or n_best <= 0:
        raise ValueError("iterations, rollouts, and n_best must be positive")
    if not 0.0 <= fusion_probability <= 1.0:
        raise ValueError("fusion_probability must be in [0, 1]")
    if not 0.0 <= value_mix <= 1.0:
        raise ValueError("value_mix must be in [0, 1]")
    if value_mix and policy_value_evaluator is None:
        raise ValueError("value_mix requires policy_value_evaluator")

    rng = random.Random(seed)
    space = LegalFusionSearchSpace(graph, n_devs, registry)
    root = _LegalMCTSNode(space.initial_state())
    best: Dict[tuple, Tuple[float, Plan]] = {}

    def remember(plan: Plan, score: float) -> None:
        key = _plan_key(plan)
        if key not in best or score < best[key][0]:
            best[key] = (score, plan)

    for _ in range(iterations):
        node = root
        path = [node]
        while node.expanded and not node.terminal:
            child = node.select_child(exploration)
            if child is None:
                break
            node = child
            path.append(node)

        node.expand(space, policy_value_evaluator)
        if not node.terminal:
            child = node.select_child(exploration)
            if child is None:
                break
            node = child
            path.append(node)

        reward = 0.0
        if space.is_terminal(node.state):
            plan = space.to_plan(node.state, name="legal-mcts")
            score = float(scorer(plan))
            remember(plan, score)
            reward = -score
        else:
            rollout_scores: List[float] = []
            for rollout_index in range(rollouts):
                if guided_rollout and rollout_index == 0:
                    plan = _guided_rollout_legal_plan(space, node.state)
                else:
                    plan = rollout_legal_plan(
                        space,
                        node.state,
                        rng,
                        fusion_probability=fusion_probability,
                        name="legal-mcts-rollout",
                        validate=False,
                    )
                score = float(scorer(plan))
                remember(plan, score)
                rollout_scores.append(score)
            best_rollout = min(rollout_scores)
            if value_mix:
                leaf_actions = space.actions(node.state)
                predicted_value, _priors = policy_value_evaluator(
                    space, node.state, leaf_actions
                )
                if not math.isfinite(float(predicted_value)):
                    raise ValueError("predicted value must be finite")
                backed_up = (
                    (1.0 - value_mix) * best_rollout
                    + value_mix * float(predicted_value)
                )
            else:
                backed_up = best_rollout
            reward = -backed_up

        for visited in path:
            visited.update(reward)

    if not best:
        plan = rollout_legal_plan(
            space,
            root.state,
            rng,
            fusion_probability=fusion_probability,
            name="legal-mcts-fallback",
            validate=False,
        )
        best[_plan_key(plan)] = (float(scorer(plan)), plan)
    result = [plan for _, plan in sorted(best.values(), key=lambda item: item[0])[:n_best]]
    for plan in result:
        validate_fusion_plan(graph, plan, n_devs, registry=space.registry)
    return result
