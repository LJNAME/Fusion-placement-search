"""Beam and MCTS over graph-local fusion-pattern candidates.

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
    anchor_index: int = 0

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
        require_executable: bool = True,
        allowed_devices: Optional[Sequence[int]] = None,
        tie_shared_params: bool = False,
    ) -> None:
        if n_devs <= 0:
            raise ValueError("n_devs must be positive")
        self.graph = graph
        self.n_devs = int(n_devs)
        if allowed_devices is None:
            self.allowed_devices = tuple(range(self.n_devs))
        else:
            devices = tuple(dict.fromkeys(int(device) for device in allowed_devices))
            if not devices:
                raise ValueError("allowed_devices must not be empty")
            if any(device < 0 or device >= self.n_devs for device in devices):
                raise ValueError(
                    f"allowed_devices must be in [0, {self.n_devs}), got {devices}"
                )
            self.allowed_devices = devices
        self.registry = registry or default_fusion_registry()
        self.require_executable = bool(require_executable)
        self.order = tuple(graph.topological_order())
        self.position = {node_id: index for index, node_id in enumerate(self.order)}
        self.candidates = tuple(
            enumerate_fusion_candidates(
                graph,
                self.registry,
                require_executable=self.require_executable,
            )
        )
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
        # Shared parameters (tied embedding / lm_head) must stay on one device:
        # a split needs a replica and a gradient sync that nothing models. With
        # tie_shared_params the constraint is an action filter, so search never
        # spends budget on such plans. Default off so earlier runs reproduce.
        self.tie_shared_params = bool(tie_shared_params)
        self.tie_group: Dict[str, str] = {}
        if self.tie_shared_params:
            from .training_memory import parameter_id

            members: Dict[str, List[str]] = {}
            for node_id in self.order:
                members.setdefault(parameter_id(graph.nodes[node_id]), []).append(node_id)
            self.tie_group = {
                node_id: pid for pid, ids in members.items() if len(ids) > 1 for node_id in ids
            }

    def bound_devices(self, state: LegalPartialState) -> Dict[str, int]:
        """Shared-parameter ID -> device already chosen for one of its users."""

        if not self.tie_group:
            return {}
        bound: Dict[str, int] = {}
        for choice in state.choices:
            for node_id in choice.candidate.node_ids:
                pid = self.tie_group.get(node_id)
                if pid is not None:
                    bound.setdefault(pid, choice.device)
        return bound

    def _tied_devices(self, candidate: FusionCandidate, bound: Dict[str, int]) -> Optional[set]:
        """Devices forced on a candidate by tied parameters; None = unconstrained."""

        forced = {bound[pid] for pid in (self.tie_group.get(n) for n in candidate.node_ids)
                  if pid is not None and pid in bound}
        return forced or None

    def initial_state(self) -> LegalPartialState:
        return LegalPartialState()

    def is_terminal(self, state: LegalPartialState) -> bool:
        return len(state.assigned) == len(self.order)

    def anchor(self, state: LegalPartialState) -> Optional[str]:
        if state.anchor_index >= len(self.order):
            return None
        return self.order[state.anchor_index]

    def actions(self, state: LegalPartialState) -> List[LegalChoice]:
        anchor = self.anchor(state)
        if anchor is None:
            return []
        result: List[LegalChoice] = []
        bound = self.bound_devices(state)
        for candidate in self.by_node[anchor]:
            if any(node_id in state.assigned for node_id in candidate.node_ids):
                continue
            devices = self.allowed_devices
            if bound:
                forced = self._tied_devices(candidate, bound)
                if forced is not None:
                    if len(forced) > 1:
                        continue  # two tied parameters already on different devices
                    devices = tuple(d for d in devices if d in forced)
            for device in devices:
                result.append(LegalChoice(candidate, device))
        return result

    def apply(self, state: LegalPartialState, choice: LegalChoice) -> LegalPartialState:
        if choice.device not in self.allowed_devices:
            raise ValueError(f"device is not allowed: {choice.device}")
        anchor = self.anchor(state)
        if (
            anchor is None
            or choice.candidate not in self.by_node[anchor]
            or any(node_id in state.assigned for node_id in choice.candidate.node_ids)
        ):
            raise ValueError(
                f"choice is not legal at anchor {anchor}: "
                f"{choice.candidate.pattern_id}{choice.candidate.node_ids}"
            )
        if self.tie_group:
            forced = self._tied_devices(choice.candidate, self.bound_devices(state))
            if forced is not None and forced != {choice.device}:
                raise ValueError(
                    f"shared parameter of {choice.candidate.node_ids} is bound to {sorted(forced)}"
                )
        assigned = state.assigned.union(choice.candidate.node_ids)
        anchor_index = state.anchor_index + 1
        while anchor_index < len(self.order) and self.order[anchor_index] in assigned:
            anchor_index += 1
        return LegalPartialState(
            choices=state.choices + (choice,),
            assigned=assigned,
            anchor_index=anchor_index,
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
            extra={
                "search_space": "legal_fusion_v1",
                "require_executable": self.require_executable,
                "allowed_devices": list(self.allowed_devices),
            },
        )
        if validate:
            validate_fusion_plan(
                self.graph,
                plan,
                self.n_devs,
                registry=self.registry,
                require_executable=self.require_executable,
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

    The completion prefers the largest available candidate at the current
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


def _unfused_guided_rollout_legal_plan(
    space: LegalFusionSearchSpace,
    state: LegalPartialState,
    name: str = "legal-unfused-guided-rollout",
) -> Plan:
    """Deterministic counterpart of the fuse-and-colocate completion.

    It prefers the smallest legal candidate (a singleton whenever one exists)
    and keeps the most recently selected device. Every action comes from
    ``space.actions``, so the completion is legal by construction.
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
                len(choice.candidate.node_ids),
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


def _uniform_singleton_plan(space: LegalFusionSearchSpace, device: int) -> Plan:
    """Construct a reference inside this space's registry/device constraints."""
    state = space.initial_state()
    while not space.is_terminal(state):
        action = next(action for action in space.actions(state)
                      if len(action.candidate.node_ids) == 1 and action.device == device)
        state = space.apply(state, action)
    return space.to_plan(state, name="legal-beam-single-card-reference")


def legal_beam_search(
    graph: OpGraph,
    scorer: Callable[[Plan], float],
    n_devs: int,
    beam: int = 8,
    rollouts: int = 3,
    fusion_probability: float = 0.5,
    seed: int = 0,
    registry: Optional[FusionPatternRegistry] = None,
    require_executable: bool = True,
    allowed_devices: Optional[Sequence[int]] = None,
    evaluator: Optional[Callable[[Plan], object]] = None,
    tie_shared_params: bool = False,
) -> List[Plan]:
    """Beam search with finite single-card singleton references retained.

    For a fixed deterministic scorer, the best result cannot be worse than the
    best finite reference on an allowed device. In a contracted search space,
    singleton macro nodes preserve the caller's frozen fusion partition.

    Hard constraint (opt-in): ``evaluator(plan)`` returns an object with
    ``feasible`` and ``latency`` and replaces ``scorer``. Infeasible plans are
    never kept; a partial state is ranked by the mean latency of its feasible
    rollouts (``inf`` if none). The result may then be empty.
    """

    if evaluator is None and scorer is None:
        raise ValueError("scorer is required without an evaluator")
    if beam <= 0 or rollouts <= 0:
        raise ValueError("beam and rollouts must be positive")
    if not 0.0 <= fusion_probability <= 1.0:
        raise ValueError("fusion_probability must be in [0, 1]")

    rng = random.Random(seed)
    space = LegalFusionSearchSpace(
        graph,
        n_devs,
        registry,
        require_executable=require_executable,
        allowed_devices=allowed_devices,
        tie_shared_params=tie_shared_params,
    )

    def evaluate(plan: Plan) -> Optional[float]:
        if evaluator is None:
            return float(scorer(plan))
        result = evaluator(plan)
        return float(result.latency) if result.feasible else None

    states = [space.initial_state()]
    completed: Dict[tuple, Tuple[float, Plan]] = {}
    # These references do not consume random draws or reserve Beam slots.
    # Keep each allowed card, including the only card in fusion-only mode.
    for device in space.allowed_devices:
        plan = _uniform_singleton_plan(space, device)
        score = evaluate(plan)
        if score is not None and math.isfinite(score):
            completed[_plan_key(plan)] = (score, plan)

    for _ in range(len(space.order)):
        ranked_states: List[Tuple[float, tuple, LegalPartialState]] = []
        for state in states:
            for action in space.actions(state):
                next_state = space.apply(state, action)
                if space.is_terminal(next_state):
                    plan = space.to_plan(next_state, name="legal-beam")
                    score = evaluate(plan)
                    if score is None:
                        continue
                    key = _plan_key(plan)
                    if key not in completed or score < completed[key][0]:
                        completed[key] = (score, plan)
                    continue

                score = 0.0
                feasible_rollouts = 0
                for _rollout in range(rollouts):
                    plan = rollout_legal_plan(
                        space,
                        next_state,
                        rng,
                        fusion_probability=fusion_probability,
                        validate=False,
                    )
                    value = evaluate(plan)
                    if value is not None:
                        score += value
                        feasible_rollouts += 1
                score = score / feasible_rollouts if feasible_rollouts else math.inf
                ranked_states.append((score, next_state.key(), next_state))

        ranked_states.sort(key=lambda item: (item[0], item[1]))
        states = [state for _, _, state in ranked_states[:beam]]
        if not states:
            break

    if not completed and evaluator is not None:
        return []
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
        validate_fusion_plan(
            graph,
            plan,
            n_devs,
            registry=space.registry,
            require_executable=space.require_executable,
        )
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
    exhausted: bool = False  # every leaf below was evaluated (used with skip_exhausted)

    def update(self, reward: float) -> None:
        self.n += 1
        # This tree is used for deterministic minimisation, not for estimating
        # an environment's expected return.  Back up the best completion seen
        # below the node so selection targets the best reachable plan.
        self.q = reward if self.n == 1 else max(self.q, reward)

    def update_infeasible(self) -> None:
        """Count a visit that found no feasible plan; it never raises Q."""

        self.n += 1
        if self.n == 1:
            self.q = -math.inf

    def expand(
        self,
        space: LegalFusionSearchSpace,
        policy_value_evaluator: Optional[Callable] = None,
        prune: Optional[Callable] = None,
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
        if prune is not None:
            # Sound pruning only: prune(space, state) must be True only when
            # every completion of the state is infeasible.
            self.children = [c for c in self.children if not prune(space, c.state)]

    def select_child(self, exploration: float,
                     value: Optional[Callable[[float], float]] = None,
                     skip_exhausted: bool = False) -> Optional["_LegalMCTSNode"]:
        children = [c for c in self.children if not c.exhausted] if skip_exhausted else self.children
        unvisited = [child for child in children if child.n == 0]
        if unvisited:
            return max(unvisited, key=lambda child: child.prior)
        if not children:
            return None
        log_parent = math.log(self.n + 1.0)
        q = value or (lambda x: x)
        return max(
            children,
            key=lambda child: q(child.q)
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
    require_executable: bool = True,
    allowed_devices: Optional[Sequence[int]] = None,
    unfused_guided_rollout: bool = False,
    evaluator: Optional[Callable[[Plan], object]] = None,
    prune: Optional[Callable] = None,
    tie_shared_params: bool = False,
    unfused_incumbent: bool = False,
    normalize_q: bool = False,
    skip_exhausted: bool = False,
) -> List[Plan]:
    """MCTS over legal fusion candidates and device choices.

    With ``guided_rollout=True``, the first completion at every visited leaf is
    a deterministic fuse-and-colocate baseline; remaining completions are
    stochastic.  The best completion is backed up because search minimises the
    best plan in a subtree rather than the mean random completion cost.

    ``unfused_guided_rollout=True`` (default off, so earlier runs reproduce)
    adds one more deterministic completion per leaf: no fusion, colocated. It
    is scored in addition to ``rollouts``, so each non-terminal leaf costs one
    extra scorer call. Selection and backup rules are unchanged.

    Hard constraints (opt-in, default off so earlier runs reproduce):
    ``evaluator(plan)`` returns an object with ``feasible`` and ``latency``
    (``training_memory.Evaluation``) and replaces ``scorer``. Infeasible plans
    are never backed up into Q and never returned. ``prune(space, state)``
    removes children whose every completion is infeasible. With an evaluator
    the result may be empty when no feasible plan was found.

    ``unfused_incumbent=True`` (default off) scores the unfused colocated plan
    once before the first iteration and makes it the initial best candidate.
    It is not a tree node and does not enter selection or backup; a fused plan
    is returned ahead of it only if it scores strictly better. Costs one extra
    evaluation.

    ``normalize_q=True`` (default off) maps Q to [0, 1] before UCT selection,
    using the best and worst feasible rewards seen so far; an infeasible
    subtree (Q = -inf) scores 0 instead of -inf. Raw makespans (~1e-4 s) are
    otherwise swamped by the exploration term, and a subtree whose first
    rollouts were infeasible is never revisited. Meant for hard-constrained
    search (``evaluator``) or penalty-free spaces; finite penalty scores would
    widen the bounds.

    ``skip_exhausted=True`` (default off) marks an evaluated terminal leaf, and
    any node whose children are all marked, as exhausted; selection skips them
    so an iteration is not spent re-reaching an already scored plan. Search
    stops early once the root is exhausted (the whole space was enumerated).
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
    space = LegalFusionSearchSpace(
        graph,
        n_devs,
        registry,
        require_executable=require_executable,
        allowed_devices=allowed_devices,
        tie_shared_params=tie_shared_params,
    )
    if evaluator is None and scorer is None:
        raise ValueError("scorer is required without an evaluator")
    if evaluator is not None and value_mix:
        raise ValueError("value_mix is not defined for hard-constrained search")

    def evaluate(plan: Plan) -> Optional[float]:
        """Latency for feasible plans, None for infeasible ones."""

        if evaluator is None:
            return float(scorer(plan))
        result = evaluator(plan)
        if not result.feasible:
            return None
        return float(result.latency)

    root = _LegalMCTSNode(space.initial_state())
    best: Dict[tuple, Tuple[float, Plan]] = {}
    bounds = [math.inf, -math.inf]  # feasible rewards seen (reward = -score)

    def normalized(q: float) -> float:
        if not math.isfinite(q):
            return 0.0
        lo, hi = bounds
        return 1.0 if hi <= lo else (q - lo) / (hi - lo)

    value = normalized if normalize_q else None
    if prune is not None and prune(space, root.state):
        return []

    def remember(plan: Plan, score: float) -> None:
        key = _plan_key(plan)
        if key not in best or score < best[key][0]:
            best[key] = (score, plan)

    if unfused_incumbent:
        incumbent = _unfused_guided_rollout_legal_plan(space, root.state, name="legal-mcts-unfused-incumbent")
        incumbent_score = evaluate(incumbent)
        if incumbent_score is not None:
            remember(incumbent, incumbent_score)

    for _ in range(iterations):
        node = root
        path = [node]
        while node.expanded and not node.terminal:
            child = node.select_child(exploration, value, skip_exhausted)
            if child is None:
                break
            node = child
            path.append(node)

        node.expand(space, policy_value_evaluator, prune)
        if not node.terminal:
            child = node.select_child(exploration, value, skip_exhausted)
            if child is None:
                if prune is None:
                    break
                # Every child was pruned: this subtree has no feasible plan.
                for visited in path:
                    visited.update_infeasible()
                if node is root:
                    break
                continue
            node = child
            path.append(node)

        reward: Optional[float] = 0.0
        if space.is_terminal(node.state):
            plan = space.to_plan(node.state, name="legal-mcts")
            score = evaluate(plan)
            if score is None:
                reward = None
            else:
                remember(plan, score)
                reward = -score
        elif evaluator is not None:
            feasible_scores: List[float] = []
            for rollout_index in range(rollouts):
                if guided_rollout and rollout_index == 0:
                    plan = _guided_rollout_legal_plan(space, node.state)
                else:
                    plan = rollout_legal_plan(
                        space, node.state, rng, fusion_probability=fusion_probability,
                        name="legal-mcts-rollout", validate=False,
                    )
                score = evaluate(plan)
                if score is not None:
                    remember(plan, score)
                    feasible_scores.append(score)
            if unfused_guided_rollout:
                plan = _unfused_guided_rollout_legal_plan(space, node.state)
                score = evaluate(plan)
                if score is not None:
                    remember(plan, score)
                    feasible_scores.append(score)
            reward = -min(feasible_scores) if feasible_scores else None
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
            if unfused_guided_rollout:
                plan = _unfused_guided_rollout_legal_plan(space, node.state)
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

        if reward is not None and math.isfinite(reward):
            bounds[0], bounds[1] = min(bounds[0], reward), max(bounds[1], reward)
        for visited in path:
            if reward is None:
                visited.update_infeasible()
            else:
                visited.update(reward)
        if skip_exhausted and space.is_terminal(node.state):
            node.exhausted = True
            for parent in reversed(path[:-1]):
                if parent.children and all(c.exhausted for c in parent.children):
                    parent.exhausted = True
                else:
                    break
            if root.exhausted:
                break

    if not best and evaluator is not None:
        return []
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
        validate_fusion_plan(
            graph,
            plan,
            n_devs,
            registry=space.registry,
            require_executable=space.require_executable,
        )
    return result
