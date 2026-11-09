"""Sound resident-weight lower bounds for the current forward memory model.

No timing prediction is performed. These bounds do not claim training or
physical allocator feasibility; activations/transfers can still exceed them.
"""
from __future__ import annotations

from collections import defaultdict


def replay_plan(space, plan):
    """Yield the actual legal action path, rejecting out-of-domain plans."""
    target = {n: group for group in plan.groups for n in group.node_ids}
    state = space.initial_state()
    yield state
    while not space.is_terminal(state):
        node = space.anchor(state)
        group = target[node]
        action = next((a for a in space.actions(state)
                       if a.device == group.device
                       and set(a.candidate.node_ids) == set(group.node_ids)
                       and a.candidate.pattern_id == group.extra["pattern_id"]), None)
        if action is None:
            raise ValueError(f"plan outside action space at {node}")
        state = space.apply(state, action)
        yield state


class ForwardResidentBound:
    """Prune only a proved overflow of resident weights plus tied terminal output.

    Hybrid special case: if the first/last trunk nodes share a parameter, the
    existing no-return rule forces every trunk node to the first node's card.
    A branch placement binds every node of its branch, including future weights.
    No such forced trunk/branch inference is used for the original node domain.
    """

    def __init__(self, capacity):
        self.capacity = tuple(float(x) for x in capacity)
        self.calls = self.blocked = 0
        self.first_blocked = None
        self._spaces = {}

    def _metadata(self, space):
        if id(space) not in self._spaces:
            trunk_order = [n for n in space.order if n in space.trunk_nodes]
            first, last = trunk_order[0], trunk_order[-1]
            tied_tail = (space.tie_shared_params and first in space.tie_group
                         and space.tie_group.get(first) == space.tie_group.get(last)
                         and not space.graph.successors(last)
                         and all(len(c.node_ids) == 1 for c in space.by_node[last]))
            unit_params = defaultdict(float)
            for n, unit in space.branch_unit.items():
                unit_params[unit] += float(space.graph.nodes[n].param_bytes)
            self._spaces[id(space)] = (space, first, last, tied_tail, dict(unit_params),
                sum(float(space.graph.nodes[n].param_bytes) for n in space.trunk_nodes))
        return self._spaces[id(space)]

    def explain(self, space, state):
        _, first, last, tied_tail, unit_params, trunk_params = self._metadata(space)
        assigned = [0.] * space.n_devs
        units, owner = {}, None
        for choice in state.choices:
            for n in choice.candidate.node_ids:
                assigned[choice.device] += float(space.graph.nodes[n].param_bytes)
                if n == first:
                    owner = choice.device
                if n in space.branch_unit:
                    units[space.branch_unit[n]] = choice.device
        forced = list(assigned)
        pinned = space.placement_policy == "hybrid" and tied_tail and owner is not None
        if pinned:
            forced = [0.] * space.n_devs
            forced[owner] = trunk_params
            for unit, device in units.items():
                forced[device] += unit_params[unit]
        lower = list(forced)
        if tied_tail and owner is not None:
            lower[owner] += float(space.graph.nodes[last].out_bytes)
        overflow = [d for d, (v, cap) in enumerate(zip(lower, self.capacity)) if v > cap]
        return {"peak_lower_bound_bytes": lower, "forced_parameter_bytes": forced,
                "overflow_devices": overflow, "trunk_pinned": pinned, "owner": owner,
                "branch_units_bound": len(units),
                "branch_units_on_owner": sum(d == owner for d in units.values()) if pinned else None,
                "assigned_nodes": len(state.assigned), "depth": len(state.choices)}

    def __call__(self, space, state):
        self.calls += 1
        detail = self.explain(space, state)
        if detail["overflow_devices"]:
            self.blocked += 1
            if self.first_blocked is None:
                self.first_blocked = detail
            return True
        return False

    def summary(self):
        return {"calls": self.calls, "blocked_action_results": self.blocked,
                "first_blocked": self.first_blocked,
                "scope": "sound lower bound, not sufficient for full memory feasibility"}
