"""Explain queried legal-action sets without scoring or changing them."""
from collections import Counter, defaultdict


def explain_actions(space, state, actions):
    anchor = space.anchor(state)
    available = [c for c in space.by_node.get(anchor, ()) if not set(c.node_ids) & state.assigned]
    groups = {(c.candidate.pattern_id, c.candidate.node_ids) for c in actions}
    devices = {c.device for c in actions}
    kind = "dead_end" if not actions else "forced" if len(actions) == 1 else (
        "fusion_and_placement" if len(groups) > 1 and len(devices) > 1 else
        "fusion" if len(groups) > 1 else "placement")
    causes = set()
    if len(actions) == 1:
        selected = actions[0].candidate
        placements = space._placement_devices(state, selected)
        if len(space.allowed_devices) == 1:
            causes.add("single_allowed_device")
        relevant = space.trunk_nodes if space.placement_policy == "hybrid" else None
        uses_stage = space.placement_policy == "contiguous_stages" or (
            space.placement_policy == "hybrid" and space.trunk_nodes.intersection(selected.node_ids))
        if uses_stage and len(space._stage_devices(state, relevant)) < len(space.allowed_devices):
            chosen = state.choices if relevant is None else tuple(c for c in state.choices if relevant.intersection(c.candidate.node_ids))
            assigned = state.assigned if relevant is None else state.assigned.intersection(relevant)
            used = {c.device for c in chosen}
            bound = space.bound_devices(state)
            if any(space.position[n] >= state.anchor_index for n in assigned):
                causes.add("unfinished_fusion_span")
            elif any(n not in state.assigned and (relevant is None or n in relevant) and bound.get(pid) in used
                     for n, pid in space.tie_group.items()):
                causes.add("pending_shared_parameter_blocks_stage_exit")
            else:
                causes.add("completed_stage_no_return")
        if space.placement_policy in ("branch_aware", "hybrid"):
            bound = {space.branch_unit[n]: c.device for c in state.choices for n in c.candidate.node_ids if n in space.branch_unit}
            if any(n in space.branch_unit and space.branch_unit[n] in bound for n in selected.node_ids):
                causes.add("branch_unit_device_binding")
        tied = space._tied_devices(selected, space.bound_devices(state))
        if tied and len(set(placements) & tied) < len(placements):
            causes.add("direct_shared_parameter_binding")
        before_prune = 0
        for candidate in available:
            forced = space._tied_devices(candidate, space.bound_devices(state))
            if forced is None or len(forced) == 1:
                before_prune += len(set(space._placement_devices(state, candidate)) & (forced or set(space.allowed_devices)))
        if space.partial_prune is not None and before_prune > len(actions):
            causes.add("partial_feasibility_pruning")
        if len(available) == 1:
            causes.add("one_available_candidate")
        if any(len(c.node_ids) > 1 and set(c.node_ids) & state.assigned for c in space.by_node.get(anchor, ())):
            causes.add("fusion_alternative_already_covered")
    priority = ("partial_feasibility_pruning", "direct_shared_parameter_binding", "branch_unit_device_binding",
                "unfinished_fusion_span", "pending_shared_parameter_blocks_stage_exit", "completed_stage_no_return",
                "single_allowed_device", "one_available_candidate")
    primary = next((c for c in priority if c in causes), "other") if len(actions) == 1 else None
    return {"anchor": anchor, "assigned_nodes": len(state.assigned), "legal_actions": len(actions),
            "available_candidates": len(available), "kind": kind, "primary_forced_cause": primary,
            "causes": sorted(causes)}


class ActionAudit:
    """Aggregate every actions() query; repeated states are intentionally counted.

    A single reference path is also stored separately as exact per-state rows.
    This is not an enumeration of the full graph's exponential state space.
    """
    def __init__(self):
        self.queries = 0
        self.widths, self.kinds, self.causes = Counter(), Counter(), Counter()
        self.by_anchor = defaultdict(Counter)

    def __call__(self, space, state, actions):
        row = explain_actions(space, state, actions)
        self.queries += 1
        self.widths[str(len(actions))] += 1
        self.kinds[row["kind"]] += 1
        if row["primary_forced_cause"]:
            self.causes[row["primary_forced_cause"]] += 1
        self.by_anchor[str(row["anchor"])][str(len(actions))] += 1

    def to_dict(self):
        return {"action_queries": self.queries, "legal_action_count_histogram": dict(self.widths),
                "query_kinds": dict(self.kinds), "forced_primary_causes": dict(self.causes),
                "by_anchor": {n: dict(v) for n, v in sorted(self.by_anchor.items())},
                "counting": "all action queries, including repeat queries of the same state; not unique states"}
