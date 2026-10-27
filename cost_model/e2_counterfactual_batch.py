"""Exact batched versions of the original Impact and target-only Proxy scores."""

import math
import statistics

from .e2_srop_proxy import ObjectCounterfactualProfile


def counterfactual_scores(policy, profile, uncertainty, remaining, active_protocol, ranker, baseline):
    keys, scenarios = [], []
    delta = abs(float(active_protocol["impact_relative_delta"]))
    for key in remaining:
        if policy == "impact_only":
            samples = [ObjectCounterfactualProfile(profile, key, value) for value in (-delta, delta)]
        elif policy == "srop_proxy":
            sigma = max(float(uncertainty[key]), 0.0) / float(active_protocol["uncertainty_q90_normalizer"])
            samples = []
            for z in active_protocol["srop_proxy_standard_normal_quantiles"]:
                multiplier = math.exp(float(z)*sigma)
                relative = multiplier-1.0
                samples.append(ObjectCounterfactualProfile(profile, key, relative,
                    multiplier=multiplier if 1.0+relative <= 0 else None))
        else:
            raise ValueError("only Impact and target-only Proxy use this scorer")
        keys.extend([key]*len(samples))
        scenarios.extend(samples)
    baseline_costs = {row.plan_id: row.makespan for row in baseline}
    losses = {key: [] for key in remaining}
    for key, ranked in zip(keys, ranker.rank_many(scenarios)):
        if policy == "impact_only":
            responses = [row.makespan-baseline_costs[row.plan_id] for row in ranked]
            losses[key].append((max(responses)-min(responses))*1e3)
        else:
            current = next(row.makespan for row in ranked if row.plan_id == baseline[0].plan_id)
            losses[key].append(max(0.0, current-ranked[0].makespan)*1e3)
    if policy == "impact_only":
        return {key: max(values)/delta for key, values in losses.items()}
    return {key: statistics.fmean(values) for key, values in losses.items()}
