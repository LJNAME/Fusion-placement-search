"""Update-aware SROP decision-loss proxy using hypothetical NN refits.

This is not a Bayesian posterior or a Knowledge Gradient guarantee. A sampled
observation changes the nearest-shape mean predictions of every unmeasured
object. Remaining uncertainty is not integrated jointly. Cost normalization,
five quadrature points and the zero-score stop are unchanged for the ablation.
"""

from __future__ import annotations

import copy
import math
import statistics

from .e2_residual import NearestShapePredictor


class NearestRefitResponse:
    """Compile exactly the mean response of adding one public-feature row.

    Standardization and nearest-neighbor identities depend on features, not
    the hypothetical target. Fit them once per queried object and reuse the
    response for all sampled labels. Real observations still do a full refit.
    """

    def __init__(self, public_records, prediction_protocol, query_object_id):
        by_id = {row["object_id"]: row for row in public_records}
        query = by_id[query_object_id]
        if "target" in query or "measurement" in query:
            raise ValueError("hypothetical query must be unmeasured")
        for row in public_records:
            access = row.get("label_access")
            if access is None or (not access["visible"] and ("target" in row or "measurement" in row)):
                raise ValueError("hypothetical refit requires a masked public dataset view")
        visible = [row for row in public_records if "target" in row]
        synthetic = copy.deepcopy(query)
        synthetic["target"] = {"log_residual": 0.0}
        predictor = NearestShapePredictor().fit_mean([*visible, synthetic], prediction_protocol)
        self.query_object_id = query_object_id
        self.query_analytic_s = float(query["analytic"]["estimate_s"])
        self.responses = {}
        for row in public_records:
            key = row["object_id"]
            if "target" in row or key == query_object_id:
                continue
            source = predictor.records[predictor._nearest_index(row)]
            self.responses[key] = (float(row["analytic"]["estimate_s"]),
                                   None if source["object_id"] == query_object_id else float(source["target"]["log_residual"]))
        self.affected_object_ids = tuple(sorted(key for key, (_, fixed) in self.responses.items() if fixed is None))

    def profile(self, baseline, observed_duration_s):
        if observed_duration_s <= 0.0:
            raise ValueError("hypothetical duration must be positive")
        residual = math.log(observed_duration_s / self.query_analytic_s)
        updated = copy.copy(baseline)
        updated.durations_s = dict(baseline.durations_s)
        updated.sources = dict(baseline.sources)
        updated._group_costs = {}
        for key, (analytic_s, fixed) in self.responses.items():
            updated.durations_s[key] = analytic_s * math.exp(residual if fixed is None else fixed)
            updated.sources[key] = "hypothetical_nearest_refit"
        updated.durations_s[self.query_object_id] = float(observed_duration_s)
        updated.sources[self.query_object_id] = "hypothetical_observation"
        return updated


def expected_refit_losses(profile, uncertainty, public_records, remaining,
                         prediction_protocol, active_protocol, ranker, baseline_ranked):
    current_top1 = baseline_ranked[0].plan_id
    values = {}
    batched = active_protocol.get("refit_batch_scenarios", True) and hasattr(ranker, "rank_many")
    scenarios = []
    scenario_keys = []
    for key in remaining:
        response = NearestRefitResponse(public_records, prediction_protocol, key)
        sigma = max(float(uncertainty[key]), 0.0) / float(active_protocol["uncertainty_q90_normalizer"])
        losses = []
        for quantile in active_protocol["srop_proxy_standard_normal_quantiles"]:
            hypothetical_s = profile.durations_s[key] * math.exp(float(quantile) * sigma)
            hypothetical = response.profile(profile, hypothetical_s)
            if batched:
                scenarios.append(hypothetical)
                scenario_keys.append(key)
                continue
            ranked = ranker.rank(hypothetical)
            current_cost = next(item.makespan for item in ranked if item.plan_id == current_top1)
            losses.append(max(0.0, current_cost - ranked[0].makespan) * 1e3)
        if not batched:
            values[key] = statistics.fmean(losses)
    if batched:
        grouped = {key: [] for key in remaining}
        for key, ranked in zip(scenario_keys, ranker.rank_many(scenarios)):
            current_cost = next(item.makespan for item in ranked if item.plan_id == current_top1)
            grouped[key].append(max(0.0, current_cost - ranked[0].makespan) * 1e3)
        values = {key: statistics.fmean(losses) for key, losses in grouped.items()}
    return values
