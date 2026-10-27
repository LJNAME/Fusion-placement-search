"""Comparable budget curves with stopping, seed averaging and harm retained."""

from __future__ import annotations

import statistics

from .e2_diagnostics import REGRET_TOLERANCE_MS


BUDGETS = (0, 1, 2, 4, 8, 16)
METRICS = ("top1_regret_ms", "normalized_top1_regret", "strict_top_k_recall",
           "tie_aware_top_k_recall", "best_of_k_regret_ms", "profiling_wall_time_s",
           "predictor_fit_wall_time_s", "selection_wall_time_s", "ranking_wall_time_s",
           "algorithm_wall_time_s")


def prefix_point(trajectory, budget, *, stopping):
    limit = len(trajectory) - 1
    if stopping:
        limit = next((index for index, point in enumerate(trajectory) if point["stop_recommended"]), limit)
    return trajectory[min(int(budget), limit)]


def compact_task(task, result):
    policies = {}
    for name, policy in result["policies"].items():
        trajectories = policy["full_trajectories"]
        modes = {}
        for mode in ("forced", "stopping"):
            curves = []
            for budget in BUDGETS:
                points = [prefix_point(trajectory, budget, stopping=mode == "stopping") for trajectory in trajectories]
                curves.append({"requested_budget": budget,
                    "measurements_mean": statistics.fmean(point["budget"] for point in points),
                    **{key: statistics.fmean(point[key] for point in points) for key in METRICS}})
            # Integrate every integer prefix on the common 0..16 horizon.
            # Pool exhaustion and stopping hold the last decision constant.
            aucs = []
            for trajectory in trajectories:
                values = [prefix_point(trajectory, budget, stopping=mode == "stopping")["top1_regret_ms"]
                          for budget in range(17)]
                aucs.append(sum((left + right) * 0.5 for left, right in zip(values, values[1:])) / 16.0)
            modes[mode] = {"curve": curves, "regret_budget_auc_ms": statistics.fmean(aucs)}
        traces = []
        for trajectory in trajectories:
            stopped = prefix_point(trajectory, len(trajectory), stopping=True)
            traces.append({
                "random_seed": trajectory[0]["random_seed"],
                "selected_object_ids": trajectory[-1]["selected_object_ids"],
                "scores_before_measurement": [point["next_selection_score"] for point in trajectory[:-1]],
                "stop_budget": stopped["budget"], "stop_regret_ms": stopped["top1_regret_ms"],
                "stop_reason": "nonpositive_score" if stopped["stop_recommended"] else "pool_exhausted",
                "full_budget_regret_ms": trajectory[-1]["top1_regret_ms"],
                "full_budget_strict_top_k_recall": trajectory[-1]["strict_top_k_recall"],
                "max_regret_ms": max(point["top1_regret_ms"] for point in trajectory),
            })
        policies[name] = {"runs": len(trajectories), "modes": modes, "traces": traces}
    return {
        "task_id": task["task_id"], "split_id": task["split_id"], "model_id": task["model_id"],
        "scale": task["scale"], "initial_train_count": task["initial_train_count"], "subset_seed": task["subset_seed"],
        "initial_regret_ms": result["initial_top1_regret_ms"],
        "measurement_pool_objects": result["measurement_pool_objects"],
        "candidate_plans": result["candidate_plans"], "policies": policies,
        "structure_preparation_wall_time_s": result["structure_preparation_wall_time_s"],
        "evaluation_wall_time_s": result["evaluation_wall_time_s"],
    }


def summarize_tasks(tasks):
    if not tasks:
        return {}
    summaries = {}
    for mode in ("forced", "stopping"):
        policies = {}
        for policy in tasks[0]["policies"]:
            aucs = [task["policies"][policy]["modes"][mode]["regret_budget_auc_ms"] for task in tasks]
            curve = []
            for index, budget in enumerate(BUDGETS):
                points = [task["policies"][policy]["modes"][mode]["curve"][index] for task in tasks]
                curve.append({"requested_budget": budget,
                    **{key: statistics.fmean(point[key] for point in points) for key in ("measurements_mean", *METRICS)},
                    "harmed_configs": sum(point["top1_regret_ms"] > task["initial_regret_ms"] + REGRET_TOLERANCE_MS
                                            for task, point in zip(tasks, points)),
                    "nonzero_regret_configs": sum(point["top1_regret_ms"] > REGRET_TOLERANCE_MS for point in points)})
            policies[policy] = {"mean_regret_budget_auc_ms": statistics.fmean(aucs), "curve": curve}
        comparisons = {}
        for baseline in ("random", "uncertainty_only", "impact_only"):
            deltas = [task["policies"]["srop_proxy"]["modes"][mode]["regret_budget_auc_ms"] -
                      task["policies"][baseline]["modes"][mode]["regret_budget_auc_ms"] for task in tasks]
            comparisons[baseline] = {
                "srop_lower_auc_configs": sum(delta < -REGRET_TOLERANCE_MS for delta in deltas),
                "srop_higher_auc_configs": sum(delta > REGRET_TOLERANCE_MS for delta in deltas),
                "tied_configs": sum(abs(delta) <= REGRET_TOLERANCE_MS for delta in deltas),
                "mean_srop_minus_baseline_auc_ms": statistics.fmean(deltas),
            }
        summaries[mode] = {"policies": policies, "srop_auc_comparisons": comparisons}
    return {"configs": len(tasks), "positive_initial_regret_configs": sum(task["initial_regret_ms"] > REGRET_TOLERANCE_MS for task in tasks),
            "unique_model_ids": len({task["model_id"] for task in tasks}), "modes": summaries}


def grouped_summaries(tasks):
    return {
        "all_configs": summarize_tasks(tasks),
        "positive_initial_regret": summarize_tasks([task for task in tasks if task["initial_regret_ms"] > REGRET_TOLERANCE_MS]),
        "zero_initial_regret": summarize_tasks([task for task in tasks if task["initial_regret_ms"] <= REGRET_TOLERANCE_MS]),
        "by_model": {model: summarize_tasks([task for task in tasks if task["model_id"] == model])
                     for model in sorted({task["model_id"] for task in tasks})},
    }
