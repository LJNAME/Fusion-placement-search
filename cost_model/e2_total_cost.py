"""Decision availability on a serial measured-CPU plus recorded-profiling clock."""

import statistics


HORIZONS_S = (1.0, 2.0, 4.0, 8.0, 16.0)
PRIMARY_HORIZON_S = 4.0


def stopped_prefix(trace, *, stopping):
    if not stopping:
        return trace
    end = next((index for index, row in enumerate(trace) if row["stop_recommended"]), len(trace)-1)
    return trace[:end+1]


def decision_timeline(trace, preparation_s, reference_regret_ms):
    events = [{"time_s": 0.0, "regret_ms": reference_regret_ms, "published_measurements": 0}]
    observations = []
    for index, point in enumerate(trace):
        ready_cpu = point["decision_ready_algorithm_wall_time_s"]
        if ready_cpu > point["algorithm_wall_time_s"] + 1e-12:
            raise ValueError("a decision cannot be published after its next acquisition finishes")
        if index:
            previous = trace[index-1]
            complete = preparation_s + previous["algorithm_wall_time_s"] + point["profiling_wall_time_s"]
            observations.append(complete)
            if ready_cpu < previous["algorithm_wall_time_s"]:
                raise ValueError("a refit must follow the previous acquisition")
        ready = preparation_s + point["profiling_wall_time_s"] + ready_cpu
        if ready < events[-1]["time_s"]:
            raise ValueError("published decisions must be time ordered")
        events.append({"time_s": ready, "regret_ms": point["top1_regret_ms"],
                       "published_measurements": point["budget"]})
    last = trace[-1]
    return {"events": events, "observation_completion_times_s": observations,
        "completion_s": preparation_s+last["profiling_wall_time_s"]+last["algorithm_wall_time_s"],
        "preparation_s": preparation_s, "cpu_s": last["algorithm_wall_time_s"],
        "profiling_s": last["profiling_wall_time_s"], "measurements": last["budget"],
        "final_regret_ms": last["top1_regret_ms"]}


def initial_only_timeline(point, preparation_s, reference_regret_ms):
    ready = preparation_s+point["algorithm_wall_time_s"]
    return {"events": [{"time_s": 0.0, "regret_ms": reference_regret_ms, "published_measurements": 0},
                       {"time_s": ready, "regret_ms": point["top1_regret_ms"], "published_measurements": 0}],
        "observation_completion_times_s": [], "completion_s": ready,
        "preparation_s": preparation_s, "cpu_s": point["algorithm_wall_time_s"], "profiling_s": 0.0,
        "measurements": 0, "final_regret_ms": point["top1_regret_ms"]}


def reference_timeline(regret_ms):
    return {"events": [{"time_s": 0.0, "regret_ms": regret_ms, "published_measurements": 0}],
        "observation_completion_times_s": [], "completion_s": 0.0, "preparation_s": 0.0,
        "cpu_s": 0.0, "profiling_s": 0.0, "measurements": 0, "final_regret_ms": regret_ms}


def cost_summary(timeline):
    events = timeline["events"]
    curve = []
    for horizon in HORIZONS_S:
        area, previous_time, regret = 0.0, 0.0, events[0]["regret_ms"]
        latest = events[0]
        for event in events[1:]:
            if event["time_s"] > horizon:
                break
            area += (event["time_s"]-previous_time)*regret
            previous_time, regret, latest = event["time_s"], event["regret_ms"], event
        area += (horizon-previous_time)*regret
        curve.append({"budget_s": horizon, "step_auc_ms": area/horizon, "regret_ms": regret,
            "published_measurements": latest["published_measurements"],
            "completed_measurements": sum(t <= horizon for t in timeline["observation_completion_times_s"])})
    stable = next((event["time_s"] for index, event in enumerate(events)
                   if all(row["regret_ms"] <= 1e-9 for row in events[index:])), None)
    return {**{key: value for key, value in timeline.items() if key not in {"events", "observation_completion_times_s"}},
        "curve": curve, "stable_zero_time_s": stable}


def mean_summaries(items):
    if not items:
        return {}
    keys = ("completion_s", "preparation_s", "cpu_s", "profiling_s", "measurements", "final_regret_ms")
    return {**{key: statistics.fmean(item[key] for item in items) for key in keys},
        "nonzero_final_fraction": statistics.fmean(item.get("nonzero_final_fraction", item["final_regret_ms"] > 1e-9) for item in items),
        "curve": [{"budget_s": horizon, **{key: statistics.fmean(item["curve"][index][key] for item in items)
                  for key in ("step_auc_ms", "regret_ms", "published_measurements", "completed_measurements")}}
                  for index, horizon in enumerate(HORIZONS_S)]}


def task_summary(runs, reference_regret_ms):
    initial_fees = {run["repeat"]: run["trace"][0]["decision_ready_algorithm_wall_time_s"]
                   for run in runs if run["policy"] == "initial_only"}
    modes = {}
    for mode in ("forced", "stopping"):
        policies = {}
        for policy in sorted({run["policy"] for run in runs}):
            summaries = []
            for run in runs:
                if run["policy"] != policy:
                    continue
                trace = run["trace"]
                if run.get("repeat") in initial_fees:
                    own_initial = trace[0]["decision_ready_algorithm_wall_time_s"]
                    common_initial = initial_fees[run["repeat"]]
                    trace = [{**point,
                        "decision_ready_algorithm_wall_time_s": common_initial+(point["decision_ready_algorithm_wall_time_s"]-own_initial),
                        "algorithm_wall_time_s": common_initial+(point["algorithm_wall_time_s"]-own_initial)} for point in trace]
                if run["policy"] == "initial_only":
                    point = {**trace[0], "algorithm_wall_time_s": trace[0]["decision_ready_algorithm_wall_time_s"]}
                    timeline = initial_only_timeline(point, run["preparation_s"], reference_regret_ms)
                else:
                    trace = stopped_prefix(trace, stopping=mode == "stopping")
                    timeline = decision_timeline(trace, run["preparation_s"], reference_regret_ms)
                summaries.append(cost_summary(timeline))
            # Average seeds/repeats within each task before averaging tasks.
            policies[policy] = mean_summaries(summaries)
        policies["reference_only"] = mean_summaries([cost_summary(reference_timeline(reference_regret_ms))])
        modes[mode] = policies
    return modes


def summarize_tasks(tasks):
    if not tasks:
        return {}
    result = {"configs": len(tasks)}
    for mode in ("forced", "stopping"):
        result[mode] = {}
        for policy in tasks[0]["modes"][mode]:
            points = [task["modes"][mode][policy] for task in tasks]
            result[mode][policy] = mean_summaries(points)
    comparisons = {}
    index = HORIZONS_S.index(PRIMARY_HORIZON_S)
    for mode in ("forced", "stopping"):
        comparisons[mode] = {}
        for baseline in ("random", "uncertainty_only", "impact_only", "srop_proxy", "initial_only", "reference_only"):
            delta = [task["modes"][mode]["srop_refit"]["curve"][index]["step_auc_ms"] -
                     task["modes"][mode][baseline]["curve"][index]["step_auc_ms"] for task in tasks]
            comparisons[mode][baseline] = {"lower_auc_configs": sum(x < -1e-9 for x in delta),
                "higher_auc_configs": sum(x > 1e-9 for x in delta), "tied_configs": sum(abs(x) <= 1e-9 for x in delta),
                "mean_auc_delta_ms": statistics.fmean(delta)}
    result["refit_vs_baselines_primary_4s"] = comparisons
    return result
