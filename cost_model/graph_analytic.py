"""CPU-only execution-stage analysis; no implicit eager-to-Graph conversion.

Method reference: https://github.com/openxla/xla/discussions/10065
This is a limited method adaptation, not an XLA implementation or DCU validation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import itertools
import math
from typing import Mapping


@dataclass(frozen=True)
class Parameter:
    value: float | None
    unit: str
    category: str
    source: str
    scope: str
    status: str = "unresolved"
    needs_calibration: bool = True

    def __post_init__(self):
        if not self.source or not self.unit or not self.scope:
            raise ValueError("Parameter provenance, unit and scope are required")
        if self.value is not None and not math.isfinite(self.value):
            raise ValueError("Parameter must be finite or unknown")
        if (self.value is None) != (self.status == "unresolved"):
            raise ValueError("Unknown values must remain unresolved")


@dataclass(frozen=True)
class Stage:
    name: str
    read_bytes: int
    write_bytes: int
    arithmetic_flops: int
    silu_elements: int


def recipes(shape=(1, 256, 3072), dtype="bf16", layout="contiguous"):
    if dtype != "bf16" or layout != "contiguous":
        raise ValueError("Only explicitly declared contiguous BF16 is supported")
    if len(shape) != 3 or any(type(d) is not int or d <= 0 for d in shape):
        raise ValueError("Expected a positive three-dimensional shape")
    n = math.prod(shape)
    size = 2 * n
    # Nonlinear SiLU service includes sigmoid; ordinary multiply is separate.
    return {
        "native": (Stage("silu", size, size, n, n),
                   Stage("mul", 2 * size, size, n, 0)),
        "fused": (Stage("pack", 2 * size, 2 * size, 0, 0),
                  Stage("lightop", 2 * size, size, 2 * n, n)),
    }


def parameters(memory_bandwidth_Bps: float, source: str) -> dict[str, Parameter]:
    def unknown(unit, category):
        return Parameter(None, unit, category, "not established", "target Graph BF16 implementation")

    return {
        "memory_bandwidth_Bps": Parameter(memory_bandwidth_Bps, "B/s", "hardware",
            source, "historical device measurement; current effective rate unverified",
            "historical_proxy"),
        "bandwidth_utilization": unknown("ratio", "calibration"),
        "vector_flops_per_s": unknown("FLOP/s", "hardware"),
        "silu_elements_per_s": unknown("element/s", "calibration"),
        "compute_utilization": unknown("ratio", "calibration"),
        "lightop_compute_multiplier": unknown("ratio", "calibration"),
        "read_utilization": Parameter(1., "ratio", "analytical", "Recipe logical operands",
            "one logical read; actual rereads remain an implementation assumption", "declared", False),
        "coalescing_efficiency": unknown("ratio", "analytical"),
        "cache_hit_fraction": unknown("ratio", "analytical"),
        "cache_bandwidth_Bps": unknown("B/s", "hardware"),
        "graph_stage_overhead_us": unknown("us/stage", "execution"),
        "graph_call_overhead_us": unknown("us/call", "execution"),
        "fused_extra_overhead_us": unknown("us/call", "execution"),
    }


def assume(base: Mapping[str, Parameter], values: Mapping[str, float]):
    """Explicit scenario assumptions never acquire measured/calibrated status."""
    result = dict(base)
    for name, value in values.items():
        if name not in result:
            raise ValueError("Unknown parameter: " + name)
        result[name] = replace(result[name], value=float(value), status="scenario_assumption",
            source="explicit CPU scenario; not fitted to local_pair", needs_calibration=True)
    validate(result)
    return result


def validate(p):
    fractions = {"bandwidth_utilization", "compute_utilization", "coalescing_efficiency"}
    overheads = {"graph_stage_overhead_us", "graph_call_overhead_us", "fused_extra_overhead_us"}
    for name, item in p.items():
        v = item.value
        if v is None:
            continue
        if name in fractions and not 0 < v <= 1:
            raise ValueError(name + " must be in (0, 1]")
        if name == "cache_hit_fraction" and not 0 <= v <= 1:
            raise ValueError("cache_hit_fraction must be in [0, 1]")
        if name in overheads and v < 0:
            raise ValueError(name + " cannot be negative")
        if name not in fractions | overheads | {"cache_hit_fraction"} and v <= 0:
            raise ValueError(name + " must be positive")


def analyze(p: Mapping[str, Parameter], *, shape=(1, 256, 3072), layout="contiguous"):
    """Unknown terms propagate as null, including totals; no zero imputation."""
    validate(p)
    arms = {}
    for arm, stages in recipes(shape, layout=layout).items():
        missing = set()

        def get(name):
            v = p[name].value
            if v is None:
                missing.add(name)
            return v

        reports = []
        for stage in stages:
            arithmetic = nonlinear = memory = None
            eta = get("compute_utilization") if stage.arithmetic_flops else None
            rate = get("vector_flops_per_s") if stage.arithmetic_flops else None
            if not stage.arithmetic_flops:
                arithmetic = 0.
            elif eta is not None and rate is not None:
                arithmetic = stage.arithmetic_flops / (rate * eta) * 1e6
            if not stage.silu_elements:
                nonlinear = 0.
            else:
                silu = get("silu_elements_per_s")
                if eta is not None and silu is not None:
                    nonlinear = stage.silu_elements / (silu * eta) * 1e6
            if stage.name == "lightop":
                multiplier = get("lightop_compute_multiplier")
                arithmetic = arithmetic / multiplier if arithmetic is not None and multiplier is not None else None
                nonlinear = nonlinear / multiplier if nonlinear is not None and multiplier is not None else None
            bw, use, coal, reads, hit = (get(k) for k in (
                "memory_bandwidth_Bps", "bandwidth_utilization", "coalescing_efficiency",
                "read_utilization", "cache_hit_fraction"))
            cache = get("cache_bandwidth_Bps") if hit is not None and hit > 0 else None
            dram_read = cache_read = write = None
            if all(v is not None for v in (bw, use, coal, reads, hit)):
                dram_read = stage.read_bytes * reads * (1 - hit) / (bw * use * coal) * 1e6
                cache_read = 0. if hit == 0 else (
                    stage.read_bytes * reads * hit / (cache * coal) * 1e6 if cache else None)
                write = stage.write_bytes / (bw * use) * 1e6
                if cache_read is not None:
                    memory = dram_read + cache_read + write
            compute = arithmetic + nonlinear if arithmetic is not None and nonlinear is not None else None
            core = max(compute, memory) if compute is not None and memory is not None else None
            graph = get("graph_stage_overhead_us")
            duration = core + graph if core is not None and graph is not None else None
            reports.append({**asdict(stage), "compute_us": compute, "arithmetic_us": arithmetic,
                "nonlinear_us": nonlinear, "dram_read_us": dram_read, "cache_read_us": cache_read,
                "write_us": write, "memory_us": memory, "core_us": core,
                "graph_stage_overhead_us": graph, "duration_us": duration})
        call = get("graph_call_overhead_us")
        extra = get("fused_extra_overhead_us") if arm == "fused" else 0.
        complete = not missing
        arms[arm] = {"stages": reports, "graph_call_overhead_us": call,
            "fused_extra_overhead_us": extra, "missing_parameters": sorted(missing),
            "duration_us": sum(s["duration_us"] for s in reports) + call + extra if complete else None}
    n, f = (arms[a]["duration_us"] for a in ("native", "fused"))
    delta = f - n if n is not None and f is not None else None
    return {"scope": "cpu_analytical_scenario_not_profile_admission", "shape": list(shape),
        "dtype": "bf16", "layout": layout, "arms": arms,
        "delta_fused_minus_native_us": delta, "choice": choice(delta),
        "parameters": {k: asdict(v) for k, v in sorted(p.items())}}


def choice(delta):
    if delta is None:
        return "unknown"
    if abs(delta) <= 1e-10:
        return "tie"
    return "native" if delta > 0 else "fused"


def sensitivity(base, assumptions, axes, *, complete_fused_us=None):
    """Full factorial + one-factor changes; rates are declared hypotheses.

    complete_fused_us must be obtained through existing strict admission by the
    caller. It replaces the ENTIRE fused duration, not an individual stage.
    """
    if complete_fused_us is not None and (not math.isfinite(complete_fused_us) or complete_fused_us <= 0):
        raise ValueError("Invalid complete-call duration")
    if any(not values for values in axes.values()):
        raise ValueError("Sensitivity axes cannot be empty")
    modes = ["analytic_both"] + (["exact_complete_fused"] if complete_fused_us is not None else [])

    def row(values):
        result = analyze(assume(base, values))
        native = result["arms"]["native"]["duration_us"]
        fused = result["arms"]["fused"]["duration_us"]
        if native is None or fused is None:
            raise ValueError("Sensitivity scenarios must resolve all active parameters")
        scores = {"analytic_both": {"native_us": native, "fused_us": fused,
            "delta_us": fused - native, "choice": choice(fused - native)}}
        if complete_fused_us is not None:
            scores["exact_complete_fused"] = {"native_us": native, "fused_us": complete_fused_us,
                "delta_us": complete_fused_us - native, "choice": choice(complete_fused_us - native)}
        return {"assumptions": dict(sorted(values.items())), "scores": scores}

    baseline = row(assumptions)
    names = sorted(axes)
    rows = [row({**assumptions, **dict(zip(names, values))})
            for values in itertools.product(*(axes[n] for n in names))]
    effects = {}
    for name in names:
        one = [row({**assumptions, name: value}) for value in axes[name]]
        per_mode = {}
        for mode in modes:
            buckets = {}
            for r in rows:
                context = tuple((k, r["assumptions"][k]) for k in names if k != name)
                buckets.setdefault(context, []).append(r)
            spans = [max(v["scores"][mode]["delta_us"] for v in b) -
                     min(v["scores"][mode]["delta_us"] for v in b) for b in buckets.values()]
            switches = [b for b in buckets.values() if
                        {"native", "fused"} <= {v["scores"][mode]["choice"] for v in b}]
            witness = None
            if switches:
                b = switches[0]
                witness = [next(v for v in b if v["scores"][mode]["choice"] == c) for c in ("native", "fused")]
            per_mode[mode] = {"max_delta_span_us": max(spans),
                "contexts_with_sign_flip": len(switches), "sign_flip_witness": witness}
        effects[name] = {"one_factor_rows": one, "modes": per_mode}
    shortlist = {mode: [name for name in names
                        if effects[name]["modes"][mode]["contexts_with_sign_flip"]] for mode in modes}
    return {"scope": "hypothesis_sensitivity_not_measured_ranking", "baseline": baseline,
        "axes": axes, "joint_rows": rows, "effects": effects,
        "conditional_calibration_shortlist": shortlist,
        "shortlist_is_not_a_frozen_gpu_task_list": True,
        "ranges_are_assumptions_not_confidence_intervals": True}
