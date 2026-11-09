"""P2: legal MCTS with the training-memory hard constraint (CPU only).

Every candidate is checked per device against
``utilization * max_mem`` (default 95%) using the LLMem-based training
estimator (M_base + training state + activation lifetime peak). Infeasible
plans never enter MCTS Q and are never returned; partial states whose placed
parameters already overflow are pruned.

Fused groups are only admitted when their backward is listed in
``--verified-backward`` (pattern=save|recompute); by default none is, so
search covers placement of unfused ops.

Example:
    python -m cost_model.memory_constrained_search --seq-length 768 \
        --output data/memory_constrained_search_v1/report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Optional, Sequence

from simulator import FusionGroup, OpGraph, Plan
from simulator.dataset import plan_to_dict
from simulator.hardware import load_hardware
from simulator.model_graphs import build_model_graph

from .backend_semantics import get_semantics
from .hardware_profile import AnalyticHardwareProfile
from .legal_search import legal_mcts_search
from .measured_profile import MeasuredHardwareProfile
from .search_evaluation import qwen25_3b_spec
from .structural import StructuralCostModel
from .training_memory import (
    DEFAULT_MEMORY_UTILIZATION,
    Evaluation,
    MemoryBaseline,
    TrainingLifetimeConfig,
    make_static_memory_prune,
    make_training_evaluator,
    memory_budget_bytes,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HARDWARE = ROOT / "data" / "bw1000_2card" / "bw1000_2card.calibrated.json"
DEFAULT_RAW = ROOT / "data" / "bw1000_2card" / "raw_measurements.json"
DEFAULT_M_BASE = ROOT / "data" / "training_memory_p1d_v2" / "m_base.json"
DEFAULT_SEMANTICS = "hf_qwen2_eager_saved_v1"


def _evaluation_summary(ev: Evaluation, budget: Sequence[float]) -> dict:
    peaks = [ev.peak_per_device[d] for d in range(len(budget))]
    return {
        "feasible": ev.feasible,
        "reason": ev.reason,
        "latency_s": ev.latency,
        "peak_bytes": peaks,
        "budget_bytes": list(budget),
        "headroom_bytes": [b - p for b, p in zip(budget, peaks)],
        "overflow_devices": list(ev.overflow_devices),
        "breakdown": ev.breakdown,
    }


def _single_device_plan(graph: OpGraph, device: int = 0) -> Plan:
    return Plan(name=f"unfused-dev{device}",
                groups=[FusionGroup([n], device, {"pattern_id": "unfused"})
                        for n in graph.topological_order()])


def run_memory_constrained_search(
    graph: OpGraph,
    profile,
    *,
    utilization: float = DEFAULT_MEMORY_UTILIZATION,
    baseline: Optional[MemoryBaseline] = None,
    lifetime_config: TrainingLifetimeConfig = TrainingLifetimeConfig(),
    verified_backward_patterns: Mapping[str, str] | Sequence[str] = (),
    iterations: int = 64,
    rollouts: int = 3,
    seed: int = 23,
    n_best: int = 4,
    prune: bool = True,
    require_executable: bool = True,
    lifetime_model=None,
) -> dict:
    """Hard-constrained MCTS; returns plans plus a JSON-ready report."""

    budget = memory_budget_bytes(profile, utilization)
    baseline = baseline or MemoryBaseline.uncalibrated(len(budget))
    evaluator = make_training_evaluator(
        graph, profile, budget,
        baseline=baseline,
        lifetime_model=lifetime_model or StructuralCostModel(),
        lifetime_config=lifetime_config,
        verified_backward_patterns=verified_backward_patterns,
    )
    prune_fn = (make_static_memory_prune(graph, budget, baseline=baseline,
                                         verified_backward_patterns=verified_backward_patterns)
                if prune else None)
    started = time.perf_counter()
    plans = legal_mcts_search(
        graph, None, len(budget), iterations=iterations, rollouts=rollouts,
        fusion_probability=0.7, seed=seed, n_best=n_best,
        require_executable=require_executable, evaluator=evaluator, prune=prune_fn,
    )
    elapsed = time.perf_counter() - started
    evaluated = list(evaluator.cache.values())
    reference = _single_device_plan(graph, 0)
    report = {
        "scope": "memory_constrained_mcts_v1",
        "utilization": float(utilization),
        "budget_bytes": list(budget),
        "max_mem_bytes": [float(d.max_mem) for d in getattr(profile, "hardware", profile).devices],
        "m_base": baseline.to_dict(),
        "activation_semantics": lifetime_config.activation_semantics,
        "verified_backward_patterns": (dict(verified_backward_patterns)
                                       if isinstance(verified_backward_patterns, Mapping)
                                       else list(verified_backward_patterns)),
        "search": {"iterations": iterations, "rollouts": rollouts, "seed": seed,
                   "n_best": n_best, "prune": prune, "elapsed_s": elapsed},
        "evaluated_plans": len(evaluated),
        "feasible_evaluated_plans": sum(e.feasible for e in evaluated),
        "infeasible_reasons": _count(e.reason for e in evaluated if not e.feasible),
        "reason": None if plans else "evaluated_no_feasible",
        "reference_all_dev0": _evaluation_summary(evaluator(reference), budget),
        "candidates": [],
    }
    for rank, plan in enumerate(plans, start=1):
        ev = evaluator(plan)
        if not ev.feasible:  # legal_mcts_search must never return these
            raise AssertionError("hard-constrained search returned an infeasible plan")
        nodes_per_device = [0] * len(budget)
        for group in plan.groups:
            nodes_per_device[group.device] += len(group.node_ids)
        report["candidates"].append({
            "rank": rank,
            **_evaluation_summary(ev, budget),
            "nodes_per_device": nodes_per_device,
            "fused_groups": sum(len(g.node_ids) > 1 for g in plan.groups),
            "plan": plan_to_dict(plan),
        })
    return {"plans": plans, "report": report}


def _count(items) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        for part in str(item).split(";"):
            key = part.split(":")[0]
            out[key] = out.get(key, 0) + 1
    return out


def _parse_verified(items: Sequence[str]) -> dict[str, str]:
    out = {}
    for item in items:
        pattern, _, policy = item.partition("=")
        out[pattern] = policy or "save"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hardware", type=Path, default=DEFAULT_HARDWARE)
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--profile", choices=("measured", "analytic"), default="measured",
                    help="latency profile; memory feasibility does not depend on it")
    ap.add_argument("--m-base", type=Path, default=DEFAULT_M_BASE)
    ap.add_argument("--semantics", default=DEFAULT_SEMANTICS)
    ap.add_argument("--utilization", type=float, default=DEFAULT_MEMORY_UTILIZATION,
                    help="memory budget as a fraction of max_mem (default 0.95)")
    ap.add_argument("--seq-length", type=int, default=768)
    ap.add_argument("--micro-batch-size", type=int, default=1)
    ap.add_argument("--num-layers", type=int, default=None)
    ap.add_argument("--iterations", type=int, default=64)
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--no-prune", action="store_true")
    ap.add_argument("--verified-backward", nargs="*", default=[],
                    help="fused patterns admitted for training, as pattern=save|recompute")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    spec = replace(qwen25_3b_spec(), seq_length=args.seq_length,
                   micro_batch_size=args.micro_batch_size)
    if args.num_layers:
        spec = replace(spec, num_layers=args.num_layers)
    graph = build_model_graph(spec)
    if args.profile == "analytic":
        profile = AnalyticHardwareProfile(load_hardware(args.hardware))
    else:
        profile = MeasuredHardwareProfile.from_files(args.hardware, args.raw)
    baseline = MemoryBaseline.from_dict(json.loads(args.m_base.read_text(encoding="utf-8")))
    result = run_memory_constrained_search(
        graph, profile,
        utilization=args.utilization, baseline=baseline,
        lifetime_config=TrainingLifetimeConfig(semantics=get_semantics(args.semantics)),
        verified_backward_patterns=_parse_verified(args.verified_backward),
        iterations=args.iterations, rollouts=args.rollouts, seed=args.seed,
        n_best=args.top_k, prune=not args.no_prune,
    )
    report = result["report"]
    report["latency_profile"] = args.profile
    report["spec"] = {"model": spec.model_type, "seq_length": spec.seq_length,
                      "micro_batch_size": spec.micro_batch_size, "num_layers": spec.num_layers,
                      "nodes": len(graph.nodes)}
    report["inputs"] = {name: {"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                        for name, p in (("hardware", args.hardware), ("raw", args.raw), ("m_base", args.m_base))
                        if name != "raw" or args.profile == "measured"}
    report["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    gb = 1e9
    ref = report["reference_all_dev0"]
    print(f"budget {args.utilization:.0%}: {[round(b / gb, 3) for b in report['budget_bytes']]} GB")
    print(f"all-dev0 reference: feasible={ref['feasible']} reason={ref['reason']} "
          f"peak={[round(p / gb, 3) for p in ref['peak_bytes']]} GB")
    print(f"evaluated={report['evaluated_plans']} feasible={report['feasible_evaluated_plans']} "
          f"infeasible={report['infeasible_reasons']} elapsed={report['search']['elapsed_s']:.1f}s")
    for c in report["candidates"]:
        print(f"  #{c['rank']} latency={c['latency_s'] * 1e3:.3f} ms "
              f"peak={[round(p / gb, 3) for p in c['peak_bytes']]} GB nodes/dev={c['nodes_per_device']}")
    if not report["candidates"]:
        print("  no feasible plan found")
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
