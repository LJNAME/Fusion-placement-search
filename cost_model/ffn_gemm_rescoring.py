"""Rescore the 48 FFN plans with reviewed GEMM Profiles; Exact vs existing MCTS.

Everything except Projection GEMMs keeps its current source (GPU0 SiLU/Mul
reviewed stages, GPU0 Fused Profile, GPU1 activation proxies, link proxies).
Prediction error is reported before search effects. MCTS logic is unchanged.
"""
import argparse
import json
from pathlib import Path

from cost_model import ffn_profile_search as ffn
from cost_model.exact_search import exact_legal_search
from cost_model.fusion_kernel_profile import key_id
from cost_model.gemm_profile import MeasuredGemmFirstProfile, ReviewedGemmStore
from cost_model.legal_search import legal_mcts_search
from cost_model.local_profile_search import signature
from cost_model.structural import StructuralCostModel

SEEDS = (0, 1, 7)


def profiles(gemm_dir):
    from integration.lightop.silu_mul import adapter_registry
    config = json.loads(ffn.CONFIG.read_text(encoding="utf-8"))
    search, full, hardware, measured, _ = ffn.build_case(config)
    base = ffn.FFNProxyBase(hardware, measured, config, config["scenarios"][0])
    native = ffn.NativeMeasuredFirstProfile(full, base, measured.store, measured.bindings)
    gemm_dir = Path(gemm_dir)
    store = ReviewedGemmStore(gemm_dir / "raw", gemm_dir / "review/evaluation.json")
    review = json.loads((gemm_dir / "review/evaluation.json").read_text(encoding="utf-8"))
    bindings = {(c["case_identity"]["case"]["node"], c["case_identity"]["case"]["device"]): c["case_identity_id"]
                for c in review["identity"]["cases"].values()}
    measured_profile = MeasuredGemmFirstProfile(native, store, bindings)
    return search, full, native, measured_profile, store, review, adapter_registry()


def search_with(search, full, profile, registry):
    model = StructuralCostModel(registry=registry, scheduling_policy=ffn.SCHEDULING_POLICY)
    cache = {}

    def score(plan):
        key = signature(plan)
        if key not in cache:
            cache[key] = model.predict(full, ffn.with_boundaries(plan), profile)
        return cache[key].makespan

    exact = exact_legal_search(search, score, profile.n_devs, top_k=48, max_evaluations=48, registry=registry)
    plans = []
    for plan in exact.plans:
        est = cache[signature(plan)]
        plans.append({"plan_id": key_id(signature(plan)), "fused": any(len(g.node_ids) > 1 for g in plan.groups),
                      "placement": {n: g.device for g in plan.groups for n in g.node_ids},
                      "cross_device": est.cross_device_bytes > 0, "cost_us": est.makespan * 1e6,
                      "sources": sorted({g["source"] for g in est.details["groups"]})})
    mcts = []
    for seed in SEEDS:
        result = legal_mcts_search(search, score, profile.n_devs, registry=registry, iterations=128,
                                   rollouts=2, n_best=4, seed=seed)
        value = score(result[0])
        mcts.append({"seed": seed, "best_plan_id": key_id(signature(result[0])), "cost_us": value * 1e6,
                     "gap_to_exact_us": (value - exact.optimal_score) * 1e6,
                     "matches_exact": key_id(signature(result[0])) == plans[0]["plan_id"]})
    local = next(p for p in plans if not p["cross_device"])
    cross = next(p for p in plans if p["cross_device"])
    return {"best": plans[0], "plans": plans, "mcts": mcts, "search_space": exact.search_space_size,
            "cross_minus_local_us": cross["cost_us"] - local["cost_us"], "best_cross": cross}


def build_report(gemm_dir):
    search, full, native, measured, store, review, registry = profiles(gemm_dir)
    before = search_with(search, full, native, registry)
    after = search_with(search, full, measured, registry)
    rank_before = {p["plan_id"]: i for i, p in enumerate(before["plans"])}
    return {"scope": "48-plan rescoring with reviewed GEMM Profiles; not a measured FFN reference",
        "prediction_error_first": review["prediction_error"],
        "gemm_quality_failed_left_as_proxy": store.failed,
        "before": {k: before[k] for k in ("best", "mcts", "cross_minus_local_us", "best_cross", "search_space")},
        "after": {k: after[k] for k in ("best", "mcts", "cross_minus_local_us", "best_cross", "search_space")},
        "best_changed": before["best"]["plan_id"] != after["best"]["plan_id"],
        "fusion_changed": before["best"]["fused"] != after["best"]["fused"],
        "placement_changed": before["best"]["placement"] != after["best"]["placement"],
        "top10_after_with_rank_before": [{**{k: p[k] for k in ("plan_id", "fused", "placement", "cost_us")},
                                          "rank_before": rank_before[p["plan_id"]]} for p in after["plans"][:10]],
        "plans_after": after["plans"],
        "unchanged_sources": ["GPU0 SiLU/Mul reviewed stages", "GPU0 Fused Profile", "GPU1 SiLU/Mul/Fused proxies",
                              "cross-GPU link proxies (default SDMA unstable)"],
        "best_unchanged_is_not_accuracy_evidence": True, "mcts_logic_changed": False,
        "measured_ffn_reference_plans": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemm", type=Path, required=True, help="data directory with raw/ and review/")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_report(args.gemm)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("best_changed", "fusion_changed", "placement_changed")}))


if __name__ == "__main__":
    main()
