"""48-plan Exact/MCTS with all reviewed compute Profiles: GPU0 stages + Fused,
P2 GEMMs, and the GPU1 completion batch. Communication stays proxy (default
SDMA unstable). Down GPU0 from the GPU1 batch is a reference, not a Profile.
"""
import argparse
import json
from pathlib import Path

from cost_model.ffn_gemm_rescoring import profiles as gemm_profiles, search_with
from cost_model.gpu1_profile import MeasuredGpu1Profile, ReviewedGpu1Store

REFERENCE_ONLY = ("gemm/down/gpu0",)


def build_report(gemm_dir, gpu1_dir):
    search, full, native, gemm, gemm_store, gemm_review, registry = gemm_profiles(gemm_dir)
    gpu1_dir = Path(gpu1_dir)
    store = ReviewedGpu1Store(gpu1_dir / "raw", gpu1_dir / "review/evaluation.json", excluded_cases=REFERENCE_ONLY)
    review = json.loads((gpu1_dir / "review/evaluation.json").read_text(encoding="utf-8"))
    profile = MeasuredGpu1Profile(gemm, store)
    steps = {"proxy_compute_except_gpu0_activation": search_with(search, full, native, registry),
             "plus_p2_gemm": search_with(search, full, gemm, registry),
             "plus_gpu1_completion": search_with(search, full, profile, registry)}
    final = steps["plus_gpu1_completion"]
    first = steps["proxy_compute_except_gpu0_activation"]
    ranks = {p["plan_id"]: i for i, p in enumerate(first["plans"])}
    coverage = {}
    for p in final["plans"]:
        analytic = [s for s in p["sources"] if s.startswith("analytic_")]
        coverage[p["plan_id"]] = analytic
    return {"scope": "48-plan rescoring with every reviewed compute Profile; communication proxy",
        "proxy_error_first": {"gemm": gemm_review["prediction_error"], "gpu1": review["proxy_error"]},
        "gpu1_quality_failed_left_as_proxy": store.failed, "reference_only_cases": list(REFERENCE_ONLY),
        "steps": {name: {k: s[k] for k in ("best", "mcts", "cross_minus_local_us", "best_cross", "search_space")}
                  for name, s in steps.items()},
        "best_changed_vs_proxy": final["best"]["plan_id"] != first["best"]["plan_id"],
        "fusion_changed_vs_proxy": final["best"]["fused"] != first["best"]["fused"],
        "placement_changed_vs_proxy": final["best"]["placement"] != first["best"]["placement"],
        "top10_final_with_rank_before": [{**{k: p[k] for k in ("plan_id", "fused", "placement", "cost_us", "cross_device")},
                                          "rank_before": ranks[p["plan_id"]]} for p in final["plans"][:10]],
        "compute_coverage": {"plans_with_all_compute_measured": sum(not v for v in coverage.values()),
                             "remaining_analytic_sources": sorted({s for v in coverage.values() for s in v})},
        "communication": "analytic_ffn_link_proxy; default SDMA unstable (data/ffn_comm_instability_v1)",
        "plans_final": final["plans"], "best_unchanged_is_not_accuracy_evidence": True,
        "mcts_logic_changed": False, "measured_ffn_reference_plans": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemm", type=Path, required=True)
    parser.add_argument("--gpu1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_report(args.gemm, args.gpu1)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("best_changed_vs_proxy", "fusion_changed_vs_proxy",
                                              "placement_changed_vs_proxy", "compute_coverage")}))


if __name__ == "__main__":
    main()
