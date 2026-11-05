"""Generate a deterministic CPU stage-cost and conditional calibration audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cost_model.graph_analytic import analyze, assume, parameters, sensitivity

ROOT = Path(__file__).resolve().parents[1]
HARDWARE = ROOT / "data/bw1000_2card/bw1000_2card.calibrated.json"
SCENARIOS = ROOT / "configs/graph_analytic_scenarios.json"
RAW = ROOT / "data/lightop_profile_calibration_20261104_01/raw"


def reviewed_complete_fused(directory=RAW):
    # Import the existing CPU reviewer, not a GPU collector invocation.
    from calibration.lightop_profile_calibration import review_receipt
    decision, profile = review_receipt(Path(directory))
    if profile is None or decision["status"] != "accepted":
        raise ValueError("Complete-call Profile did not pass strict review")
    return {"duration_us": decision["cost_query"]["duration_s"] * 1e6,
        "source": decision["cost_query"]["source"],
        "kernel_key_id": decision["kernel_key_id"],
        "measurement_contract_id": decision["measurement_contract_id"],
        "admission_id": decision["admission_id"],
        "scope": "existing observed GPU0 implementation/case/contract only",
        "additional_pack_us": 0., "additional_launch_us": 0.}


def build(config_path=SCENARIOS, hardware_path=HARDWARE):
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    hw = json.loads(Path(hardware_path).read_text(encoding="utf-8"))
    device = hw["devices"][0]
    base = parameters(device["mem_bw"], str(hardware_path) + " devices[0].mem_bw; " + device["source"])
    # Do not read measured local_pair latency labels or fit any parameters.
    exact = reviewed_complete_fused()
    return {
        "scope": "cpu_graph_stage_decomposition_not_predictor_accuracy_validation_v1",
        "method": {"reference": "https://github.com/openxla/xla/discussions/10065",
            "adapted": ["stage compute/memory overlap", "read utilization", "coalescing",
                        "explicit cache split", "implementation-dependent compute service"],
            "project_additions": ["actual pack Recipe", "Graph contract parameters", "unknown propagation"],
            "not_implemented": ["XLA compiler indexing analysis", "GEMM tile selection", "occupancy inference"],
            "compute_assumption": "sum arithmetic and nonlinear service, then max with memory",
            "cache_assumption": "read hit fraction applies to logical reads; writes charged to device bandwidth",
            "parameters_fitted_to_local_pair": False, "new_gpu_execution": False},
        "recipe_provenance": {
            "shape_dtype_layout": "fixed local case declaration; not inferred for unknown tensors",
            "native": "calibration/lightop_local_pair.py NativeCall: silu.out -> mul.out",
            "fused": "integration/lightop/silu_mul.py: preallocated cat -> LightOP",
            "bytes": "BF16 2 B/element; read/write logical operands per actual stage",
            "flops": "elementwise multiplies only; nonlinear SiLU service separately unknown",
            "units": {"read_bytes": "B", "write_bytes": "B", "arithmetic_flops": "FLOP", "silu_elements": "element"}},
        "unresolved": analyze(base),
        "illustrative_scenario": analyze(assume(base, cfg["assumptions"])),
        "strict_complete_fused_profile": exact,
        "sensitivity": sensitivity(base, cfg["assumptions"], cfg["axes"], complete_fused_us=exact["duration_us"]),
        "interpretation": {
            "native_chain_profile_used_as_singletons": False,
            "exact_fused_vs_analytic_native_is_not_independent_validation": True,
            "minimal_calibration": "conditional shortlist only; resolve influential parameter combinations, not every grid cell",
            "not_swept": ["vector_flops_per_s", "coalescing_efficiency", "cache_hit_fraction", "cache_bandwidth_Bps", "read_utilization", "graph_call_overhead_us"],
            "unswept_parameters_are_not_certified_irrelevant": True,
            "matrix_flops_used_as_vector_or_sigmoid_throughput": False},
        "source_sha256": {str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / "cost_model/graph_analytic.py", ROOT / "tools/analyze_graph_cost.py", Path(config_path), Path(hardware_path))},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenarios", type=Path, default=SCENARIOS)
    args = parser.parse_args()
    result = build(args.scenarios)
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "report.json"
    with target.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"output": str(target), "scenarios": len(result["sensitivity"]["joint_rows"]),
        "conditional_calibration_shortlist": result["sensitivity"]["conditional_calibration_shortlist"]}))


if __name__ == "__main__":
    main()
