"""Does the cost model predict the relative benefit of fused vs unfused execution?

CPU only; reuses existing same-device measured pairs, no new GPU work.
For each pair the model predicts r = fused / unfused on a local fixture graph
(the fused group vs its ops as singletons, same device, same external I/O) and
is compared with the measured r. Reported per case: sign (which arm is faster),
ratio error, and, for the calibrated model, out-of-sample error where the
correction is fitted only on cases of a different shape.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

from calibration.fusion_cost_validation import build_fixture_graph, summarize_case
from calibration.probe_kerv_additivity import CASES
from cost_model.fusion_library import LibraryPattern, _one
from cost_model.hardware_profile import AnalyticHardwareProfile
from cost_model.structure import analyze_plan_structure
from simulator import OpGraph, OpNode, TensorEdge
from simulator.fusion_rules import FusionPatternRegistry, default_fusion_registry
from simulator.hardware import load_hardware

ROOT = Path(__file__).resolve().parents[1]
HARDWARE = ROOT / "data/bw1000_2card/bw1000_2card.calibrated.json"
GRAPH_LAUNCH_US = json.loads((ROOT / "configs/ffn_profile_search.json").read_text())["projection_graph_overhead_us"]
SILU_BATCH = ROOT / "data/fusion_cost_validation_v1/inputs"
KERV_ADDRMS = ROOT / "data/kerv_add_rmsnorm_timing_01/review/profile.json"
NATIVE_ADDRMS = ROOT.parent / "native_add_rmsnorm_timing_01/result.json"
TRIAL02 = ROOT / "data/hygon_silu_trial02_review/report.json"
BAND = 0.01  # |r - 1| below this is "inconclusive", as in fusion_cost_validation

PATTERNS = {
    "silu_mul": LibraryPattern(pattern_id="silu_mul", backend="fixture", implementation_status="implemented",
                               op_types=("silu", "mul"), min_nodes=2, max_nodes=2, require_same_shape=True),
    "add_rmsnorm": LibraryPattern(pattern_id="add_rmsnorm", backend="fixture", implementation_status="implemented",
                                  op_types=("add", "rmsnorm"), min_nodes=2, max_nodes=2, require_same_shape=True,
                                  require_linear_chain=False, intermediate_outputs=True),
}
IDS = {"silu_mul": ("silu", "mul"), "add_rmsnorm": ("add", "norm")}


def registry():
    singles = [p for p in default_fusion_registry().patterns() if p.max_nodes == 1]
    return FusionPatternRegistry([*singles, *PATTERNS.values()])


def add_rmsnorm_fixture(shape):
    """x + residual -> rmsnorm; the sum also feeds the next residual (dual output),
    using the model graph's flop/byte conventions (add n, rmsnorm 5n, weight 2*h)."""
    n = math.prod(shape); nbytes = 2 * n
    meta = lambda: dict(output_shape=list(shape), dtype="bf16", layout="contiguous",  # noqa: E731
                        phase="forward", layer=0, shape={"n": n})
    g = OpGraph(name="add-rmsnorm-fixture", extra={"output_node": "next"})
    for name, op, flops, params in (("x", "input", 0., 0.), ("residual", "input", 0., 0.),
                                    ("add", "add", float(n), 0.), ("norm", "rmsnorm", 5. * n, 2. * shape[-1]),
                                    ("next", "add", float(n), 0.)):
        g.add_node(OpNode(name, op, flops, nbytes, params, meta()))
    for src, dst in (("x", "add"), ("residual", "add"), ("add", "norm"), ("add", "next"), ("norm", "next")):
        g.add_edge(TensorEdge(src, dst, nbytes))
    return g


def profiles():
    hw = load_hardware(HARDWARE)
    graph_hw = copy.deepcopy(hw)
    graph_hw.launch_overhead, graph_hw.fusion_overhead = GRAPH_LAUNCH_US * 1e-6, 0.
    return {"eager_launch": AnalyticHardwareProfile(hw), "graph_launch": AnalyticHardwareProfile(graph_hw)}


def predicted_ratio(profile, graph, pattern, device):
    ids, reg = IDS[pattern], registry()

    def cost(fused):
        plan = _one(graph, ids, pattern, fused)
        for group in plan.groups:
            group.device = device
        groups = analyze_plan_structure(graph, plan, profile.n_devs, registry=reg).groups
        return sum(profile.estimate_group(graph, g).duration for g in groups if set(g.node_ids) <= set(ids))
    return cost(True) / cost(False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preference(r):
    return "inconclusive" if abs(r - 1) < BAND else ("fused" if r < 1 else "unfused")


def load_cases():
    cases = []
    for device, seq in CASES:
        raw = json.loads((SILU_BATCH / f"gpu{device}_seq{seq}/result.json").read_text())
        case = summarize_case(raw)
        med = {k: v["median_us"] for k, v in case["statistics"].items()}
        paired = [r["fused"] / r["chain"] for r in case["paired"].values()]
        cases.append(dict(case_id=f"kerv_silu_mul_{case['case_id']}", pattern="silu_mul",
                          implementation="kerv silu_mul (graph)", contract="graph_replay",
                          shape=list(case["shape"]), device=device, fused_us=med["fused"],
                          unfused_us=med["chain"], unfused_arm="same-batch silu->mul chain",
                          paired_rounds=len(paired), paired_same_sign=all(p < 1 for p in paired)
                          or all(p > 1 for p in paired), same_batch=True))
    kerv = json.loads(KERV_ADDRMS.read_text())
    native = json.loads(NATIVE_ADDRMS.read_text())
    if native["quality"]["status"] != "quality_passed":
        raise ValueError("native AddRMS chain did not pass its quality gate")
    cases.append(dict(case_id="kerv_add_rmsnorm_gpu0_seq256", pattern="add_rmsnorm",
                      implementation="kerv add_rmsnorm consume-x dual output (graph)", contract="graph_replay",
                      shape=[1, 256, 2048], device=0, fused_us=kerv["latency"]["statistics"]["median_us"],
                      unfused_us=native["quality"]["statistics"]["median_us"],
                      unfused_arm="native add -> F.rms_norm graph chain", paired_rounds=0,
                      paired_same_sign=None, same_batch=False))
    trial = json.loads(TRIAL02.read_text())["benchmark"]["arms"]
    cases.append(dict(case_id="lightop_silu_mul_split_gpu0_seq256", pattern="silu_mul",
                      implementation="lightop op.silu_and_mul_opt + pack (eager)", contract="eager",
                      shape=[1, 256, 3072], device=0, fused_us=trial["lightop_split_with_pack"]["gpu_us_per_call"],
                      unfused_us=trial["native_split"]["gpu_us_per_call"], unfused_arm="native silu, mul on split inputs",
                      paired_rounds=0, paired_same_sign=None, same_batch=True))
    return cases


ALIGNED = {"graph_replay": "graph_launch", "eager": "eager_launch"}


def evaluate(cases):
    profs = profiles()
    graphs = {}
    rows = []
    for case in cases:
        key = (case["pattern"], tuple(case["shape"]))
        if key not in graphs:
            graphs[key] = (build_fixture_graph(case["shape"]) if case["pattern"] == "silu_mul"
                           else add_rmsnorm_fixture(case["shape"]))
        graph = graphs[key]
        measured = case["fused_us"] / case["unfused_us"]
        row = dict(case, measured_ratio=measured, measured_preference=preference(measured),
                   aligned_profile=ALIGNED[case["contract"]])
        for name, prof in profs.items():
            r = predicted_ratio(prof, graph, case["pattern"], case["device"])
            row[f"{name}_ratio"] = r
            row[f"{name}_preference"] = preference(r)
            row[f"{name}_sign_ok"] = preference(r) == row["measured_preference"]
            row[f"{name}_ratio_error"] = r / measured - 1
        rows.append(row)
    # Calibrated, out of sample: c fitted on cases of the same implementation and
    # contract but a different shape; prediction = c * aligned analytic ratio.
    for row in rows:
        group = [o for o in rows if o["implementation"] == row["implementation"]
                 and o["contract"] == row["contract"] and o["shape"] != row["shape"]]
        aligned = row["aligned_profile"] + "_ratio"
        if not group:
            row.update(calibrated_ratio=None, calibrated_fit_cases=[], calibrated_note="no other-shape case to fit on")
            continue
        logc = [math.log(o["measured_ratio"] / o[aligned]) for o in group]
        c = math.exp(sum(logc) / len(logc))
        r = c * row[aligned]
        row.update(calibrated_ratio=r, calibrated_preference=preference(r),
                   calibrated_sign_ok=preference(r) == row["measured_preference"],
                   calibrated_ratio_error=r / row["measured_ratio"] - 1, calibrated_correction=c,
                   calibrated_fit_cases=[o["case_id"] for o in group])
    return rows


def summarize(rows):
    def stat(prefix, subset):
        errs = [abs(r[f"{prefix}_ratio_error"]) for r in subset if r.get(f"{prefix}_ratio") is not None]
        signs = [r[f"{prefix}_sign_ok"] for r in subset if r.get(f"{prefix}_ratio") is not None]
        return dict(cases=len(errs), sign_agreement=f"{sum(signs)}/{len(signs)}",
                    max_abs_ratio_error=max(errs) if errs else None,
                    mean_abs_ratio_error=sum(errs) / len(errs) if errs else None)
    aligned = [dict(r, aligned_ratio=r[r["aligned_profile"] + "_ratio"],
                    aligned_sign_ok=r[r["aligned_profile"] + "_sign_ok"],
                    aligned_ratio_error=r[r["aligned_profile"] + "_ratio_error"]) for r in rows]
    return {"uncalibrated_contract_aligned": stat("aligned", aligned),
            "uncalibrated_eager_launch_all": stat("eager_launch", rows),
            "calibrated_out_of_sample": stat("calibrated", rows)}


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cases = load_cases()
    rows = evaluate(cases)
    sources = {"silu_mul_batch": SILU_BATCH, "kerv_add_rmsnorm": KERV_ADDRMS,
               "native_add_rmsnorm": NATIVE_ADDRMS, "lightop_trial02": TRIAL02}
    hashes = {}
    for name, path in sources.items():
        files = sorted(p for p in Path(path).rglob("result.json")) if Path(path).is_dir() else [Path(path)]
        for f in files:
            hashes[str(f.relative_to(ROOT.parent))] = sha(f)
    shutil.copyfile(NATIVE_ADDRMS, output / "native_add_rmsnorm_result.json")  # lives outside the repo
    report = {"scope": "local fused-vs-unfused benefit on existing measured pairs; CPU only; no new GPU work",
              "hardware_sha256": sha(HARDWARE), "graph_launch_us": GRAPH_LAUNCH_US,
              "preference_band": BAND, "source_sha256": hashes,
              "code_sha256": {p: sha(ROOT / p) for p in ("calibration/fusion_benefit_validation.py",
                              "cost_model/hardware_profile.py", "cost_model/structure.py",
                              "cost_model/fusion_library.py", "simulator/shape_model.py")},
              "summary": summarize(rows), "cases": rows}
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    cols = ["case_id", "pattern", "contract", "shape", "device", "same_batch", "fused_us", "unfused_us",
            "measured_ratio", "measured_preference", "graph_launch_ratio", "eager_launch_ratio",
            "aligned_profile", "calibrated_ratio", "calibrated_fit_cases"]
    with open(output / "cases.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([json.dumps(r.get(c)) if isinstance(r.get(c), list) else r.get(c) for c in cols])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    report = run(parser.parse_args().output)
    for r in report["cases"]:
        al = r[r["aligned_profile"] + "_ratio"]
        cal = r.get("calibrated_ratio")
        print(f"{r['case_id']:38s} measured {r['measured_ratio']:.3f} ({r['measured_preference']:7s}) "
              f"aligned {al:.3f} eager {r['eager_launch_ratio']:.3f} "
              f"calibrated {'-' if cal is None else f'{cal:.3f}'}")
    print(json.dumps(report["summary"], indent=1))


if __name__ == "__main__":
    main()
