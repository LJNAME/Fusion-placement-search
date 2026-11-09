"""P1: look for coupling witnesses in existing real measurements, without new GPU work."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluate_stage_search import save


def placement_candidate(cells):
    preferred = {arm: min((0, 1), key=lambda d: cells[str(d)][arm]["median_us"]) for arm in ("native", "fused")}
    separated = {}
    for arm, d in preferred.items():
        separated[arm] = cells[str(d)][arm]["max_us"] < cells[str(1 - d)][arm]["min_us"]
    return {"preferred_device_from_medians": preferred,
            "point_estimate_placement_flip": preferred["native"] != preferred["fused"],
            "observed_ranges_separated": separated,
            "status": "point_estimate_candidate_only" if preferred["native"] != preferred["fused"] else "no_point_flip",
            "confirmed_coupling": False,
            "range_note": "min/max are observed sample ranges, not confidence intervals; overlapping ranges do not establish an ordering"}


def contract_without_uuid(contract):
    result = copy.deepcopy(contract)
    result["environment"].pop("device_uuid", None)
    return result


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT / "data/fusion_benefit_validation_v1/report.json"
    comparisons = json.loads(source.read_text(encoding="utf-8"))["cases"]
    groups = {}
    for case in comparisons:
        key = (case["implementation"], case["contract"])
        groups.setdefault(key, []).append(case)
    inventory = []
    for (implementation, contract), cases in groups.items():
        preferences = {c["measured_preference"] for c in cases} - {"inconclusive"}
        inventory.append({"implementation": implementation, "contract": contract,
                          "case_ids": [c["case_id"] for c in cases],
                          "observed_preferences": sorted(preferences),
                          "same_implementation_preference_flip": {"fused", "unfused"} <= preferences,
                          "evidence_scope": "existing comparison report; not newly admitted or independent whole-plan labels"})
    base = ROOT / "data/kerv_profile_admission_v1"
    admission = json.loads((base / "admission.json").read_text(encoding="utf-8"))
    profile_path = base / "measurements/profile.json"
    if admission["status"] != "accepted" or hashlib.sha256(profile_path.read_bytes()).hexdigest() != admission["source_profile_sha256"]:
        raise ValueError("paired admission source changed")
    sources = [Path(__file__), source, base / "admission.json", profile_path]
    cells, receipts = {}, {}
    for device in (0, 1):
        cells[str(device)] = {}
        for arm in ("native", "fused"):
            root = base / f"measurements/inputs/gpu{device}/{arm}"
            result = json.loads((root / "result.json").read_text(encoding="utf-8"))
            samples = json.loads((root / "samples.json").read_text(encoding="utf-8"))
            if len(samples) != 7 or {s["round"] for s in samples} != set(range(7)) or any(s["graph_calls"] != 50 for s in samples):
                raise ValueError("unexpected paired sample contract")
            values = [s["amortized_per_call_s"] * 1e6 for s in samples]
            if abs(statistics.median(values) - result["graph_batch_amortized_per_call_s"] * 1e6) > 1e-9:
                raise ValueError("raw median differs from receipt")
            cells[str(device)][arm] = {"median_us": statistics.median(values), "min_us": min(values),
                                       "max_us": max(values), "samples_us": values,
                                       "measurement_contract_id": result["measurement_contract_id"]}
            receipts[(device, arm)] = result
            sources.extend((root / "result.json", root / "samples.json"))
    first = receipts[(0, "fused")]
    for result in receipts.values():
        if result["input"] != first["input"] or contract_without_uuid(result["measurement_contract"]) != contract_without_uuid(first["measurement_contract"]):
            raise ValueError("input or non-UUID execution conditions differ")
        if result["kernel_source_sha256"] != first["kernel_source_sha256"]:
            raise ValueError("kernel implementation source differs")
    candidate = {"implementation": "KERV silu_mul; Native complete SiLU->Mul comparison chain",
                 "shape": first["input"]["shape"], "dtype": first["input"]["dtype"],
                 "layout": first["input"]["layout"], "cells": cells, **placement_candidate(cells),
                 "identity_check": "same inputs, implementation source and full contract except physical UUID; physical receipts unchanged",
                 "admission_scope": admission["scope"],
                 "matches_current_3b_ffn": False,
                 "three_b_gap": "measured width 3072 vs 3B FFN width 11008; not transferred to 3B or current LightOP model",
                 "followup_needed": "only if pursued: paired cross-device order validation for this exact kernel/shape, then graph-level reference; no new job created"}
    report = {"scope": "P1 bounded existing-evidence audit; no model/cost changes, new GPU, admission or whole-graph execution",
              "comparison_inventory": inventory, "placement_candidate": candidate,
              "confirmed_real_preference_flips": 0, "confirmed_real_whole_plan_placement_changes": 0,
              "excluded_inferences": ["KERV vs LightOP is an implementation change, not a same-kernel preference flip",
                                      "eager vs Graph is a contract change",
                                      "GPU1 singleton sums are not a matched Native complete chain",
                                      "hardware-class reuse is not an independent second-card measurement",
                                      "overlapping cross-card sample ranges do not confirm an optimal-device change"],
              "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    save(output / "report.json", report)
    print(json.dumps({"inventory_groups": len(inventory), "point_candidate": {k: v for k, v in candidate.items()
                       if k in ("preferred_device_from_medians", "point_estimate_placement_flip", "observed_ranges_separated", "confirmed_coupling", "shape")}}))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
