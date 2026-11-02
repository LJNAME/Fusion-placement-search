"""CPU-frozen B3 evidence matrix, reuse review, buffers and quality rules.

No collector, GPU smoke, latency label, or admission decision is run here.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import statistics as st

from cost_model.ffn_placement_b1 import COMPUTE, FUSED, serialized
from cost_model.ffn_communication_b2 import B2_FILES, load_b1, verify as verify_b2
from cost_model.fusion_kernel_profile import key_id
from cost_model.kerv_admission import USE, load_admitted_store
from cost_model.kerv_profile import digest

ROOT = Path(__file__).resolve().parents[1]
E = [1, 256, 3072]
H = [1, 256, 1024]
RECIPE = {
    "mode": "graph_replay", "valid_rounds": 7, "calls_per_round": 50, "warmup_replays_per_arm": 10,
    "numeric_contract": {"rtol": .01, "atol": .001, "copy_and_source_preservation": "bitwise exact",
                         "fused_reference": "Native and FP32-expression rounded to BF16; no native fallback"},
    "named_fixtures": {"generation": "separate CPU Torch generator per named tensor; BF16 normal; upload outside capture/timing",
                       "producer_V": {"seed": 1234, "scale": .02}, "W_down": {"seed": 1235, "scale": .02},
                       "W_gate": {"seed": 1236, "scale": .02}, "W_up": {"seed": 1237, "scale": .02},
                       "gate_original": {"seed": 1238, "scale": 1.0}, "up": {"seed": 1239, "scale": 1.0},
                       "anchor": {"seed": 1240, "scale": 1.0},
                       "Y": "exact GPU0 reference down-projection result from producer_V/W_down, prepared outside timing; same Y used for both GPUs and link components"},
    "max_attempts": 9, "seed": 1234, "allocation_included": False,
    "unit": "us/call (one copy is one transfer)",
    "primary_selection": "first seven anchor-valid attempts in chronological order; never replace or trim",
    "anchor_drift_limit": 0.10, "anchor_formula": "abs(post-pre)/pre; any required GPU above 0.10 invalidates the whole round group",
    "iqr_relative_limit": 0.10, "iqr_method": "statistics.quantiles(n=4, method=inclusive); (Q3-Q1)/median",
    "dispersion_failure": "first seven primary samples remain immutable; excess dispersion stops admission; at most two extra diagnostic rounds cannot rescue them",
    "fatal": ["numeric/source-preservation failure", "capture/replay failure", "native fallback on fused arm", "host staging on timed transfer", "identity/layout mismatch"],
    "retention": "all 50 outputs checked for every arm and every attempt, including invalid/diagnostic attempts",
    "restore": "after capture and before EVERY warmup or timed replay; gate reset and synchronization outside Event interval",
    "anchor": {"operator": "F.silu", "shape": E, "dtype": "bf16", "source_count": 1,
               "outputs_retained": 50, "calls": 50, "warmup_replays": 10,
               "position": "before/after the entire round group on every participating GPU, ascending GPU order",
               "is_cost_label": False},
    "admission_depends_on_speedup": False,
    "activation_orders": [["silu", "mul", "fused", "native"], ["mul", "native", "silu", "fused"],
                          ["native", "fused", "mul", "silu"], ["fused", "silu", "native", "mul"]],
    "link_orders": [["producer", "copy", "chain", "consumer"], ["copy", "consumer", "producer", "chain"],
                    ["consumer", "chain", "copy", "producer"], ["chain", "producer", "consumer", "copy"]],
    "rotation": "attempt_id modulo four, including invalid attempts",
    "no_automatic_admission": True,
}
NAMES = ("identity.json", "recipe.json", "measurement_matrix.json", "reuse_audit.json", "buffer_audit.json", "freeze_checklist.json")


def quality_review(attempts, required_roles, required_devices):
    """Quality eligibility only; synthetic unit samples are never Profile data."""
    if len(attempts) > RECIPE["max_attempts"]:
        raise ValueError("attempt budget exceeded")
    primary, invalid = [], []
    for index, row in enumerate(attempts):
        if row["attempt"] != index:
            raise ValueError("all attempts, including invalid ones, must be chronological and complete")
        if not all(row.get(k) is True for k in ("numeric_ok", "capture_ok", "path_ok")):
            return {"status": "stop_fatal", "failed_attempt": index, "timing_eligible": False}
        if set(row["arm_us"]) != set(required_roles) or set(row["anchors"]) != set(required_devices):
            raise ValueError("round group arms/devices differ from frozen recipe")
        if any(not math.isfinite(v) or v <= 0 for v in row["arm_us"].values()):
            raise ValueError("invalid duration")
        drift = []
        for anchor in row["anchors"].values():
            before, after = anchor["before_us"], anchor["after_us"]
            if not all(math.isfinite(v) and v > 0 for v in (before, after)):
                raise ValueError("invalid anchor")
            drift.append(abs(after - before) / before)
        if max(drift) > RECIPE["anchor_drift_limit"]:
            invalid.append(index)
        elif len(primary) < RECIPE["valid_rounds"]:
            primary.append(index)
    if len(primary) < 7:
        return {"status": "needs_more_attempts" if len(attempts) < 9 else "stop_insufficient_valid_rounds",
                "primary_attempts": primary, "invalid_attempts": invalid, "timing_eligible": False}
    statistics = {}
    for role in required_roles:
        values = [attempts[i]["arm_us"][role] for i in primary]
        q1, _, q3 = st.quantiles(values, n=4, method="inclusive")
        median = st.median(values)
        statistics[role] = {"median_us": median, "iqr_over_median": (q3-q1)/median, "n_rounds": 7, "n_calls": 350}
    stable = all(s["iqr_over_median"] <= .10 for s in statistics.values())
    return {"status": "eligible_for_later_admission" if stable else "stop_dispersion", "timing_eligible": stable,
            "primary_attempts": primary, "invalid_attempts": invalid, "statistics": statistics,
            "diagnostic_attempts": [i for i in range(len(attempts)) if i not in primary and i not in invalid],
            "automatic_admission": False}


def link_residual(chain_us, producer_us, copy_us, consumer_us):
    if any(not math.isfinite(v) or v <= 0 for v in (chain_us, producer_us, copy_us, consumer_us)):
        raise ValueError("invalid chain/component duration")
    residual = chain_us - producer_us - copy_us - consumer_us
    return {"R_link_us": residual, "R_link_over_chain": residual / chain_us,
            "engineering_action": "inspect_dependencies_and_composition" if abs(residual) >= .5 else "locally_allow_additivity",
            "global_additivity_admitted": False}


def buffer(name, shape, count, gpu, *, alias_of=None):
    nbytes = math.prod(shape) * 2 * count
    return {"name": name, "shape_per_call": shape, "count": count, "gpu": gpu, "dtype": "bf16",
            "allocated_bytes": 0 if alias_of else nbytes, "alias_of": alias_of,
            "allocation": "one backing pool with disjoint per-call views" if count == 50 and not alias_of else "declared tensor or alias"}


def buffer_inventory():
    activation = [buffer(n, E, c, "active_gpu", alias_of=a) for n, c, a in (
        ("gate_pool", 50, None), ("mul_left_silu_pool", 50, None), ("silu_outputs", 50, None),
        ("mul_outputs", 50, None), ("native_outputs", 50, None), ("fused_outputs", 50, "gate_pool"),
        ("gate_original", 1, None), ("up_shared_readonly", 1, None), ("up_backup", 1, None), ("silu_original", 1, None))]
    anchor = [buffer("anchor_source_readonly", E, 1, "each_participating_gpu"), buffer("anchor_outputs", E, 50, "each_participating_gpu")]
    chain = [buffer(name, shape, count, gpu) for name, shape, count, gpu in (
        ("producer_inputs", E, 50, 0), ("producer_weight", [3072,1024], 1, 0), ("consumer_weight", [1024,3072], 1, 1),
        ("producer_isolated_outputs", H, 50, 0), ("copy_isolated_sources", H, 50, 0), ("copy_isolated_outputs", H, 50, 1),
        ("consumer_isolated_inputs", H, 50, 1), ("consumer_isolated_outputs", E, 50, 1),
        ("chain_producer_outputs", H, 50, 0), ("chain_copy_outputs", H, 50, 1), ("chain_consumer_outputs", E, 50, 1),
        ("producer_original", E, 1, 0), ("consumer_original_Y", H, 1, 0))]
    return {"scope": "explicit retained buffer lower bounds; not measured peak HBM; excludes graph-private temporaries, allocator slack and unlisted numeric oracles",
            "activation_group": activation, "activation_controlled_bytes": sum(b["allocated_bytes"] for b in activation),
            "anchor_per_gpu": anchor, "anchor_controlled_bytes_per_gpu": sum(b["allocated_bytes"] for b in anchor),
            "link_quartet": chain, "link_quartet_controlled_bytes": sum(b["allocated_bytes"] for b in chain),
            "allocation_lifetime": "only current round group plus anchors; all captures initialized before timing; retain all outputs until checked",
            "gate_lifecycle": ["initialize 50 disjoint gate views from gate_original", "restore after capture and before each warmup/replay",
                               "restore and synchronize outside timing", "Fused overwrites each gate exactly once per replay", "check all 50 outputs and up preservation before next reset"],
            "readonly_up_count": 1, "fifty_up_buffers_required": False,
            "singleton_policy": "50 disjoint inputs and 50 retained outputs; one resident weight for GEMM; Mul shares the same immutable up",
            "copy_policy": "50 disjoint sources and 50 disjoint retained destinations; no host staging inside capture",
            "gpu_buffer_alias_and_lifetime_verified": False}


def build():
    b1, b1_hashes = load_b1()
    b2_root = ROOT / "data/ffn_communication_b2"
    b2 = {name: json.loads((b2_root / name).read_text(encoding="utf-8")) for name in B2_FILES}
    b2_saved = json.loads((b2_root / "verification.json").read_text(encoding="utf-8"))
    if key_id(verify_b2(b2)) != key_id(b2_saved):
        raise ValueError("B2 verification changed")
    recipe = copy.deepcopy(RECIPE)
    recipe["recipe_id"] = key_id(RECIPE)
    graph = b1["identity.json"]["subgraph"]
    nodes = {n["id"]: n for n in graph["nodes"]}
    rows = []
    for gpu in (0, 1):
        for node in COMPUTE:
            local = node.rsplit(".", 1)[-1]
            rows.append({"row_id": f"{local}@gpu{gpu}", "kind": "singleton", "node_ids": [node],
                         "device_id": f"gpu{gpu}", "shape": nodes[node]["extra"]["output_shape"], "dtype": "bf16",
                         "layout": "constructor_declared_contiguous; must observe before timing", "action": "new_measurement"})
        for arm in ("native", "fused"):
            rows.append({"row_id": f"silu_mul/{arm}@gpu{gpu}", "kind": "native_chain" if arm == "native" else "fused_kernel",
                         "node_ids": list(FUSED), "device_id": f"gpu{gpu}", "shape": E, "dtype": "bf16",
                         "layout": "constructor_declared_contiguous; must observe before timing", "action": "new_measurement"})
    requests = b2["communication_requests.json"]["requests"]
    for request in requests:
        size = request["source_carrier"]["tensor_bytes"]
        rows.append({"row_id": f"copy/{size}/{request['source_gpu']}->{request['destination_gpu']}", "kind": "communication",
                     "device_id": f"gpu{request['source_gpu']}->gpu{request['destination_gpu']}", "shape": request["source_carrier"]["shape"],
                     "dtype": "bf16", "b2_request_id": request["request_id"], "action": "new_measurement"})
    diagnostic = {"row_id": "diagnostic/down0-copy524288-gate1", "kind": "link_chain_diagnostic", "device_id": "gpu0->gpu1",
                  "action": "new_measurement", "components": ["down_proj@gpu0", "copy/524288/0->1", "gate_proj@gpu1"],
                  "scope": "shape-compatible diagnostic proxy; not an original FFN edge or cross-layer fusion",
                  "R_link": "chain - (producer + copy + consumer), per matched round then summarized",
                  "relative_residual": "R_link / chain", "threshold_us": .5, "boundary_action": "abs(R_link)>=0.5 inspect; no automatic correction",
                  "chain_timer": "same GPU1 Event pair; start fence to GPU0 producer and completion fence back to GPU1; B4 smoke must prove ordering",
                  "producer_fixture": "50 independent V tensors and resident W_down, seed1234; same inputs/weight as isolated down0",
                  "consumer_fixture": "50 copies of exact canonical Y=down0(V,W_down), prepared outside timing; same values as chain copy destination",
                  "copy_fixture": "50 independent source Y tensors, same values as producer outputs; sources unchanged in isolated copy",
                  "scope_limit": "does not certify reverse direction, 1.5MiB links or complete FFN composition"}
    groups = []
    used = set()
    for gpu in (0, 1):
        members = [f"silu@gpu{gpu}", f"gated_mul@gpu{gpu}", f"silu_mul/native@gpu{gpu}", f"silu_mul/fused@gpu{gpu}"]
        groups.append({"group_id": f"activation_gpu{gpu}", "members": members, "roles": ["silu", "mul", "native", "fused"],
                       "anchor_devices": [f"gpu{gpu}"], "orders": recipe["activation_orders"],
                       "fixture": "synthetic gate_original/up repeated consistently across four arms; Mul input is precomputed SiLU(gate_original)"})
        used.update(members)
    groups.append({"group_id": "link_quartet", "members": [*diagnostic["components"], diagnostic["row_id"]],
                   "roles": ["producer", "copy", "consumer", "chain"], "anchor_devices": ["gpu0", "gpu1"],
                   "orders": recipe["link_orders"], "fixture": "same V/W_down/Y/W_gate across all four arms; paired attempt IDs and anchors"})
    used.update(groups[-1]["members"])
    for row in rows:
        if row["row_id"] not in used:
            devices = ["gpu0", "gpu1"] if row["kind"] == "communication" else [row["device_id"]]
            groups.append({"group_id": row["row_id"], "members": [row["row_id"]], "roles": ["body"], "anchor_devices": devices, "orders": [["body"]]})
    contract_template = {"mode": "graph_replay", "calls_per_round": 50, "rounds": 7, "warmup_replays": 10,
                         "allocation_included": False, "unit": "us/call", "restore_scope": "outside Event interval",
                         "retention": "all 50 outputs", "recipe_id": recipe["recipe_id"], "runtime_environment": "bind actual B4 observation"}
    for row in [*rows, diagnostic]:
        row["candidate_id"] = key_id({"graph_hash": b1["identity.json"]["source_graph_hash"],
                                       "node_ids": row.get("node_ids"), "communication_request": row.get("b2_request_id"),
                                       "diagnostic_components": row.get("components")})
        row["recipe_id"] = recipe["recipe_id"]
        row["measurement_contract_id"] = None
        row["round_group_id"] = next(g["group_id"] for g in groups if row["row_id"] in g["members"])
        row["gpu_measurement_ready"] = False
        row["pair_key"] = None
        if row["kind"] == "singleton":
            local = row["node_ids"][0].rsplit(".", 1)[-1]
            row["input_policy"] = {
                "gate_proj": "50 disjoint Y inputs; one resident W_gate; all 50 outputs retained",
                "up_proj": "50 disjoint Y inputs; one resident W_up; all 50 outputs retained",
                "down_proj": "50 disjoint producer_V inputs; one resident W_down; all 50 outputs retained",
                "silu": "same 50 gate views as Native/Fused; reset outside each replay; all 50 outputs retained",
                "gated_mul": "50 disjoint precomputed SiLU(gate_original) inputs plus one shared read-only up; all 50 outputs retained",
            }[local]
        elif row["kind"] in {"native_chain", "fused_kernel"}:
            row["input_policy"] = "same 50 disjoint gate views and one shared read-only up; reset gate outside EVERY replay; Fused outputs alias gates"
        elif row["kind"] == "communication":
            row["input_policy"] = "50 disjoint immutable sources and 50 retained destinations; Y for 0.5MiB, gate_original for 1.5MiB; preparation outside timing"
    for gpu in (0, 1):
        pair = [r for r in rows if r["row_id"] in (f"silu_mul/native@gpu{gpu}", f"silu_mul/fused@gpu{gpu}")]
        key = {"candidate_id": pair[0]["candidate_id"], "device_id": f"gpu{gpu}", "shape": E, "dtype": "bf16",
               "layout": "declared_contiguous", "measurement_contract": {"template_id": key_id(contract_template), "runtime_contract_id": None},
               "recipe_id": recipe["recipe_id"]}
        for row in pair:
            row["pair_key"] = copy.deepcopy(key)
            row["pair_key_template_id"] = key_id(key)
    store = load_admitted_store(ROOT / "data/kerv_profile_admission_v1/measurements/profile.json", ROOT / "data/kerv_profile_admission_v1/admission.json", usage_scope=USE)
    reuse = []
    for r in store.records.values():
        if r["role"] != "fused_kernel":
            continue
        reuse.append({"device_id": r["case_id"], "record_id": r["record_id"], "original_admission": r["admission"],
                      "recorded_warmup": r["measurement_contract"]["warmup"]["replays_per_arm"],
                      "recorded_rounds": r["measurement_contract"]["rounds"], "recorded_calls": r["measurement_contract"]["calls_per_round"],
                      "recorded_working_set": r["measurement_contract"]["working_set"],
                      "restore_outside_timing_recorded": r["measurement_contract"].get("restore_scope") == "outside Event interval",
                      "anchor_observations_present": any("anchors" in sample for sample in r["samples"]),
                      "eligible_for_b3_reuse": False, "missing_evidence": ["pre/post anchor observations", "new recipe_id and chronological attempt eligibility"],
                      "changed_context": "new named fixtures and four-arm group residency/order; old evidence is a two-arm pair",
                      "original_profile_unchanged": True, "action": "remeasure_under_new_recipe"})
    zero = b2["coverage.json"]["zero_transfer_plans"]
    checklist = [{"item": item, "status": "cpu_review_complete"} for item in (
        "18 core rows and within-device pair keys", "old Fused reuse audited (both ineligible)",
        "four isolated copy directions/sizes", "one shape-compatible link quartet",
        "R_link formula/relative value/0.5us boundary", "anchor/IQR/chronological attempts and fatal errors", "input restore and controlled buffer lifetimes")]
    checklist.append({"item": "HIP Graph capture and cross-device route/event smoke", "status": "pending_B4_GPU_smoke", "static_plan_available": True})
    smoke_plan = {
        "phase": "B4 before formal timing; no eager/host-staging fallback",
        "replays_per_body": 3, "calls_in_each_capture": 50,
        "activation": "capture all four arms on both GPUs; reset gates outside every replay; check every output, per-gate alias, invalidation and shared-up preservation",
        "communication": "all four B2 requests, including both 1.5MiB directions; 50 independent sources/destinations; vary source payloads outside each replay to detect stale copies",
        "chain": "producer/copy/consumer quartet and GPU1 start/end fences; no cross-device Event subtraction",
        "event_proof": "destination-stream consumer guard must be causally after copy and before end Event; retain route/dependency trace; a host-synchronized output check alone is insufficient",
        "identity": "actual loaded Torch/HIP/FlagScale implementation, both UUIDs, strides/aliases, input fingerprints and peer route",
        "failure": "stop the affected whole round group, preserve artifacts, no timing labels; do not silently drop an arm or replace Graph with eager",
        "verified_on_gpu": False,
    }
    identity = {"format": "fusion-placement-search.ffn-measurement-b3", "version": "1.0", "b3_cpu_recipe_frozen": True,
                "b1_source_sha256": b1_hashes, "b2_source_sha256": {name: digest(b2_root/name) for name in (*B2_FILES, "verification.json")},
                "source_sha256": {name: digest(ROOT/name) for name in ("cost_model/ffn_measurement_b3.py", "calibration/measure_kerv_profile_pair.py",
                    "data/kerv_profile_admission_v1/measurements/profile.json", "data/kerv_profile_admission_v1/admission.json")},
                "boundary_sanity": {"input_gpu": 0, "output_gpu": 0, "zero_communication_plan_ids": zero,
                                    "scope": "only GPU0/GPU0 B1 scene; not a four-boundary-scenario count",
                                    "full_ffn_measurements": "two deferred B6 sanity tasks, not included in these 19 evidence slots"},
                "b4_smoke_required": True, "formal_gpu_timing_allowed": False, "new_gpu_measurements": 0,
                "new_profiles": 0, "search_cost_items_admitted": 0}
    matrix = {"core_rows": rows, "diagnostics": [diagnostic], "round_groups": groups, "contract_template": contract_template,
              "summary": {"singleton_rows": 10, "native_rows": 2, "fused_rows": 2, "communication_rows": 4,
                          "core_rows": len(rows), "diagnostic_rows": 1, "total_evidence_slots": len(rows)+1,
                          "eligible_fused_reuse_rows": 0, "new_measurement_tasks": len(rows)+1, "hypothetical_tasks_if_two_fused_reusable": 17},
              "pair_finalization": "B4 binds actual UUID/build and exact measurement_contract_id on both arms before timing",
              "profiling_scope": "synthetic boundary/reference inputs of real Builder shapes, not captured model weights or complete FFN execution"}
    return copy.deepcopy(dict(zip(NAMES, (identity, recipe, matrix, {"records": reuse}, buffer_inventory(),
                            {"items": checklist, "cpu_checks_complete": 7, "gpu_smoke_pending": 1, "all_eight_passed": False,
                             "b4_smoke_plan": smoke_plan}))))


def verify(bundle):
    expected = build()
    if any(serialized(bundle[n]) != serialized(expected[n]) for n in NAMES):
        raise ValueError("B3 freeze or source identities changed")
    matrix = bundle["measurement_matrix.json"]
    rows = matrix["core_rows"] + matrix["diagnostics"]
    ids = [r["row_id"] for r in rows]
    membership = [r for g in matrix["round_groups"] for r in g["members"]]
    if len(ids) != 19 or len(set(ids)) != 19 or sorted(ids) != sorted(membership):
        raise ValueError("matrix missing/duplicate rows or round groups")
    for gpu in (0, 1):
        pair = [r for r in rows if r["row_id"] in (f"silu_mul/native@gpu{gpu}", f"silu_mul/fused@gpu{gpu}")]
        if len(pair) != 2 or pair[0]["pair_key"] != pair[1]["pair_key"]:
            raise ValueError("Native/Fused pair identity mismatch")
    return {"pass_checks": True, **matrix["summary"], "cpu_freeze_checks_complete": 7, "gpu_smoke_pending": 1,
            "round_groups": len(matrix["round_groups"]), "repeat_byte_equivalent": True,
            "output_payload_ids": {n: key_id(bundle[n]) for n in NAMES}, "new_gpu_measurements": 0,
            "formal_gpu_timing_allowed": False, "search_cost_items_admitted": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/ffn_measurement_b3")
    args = parser.parse_args()
    bundle = build()
    result = verify(bundle)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in {**bundle, "verification.json": result}.items():
        with (args.output/name).open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized(value))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
