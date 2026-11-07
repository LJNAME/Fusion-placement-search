"""Offline admission and exact snapshot lookup for one KERV Graph cost."""
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import re
import subprocess

from calibration.kerv_add_rmsnorm_probe import ROOT, prepare_adapter, sha, save
from calibration.kerv_add_rmsnorm_timing import SOURCES, kernel_identity, plan, summarize
from calibration.kerv_silu_mul import DEFAULT_FLAGSCALE
from cost_model.fusion_kernel_profile import GroupKernelKeys, KernelLatency, KeyedKernelProfile, key_id
from cost_model.hardware_profile import AnalyticHardwareProfile
from cost_model.measurement_labels import MeasurementContractError, measurement_contract_id


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class _ObservedGroupKeys(GroupKernelKeys):
    def describe(self, group):
        if (tuple(group.node_ids), group.device) != self.selected:
            raise MeasurementContractError("admission covers only the observed group and device")
        return super().describe(group)


def _keys(raw, flagscale_root):
    bound, descriptor = prepare_adapter(flagscale_root, "cuda:0")
    frozen, identity, runtime = (read(raw / n) for n in ("plan.json", "identity.json", "runtime.json"))
    if descriptor != frozen["adapter"]:
        raise MeasurementContractError("selected placement/implementation differs from measurement")
    keys = kernel_identity(bound, identity["tensor_observations"], runtime["runtime"], runtime["device"], return_keys=True)
    group = bound.placed.plan.groups[bound.index]
    key, gaps = keys.describe(group)
    if gaps or key != identity["kernel_key"] or key_id(key) != identity["kernel_key_id"]:
        raise MeasurementContractError("canonical key does not reproduce from actual observations")
    selected = _ObservedGroupKeys(keys.graph, keys.hardware, keys.bindings, keys.runtime)
    selected.selected = (tuple(group.node_ids), group.device)
    return selected, group


def review(directory, flagscale_root=DEFAULT_FLAGSCALE):
    """Explicit CPU review writes an accepted label; never invoked by lookup."""
    directory = Path(directory); raw = directory / "raw"
    expected = {"plan.json", "identity.json", "runtime.json", "result.json", "progress.json", "fatal_traceback.log"}
    expected |= {f"round_{i}.json" for i in range(7)}
    if {p.name for p in raw.iterdir()} != expected or (raw / "fatal_traceback.log").read_bytes():
        raise MeasurementContractError("raw files incomplete or fatal failure recorded")
    frozen, identity, runtime, result = (read(raw/n) for n in ("plan.json", "identity.json", "runtime.json", "result.json"))
    current = plan(flagscale_root)
    for field in ("scope", "adapter", "shape", "dtype", "source_sha256", "correctness_evidence_sha256", "pool", "contract_template", "quality"):
        if frozen[field] != current[field]:
            raise MeasurementContractError("frozen measurement scope changed: " + field)
    revision = frozen["collector_revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or set(frozen["collector_sources"]) != set(SOURCES):
        raise MeasurementContractError("invalid collector source identity")
    import hashlib
    for name, digest in frozen["collector_sources"].items():
        source = subprocess.check_output(["git", "-C", str(ROOT), "show", revision + ":" + name])
        if hashlib.sha256(source).hexdigest() != digest:
            raise MeasurementContractError("collector differs from committed source: " + name)
    rows = [read(raw/f"round_{i}.json") for i in range(7)]
    quality = summarize(rows)
    if quality["status"] != "quality_passed" or result["records"] != rows or result["quality"] != quality:
        raise MeasurementContractError("round records or recomputed quality failed")
    if not all(type(r[k]) is int for r in rows for k in ("started_monotonic_ns", "finished_monotonic_ns")):
        raise MeasurementContractError("missing round chronology")
    if (any(r["started_monotonic_ns"] >= r["finished_monotonic_ns"] for r in rows)
            or any(a["finished_monotonic_ns"] > b["started_monotonic_ns"] for a,b in zip(rows,rows[1:]))):
        raise MeasurementContractError("rounds overlap or are reordered")
    if (not all(result[k] is True for k in ("complete", "gpu_executed", "graph_capture_verified", "timing_attempted"))
            or result["identity"] != identity or result["automatic_admission"] is not False):
        raise MeasurementContractError("incomplete Graph capture/timing receipt")
    contract = {**frozen["contract_template"], "environment": runtime["environment"]}
    contract_id = measurement_contract_id(contract)
    if identity["measurement_contract"] != contract or identity["measurement_contract_id"] != contract_id:
        raise MeasurementContractError("measurement contract identity mismatch")
    rt, env = runtime["runtime"], runtime["environment"]
    if (rt["accelerator"] != "HIP" or rt["grad_enabled"] is not False or rt["autocast_enabled"] is not False
            or rt["accelerator_version"] != env["hip"] or rt["driver_version"] != env["driver_version"]
            or rt["triton"] != env["triton"] or runtime["device"]["architecture"] != env["device_architecture"]):
        raise MeasurementContractError("runtime and contract environment mismatch")
    if not all(rt.get(k) for k in ("torch_git_revision", "torch_extension_sha256", "torch_build_sha256", "driver_version")):
        raise MeasurementContractError("runtime source/build identity missing")
    _keys(raw, flagscale_root)
    candidate = KernelLatency(**result["profile_candidate"])
    if (candidate.admission != "conditional_dev" or candidate.kind != "measured"
            or candidate.measurement_contract_id != contract_id or candidate.statistics != quality["statistics"]
            or candidate.duration_s != quality["statistics"]["median_us"] / 1e6):
        raise MeasurementContractError("candidate does not match the measured rounds")
    raw_sha = {name:sha(raw/name) for name in sorted(expected)}
    accepted = replace(candidate, admission="accepted", provenance="reviewed_kerv_add_rmsnorm:" + key_id(raw_sha))
    decision = {"scope":"one GPU0 observed AddRMS group, Graph isolated replay contract; not whole-plan admission",
                "admission":"accepted", "kernel_key":identity["kernel_key"], "kernel_key_id":identity["kernel_key_id"],
                "measurement_contract":contract, "measurement_contract_id":contract_id,
                "latency":asdict(accepted), "quality":quality, "raw_sha256":raw_sha,
                "verified_collector_revision":revision, "reviewer_source_sha256":sha(__file__),
                "accepted_profiles_added":1, "cross_layer_reuse_admitted":False,
                "whole_plan_admitted":False, "gpu1_admitted":False,
                "uuid_used_for_identity":False, "new_gpu_work_during_review":False,
                "numerics_verified_as_source_bound_receipts":True}
    (directory/"review").mkdir(exist_ok=True)
    save(directory/"review/profile.json", decision)
    return decision


def load_profile(directory, flagscale_root=DEFAULT_FLAGSCALE, *, contract_id=None):
    """Read the explicit admission; exact canonical lookup, no promotion/fallback."""
    directory = Path(directory); raw = directory / "raw"
    record = read(directory/"review/profile.json")
    if record["admission"] != "accepted" or record["latency"]["admission"] != "accepted":
        raise MeasurementContractError("profile not accepted")
    for name, digest in record["raw_sha256"].items():
        if Path(name).name != name or sha(raw/name) != digest:
            raise MeasurementContractError("reviewed raw receipt changed: " + name)
    identity = read(raw/"identity.json")
    expected = measurement_contract_id(record["measurement_contract"])
    if (record["kernel_key"] != identity["kernel_key"] or key_id(record["kernel_key"]) != record["kernel_key_id"]
            or expected != record["measurement_contract_id"] or expected != identity["measurement_contract_id"]
            or expected != record["latency"]["measurement_contract_id"] or (contract_id is not None and contract_id != expected)):
        raise MeasurementContractError("reviewed key/contract mismatch")
    keys, group = _keys(raw, flagscale_root)
    entry = KernelLatency(**copy.deepcopy(record["latency"]))
    if (entry.kind != "measured" or record["quality"] != read(raw/"result.json")["quality"] or entry.statistics != record["quality"]["statistics"]
            or entry.duration_s != record["quality"]["statistics"]["median_us"] / 1e6):
        raise MeasurementContractError("accepted latency differs from the reviewed measurements")
    profile = KeyedKernelProfile(keys, {record["kernel_key_id"]:entry}, AnalyticHardwareProfile(keys.hardware),
                                 strict=True, measurement_contract_id=expected, allow_unbound=False)
    return profile, group
