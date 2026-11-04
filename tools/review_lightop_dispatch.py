"""Read-only CPU audit of hipprof's marked Graph replay. No latency labels."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sqlite3

from calibration.lightop_dispatch import MARKERS, SCOPE, plan
from cost_model.fusion_kernel_profile import key_id
from cost_model.lightop_profile import LightopKernelKeys, measurement_contract
from cost_model.measurement_labels import measurement_contract_id
from integration.lightop.silu_mul import adapter_registry
from simulator.placement_artifacts import load_placed_op_graph


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def database_events(path):
    """Known DTK hipprof schema, decoded by (PID, STR_ID), never name guesses."""
    with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True) as db:
        tables = [row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        def table(prefix):
            names = [name for name in tables if name.startswith(prefix)]
            if len(names) != 1:
                raise ValueError("expected exactly one " + prefix + " table")
            return quote(names[0])
        strings = {(pid, str(sid)): name for pid, sid, name in
                   db.execute("SELECT PID,STR_ID,STR_NAME FROM STR_TABLE WHERE TYPE=6")}
        rows = db.execute("SELECT BeginNs,EndNs,dev_id,queue_id,Name,pid,_Index,DurationNs FROM "
                          + table("HIPOPS_") + " ORDER BY BeginNs,_Index").fetchall()
        copies = db.execute("SELECT BeginNs,EndNs,dev_id,queue_id,pid,Bytes FROM "
                            + table("HIPCOPY_") + " ORDER BY BeginNs").fetchall()
    events = []
    for begin, end, device, queue, sid, pid, index, duration in rows:
        if duration <= 0 or end - begin != duration:
            raise ValueError("invalid raw kernel timestamps")
        events.append({"begin_ns": begin, "end_ns": end, "device": device, "queue": str(queue),
                       "pid": pid, "correlation_index": index, "duration_ns": duration,
                       "kernel_name": strings[(pid, str(sid))]})
    return events, copies


def csv_consistent(path, events):
    counts, durations = Counter(), defaultdict(int)
    for event in events:
        counts[event["kernel_name"]] += 1
        durations[event["kernel_name"]] += event["duration_ns"]
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["Name"] != "Total"]
    if (len(rows) != len(counts) or {row["Name"] for row in rows} != set(counts)
            or any(int(row["Calls"]) != counts[row["Name"]]
                   or int(row["TotalDurationNs"]) != durations[row["Name"]] for row in rows)):
        raise ValueError("kernel CSV disagrees with raw SQLite events")


def kernel_role(name):
    if "act_and_mul_kernel_opt1" in name and "BFloat16" in name and "silu_kernel" in name:
        return "lightop_silu_mul"
    if "CatArrayBatchedCopy_contig" in name and "OpaqueType<2" in name:
        return "split_input_pack"
    return "unexpected"


def target_region(events, copies, pid, device):
    markers = []
    for size in MARKERS:
        matching = [row for row in copies if row[4] == pid and row[2] == device and row[5] == size]
        if len(matching) != 1:
            raise ValueError("marker copy missing or ambiguous: " + str(size))
        markers.append(matching[0])
    start, end = markers[0][1], markers[1][0]
    if any(row[1] < row[0] for row in markers) or start >= end:
        raise ValueError("invalid marker order")
    target = []
    for event in events:
        if event["pid"] != pid:
            continue
        begin, finish = event["begin_ns"], event["end_ns"]
        if begin < start < finish or begin < end < finish:
            raise ValueError("kernel straddles a marker boundary")
        if start <= begin and finish <= end:
            if event["device"] != device:
                raise ValueError("target replay used another device")
            target.append({**event, "role": kernel_role(event["kernel_name"])})
    if any(row[4] == pid and start <= row[0] and row[1] <= end for row in copies if row not in markers):
        raise ValueError("unexpected copy in target replay")
    expected = [role for _ in range(50) for role in ("split_input_pack", "lightop_silu_mul")]
    if [event["role"] for event in target] != expected:
        counts = dict(Counter(event["role"] for event in target))
        raise ValueError("target must contain fifty ordered pack/LightOP pairs; observed " + repr(counts))
    if len({event["queue"] for event in target}) != 1:
        raise ValueError("target did not use one GPU queue")
    return target, {"begin_ns": start, "end_ns": end, "marker_bytes": list(MARKERS),
                    "target_kernel_events": len(target), "roles": dict(Counter(e["role"] for e in target))}


def review(directory):
    directory = Path(directory)
    frozen, worker, identity, actual, prep = [read(directory / name) for name in
        ("plan.json", "profile_worker.json", "identity.json", "runtime.json", "preparation.json")]
    local = plan()
    # A later docs/report-only commit must not invalidate an unchanged collector.
    local["collector"]["git_revision"] = frozen["collector"]["git_revision"]
    if frozen != local:
        raise ValueError("trace plan or collector sources differ from local review code")
    if (worker["scope"] != SCOPE or worker["complete"] is not True or worker["target_replays"] != 1
            or worker["calls_per_replay"] != 50 or worker["marker_copy_bytes"] != list(MARKERS)
            or worker["device_index"] != 0 or worker["timing_labels_emitted"] is not False
            or worker["automatic_admission"] is not False or prep["call_count"] != 50
            or worker["collector"] != frozen["collector"] or identity["collector"] != frozen["collector"]):
        raise ValueError("worker scope, source identity or completion mismatch")
    if worker["checks"] != {"outputs_checked": 50, "references_per_output": 2,
                            "input_bytes_unchanged": True, "outputs_independent": True}:
        raise ValueError("target replay correctness/alias checks incomplete")
    placed = load_placed_op_graph(directory / "placed.json", registry=adapter_registry())
    keys = LightopKernelKeys(placed, identity["group_index"], prep["one_call"], actual)
    key, gaps = keys.describe(placed.plan.groups[identity["group_index"]])
    if (gaps or key != identity["kernel_key"] or identity["identity_gaps"]
            or key_id(key) != identity["kernel_key_id"] or key_id(key) != worker["kernel_key_id"]
            or key_id(key) != frozen["expected_kernel_key_id"]):
        raise ValueError("trace Kernel identity differs from the smoked implementation/case")
    contract = measurement_contract(actual["environment"])
    if (contract != identity["smoke_contract"]
            or measurement_contract_id(contract) != identity["smoke_contract_id"]
            or measurement_contract_id(contract) != frozen["expected_smoke_contract_id"]):
        raise ValueError("trace environment or adapter recipe differs from Smoke")
    databases, summaries = list(directory.glob("*.db")), list(directory.glob("*.hipkernel.csv"))
    if len(databases) != 1 or len(summaries) != 1:
        raise ValueError("expected one hipprof SQLite database and kernel CSV")
    events, copies = database_events(databases[0])
    csv_consistent(summaries[0], events)
    target, region = target_region(events, copies, worker["pid"], worker["device_index"])
    return {"scope": SCOPE, "dispatch_observed": True, "kernel_key_id": key_id(key),
            "smoke_contract_id": measurement_contract_id(contract), "region": region,
            "database_csv_consistent": True, "all_process_kernel_events": len(events),
            "reviewer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "validation_and_warmup_excluded": True, "target_correctness": worker["checks"],
            "trace_file_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (databases[0], summaries[0])},
            "timing_labels_emitted": False, "usable_as_latency_label": False, "automatic_admission": False}, target
