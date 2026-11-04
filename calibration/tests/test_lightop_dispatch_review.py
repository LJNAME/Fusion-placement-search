"""Synthetic SQLite/CSV trace fixtures; these never certify real GPU dispatch."""
from collections import Counter, defaultdict
import copy
import csv
import hashlib
import json
import sqlite3

import pytest

from tools import review_lightop_dispatch as audit
from calibration import lightop_dispatch as dispatch
from cost_model.fusion_kernel_profile import key_id
from cost_model.lightop_profile import LightopKernelKeys, measurement_contract
from cost_model.measurement_labels import measurement_contract_id
from cost_model.tests.test_lightop_profile import fixture_case
from simulator.placement_artifacts import save_placed_op_graph

PACK = "void at::native::CatArrayBatchedCopy_contig<OpaqueType<2u>, unsigned>()"
FUSED = "void at::native::act_and_mul_kernel_opt1<c10::BFloat16, &silu_kernel<c10::BFloat16>>()"


def trace_fixture(directory):
    path = directory / "fixture.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE STR_TABLE(PID,TYPE,STR_ID,STR_NAME)")
        db.executemany("INSERT INTO STR_TABLE VALUES(?,?,?,?)",
                       [(100,6,1,PACK),(100,6,2,FUSED),(100,6,3,"native_silu_kernel"),(99,6,1,"foreign_kernel")])
        db.execute("CREATE TABLE HIPOPS_fixture(BeginNs,EndNs,dev_id,queue_id,Name,pid,_Index,DurationNs)")
        rows = [(50,55,0,'7','3',100,0,5), (3100,3105,0,'7','3',100,101,5), (1500,1505,0,'8','1',99,102,5)]
        rows += [(1010+i*10,1015+i*10,0,'7',str(1+i%2),100,i+1,5) for i in range(100)]
        db.executemany("INSERT INTO HIPOPS_fixture VALUES(?,?,?,?,?,?,?,?)", rows)
        db.execute("CREATE TABLE HIPCOPY_fixture(BeginNs,EndNs,dev_id,queue_id,pid,Bytes)")
        db.executemany("INSERT INTO HIPCOPY_fixture VALUES(?,?,?,?,?,?)",[(1000,1001,0,'7',100,257),(3000,3001,0,'7',100,263)])
    events, copies = audit.database_events(path)
    counts, times = Counter(), defaultdict(int)
    for event in events:
        counts[event["kernel_name"]] += 1
        times[event["kernel_name"]] += event["duration_ns"]
    summary = directory / "fixture.hipkernel.csv"
    with summary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["Name", "Calls", "TotalDurationNs"])
        writer.writeheader()
        writer.writerows({"Name": name, "Calls": count, "TotalDurationNs": times[name]} for name,count in counts.items())
    return path, summary, events, copies


def test_pid_decoding_marker_isolation_and_csv_consistency_are_read_only(tmp_path):
    path, summary, events, copies = trace_fixture(tmp_path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    audit.csv_consistent(summary, events)
    target, region = audit.target_region(events, copies, 100, 0)
    assert len(target) == 100 and region["roles"] == {"split_input_pack":50,"lightop_silu_mul":50}
    assert all(event["pid"] == 100 and event["kernel_name"] != "native_silu_kernel" for event in target)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("failure", ["missing_fused", "unknown_kernel", "native_fallback", "wrong_dtype",
                                    "wrong_device", "duplicate_marker", "missing_marker", "reverse_markers",
                                    "straddle", "other_queue", "extra_copy"])
def test_incomplete_or_ambiguous_trace_is_not_certified(tmp_path, failure):
    _, _, events, copies = trace_fixture(tmp_path)
    one = next(event for event in events if event["kernel_name"] == FUSED)
    if failure == "missing_fused": events.remove(one)
    elif failure == "unknown_kernel": events.insert(2, {**one,"begin_ns":1004,"end_ns":1008,"kernel_name":"unknown"})
    elif failure == "native_fallback": one["kernel_name"] = "native_silu_kernel"
    elif failure == "wrong_dtype": one["kernel_name"] = FUSED.replace("BFloat16","Half")
    elif failure == "wrong_device": one["device"] = 1
    elif failure == "duplicate_marker": copies.append(copies[0])
    elif failure == "missing_marker": copies.pop()
    elif failure == "reverse_markers": copies[1]=(500,501,0,'7',100,263)
    elif failure == "straddle": one.update(begin_ns=999,end_ns=1002)
    elif failure == "other_queue": one["queue"] = "other"
    else: copies.append((2500,2501,0,'7',100,1024))
    with pytest.raises(ValueError):
        audit.target_region(events,copies,100,0)


def test_csv_mismatch_is_rejected(tmp_path):
    _, summary, events, _ = trace_fixture(tmp_path)
    events[0]["duration_ns"] += 1
    with pytest.raises(ValueError, match="disagrees"):
        audit.csv_consistent(summary,events)


def full_receipt(tmp_path, monkeypatch):
    placed,index,prep,actual = fixture_case()
    keys = LightopKernelKeys(placed,index,prep,actual)
    key,gaps = keys.describe(placed.plan.groups[index])
    contract = measurement_contract(actual["environment"])
    frozen = dispatch.plan()
    frozen.update(expected_kernel_key_id=key_id(key),expected_smoke_contract_id=measurement_contract_id(contract))
    monkeypatch.setattr(audit,"plan",lambda:copy.deepcopy(frozen))
    identity = {"kernel_key":key,"kernel_key_id":key_id(key),"identity_gaps":list(gaps),
                "smoke_contract":contract,"smoke_contract_id":measurement_contract_id(contract),
                "group_index":index,"collector":frozen["collector"]}
    worker = {"scope":dispatch.SCOPE,"complete":True,"pid":100,"device_index":0,"target_replays":1,
              "calls_per_replay":50,"marker_copy_bytes":[257,263],"timing_labels_emitted":False,
              "automatic_admission":False,"collector":frozen["collector"],"kernel_key_id":key_id(key),
              "checks":{"outputs_checked":50,"references_per_output":2,"input_bytes_unchanged":True,"outputs_independent":True}}
    receipt = tmp_path/"worker"
    receipt.mkdir()
    for name,value in [("plan.json",frozen),("identity.json",identity),("runtime.json",actual),
                       ("preparation.json",{"one_call":prep,"call_count":50}),("profile_worker.json",worker)]:
        dispatch.save(receipt/name,value)
    save_placed_op_graph(receipt/"placed.json",placed)
    trace_fixture(tmp_path)
    return worker,identity,frozen


def test_full_cpu_fixture_reconstructs_identity_without_emitting_latency_labels(tmp_path, monkeypatch):
    full_receipt(tmp_path,monkeypatch)
    report,target = audit.review(tmp_path)
    assert report["dispatch_observed"] and len(target) == 100
    assert not report["usable_as_latency_label"] and not report["automatic_admission"]


@pytest.mark.parametrize("source_changed", [False, True])
def test_later_commit_is_allowed_only_when_collector_sources_match(tmp_path,monkeypatch,source_changed):
    _,_,frozen = full_receipt(tmp_path,monkeypatch)
    local = copy.deepcopy(frozen)
    local["collector"]["git_revision"] = "later-documentation-commit"
    if source_changed:
        local["collector"]["source_sha256"]["calibration/lightop_dispatch.py"] = "changed-source"
    monkeypatch.setattr(audit,"plan",lambda:copy.deepcopy(local))
    if source_changed:
        with pytest.raises(ValueError,match="collector sources"):
            audit.review(tmp_path)
    else:
        assert audit.review(tmp_path)[0]["dispatch_observed"]


@pytest.mark.parametrize("failure",["worker_incomplete","numeric_failure","kernel_key","contract","collector"])
def test_trace_counts_do_not_override_bad_worker_or_identity(tmp_path,monkeypatch,failure):
    worker,identity,_ = full_receipt(tmp_path,monkeypatch)
    if failure == "worker_incomplete": worker["complete"] = False
    elif failure == "numeric_failure": worker["checks"]["input_bytes_unchanged"] = False
    elif failure == "kernel_key": identity["kernel_key_id"] = "other"
    elif failure == "contract": identity["smoke_contract_id"] = "other"
    else: worker["collector"] = {"source_sha256":{}}
    dispatch.save(tmp_path/"worker/profile_worker.json",worker)
    dispatch.save(tmp_path/"worker/identity.json",identity)
    with pytest.raises(ValueError): audit.review(tmp_path)
