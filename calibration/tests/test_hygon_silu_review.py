"""Synthetic hipprof decoding and unit checks; not GPU performance evidence."""
import csv
import sqlite3

import pytest

from tools.review_hygon_silu_trial import profile_evidence, benchmark_evidence
from calibration.hygon_silu_trial import ARMS, SCOPE


def fixture(directory, arm, mutation=None):
    names = (["void act_and_mul_kernel_opt1<BFloat16>"] if arm == "lightop_packed" else [
        "elementwise_kernel_manual_unroll silu_kernel",
        "elementwise_kernel_manual_unroll BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16>"])
    count = 10 if mutation == "count" else 11
    db = sqlite3.connect(directory / "trace.db")
    db.execute("CREATE TABLE STR_TABLE (PID INTEGER,STR_ID INTEGER,STR_NAME TEXT,TYPE INTEGER)")
    db.execute("CREATE TABLE HIPOPS_fixture (BeginNs INTEGER,EndNs INTEGER,dev_id INTEGER,queue_id TEXT,Name TEXT,pid INTEGER,_Index INTEGER,DurationNs INTEGER)")
    for i, name in enumerate(names, 1):
        db.execute("INSERT INTO STR_TABLE VALUES (?,?,?,6)", (7, i, name))
    for i in range(count * len(names)):
        db.execute("INSERT INTO HIPOPS_fixture VALUES (?,?,?,?,?,?,?,?)",
            (i * 1000, i * 1000 + 500, int(mutation == "device"), "0", str(i % len(names) + 1), 7, i, 501 if mutation == "duration" else 500))
    db.commit(); db.close()
    with (directory / "result.hipkernel.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["Name", "Calls", "TotalDurationNs"])
        writer.writeheader()
        for name in names:
            writer.writerow({"Name": name, "Calls": count + int(mutation == "csv"), "TotalDurationNs": 500 * count})


@pytest.mark.parametrize("arm", ["native_packed", "lightop_packed"])
def test_kernel_string_ids_are_decoded_and_actual_operator_events_counted(tmp_path, arm):
    fixture(tmp_path, arm)
    result, rows = profile_evidence(tmp_path, arm)
    assert result["target_counts"] == ({"lightop_fused": 11} if arm == "lightop_packed" else {"native_silu": 11, "native_mul": 11})
    assert len(rows) == (11 if arm == "lightop_packed" else 22)
    assert result["database_csv_counts_and_durations_match"]


@pytest.mark.parametrize("mutation", ["count", "device", "duration", "csv"])
def test_wrong_counts_device_duration_or_csv_are_rejected(tmp_path, mutation):
    fixture(tmp_path, "lightop_packed", mutation)
    with pytest.raises(ValueError): profile_evidence(tmp_path, "lightop_packed")


def test_a_correct_batch_duration_cannot_be_mislabeled_as_us_per_call():
    value = {"scope": SCOPE, "complete": True, "profiler_active": False, "automatic_admission": False,
        "independent_batches_per_arm": 1,
        "arms": {n: {"calls": 100, "warmup_calls": 10, "numeric_ok": True, "input_bytes_unchanged": True,
            "gpu_batch_ms": 1., "gpu_us_per_call": 10., "synchronized_wall_us_per_call": 12.} for n in ARMS}}
    assert benchmark_evidence(value)["packed"]["native_over_lightop"] == 1
    value["arms"]["lightop_packed"]["gpu_us_per_call"] = 1000
    with pytest.raises(ValueError): benchmark_evidence(value)
