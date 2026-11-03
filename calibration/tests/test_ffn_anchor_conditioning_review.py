"""CPU audit tests with synthetic recordings, not GPU performance evidence."""
import subprocess
from pathlib import Path

import pytest

from calibration import ffn_anchor_conditioning as collector
from calibration.tests.test_ffn_anchor_conditioning import synthetic
from calibration.tests import test_ffn_anchor_conditioning as fixture_module
from tools.review_ffn_anchor_conditioning import inventory, review


@pytest.fixture
def recording(tmp_path, monkeypatch):
    base = fixture_module.FakeRuntime
    class Runtime(base):
        def __init__(self, *args):
            super().__init__(*args)
            self.identity["peer_capabilities"] = {}
    monkeypatch.setattr(fixture_module, "FakeRuntime", Runtime)
    frozen = collector.plan()
    collector.save(tmp_path / "plan.json", frozen)
    collector.write_report(tmp_path, {c["id"]: {"status": "not_run"} for c in frozen["design"]["cells"]}, template=True)
    def run(command, **kwargs):
        cell = next(c for c in frozen["design"]["cells"] if c["id"] == command[command.index("--worker") + 1])
        directory = command[command.index("--output") + 1]
        synthetic(directory, cell, ticks_start=1000 * (1 + frozen["design"]["cells"].index(cell)),
                  observe_override={"fixture": {"value": None, "error": "synthetic"}})
        path = Path(directory)
        collector.save(path / "runtime_identity.json", collector.read(path / "identity.json")["identity"]["target"])
        return subprocess.CompletedProcess(command, 0)
    collector.collect(tmp_path, "unused", frozen, run_worker=run)
    return tmp_path


def test_complete_audit_rebuilds_all_reports_preserving_originals_and_roles(recording):
    original = inventory(recording)
    report, timeline, endpoints = review(recording)
    assert report["raw_file_sha256"] == original == inventory(recording)
    assert report["measured_replays"] == 196 and report["conditioning_replays"] == 28
    assert len(timeline) == 224 and len(endpoints) == 56
    assert sum(not row["primary_measurement"] for row in timeline) == 28
    assert report["primary_timed_function_invocations"] == 9800
    assert report["conditioning_timed_function_invocations"] == 1400
    # Conditioning is deliberately 999 us in the fixture; the target stays 20.
    assert all(c["target_median_us"] == 20 for c in report["cells"].values())
    assert not report["protocol_adopted"] and report["new_accepted_profiles"] == 0


@pytest.mark.parametrize("mutation", ["source", "runtime", "summary", "report"])
def test_audit_rejects_changes_outside_per_cell_sample_validator(recording, mutation):
    if mutation == "source":
        path = recording / "plan.json"
        value = collector.read(path)
        value["source_sha256"]["calibration/ffn_anchor_conditioning.py"] = "0" * 64
    elif mutation == "runtime":
        path = recording / "block1_A/runtime_identity.json"
        value = collector.read(path)
        value["actual_torch_build"]["torch_extension_sha256"] = "changed"
    elif mutation == "summary":
        path = recording / "summary.json"
        value = collector.read(path)
        value["cells"]["block1_A"]["target_median_us"] += 1
    else:
        path = recording / "report.json"
        value = collector.read(path)
        value["protocol_adopted"] = True
    collector.save(path, value)
    with pytest.raises(ValueError):
        review(recording)
