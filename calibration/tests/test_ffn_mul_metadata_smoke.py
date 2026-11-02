"""CPU-only checks of the metadata smoke boundary, not GPU evidence."""
import copy
import subprocess
from types import SimpleNamespace

import pytest

from calibration import ffn_anchor_control as pilot
from calibration import ffn_mul_metadata_smoke as smoke
from calibration.tests.test_ffn_anchor_control import FakeRuntime, synthetic_cell
from cost_model.fusion_kernel_profile import key_id


def completed_pilot(root):
    root.mkdir()
    frozen = pilot.pilot_plan()
    pilot.save(root / "plan.json", frozen)
    def run(command, **kwargs):
        name = command[command.index("--worker") + 1]
        cell = next(c for c in pilot.PROTOCOL["cells"] if c["id"] == name)
        synthetic_cell(command[command.index("--output") + 1], cell)
        return subprocess.CompletedProcess(command, 0)
    pilot.collect(root, "fixture-unused", frozen, run_worker=run)
    return smoke.verify_pilot_for_smoke(root)


def metadata():
    def meta(ptr):
        return {"shape": [1, 256, 3072], "dtype": "torch.bfloat16", "strides": [786432, 3072, 1],
                "device": "cuda:0", "data_ptr": ptr, "layout_source": "runtime Tensor observation"}
    identity = copy.deepcopy(FakeRuntime().identity)
    identity.update(inputs={"left": [meta(100 + i) for i in range(50)],
                            "gates": [meta(200 + i) for i in range(50)], "up": meta(300)},
                    output_layouts=[meta(400 + i) for i in range(50)])
    return identity


class SmokeRuntime:
    def __init__(self):
        self.identity = metadata()
        self.progress, self.made = None, []
    def activation(self, gpu, **kwargs):
        assert gpu == 0 and kwargs == {"mul_smoke_only": True}
        self.made.append("mul")
        case = SimpleNamespace(identity=self.identity, capture=lambda: None)
        def replay(index, diagnostic):
            assert diagnostic
            return {"elapsed_ms": None if index == 0 else 1e12, "error": "fixture timing unavailable", "used_for_pass_fail": False}, {name: True for name in smoke.REQUIRED_CHECKS}
        case.replay = replay
        return {"gated_mul@gpu0": case}
    def checkpoint(self, *a, **kwargs): pass
    def snapshots(self): return {"0": {"fixture": True}}


def signature(identity):
    return key_id({"device": identity["devices"]["0"], "build": identity["actual_torch_build"],
                   "loaded_source": identity["loaded_kerv_source"]})


def test_smoke_requires_all_original_pilot_evidence_before_starting_gpu(tmp_path):
    root = tmp_path / "pilot"
    prerequisite = completed_pilot(root)
    assert prerequisite["runtime_signature"] == signature(metadata())
    summary = pilot.read(root / "summary.json")
    summary["pilot_passed"] = False
    pilot.save(root / "summary.json", summary)
    with pytest.raises(ValueError, match="complete current pilot"):
        smoke.verify_pilot_for_smoke(root)


def test_smoke_timings_never_affect_boolean_pass_and_metadata_is_persisted(tmp_path):
    runtime = SmokeRuntime()
    prerequisite = {"runtime_signature": signature(runtime.identity), "plan_id": "fixture"}
    result = smoke.execute_smoke(runtime, tmp_path, prerequisite)
    checked = smoke.review_smoke(tmp_path, prerequisite)
    assert result == checked and result["status"] == "passed" and runtime.made == ["mul"]
    assert not result["timing_used_for_pass_fail"] and not result["automatic_admission"]
    assert not result["followup_matrix_open"]
    assert pilot.read(tmp_path / "identity.json")["identity"]["inputs"]["left"]
    replays = pilot.read(tmp_path / "replays.json")
    del replays[0]["invariants"]["mul_left_bytes_unchanged"]
    pilot.save(tmp_path / "replays.json", replays)
    with pytest.raises(ValueError, match="replay evidence"):
        smoke.review_smoke(tmp_path, prerequisite)


@pytest.mark.parametrize("mutation", ["gate_substitution", "missing_stride", "output_alias"])
def test_actual_operand_proof_rejects_borrowed_or_missing_metadata(mutation):
    identity = metadata()
    if mutation == "gate_substitution": identity["inputs"]["left"] = identity["inputs"]["gates"]
    elif mutation == "missing_stride": identity["inputs"]["left"][0]["strides"] = None
    else: identity["output_layouts"] = identity["inputs"]["left"]
    with pytest.raises(ValueError): smoke.metadata_proof(identity)


def test_crashed_smoke_is_not_restarted_and_preserves_cpu_failure_package(tmp_path):
    root, output = tmp_path / "pilot", tmp_path / "smoke"
    completed_pilot(root)
    output.mkdir()
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        directory = output / "worker"
        pilot.save(directory / "last_stage.json", {"stage": "capture_end", "pool_config": {"fixture": True}, "memory_before_capture": {"0": {"allocated": 123}}})
        return subprocess.CompletedProcess(command, -11)
    result = smoke.collect(output, root, "fixture-unused", smoke.smoke_plan(root), run_worker=run)
    failure = pilot.read(output / "worker/parent_failure.json")
    assert len(calls) == 1 and result["status"] == "failed" and result["returncode"] == -11
    assert failure["pool_config"] == {"fixture": True} and failure["memory"]["0"]["allocated"] == 123
    assert "traceback" in failure and not result["followup_matrix_open"]


def test_successful_separate_smoke_still_does_not_open_formal_matrix(tmp_path):
    root, output = tmp_path / "pilot", tmp_path / "smoke"
    prerequisite = completed_pilot(root)
    output.mkdir()
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        smoke.execute_smoke(SmokeRuntime(), output / "worker", prerequisite)
        return subprocess.CompletedProcess(command, 0)
    result = smoke.collect(output, root, "fixture-unused", smoke.smoke_plan(root), run_worker=run)
    assert len(calls) == 1 and result["status"] == "passed"
    assert not result["automatic_admission"] and not result["followup_matrix_open"]
