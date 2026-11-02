"""CPU ordering checks for v3; fixture times are never device measurements."""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from calibration.ffn_b4 import execute_formal


class FakeRuntime:
    def __init__(self, root):
        self.root, self.context, self.cases = root, {}, {}
        self.profiled, self.timings = False, 0
        self.raw_at_trace = {}

    def anchor(self, gpu): return FakeCase(self, "anchor"+str(gpu))
    def release(self, group): pass
    def snapshots(self): return {}
    def checkpoint(self, *args, **kwargs): pass
    def make(self, row):
        assert not row.startswith("diagnostic/"), "blocked chain must never be constructed"
        self.cases[row] = FakeCase(self, row)
        return self.cases[row]


class FakeCase:
    def __init__(self, runtime, row):
        self.runtime, self.row, self.identity = runtime, row, {"fixture_only": row}
        self.replays = 0
    def capture(self): pass
    def replay(self, index):
        assert not self.runtime.profiled, "timing occurred after Profiler"
        self.runtime.timings += 1
        self.replays += 1
        return {"elapsed_ms": .5}, {"fixture_numeric": True}
    def trace(self, path):
        assert all(case.replays == 17 for case in self.runtime.cases.values())
        self.runtime.profiled = True
        for row in self.runtime.cases:
            path = self.runtime.root/row.replace("/", "_").replace(">", "to")/"samples.json"
            self.runtime.raw_at_trace[row] = path.read_bytes()
        return {"device_copy_trace_observed": True}


def run(blocked):
    rows = ["silu_mul/native@gpu0", "silu_mul/fused@gpu0"]
    if blocked: rows += ["diagnostic/down0-copy524288-gate1"]
    frozen = {"active_recipe": {"recipe_id": "fixture"},
              "policy": {"formal": {"blocked_rows": {rows[-1]: {"reason": "fixture reviewed crash"}} if blocked else {}}},
              "matrix": {"core_rows": [{"row_id": r, "pair_key": None} for r in rows], "diagnostics": [],
                         "round_groups": [{"group_id": "fixture", "members": rows, "roles": rows,
                                           "orders": [rows], "anchor_devices": ["gpu0"]}]}}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        runtime = FakeRuntime(root)
        with patch("calibration.ffn_b4.contract", return_value={"fixture": True}), \
             patch("calibration.ffn_b4.measurement_contract_id", return_value="fixture"):
            result = execute_formal(runtime, frozen, root, group_id="fixture")
        for row, original in runtime.raw_at_trace.items():
            assert (root/row.replace("/", "_").replace(">", "to")/"samples.json").read_bytes() == original
            samples = json.loads(original)
            assert len(samples) == 7 and all(s["path_evidence_state"] == "pending_post_timing_trace" for s in samples)
        assert runtime.profiled
        return result


def test_every_arm_and_anchor_finishes_before_trace_without_rewriting_samples():
    result = run(False)
    assert all(r["status"] == "quality_passed" for r in result["cells"].values())
    assert all(r["path_evidence_phase"] == "after_all_group_timing" for r in result["cells"].values())


def test_known_crashing_chain_is_blocked_before_construction_components_still_measured():
    result = run(True)
    blocked = result["cells"]["diagnostic/down0-copy524288-gate1"]
    assert blocked["execution_status"] == "blocked_backend_capture" and blocked["attempted_calls"] == 0
    assert result["quality_failed"] == 1 and result["recorded_cells"] == 3
