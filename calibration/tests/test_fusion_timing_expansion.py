"""CPU fixtures never constitute GPU evidence."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from calibration import fusion_timing_expansion as f


def rows(values=None, outputs=2):
    values = values or [1.] * 7
    return [{"round": i, "timing": {"elapsed_ms_per_50_calls": v, "error": None,
        "event_boundary": {"device": "cuda:0", "start_device": "cuda:0", "end_device": "cuda:0",
                           "calls_per_replay": 50, "stream": 123}},
        "checks": {"calls_checked": 50, "outputs_per_call": outputs, "references_per_output": 2,
                   "input_bytes_unchanged": True, "private_buffers_independent": True,
                   "all_outputs_finite": True}} for i, v in enumerate(values)]


class Expansion(unittest.TestCase):
    def test_three_cases_match_graph_and_passed_vendor_apis(self):
        frozen = f.plan()
        self.assertEqual(len(frozen["cases"]), 3)
        self.assertEqual(frozen["cases"]["silu_mul_3b"]["shape"][-1], 11008)
        self.assertEqual(frozen["cases"]["add_rmsnorm_3b"]["shape"][-1], 2048)
        self.assertEqual(frozen["cases"]["rmsnorm_rope_qwen3"]["key_shape"], [1, 256, 8, 128])
        self.assertFalse(frozen["cases"]["rmsnorm_rope_qwen3"]["rotary"]["graph_semantic_equivalence_verified"])
        self.assertFalse(frozen["old_native_and_kerv_remeasurement"])
        for c in frozen["cases"].values():
            self.assertTrue(c["prior_vendor_evidence"])
            self.assertFalse(c["automatic_search_registration"])

    def test_integer_positions_have_actual_identity(self):
        obs = f.observe_tensor(torch.arange(4, dtype=torch.int64).view(1, 4))
        self.assertEqual((obs["dtype"], obs["shape"], obs["strides"]), ("int64", [1, 4], [4, 1]))

    def test_neox_rotation_uses_half_split(self):
        x = torch.tensor([[[[1., 2., 3., 4.]]]])
        cache = torch.tensor([[0., 0., 1., 1.]])
        self.assertTrue(torch.equal(f.rotate(torch, x, cache), torch.tensor([[[[-3., -4., 1., 2.]]]])))

    def test_private_buffers_refresh_and_sources_remain_readonly(self):
        for case_id, spec in f.CASES.items():
            with self.subTest(case=case_id), torch.no_grad():
                inputs = f.build_inputs(torch, case_id, torch.device("cpu"), 23)
                snaps = {n: t.view(torch.uint8).clone() for n, t in inputs.items()}
                refs = f.references(torch, case_id, inputs)
                if case_id == "silu_mul_3b":
                    def op(out, packed):
                        a, b = packed.chunk(2, -1)
                        out.copy_(torch.nn.functional.silu(a) * b)
                elif case_id == "add_rmsnorm_3b":
                    def op(a, b, weight, eps):
                        b.copy_(a + b)
                        a.copy_(torch.nn.functional.rms_norm(b, (b.shape[-1],), weight, eps))
                else:
                    def op(pos, q, k, size, cache, neox, wq, wk, rq, rk, eps):
                        for v, w in ((q, wq), (k, wk)):
                            norm = v.float() * torch.rsqrt(v.float().square().mean(-1, keepdim=True) + eps) * w.float()
                            v.copy_(f.rotate(torch, norm, cache, neox))
                calls = [f.PreparedCall(torch, case_id, inputs, op) for _ in range(2)]
                for _ in range(2):
                    for call in calls:
                        call.invalidate(); call()
                    checks = f.verify_outputs(torch, spec, calls, inputs, snaps, refs)
                    self.assertTrue(checks["input_bytes_unchanged"])
                calls[0].invalidate()
                with self.assertRaises(AssertionError):
                    f.verify_outputs(torch, spec, calls, inputs, snaps, refs)

    def test_source_mutation_rejected(self):
        spec = f.CASES["add_rmsnorm_3b"]
        inp = f.build_inputs(torch, "add_rmsnorm_3b", torch.device("cpu"), 0)
        snap = {n: t.view(torch.uint8).clone() for n, t in inp.items()}
        refs = f.references(torch, "add_rmsnorm_3b", inp)
        inp["x"][0, 0, 0] = 999
        with self.assertRaisesRegex(AssertionError, "read-only"):
            f.verify_outputs(torch, spec, [], inp, snap, refs)

    def test_quality_exact_seven_no_replacement(self):
        self.assertEqual(f.summarize(rows())["statistics"]["median_us"], 20.)
        self.assertEqual(f.summarize(rows([1, 1, 1, 2, 2, 2, 2]))["status"], "quality_failed")
        for sample in (rows()[:-1], rows() + rows()[:1], list(reversed(rows()))):
            with self.assertRaises(ValueError): f.summarize(sample)

    def test_wrong_boundary_or_missing_output_rejected(self):
        sample = rows(); sample[0]["timing"]["event_boundary"]["end_device"] = "cuda:1"
        with self.assertRaises(ValueError): f.summarize(sample)
        with self.assertRaises(ValueError): f.summarize(rows(outputs=1), outputs_per_call=2)
        sample = rows(); sample[0]["checks"]["input_bytes_unchanged"] = False
        with self.assertRaises(ValueError): f.summarize(sample)

    def test_timing_failure_has_no_latency(self):
        sample = rows(); sample[0]["timing"]["error"] = "Event unavailable"
        q = f.summarize(sample)
        self.assertEqual(q["status"], "quality_failed")
        self.assertIsNone(q["statistics"])

    def test_cpu_cannot_emit_gpu_measurement(self):
        with tempfile.TemporaryDirectory() as d, patch.object(torch.version, "hip", None):
            with self.assertRaisesRegex(RuntimeError, "HIP GPU required"):
                f.run_case("silu_mul_3b", Path(d), {})
            self.assertFalse((Path(d) / "result.json").exists())

    def test_isolated_processes_continue_after_one_crash(self):
        commands = []
        def fake(cmd, **kwargs):
            commands.append(cmd)
            folder = Path(cmd[cmd.index("--output") + 1])
            if len(commands) == 1: return SimpleNamespace(returncode=-11)
            (folder / "result.json").write_text(json.dumps({"status": "quality_passed"}), encoding="utf-8")
            return SimpleNamespace(returncode=0)
        with tempfile.TemporaryDirectory() as d, patch.object(f.subprocess, "run", side_effect=fake):
            self.assertFalse(f.run(Path(d), {"cases": f.CASES}))
            result = json.loads((Path(d) / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(commands), 3)
            self.assertEqual(commands[0][commands[0].index("-m") + 1], "calibration.fusion_timing_expansion")
            self.assertEqual(result["cases"]["silu_mul_3b"]["status"], "runtime_failed")
            self.assertEqual(result["cases"]["rmsnorm_rope_qwen3"]["status"], "quality_passed")

    def fixture(self, root):
        frozen = json.loads(json.dumps(f.plan()))
        f.save(root / "plan.json", frozen)
        statuses = {}
        for case_id, spec in frozen["cases"].items():
            folder = root / case_id; folder.mkdir()
            sample = rows(outputs=1 if case_id == "silu_mul_3b" else 2)
            identity = {"case": spec, "runtime": {
                "native_sha256": spec["implementation"]["native_sha256"],
                "lightop_version": spec["implementation"]["version"],
                "hip": "synthetic_fixture_not_GPU_evidence", "device": {"architecture": "gfx936"}}}
            for row in sample: f.save(folder / f"round_{row['round']}.json", row)
            f.save(folder / "smoke.json", {"passed": True, "records": [{}] * 3})
            f.save(folder / "result.json", {"case": case_id, "identity": identity,
                "execution_key_id": f.key_id(identity), "gpu_executed": True,
                "contract": frozen["contract"], "quality": f.summarize(sample), "status": "quality_passed"})
            statuses[case_id] = {"returncode": 0, "status": "quality_passed"}
        f.save(root / "summary.json", {"cases": statuses})

    def test_cpu_review_recomputes_quality_and_never_admits(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.fixture(root)
            reviewed = f.review(root, root / "review")
            self.assertTrue(all(r["status"] == "quality_passed" and not r["accepted"] for r in reviewed.values()))

    def test_modified_round_cannot_keep_original_summary(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.fixture(root)
            p = root / "silu_mul_3b/round_0.json"
            changed = json.loads(p.read_text(encoding="utf-8"))
            changed["timing"]["elapsed_ms_per_50_calls"] = 100.
            f.save(p, changed)
            with self.assertRaisesRegex(ValueError, "result differs"):
                f.review(root, root / "review")


if __name__ == "__main__":
    unittest.main()
