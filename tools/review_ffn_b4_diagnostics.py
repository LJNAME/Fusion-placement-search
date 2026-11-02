"""Recompute diagnostic summaries without importing or executing returned code."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

from calibration.ffn_b4 import save

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def review(root):
    root = Path(root)
    plan, summary = read(root/"plan.json"), read(root/"summary.json")
    checks = {}
    for name, expected in plan["source_sha256"].items():
        blob = subprocess.run(["git", "-C", str(ROOT), "show", "10c168c:"+name], capture_output=True, check=True).stdout
        checks[name] = hashlib.sha256(blob).hexdigest() == expected
    assert all(checks.values()) and not summary["formal_measurements"] and not summary["automatic_admission"]
    timing = {}
    for name in ("timing_control", "timing_profile"):
        record = read(root/name/"result.json")
        assert summary["cases"][name]["returncode"] == 0
        phases = {}
        for phase in ("before", "after"):
            samples = record[phase]
            assert len(samples) == 3
            assert all(s["calls"] == 50 and not s["formal_sample"] and all(s["checks"].values()) for s in samples)
            assert all(s["event_device"] == s["end_device"] == "cuda:0" for s in samples)
            phases[phase] = {key: statistics.median(s[key] for s in samples) for key in
                             ("event_batch_ms", "event_us_per_call", "all_device_completion_wall_us", "extra_device_sync_us")}
        phases["event_before_over_after"] = phases["before"]["event_batch_ms"]/phases["after"]["event_batch_ms"]
        phases["wall_before_over_after"] = phases["before"]["all_device_completion_wall_us"]/phases["after"]["all_device_completion_wall_us"]
        timing[name] = phases
    profile = read(root/"timing_profile/result.json")
    assert profile["intervention"]["kernel_names"] == ["_silu_mul_kernel"]*50
    hardware = [read(root/name/"identity.json")["devices"]["0"] for name in timing]
    assert hardware[0] == hardware[1]
    chain = {}
    for name in ("chain_allocating_1", "chain_preallocated_1"):
        assert summary["cases"][name]["returncode"] == -11
        stage = read(root/name/"last_stage.json")
        log = (root/name/"worker.log").read_text(encoding="utf-8")
        assert stage["stage"] == "capture_body_returned" and "in capture_end" in log
        chain[name] = {"returncode": -11, "calls": 1, "last_stage": stage,
                       "python_fault_stack": log.split("Extension modules:")[0].strip()}
    assert summary["cases"]["chain_preallocated_50"]["worker_attempts"] == 0
    archive = root.parent/(root.name+".tar.gz")
    return {"source_ref": "10c168c", "source_checks": checks,
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "timing": timing, "same_gpu0_environment": hardware[0],
            "profiled_fused_kernel_count": 50, "chain": chain,
            "chain_preallocated_50": summary["cases"]["chain_preallocated_50"],
            "conclusions": {"profiled_existing_graph_timing_state_requires_isolation": True,
                            "synchronous_wall_is_not_kernel_profile": True,
                            "explicit_output_preallocation_did_not_resolve_capture_end_crash": True,
                            "backend_internal_root_cause_known": False},
            "new_accepted_profiles": 0,
            "raw_file_sha256": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(root.rglob("*")) if p.is_file()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.root)
    save(args.output, result)
    print(json.dumps({"timing": result["timing"], "source_checks": result["source_checks"], "new_accepted_profiles": 0}))


if __name__ == "__main__":
    main()
