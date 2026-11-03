"""Run unmodified vendor test scripts and retain their own measurements."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from integration.lightop.catalog import catalog, check_installed

MATRIX = ROOT / "data/lightop_fusion_catalog_v1/vendor_jobs.json"
MANIFEST = ROOT / "data/hygon_lightop_wheel_v1/manifest.json"


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")


def plan():
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    return {"scope": matrix["scope"], "catalog_sha256": hashlib.sha256((ROOT / "data/lightop_fusion_catalog_v1/capabilities.json").read_bytes()).hexdigest(),
            "matrix_sha256": hashlib.sha256(MATRIX.read_bytes()).hexdigest(), "wheel_sha256": matrix["wheel_sha256"],
            "jobs": matrix["jobs"], "one_attempt_per_job": True, "script_bytes_modified": False,
            "measurement_method": "vendor defaults and explicit vendor CLI variants; per-script contracts",
            "automatic_profile_admission": False, "cross_batch_uuid_check": False}


def extract_vendor_tests(wheel, target):
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    if hashlib.sha256(Path(wheel).read_bytes()).hexdigest() != matrix["wheel_sha256"]:
        raise ValueError("Wheel SHA256 mismatch")
    with zipfile.ZipFile(wheel) as z:
        for relative, expected in matrix["test_tree_sha256"].items():
            parts = PurePosixPath(relative).parts
            if not parts or parts[0] != "test" or any(p in ("..", "") for p in parts):
                raise ValueError("Unsafe vendor test path")
            raw = z.read(relative)
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError("Vendor test source mismatch: " + relative)
            path = target.joinpath(*parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)


def output_excerpt(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    return {"timing_lines": [s for s in lines if any(k in s.lower() for k in (" us", " ms", "time is", "\u8017\u65f6", "latency", "speedup", "gb/s"))],
            "validation_lines": [s for s in lines if s.strip() in ("True", "False") or any(k in s.lower() for k in ("pass", "fail", "allclose", "correctness", "\u7cbe\u5ea6", "\u901a\u8fc7\u7387", "\u5f02\u5e38"))],
            "validation_status": "vendor_output_requires_review"}


def run_job(job, output, vendor_tree, launcher, timeout=None):
    directory = output / "jobs" / job["id"]
    directory.mkdir(parents=True)
    # A private copy preserves sibling imports and isolates vendor-generated files.
    shutil.copytree(vendor_tree, directory / "vendor")
    script = (directory / "vendor" / job["script"]).resolve()
    command = [launcher, "-X", "faulthandler", "-u", str(script), *job["args"]]
    result = {"id": job["id"], "script": job["script"], "args": job["args"],
              "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(), "command": command,
              "capability_references": job["capability_references"], "attempts": 1,
              "started_unix": time.time(), "status": "not_started", "returncode": None,
              "gpu_devices": "vendor script default; see logs", "validation_status": "vendor_output_requires_review"}
    if job["launch_mode"] != "script":
        result.update(status="no_main_requires_review", attempts=0)
    else:
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONFAULTHANDLER="1")
        with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
            try:
                process = subprocess.run(command, cwd=script.parent, env=env, stdout=stdout, stderr=stderr, timeout=timeout)
                result.update(returncode=process.returncode, status="process_completed" if process.returncode == 0 else "process_failed")
            except subprocess.TimeoutExpired:
                result.update(status="timeout")
            except Exception:
                result.update(status="launch_failed", traceback=traceback.format_exc())
        result.update(output_excerpt(directory / "stdout.log"))
    result["finished_unix"] = time.time()
    save(directory / "result.json", result)
    return result


def run(output, wheel, launcher, selected=None, timeout=None):
    frozen = plan()
    jobs = frozen["jobs"]
    if selected:
        unknown = set(selected) - {j["id"] for j in jobs}
        if unknown:
            raise ValueError("Unknown jobs: " + ", ".join(sorted(unknown)))
        jobs = [j for j in jobs if j["id"] in selected]
    frozen["selected_jobs"] = [j["id"] for j in jobs]
    save(output / "plan.json", frozen)
    extract_vendor_tests(wheel, output / "vendor_original")
    identity = check_installed()
    identity.update(python=sys.version, executable=sys.executable, packages={})
    for name in ("torch", "triton", "vllm", "pytest"):
        try:
            identity["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            identity["packages"][name] = None
    identity["visible_devices"] = {k: os.environ.get(k) for k in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")}
    save(output / "identity.json", identity)
    results = []
    for job in jobs:
        print(json.dumps({"job_start": job["id"]}), flush=True)
        result = run_job(job, output, output / "vendor_original", launcher, timeout)
        results.append(result)
        save(output / "results.json", results)
        print(json.dumps({"job": job["id"], "status": result["status"], "returncode": result["returncode"]}), flush=True)
    summary = {"scope": frozen["scope"], "selected_jobs": len(jobs), "attempted_jobs": len(results),
               "process_completed": sum(r["status"] == "process_completed" for r in results),
               "other_results": sum(r["status"] != "process_completed" for r in results),
               "correctness_and_timing": "review per-job vendor stdout and generated artifacts; exit code alone is insufficient",
               "implemented_interfaces": catalog()["fusion_interface_count"], "automatic_profile_admission": False}
    save(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if not summary["other_results"] else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("plan", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--launcher", default=sys.executable)
    parser.add_argument("--jobs", nargs="+")
    parser.add_argument("--timeout-seconds", type=float, default=None)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        if args.phase == "plan":
            save(args.output / "plan.json", plan())
            print(json.dumps({"phase": "plan", "jobs": len(plan()["jobs"])}))
            return 0
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        wheel = args.wheel or Path.home() / "private_data/LJ/lightop_wheels" / manifest["filename"]
        return run(args.output, wheel, args.launcher, args.jobs, args.timeout_seconds)
    except Exception:
        save(args.output / "failure.json", {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    sys.exit(main())
