"""Temporarily expose the verified HyHAL SDK to the original vendor benchmarks."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import sysconfig
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import prepare_lightop_bench_followup as prepare
from tools.recover_lightop_smi import run_child

MANIFEST = ROOT / "data/lightop_host_smi_v1/manifest.json"


def manifest():
    data = json.loads(MANIFEST.read_text())
    report = ROOT / "data/lightop_smi_diagnostics01_review/report.json"
    if prepare.sha(report) != data["diagnostics_report_sha256"]:
        raise ValueError("Host SDK evidence changed")
    return data


def check_sources(data):
    package = Path(data["sdk_package_path"])
    for relative, digest in data["python_sources_sha256"].items():
        if prepare.sha(package / relative) != digest:
            raise ValueError("Host SMI wrapper differs from the verified source: " + relative)
    if prepare.sha(data["library_path"]) != data["library_sha256"]:
        raise ValueError("Host SMI library differs from the verified source")


@contextmanager
def sdk_search_path(site_packages, package, output):
    """Python startup handles .pth in every child, including vendor subprocesses."""
    path = site_packages / "fusion_lightop_host_smi.pth"
    content = str(package.parent) + "\n"
    # Never overwrite a pre-existing environment binding.
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
    try:
        prepare.save(output / "path_binding.json", {"path": str(path), "content": content,
            "lifetime": "removed in finally after probe or benchmark completion"})
        yield path
    finally:
        if path.read_text() != content:
            raise ValueError("SMI path binding changed during the run; retained for inspection")
        path.unlink()
        prepare.save(output / "path_cleanup.json", {"removed": True, "path": str(path)})


def probe_host(output):
    data = manifest()
    check_sources(data)
    import amdsmi
    import amdsmi.amdsmi_wrapper as wrapper
    from amdsmi._version import __version__
    if Path(amdsmi.__file__).resolve().parent != Path(data["sdk_package_path"]).resolve():
        raise ValueError("SMI import is shadowed by another package")
    actual_library = Path(wrapper._libraries["libamd_smi.so"]._name).resolve()
    if prepare.sha(actual_library) != data["library_sha256"] or __version__ != data["sdk_version"]:
        raise ValueError("Loaded SMI native library or wrapper version differs")
    amdsmi.amdsmi_init()
    try:
        count = len(amdsmi.amdsmi_get_processor_handles())
    finally:
        amdsmi.amdsmi_shut_down()
    if count < 1:
        raise ValueError("SMI did not enumerate any processors")
    prepare.save(output / "host_smi.json", {"module": str(amdsmi.__file__), "version": __version__,
        "loaded_library": str(actual_library), "library_sha256": data["library_sha256"], "processor_count": count})
    prepare.probe(output)


def run(output, wheel):
    data = manifest()
    before = prepare.snapshot()
    prepare.save(output / "before.json", before)
    if sys.platform != "linux" or sys.version_info[:2] != (3, 10):
        raise ValueError("Requires the existing Linux Python3.10 vendor launcher")
    if prepare.protected(before, prepare.manifest()) != data["protected_versions"]:
        raise ValueError("Protected numeric versions differ from the diagnostic receipt")
    if prepare.version("amdsmi") is not None:
        raise ValueError("Expected the previous conflicting amdsmi distribution to remain removed")
    check_sources(data)
    prepare.save(output / "host_smi_plan.json", data)
    site_packages = Path(sysconfig.get_path("purelib")).resolve()
    if not site_packages.is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError("Refuse to write a path binding outside the private Python environment")
    preflight = output / "preparation"
    preflight.mkdir()
    code = 1
    try:
        with sdk_search_path(site_packages, Path(data["sdk_package_path"]), output):
            probe = run_child(preflight, "probe", [str(Path(__file__).resolve()),
                "probe", "--output", str(preflight)], timeout=180)
            if probe["returncode"] != 0 or not (preflight / "probe.json").is_file():
                raise RuntimeError("Host SMI/baseline probe failed; see preparation/probe.log and probe.process.json; no benchmarks started")
            command = [sys.executable, "-X", "faulthandler", str(ROOT / "calibration/lightop_vendor_suite.py"),
                "run", "--output", str(output / "measurements"), "--launcher", sys.executable,
                "--jobs", *prepare.manifest()["jobs"]]
            if wheel is not None:
                command.extend(["--wheel", str(wheel.resolve())])
            code = subprocess.run(command).returncode
            prepare.save(output / "summary.json", {"scope": "original_vendor_dependency_followup_using_host_smi",
                "baseline_ready": True, "benchmark_runner_returncode": code, "automatic_profile_admission": False})
    finally:
        after = prepare.snapshot()
        prepare.save(output / "after.json", after)
        if before != after:
            raise ValueError("Package state changed during the host SDK benchmark run")
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("plan", "run", "probe"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheel", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.phase != "probe":
        output.mkdir(parents=True, exist_ok=False)
    try:
        if args.phase == "plan":
            prepare.save(output / "plan.json", {"host_smi": manifest(), "jobs": prepare.manifest()["jobs"]})
            return 0
        if args.phase == "probe":
            probe_host(output)
            return 0
        return run(output, args.wheel)
    except Exception:
        prepare.save(output / ("failure_" + args.phase + ".json"), {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
