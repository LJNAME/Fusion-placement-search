"""Undo only receipt03's newly introduced SMI wrapper; inspect the host SDK."""
import argparse
import ctypes
import glob
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import prepare_lightop_bench_followup as prepare


def validate_receipt(receipt, current, executable, data):
    before = json.loads((receipt / "before.json").read_text())
    after = json.loads((receipt / "after.json").read_text())
    if prepare.sha(receipt / "plan.json") != prepare.sha(prepare.MANIFEST):
        raise ValueError("Receipt plan differs from the frozen followup")
    if before["executable"] != executable or after["executable"] != executable:
        raise ValueError("Use the same private Python that performed receipt03's install")
    if before["packages"].get("amdsmi") is not None:
        raise ValueError("Refuse to remove an SMI package that existed before this receipt")
    bad = next(w for w in data["wheels"] if w["distribution"] == "amdsmi")
    if after["packages"].get("amdsmi") != bad["version"] or current["packages"].get("amdsmi") != bad["version"]:
        raise ValueError("Current/receipt SMI is not the exact newly installed failing version")
    if prepare.protected(current, data) != prepare.protected(before, data):
        raise ValueError("Numeric versions differ from the original receipt")
    log = (receipt / "probe.log").read_text()
    if "/opt/hyhal/lib/libamd_smi.so: undefined symbol: amdsmi_get_node_handle" not in log:
        raise ValueError("This recovery is specific to receipt03's HyHAL symbol failure")
    return bad


def run_child(output, label, arguments, timeout=60):
    print(json.dumps({"step": label}), flush=True)
    with (output / (label + ".log")).open("wb") as stream:
        try:
            result = subprocess.run([sys.executable, "-X", "faulthandler", *arguments],
                stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
            return {"returncode": result.returncode, "timed_out": False}
        except subprocess.TimeoutExpired:
            return {"returncode": None, "timed_out": True}


def sdk_paths():
    patterns = [
        "/usr/local/lib/python*/site-packages/amdsmi", "/usr/local/lib/python*/dist-packages/amdsmi",
        "/usr/lib/python*/site-packages/amdsmi", "/usr/lib/python*/dist-packages/amdsmi",
        "/opt/dtk*/lib/python*/site-packages/amdsmi", "/opt/dtk*/lib64/python*/site-packages/amdsmi",
        "/opt/dtk*/share/amd_smi/amdsmi", "/opt/rocm*/share/amd_smi/amdsmi",
        "/opt/hyhal/share/amd_smi/amdsmi", "/opt/conda/lib/python*/site-packages/amdsmi",
    ]
    return sorted({str(Path(p).resolve()) for pattern in patterns for p in glob.glob(pattern)
                   if (Path(p) / "__init__.py").is_file()})


def inspect_sdk(candidate, output):
    # A separate process prevents SDK library state from leaking into Torch.
    candidate = candidate.resolve()
    sys.path.insert(0, str(candidate.parent))
    import amdsmi
    if Path(amdsmi.__file__).resolve().parent != candidate:
        raise ValueError("SMI import did not resolve to the selected SDK")
    result = {"module": str(Path(amdsmi.__file__).resolve()), "initialized": False}
    amdsmi.amdsmi_init()
    try:
        result["initialized"] = True
        result["processor_count"] = len(amdsmi.amdsmi_get_processor_handles())
    finally:
        amdsmi.amdsmi_shut_down()
    prepare.save(output, result)


def inspect_library(output):
    library = Path("/opt/hyhal/lib/libamd_smi.so")
    record = {"path": str(library), "exists": library.is_file()}
    if library.is_file():
        record.update(resolved=str(library.resolve()), sha256=prepare.sha(library))
        lib = ctypes.CDLL(str(library))
        record["symbols"] = {s: hasattr(lib, s) for s in (
            "amdsmi_get_node_handle", "amdsmi_init", "amdsmi_shut_down",
            "amdsmi_get_processor_handles", "amdsmi_get_gpu_asic_info")}
    prepare.save(output, record)


def recover(receipt, output):
    current = prepare.snapshot()
    prepare.save(output / "before.json", current)
    if sys.platform != "linux" or sys.version_info[:2] != (3, 10):
        raise ValueError("Requires the original Linux Python3.10 launcher")
    data = prepare.manifest()
    bad = validate_receipt(receipt, current, sys.executable, data)
    dist = metadata.distribution("amdsmi")
    for relative, digest in bad["source_sha256"].items():
        if prepare.sha(dist.locate_file(relative)) != digest:
            raise ValueError("Installed SMI source differs; refuse automatic uninstall: " + relative)
    # Preserve the entire distribution before pip removes it, without loading it.
    backup = output / "removed_distribution"
    copied = {}
    for relative in dist.files or []:
        source = Path(dist.locate_file(relative)).resolve()
        if not source.is_file():
            continue
        if not source.is_relative_to(Path(sys.prefix).resolve()) or ".." in Path(str(relative)).parts:
            raise ValueError("SMI installation has a file outside the private environment")
        target = backup / str(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied[str(relative)] = prepare.sha(source)
    if not copied:
        raise ValueError("No installed SMI files available to back up")
    prepare.save(output / "removed_distribution_sha256.json", copied)
    prepare.pip(output, "remove_receipt03_smi", ["uninstall", "-y", "amdsmi"])
    after = prepare.snapshot()
    prepare.save(output / "after.json", after)
    if prepare.version("amdsmi") is not None or prepare.protected(after, data) != prepare.protected(current, data):
        raise ValueError("Recovery did not preserve numeric versions or remove the selected SMI")
    expected_packages = dict(current["packages"])
    expected_packages.pop("amdsmi")
    if after["packages"] != expected_packages:
        raise ValueError("Packages other than the receipt03 SMI changed during recovery")
    restored = run_child(output, "torch_import", ["-c",
        "import torch, lightop; print(torch.__version__); print(torch.version.hip); print(lightop.__file__)"])
    if restored["returncode"] != 0:
        raise RuntimeError("Torch import is still blocked; see torch_import.log; no further mutation attempted")
    candidates = []
    for index, name in enumerate(sdk_paths()):
        candidate = Path(name)
        target = output / ("sdk_%02d" % index)
        target.mkdir()
        # Copy text only for offline review; do not copy or install host libraries.
        files = {}
        for source in sorted(candidate.rglob("*.py")):
            destination = target / "source" / source.relative_to(candidate)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            files[str(source.relative_to(candidate))] = prepare.sha(source)
        extension_modules = [str(p) for p in candidate.rglob("*.so") if "cpython-" in p.name]
        entry = {"path": name, "python_sources_sha256": files, "python_extensions": extension_modules}
        if extension_modules:
            entry["inspection"] = {"skipped": "foreign Python extension; do not import into Python3.10"}
        else:
            entry["inspection"] = run_child(target, "import_and_smi_init", [str(Path(__file__).resolve()),
                "inspect-sdk", "--candidate", name, "--output", str(target / "result.json")], timeout=20)
        candidates.append(entry)
    library_probe = run_child(output, "host_library", [str(Path(__file__).resolve()),
        "inspect-library", "--output", str(output / "host_library.json")], timeout=20)
    prepare.save(output / "summary.json", {"scope": "receipt03_recovery_and_readonly_sdk_inventory",
        "torch_and_lightop_import_restored": True, "removed_distribution": bad["version"],
        "protected_versions_unchanged": True, "sdk_candidates": candidates, "host_library_probe": library_probe,
        "baseline_ready": False, "gpu_measurement_jobs": 0, "automatic_admission": False})
    print(json.dumps({"torch_and_lightop_import_restored": True, "sdk_candidates": len(candidates),
        "baseline_ready": False, "gpu_measurement_jobs": 0}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("recover", "inspect-sdk", "inspect-library"))
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "inspect-sdk":
        inspect_sdk(args.candidate, args.output)
    elif args.phase == "inspect-library":
        inspect_library(args.output)
    else:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        try:
            recover((args.receipt.resolve() / "preparation"), output)
        except Exception:
            prepare.save(output / "failure.json", {"traceback": traceback.format_exc()})
            prepare.save(output / "after_failure.json", prepare.snapshot())
            raise


if __name__ == "__main__":
    main()
