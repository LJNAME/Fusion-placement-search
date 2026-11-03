"""Download a Linux CPython3.10 wheelhouse without installing or importing it."""
import argparse
from email.parser import Parser
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.prepare_lightop_bench_followup import MANIFEST, manifest, pip_environment, save, sha


def linux310_closure(records, requirements, installed):
    environment = default_environment()
    environment.update(python_version="3.10", python_full_version="3.10.22", implementation_name="cpython",
                       implementation_version="3.10.22", platform_python_implementation="CPython",
                       os_name="posix", sys_platform="linux", platform_system="Linux", platform_machine="x86_64", extra="")
    versions = {canonicalize_name(n): v for n, v in installed.items()}
    versions.update({canonicalize_name(r["distribution"]): r["version"] for r in records})
    missing = []
    for text in requirements:
        requirement = Requirement(text)
        if requirement.marker and not requirement.marker.evaluate(environment):
            continue
        version = versions.get(canonicalize_name(requirement.name))
        if version is None or (requirement.specifier and not requirement.specifier.contains(version, prereleases=True)):
            missing.append(text)
    return sorted(set(missing))


def build(output, vendor_cache, index_url=None, download_cache=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    data = manifest()
    index_url = index_url or data["host_index_url"]
    downloads = output / "downloaded"
    downloads.mkdir()
    constraints = output / "download_constraints.txt"
    constraints.write_text("numpy==1.26.4\n", encoding="utf-8")
    # Include <3.11 pytest dependencies explicitly when building under Python3.11.
    cross_roots = ["exceptiongroup>=1.0", "tomli>=1.0", "async-timeout>=4.0,<6.0", "hf-xet>=1.1.3,<2.0.0"]
    roots = data["host_requirements"] + cross_roots
    command = [sys.executable, "-m", "pip", "--isolated", "--disable-pip-version-check", "download",
        "--dest", str(downloads), "--only-binary=:all:", "--python-version", "3.10", "--implementation", "cp",
        "--abi", "cp310", "--abi", "abi3", "--abi", "none", "--platform", "manylinux_2_28_x86_64",
        "--platform", "manylinux2014_x86_64", "--platform", "manylinux_2_17_x86_64",
        "--constraint", str(constraints), "--index-url", index_url, "--retries", "0", *roots]
    if download_cache:
        command.extend(["--find-links", str(Path(download_cache).resolve())])
    save(output / "build_command.json", command)
    with (output / "download.log").open("wb") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=pip_environment())
    if result.returncode:
        raise RuntimeError("Wheelhouse resolution failed; see download.log")
    host, vendor = output / "host", output / "vendor"
    host.mkdir()
    vendor.mkdir()
    records, requirements = [], list(roots)
    for path in sorted(downloads.glob("*.whl")):
        with zipfile.ZipFile(path) as archive:
            entry, = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
            metadata = Parser().parsestr(archive.read(entry).decode("utf-8"))
        name = metadata["Name"].lower().replace("_", "-")
        if name in data["protected_distributions"]:
            if name != "numpy":
                raise ValueError("Unexpected numeric library in host dependency set: " + name)
            continue
        if name.startswith(("nvidia-", "pytorch-triton")):
            raise ValueError("Unexpected GPU library: " + name)
        shutil.copyfile(path, host / path.name)
        records.append({"path": "host/" + path.name, "distribution": name, "version": metadata["Version"], "sha256": sha(path)})
        requirements.extend(metadata.get_all("Requires-Dist", []))
    received = json.loads((ROOT / "data/lightop_vendor_suite02_review/report.json").read_text(encoding="utf-8"))
    missing = linux310_closure(records, requirements, received["before_packages"])
    save(output / "host_closure.json", {"missing": missing, "checked_requirements": len(requirements)})
    if missing:
        raise ValueError("Incomplete Linux/Python3.10 host closure: " + str(missing))
    for wheel in data["wheels"]:
        path = Path(vendor_cache) / wheel["filename"]
        if sha(path) != wheel["sha256"]:
            raise ValueError("Vendor cache differs: " + wheel["filename"])
        shutil.copyfile(path, vendor / path.name)
        records.append({"path": "vendor/" + path.name, "distribution": wheel["distribution"], "version": wheel["version"], "sha256": sha(path)})
    bundle = {"schema_version": 1, "target": "CPython3.10 Linux x86_64; glibc>=2.28", "source_manifest_sha256": sha(MANIFEST),
        "source_index_url": index_url, "host_requirements": data["host_requirements"],
        "extra_cross_build_roots": cross_roots, "protected_numeric_wheels_excluded": True,
        "host_closure_missing": missing, "host_closure_checked_requirements": len(requirements),
        "runtime_validated": False, "files": records}
    save(output / "bundle.json", bundle)
    archive_path = output.with_name(output.name + ".tar.gz")
    with tarfile.open(archive_path, "w:gz", compresslevel=1) as archive:
        for relative in ["bundle.json", *[r["path"] for r in records]]:
            archive.add(output / relative, arcname=output.name + "/" + relative)
    print(json.dumps({"archive": str(archive_path), "sha256": sha(archive_path), "files": len(records), "bytes": archive_path.stat().st_size}), flush=True)
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vendor-cache", type=Path, required=True)
    parser.add_argument("--index-url")
    parser.add_argument("--download-cache", type=Path)
    args = parser.parse_args()
    build(args.output, args.vendor_cache, args.index_url, args.download_cache)
