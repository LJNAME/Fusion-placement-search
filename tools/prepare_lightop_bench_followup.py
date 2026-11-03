"""Prepare the vendor baseline dependencies, then check APIs without timing kernels."""
import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import inspect
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MANIFEST = ROOT / "data/lightop_bench_followup_v1/manifest.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")


def version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def manifest():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    review = ROOT / "data/lightop_vendor_suite01_review/report.json"
    if sha(review) != data["review_sha256"]:
        raise ValueError("Followup input review changed")
    selected = [r["job_id"] for r in json.loads(review.read_text(encoding="utf-8"))["records"] if r["review_status"] == "missing_dependency"]
    if selected != data["jobs"]:
        raise ValueError("Followup must select exactly the original dependency failures")
    return data


def snapshot():
    return {"python": sys.version, "executable": sys.executable, "prefix": sys.prefix,
            "packages": {d.metadata["Name"].lower(): d.version for d in metadata.distributions() if d.metadata["Name"]}}


def protected(snapshot_record, data):
    return {n: snapshot_record["packages"].get(n) for n in data["protected_distributions"]}


def approved_lock(report, protected_names):
    """Reject numeric-stack replacements before pip makes any mutation."""
    locked = []
    for item in report["install"]:
        name = re.sub(r"[-_.]+", "-", item["metadata"]["name"]).lower()
        if name in protected_names or name.startswith(("pytorch-triton", "nvidia-")) or name in ("torchvision", "torchaudio", "flash-attn", "flash-mla"):
            raise ValueError("Dependency plan would replace a protected numeric package: " + name)
        download = item["download_info"]
        url = download["url"]
        digest = download["archive_info"]["hashes"]["sha256"]
        if not url.startswith("https://") or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("Dependency URL/hash unavailable: " + name)
        locked.append(name + " @ " + url + " --hash=sha256:" + digest)
    return locked


def pip(output, label, args):
    print(json.dumps({"step": label}), flush=True)
    command = [sys.executable, "-m", "pip", "--disable-pip-version-check", *args]
    with (output / (label + ".log")).open("wb") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(label + " failed; see " + str(output / (label + ".log")))


def fetch(wheel, cache):
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / wheel["filename"]
    if not path.exists():
        temporary = cache / (wheel["filename"] + ".part")
        with urllib.request.urlopen(wheel["url"], timeout=60) as source, temporary.open("wb") as stream:
            while chunk := source.read(1024 * 1024):
                stream.write(chunk)
        if sha(temporary) != wheel["sha256"]:
            raise ValueError("Downloaded wheel SHA mismatch: " + wheel["filename"])
        temporary.replace(path)
    if sha(path) != wheel["sha256"]:
        raise ValueError("Cached wheel SHA mismatch: " + wheel["filename"])
    return path


def probe(output):
    data = manifest()
    checks = {}
    import torch
    from integration.lightop.catalog import check_installed
    check_installed()
    for wheel in data["wheels"]:
        if not wheel["install_if_missing"] and version(wheel["distribution"]) != wheel["version"]:
            raise ValueError("Baseline dependency version mismatch: " + wheel["distribution"])
    ops = importlib.import_module("vllm._custom_ops")
    actual = list(inspect.signature(ops.rms_norm_opt).parameters)
    if actual != data["required_rms_norm_opt_parameters"]:
        raise ValueError("rms_norm_opt argument order differs from the original test: " + str(actual))
    checks["rms_norm_opt_parameters"] = actual
    for name in ("silu_and_mul", "gelu_and_mul", "gelu_tanh_and_mul", "silu_and_mul_opt"):
        if not callable(getattr(torch.ops._C, name)):
            raise ValueError("Missing baseline Torch operator: " + name)
    for name in ("fused_add_rms_norm_opt", "scaled_int8_quant", "rotary_embedding", "concat_and_cache_mla", "reshape_and_cache_cuda", "moe_fused_gate", "rms_norm_dynamic_per_token_quant"):
        if not callable(getattr(ops, name)):
            raise ValueError("Missing baseline API: " + name)
    platform_module = importlib.import_module("vllm.platforms")
    if not platform_module.current_platform.is_rocm():
        raise ValueError("vLLM did not detect the ROCm/Hygon platform; inspect amdsmi/DTK library paths")
    checks["vllm_platform"] = type(platform_module.current_platform).__name__
    rms = importlib.import_module("vllm.model_executor.layers.layernorm")
    quant = importlib.import_module("lmslim.layers.gemm.int8_utils")
    if not callable(rms.RMSNorm) or not callable(quant.per_token_quant_int8):
        raise ValueError("RMSNorm or BMM reference unavailable")
    for name in ("pytest", "tqdm", "pandas", "openpyxl"):
        importlib.import_module(name)
    # Verify the installed baseline files as well as distribution names.
    files = {}
    for wheel in data["wheels"]:
        if wheel["install_if_missing"]:
            continue
        dist = metadata.distribution(wheel["distribution"])
        for relative, expected in wheel["source_sha256"].items():
            actual_hash = sha(dist.locate_file(relative))
            if actual_hash != expected:
                raise ValueError("Installed baseline source mismatch: " + relative)
            files[relative] = actual_hash
    save(output / "probe.json", {"ready": True, "checks": checks, "baseline_files_sha256": files,
        "packages": {n: version(n) for n in ("torch", "triton", "lightop", "numpy", "vllm", "lmslim", "amdsmi", "pytest", "tqdm")},
        "scope": "imports_and_interfaces_only_not_a_kernel_measurement"})


def install(output):
    data = manifest()
    before = snapshot()
    save(output / "before.json", before)
    if sys.version_info[:2] != (3, 10) or platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("Requires the existing Linux x86_64 Python3.10 vendor launcher")
    versions = protected(before, data)
    if not versions["torch"] or not versions["torch"].startswith("2.9.0+das.") or "dtk2604" not in versions["torch"]:
        raise ValueError("Requires the existing Torch2.9 DTK2604 numeric stack")
    from integration.lightop.catalog import check_installed
    check_installed()
    for wheel in data["wheels"]:
        existing = version(wheel["distribution"])
        if existing and existing != wheel["version"] and not wheel["install_if_missing"]:
            raise ValueError("Preserve existing different baseline package: " + wheel["distribution"] + "==" + existing)
    constraints = output / "constraints.txt"
    constraints.write_text("\n".join(n + "===" + v for n, v in versions.items() if v) + "\n", encoding="utf-8", newline="\n")
    report_path = output / "host_resolution.json"
    pip(output, "resolve_host_dependencies", ["install", "--dry-run", "--only-binary=:all:", "--retries", "0", "--report", str(report_path), "--constraint", str(constraints), "--index-url", data["host_index_url"], *data["host_requirements"]])
    locked = approved_lock(json.loads(report_path.read_text(encoding="utf-8")), data["protected_distributions"])
    lock = output / "host_requirements.lock"
    lock.write_text("\n".join(locked) + "\n", encoding="utf-8", newline="\n")
    if locked:
        pip(output, "install_host_dependencies", ["install", "--no-deps", "--require-hashes", "--only-binary=:all:", "--retries", "0", "--report", str(output / "host_installed.json"), "-r", str(lock)])
    cache = Path.home() / "private_data/LJ/lightop_bench_wheels"
    paths = [str(fetch(w, cache)) for w in data["wheels"] if not(w["install_if_missing"] and version(w["distribution"]))]
    pip(output, "install_vendor_baselines", ["install", "--no-deps", "--no-index", *paths])
    after = snapshot()
    save(output / "after.json", after)
    if protected(after, data) != versions:
        raise ValueError("Protected numeric stack changed")
    check_installed()
    print(json.dumps({"step": "probe_baseline_interfaces"}), flush=True)
    with (output / "probe.log").open("wb") as stream:
        result = subprocess.run([sys.executable, "-X", "faulthandler", str(Path(__file__).resolve()), "probe", "--output", str(output)], stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError("Baseline import/API probe failed; see probe.log and failure_probe.json; no GPU benchmarks started")
    save(output / "summary.json", {"ready": True, "jobs": data["jobs"], "protected_versions_unchanged": True,
        "known_metadata_differences": data["known_metadata_differences"], "automatic_admission": False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("plan", "install", "probe"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.phase != "probe":
        output.mkdir(parents=True, exist_ok=False)
    try:
        if args.phase == "plan":
            save(output / "plan.json", manifest())
        elif args.phase == "install":
            save(output / "plan.json", manifest())
            install(output)
        else:
            probe(output)
    except Exception:
        save(output / ("failure_" + args.phase + ".json"), {"traceback": traceback.format_exc()})
        if args.phase == "install":
            save(output / "after_failure.json", snapshot())
        raise


if __name__ == "__main__":
    main()
