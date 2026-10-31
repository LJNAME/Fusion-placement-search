"""Preserve legacy bytes/IDs and demonstrate rejection by the real search CLI."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics as st
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from calibration.e2_manifest import measurement_object_id
from cost_model.fusion_kernel_profile import key_id
from cost_model.measurement_labels import MeasurementLabel


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def label(identifier, median_s, samples, n_calls):
    return asdict(MeasurementLabel(identifier, None, {
        "median_us": median_s * 1e6,
        "mad_rel": st.median(abs(v - median_s) for v in samples) / median_s,
        "n_rounds": len(samples), "n_calls": n_calls}, "legacy_unbound"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--group-profile", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    raw = ROOT / "data/bw1000_e2/raw_measurements.jsonl"
    manifest = ROOT / "data/bw1000_e2/shape_manifest.json"
    old_shape_raw = ROOT / "data/bw1000_2card/raw_measurements.json"
    paths = [raw, manifest, old_shape_raw]
    if args.group_profile:
        paths += [args.group_profile, args.group_profile.with_name("frozen_manifest.json")]
    before = {str(p): sha(p) for p in paths}
    objects = {v["object_id"]: v for v in json.loads(manifest.read_text())["objects"]}
    records = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
    sidecar = []
    for row in records:
        identifier = row["object_id"]
        assert measurement_object_id(objects[identifier]["kernel_key"]) == identifier
        timing = row["timing"]
        median = timing["summary"]["median_s"]
        assert median == st.median(timing["samples"])
        sidecar.append({**label(identifier, median, timing["samples"],
            len(timing["samples"]) * timing["iterations_per_sample"]), "source_record_id": row["record_id"]})
    save(args.out_dir / "e2_legacy_labels.json", sidecar)
    group_sidecar = []
    if args.group_profile:
        profile = json.loads(args.group_profile.read_text())
        frozen = json.loads(args.group_profile.with_name("frozen_manifest.json").read_text())
        for identifier, row in profile["profiles"].items():
            assert key_id(row["kernel_key"]) == identifier
            median = row["duration_s"]
            assert median == st.median(row["block_medians_s"])
            group_sidecar.append({**label(identifier, median, row["block_medians_s"],
                len(row["block_medians_s"]) * frozen["recipe"]["iterations_per_sample"]), "slot_id": row["slot_id"]})
        save(args.out_dir / "group_legacy_labels.json", group_sidecar)
    forbidden_output = args.out_dir / "search_must_not_produce_result.json"
    command = [sys.executable, "-m", "cost_model.search_evaluation", "--scales", "small", "--quick",
               "--mcts-iterations", "8", "--output", str(forbidden_output.resolve())]
    run = subprocess.run(command, cwd=ROOT, text=True, encoding="utf-8", capture_output=True)
    for name, value in (("search.stdout.log", run.stdout), ("search.stderr.log", run.stderr)):
        with (args.out_dir / name).open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
    assert run.returncode != 0 and "MeasurementContractError" in run.stderr, run.stderr
    assert "legacy_unbound" in run.stderr and not forbidden_output.exists()
    after = {str(p): sha(p) for p in paths}
    assert before == after, "legacy artifacts changed"
    save(args.out_dir / "report.json", {
        "source_sha256_before": before, "source_sha256_after": after,
        "legacy_bytes_unchanged": True, "existing_identifiers_preserved": True,
        "e2_labels": len(sidecar), "group_labels": len(group_sidecar),
        "admission": "legacy_unbound", "measurement_contract_id": None,
        "search_command": command, "search_exit_code": run.returncode,
        "search_contract_error": True, "search_result_exists": forbidden_output.exists(),
        "scope": "CPU identity/admission verification; no GPU or new runtime-semantic claims"})
    print("PASS", len(sidecar), "E2 labels;", len(group_sidecar), "group slots; search rejected with exit", run.returncode)


if __name__ == "__main__":
    main()
