"""Lazy access to vendor implementations. Importing this module needs no GPU."""
from functools import lru_cache
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "data/lightop_fusion_catalog_v1/capabilities.json"


def catalog():
    return json.loads(CATALOG.read_text(encoding="utf-8"))


def capabilities():
    records = catalog()["capabilities"]
    review = ROOT / "data/lightop_vendor_suite01_review/capability_status.json"
    if review.exists():
        overlay = json.loads(review.read_text(encoding="utf-8"))
        for record in records:
            record["vendor_suite01"] = overlay["capabilities"][record["id"]]
    return records


def check_installed():
    data = catalog()
    dist = importlib.metadata.distribution("lightop")
    if dist.version != data["version"]:
        raise ValueError("LightOP version differs from the registered implementation")
    paths = {data["op_so_path"]: data["op_so_sha256"], "lightop/__init__.py": data["init_sha256"]}
    paths.update({c["source"]: c["source_sha256"] for c in data["capabilities"] if c["source"].startswith("lightop/")})
    for relative, expected in paths.items():
        path = Path(dist.locate_file(relative))
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("LightOP source differs from catalog: " + relative)
    return {"version": dist.version, "op_so_sha256": data["op_so_sha256"], "checked_files": len(paths)}


@lru_cache(maxsize=None)
def resolve(capability_id):
    """Return the real function/class. Unknown APIs and version mismatches fail."""
    record = next((c for c in capabilities() if c["id"] == capability_id), None)
    if record is None:
        raise KeyError(capability_id)
    check_installed()
    obj = importlib.import_module("lightop")
    for name in record["binding"].split(".")[1:]:
        obj = getattr(obj, name)
    if not callable(obj):
        raise TypeError("Registered LightOP API is not callable: " + record["binding"])
    return obj
