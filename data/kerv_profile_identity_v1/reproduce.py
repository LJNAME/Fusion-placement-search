"""Replay the archived GPU0 identity/query/cost checks without a GPU."""

import json
from pathlib import Path
import sys
import warnings


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from calibration.kerv_profile_workflow import demonstrate
from cost_model.kerv_profile import KervProfileStore
from cost_model.measurement_labels import MeasurementContractError


store = KervProfileStore.load(HERE / "profile.json")
try:
    demonstrate(HERE / "profile.json")
except MeasurementContractError:
    rejected = True
else:
    raise ValueError("strict query unexpectedly admitted a conditional record")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    replay = demonstrate(HERE / "profile.json", allow_conditional_dev=True)
expected = json.loads((HERE / "cost_query.json").read_text(encoding="utf-8"))
if replay != expected:
    raise ValueError("stored cost query differs from replay")
record = next(iter(store.records.values()))
result = {"pass_checks": True, "records": len(store.records), "strict_query_rejected": rejected,
          "diagnostic_query_matches_saved": True, "makespan_s": replay["results"][0]["makespan_s"],
          "kernel_key_id": record["kernel_key_id"], "record_id": record["record_id"],
          "identity_gaps": record["identity_gaps"], "admission": record["admission"],
          "new_gpu_measurements": 0, "accepted_records": 0,
          "scope": "original GPU0 evidence; not a complete production identity or full-plan search"}
with (HERE / "verification.json").open("w", encoding="utf-8", newline="\n") as stream:
    stream.write(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
