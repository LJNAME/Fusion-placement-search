"""CPU replay of scoped strict admission over the immutable paired receipt."""

import json
from pathlib import Path
import sys
import warnings


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from calibration.kerv_profile_workflow import demonstrate, save
from cost_model.kerv_admission import USE, load_admitted_store
from cost_model.kerv_profile import KervProfileStore, digest
from cost_model.measurement_labels import MeasurementContractError


profile = HERE / "measurements/profile.json"
decision = HERE / "admission.json"
before = digest(profile)
original = KervProfileStore.load(profile)
assert len(original.records) == 4 and all(r["admission"] == "conditional_dev" for r in original.records.values())
try:
    demonstrate(profile)
except MeasurementContractError:
    original_strict_rejected = True
else:
    raise ValueError("original conditional measurements unexpectedly passed strict query")
with warnings.catch_warnings(record=True) as emitted:
    warnings.simplefilter("always")
    replay = demonstrate(profile, admission_path=decision)
assert not emitted
assert replay == json.loads((HERE / "strict_cost_queries.json").read_text(encoding="utf-8"))
store = load_admitted_store(profile, decision, usage_scope=USE)
snapshot = json.loads((HERE / "measurements/admitted_profile.json").read_text(encoding="utf-8"))
assert snapshot["records"] == list(store.records.values())
assert digest(profile) == before
accepted = [r for r in store.records.values() if r["admission"] == "accepted"]
assert len(accepted) == 2 and all(r["role"] == "fused_kernel" for r in accepted)
assert all(not row["out_of_distribution"] for row in replay["results"])
result = {"pass_checks": True, "original_records": 4, "original_conditional_records": 4,
          "accepted_fused_records": 2, "native_comparison_records": 2,
          "strict_query_warnings": len(emitted), "original_strict_rejected": original_strict_rejected,
          "strict_query_matches_saved": True, "source_profile_unchanged": True,
          "source_profile_sha256": before, "admission_id": store.admission_id,
          "admission_use": USE,
          "strict_makespan_s": [r["makespan_s"] for r in replay["results"]],
          "out_of_distribution": [r["out_of_distribution"] for r in replay["results"]],
          "new_gpu_measurements": 0, "mcts_runs": 0, "downstream_deployable": False}
save(HERE / "verification.json", result)
print(json.dumps(result, indent=2))
