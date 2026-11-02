# Existing GPU0 receipt: exact-identity diagnostic

Input: `kerv_adapter_20261102_104316.tar.gz`, SHA256
`1c495257dfd2149c5ff598d7a0f3f0cd03b22ca8852fe0799778ce1b3604dad6`.
The archived GPU execution used Adapter revision `b219a55`, GPU0, BF16
`[1,256,3072]`, one real KERV device Kernel. Only the selected group executes;
boundary values are synthetic and have reference-model shapes.

`inputs/gpu0/fused/` preserves the exact returned report, samples, trace and
placed artifact. `profile.json` uses the existing `fusion-group-kernel.v1` Key
format, binding observed input metadata and device identity. `cost_query.json`
records the diagnostic StructuralCostModel use and Key/contract/record lineage.

The record is `conditional_dev`. Four gaps remain:

- directly observed device architecture;
- directly recorded loaded-module source provenance;
- the complete measurement recipe inside the contract;
- directly observed driver version.

These are explicitly unknown, not filled from a different run. The Key is a
draft exact identity for this receipt; it is not an accepted reusable hardware
type Profile. The new paired collector supplies the corresponding observations.

The seven per-call times are 7.180800, 7.286400, 7.260760, 7.264000, 7.238400,
7.238380 and 7.235180 us. The median is 7.238399982452393 us. These are
graph-batch-amortized times. Raw output tensors are not in the receipt: numeric
acceptance is the source-matched collector's successful assertions, not a local
GPU or per-element replay.

```bash
python -m calibration.kerv_profile_workflow query \
  --profile data/kerv_profile_identity_v1/profile.json --allow-conditional-dev
```

Strict query without the explicit flag rejects this record. No accepted labels,
new GPU timing, Native comparison, second-device sharing or MCTS results are
introduced by this dataset. See `cost_model/KERV_PROFILE.md` for the next run.
