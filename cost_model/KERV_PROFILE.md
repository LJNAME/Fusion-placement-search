# KERV Canonical Profile stage

This stage reuses `GroupKernelKeys` (`fusion-group-kernel.v1`), the existing
measurement label gate and `KeyedKernelProfile`. It adds a file-backed exact
query store, observed metadata binding and a selected-group cost adapter.
Measurement contracts remain separate from Kernel identity.

## Identity and records

`normalize_identity(placed, report)` copies the graph/hardware snapshots and
annotates the selected group's observed input layout, strides, device UUID,
architecture and runtime. The fused SiLU intermediate is explicitly described
as an internal fused expression, with no fabricated materialized strides.
Node names and layer locations retain the existing Key exclusions.

The binding includes the public operator, backend, in-place flags, loaded-module
hash and source revision. GPU collection inspects `module.__file__`, the loaded
Triton function's code filename, actual file bytes and the checkout's Git HEAD.
These observations are checked against the pinned capability. Absolute install
paths stay in provenance, outside the Key. Native's separate binding records
the actual Torch distribution, functional source hash and Torch Git revision.

Hardware observations come from the current device properties, not an inferred
marketing-name mapping. HIP driver version is read from the HIP API and checked
against the profiler trace. Unknown values stay unknown and create identity
gaps. GPU0 and GPU1 retain different UUID/device identities and independent
records; this version does not pool same-model GPU profiles.

Each record contains the Key and hash, contract and hash, raw per-round samples,
median/min/max/MAD, trace/correctness provenance, file hashes, and admission.
The contract records graph replay, 50 independent calls, 7 rounds, interleaved
arm order, 10 warmup replays, Event boundaries, synchronization, retention,
units and statistic, GPU identity, HIP/driver/Torch/Triton builds. It describes
amortized replay work, not isolated eager launch latency.

`KervProfileStore.query(kernel_key, contract_id)` requires both exact identities.
Unknown Key, shape, layout, implementation, GPU or contract does not use nearest
neighbors. Changing the measurement recipe changes the contract, not the Kernel
Key. Stored evidence hashes and reconstructed identity/statistics are checked
when loading a bundle.

Native SiLU-then-Mul has a distinct binding and role `native_chain`. Its total is
a comparison record: it is never copied into each singleton cost or labeled as
the fused Kernel. Both arms retain all 50 final outputs during capture; Native
therefore includes its real separate output writes, with capture allocation
outside replay timing. Input restore and correctness checks are outside timing.

## Admission and cost use

Every real export in this stage stays `conditional_dev`. Strict lookup rejects
it; `--allow-conditional-dev` is explicit diagnostic use with warnings, OOD and
the shared low-confidence behavior. Missing identity fields are listed on the
record and on the cost usage trace. No `legacy_unbound` table is loaded.

Changing a JSON status to `accepted` is insufficient for the query API: it also
requires verified evidence, complete identity, a same-contract counterpart arm,
and an explicitly reviewed contract passed via `approved_contract_ids`. There
is no automatic measurement promotion. An explicit review decides which replay
contract is appropriate for the intended cost-model use; complete evidence does
not establish a downstream execution contract or deployment capability.

`KervMeasuredProfile` delegates measured duration to the existing
`KeyedKernelProfile`, with launch=0, and logs Key/contract/record identity for
each measured group. A missing fused record always fails. Singleton costs can
use the explicitly recorded analytic base; the supplied demonstration uses
only two zero-cost logical boundary inputs and one measured fused group, so
its makespan equals the measured duration exactly once.

## Reproduce the existing GPU0 diagnostic

```bash
python -m calibration.kerv_profile_workflow query \
  --profile data/kerv_profile_identity_v1/profile.json --allow-conditional-dev
```

Without the flag, this exits with a measurement admission rejection. The old
GPU0 result is retained at 7.238399982 us/call. It lacks direct loaded-module,
architecture, driver and full recipe observations, which remain four explicit
gaps; no values are guessed from older experiments. This receipt cannot be
promoted by merely adding a status flag.

## GPU0/GPU1 paired collection

From the Fusion repository on SCNet, using the existing private environment:

```bash
FLAGSCALE_ROOT="$HOME/private_data/LJ/FlagScale-main" \
FUSION_PYTHON="$HOME/private_data/LJ/dcu-kernel310-20261031/run_python.sh" \
bash tools/run_kerv_profile_stage3.sh "data/kerv_profile_$(date +%Y%m%d_%H%M%S)"
```

The workflow prepares the selected group, runs both devices, verifies/exports
four records (Native/Fused on each GPU), then exercises diagnostic cost lookup.
It stops on the first error. Return the entire result directory, including
`plan.json`, raw traces, all samples, placed artifacts, comparisons and profiles.
No CPU fallback or automatic status upgrade occurs.

The recipe is fixed before collection: BF16 `[1,256,3072]`, seed1234, GPU0 then
GPU1, 7 paired rounds, 50 independent calls per graph, alternating arm order.
Both native and fused correctness are checked against native and FP32 references.
Fused requires one correlated KERV device Kernel; Native requires separate
SiLU/Mul operators and two correlated device kernels. Numerical tolerance is
the pinned capability's tolerance. Partial results and failures are retained.

Both `(median_native-median_fused)/median_native` and the median of paired
savings are reported. A descriptive preference requires seven agreeing signs
and at least 1% median paired saving magnitude. All outcomes are preserved;
there is no significance, device equivalence, or profile-sharing claim.

This stage verifies the selected candidate's measured cost. It does not rerun
MCTS or claim a changed optimum: comparable full-plan scoring additionally needs
contract-compatible singleton/link costs. The native chain total alone cannot
supply those per-operator costs, and PP layer partitioning remains downstream.

## Scoped P2 review

`data/kerv_profile_admission_v1` preserves the complete paired receipt and
records a project review decision for two exact Fused records. The accepted
use is `selected_fused_group_graph_batch_proxy`, not a general eager/full-plan
latency claim. The original four measurement records remain conditional.
An accepted derived record has a new ID and references the original record,
review decision and use. Native chains remain comparison-only.

```bash
python -m calibration.kerv_profile_workflow query \
  --profile data/kerv_profile_admission_v1/measurements/profile.json \
  --admission data/kerv_profile_admission_v1/admission.json
```

This strict path uses `load_admitted_store(..., usage_scope=...)`: an exact
bundle hash, decision identity, scope, source record, Key, contract, paired
evidence and input recipe must match. No diagnostic bypass is enabled. The
profile rejects analytic singleton and communication costs outside the review.
Whole-model deployment and MCTS search quality require their own evidence.
