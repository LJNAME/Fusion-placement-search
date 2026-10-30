# SROP E0: search freedom, Kernel identity and output constraints

This E0 belongs to the 2026-10-30 SROP validation protocol. It is separate from
the historical E0 ranking-sensitivity experiment. Its endpoint is an annotated
operator graph. It does not create downstream parallel configurations or execute
the whole graph.

Run from the repository root with the existing `fusion-dev` environment:

```powershell
conda run -n fusion-dev python -m cost_model.srop_e0
conda run -n fusion-dev python -m unittest cost_model.tests.test_srop_e0 -v
```

An upstream-adapted `OpGraph` JSON in `simulator.dataset.graph_to_dict` format can
replace the built-in graph fixtures:

```powershell
conda run -n fusion-dev python -m cost_model.srop_e0 --graph adapted_graph.json --out-dir data/my_e0
```

The default hardware and legacy raw Profile are the existing BW1000 two-card
files. `--hardware` and `--raw-profile` select another compatible pair; this CLI
does not perform live hardware discovery. Those files must be consumable by
`MeasuredHardwareProfile`. The default graph inventories are the 21-node Qwen3
layer and the 591/687-node full graph fixtures, not upstream-integration evidence.

## Controlled search check

The declared five-node fork/join fixture uses synthetic, device-dependent
latencies. Device 0 prefers separate SiLU and Mul; device 1 prefers fused
SiLU-Mul. GEMMs prefer device 0. Every value is explicitly tagged `synthetic`.

| Mode | Fusion | Placement | Exact evaluations |
|---|---|---|---:|
| Placement-only | singleton | both devices | 32 |
| Fusion-only | search | fixed device 0 | 2 |
| Sequential | first search on device 0, then freeze partition | search | 2 + 32 |
| Joint | search | search | 48 |

All modes use the existing structural scheduler. Joint selects a different
partition and devices from fusion-first. The synthetic optimum is 4.200000128 ms
for Joint and 11 ms for the other modes. This is a constructed mechanism example,
not empirical evidence that Joint or SROP improves GPU performance.

The CLI checks MCTS against each mode's exact optimum for seeds 0/1/7/23 with 128
iterations and 3 rollouts. Sequential splits its 128 iterations into 64 for fusion
and 64 for placement on a contracted graph; it cannot change the frozen partition.
Logical evaluations, unique plans and each phase's elapsed time are reported.
Equal iterations do not imply equal time or scorer calls. These are E0 diagnostics,
not the fixed-total-wall-clock experiment.

## Group Kernel identity and coverage

`fusion_kernel_profile.GroupKernelKeys` represents both singleton and fused
groups. It retains implementation/version/flags, runtime versions, physical
device identity, node operator attributes, shapes, dtypes, layout/strides,
ordered boundary inputs, repeated-input aliases and internal edges. Layer IDs,
module paths and tensor display IDs do not produce duplicate measurements.
The v1 contract supports single-output operator nodes on unchanged graph and
hardware snapshots; it is not the old `e2_manifest` key format.

`KeyedKernelProfile` looks up group latency by that complete identity and feeds
the existing `StructuralCostModel`. Missing entries raise in strict mode.
Synthetic entries remain OOD and never count as measured coverage. Measured
entries require explicit provenance and the supported identity fields; a
provenance string alone is not an independent raw-measurement audit.

The real graph inventories enumerate local legal fusion candidates times
physical devices. They do not enumerate full-model plans. Both exploratory and
executable-only inventories are included. Default multi-node patterns remain
`planned`, and the project team's downstream fusion capability file is still
unavailable.

The current BW1000 shape table can still produce `measured_exact_shape` hits.
These are reported under `legacy_lookup_sources_by_usage`, separately from
`measured_covered`. They do not establish matching implementation/version and
layout in the new namespace. No old labels are silently rebound to new keys.
Missing identity metadata and latency entries are listed per key. Communication
and scheduling assumptions remain even if all Kernel entries become measured.
This inventory uses `data/bw1000_2card`, not the later E2 complete-key dataset.
It does not invalidate E2 labels or require repeating all old measurements;
compatible records can be reused through an explicit, audited identity mapping.

## Output

- `data/srop_e0_v1/report.json`: four exact results, sixteen MCTS results, output
  constraint checks, real graph inventories, readiness flags and source/input hashes.
- `data/srop_e0_v1/synthetic_joint.placed-op-graph.json`: existing v1.0 graph artifact
  with original node identities, fusion groups and device annotations. It is
  explicitly `deployable=false` and contains synthetic cost predictions.

Tests cover artifact round trips and annotation corruption, memory infeasibility,
the planned-pattern deployability filter, fused-cost response, layout/version/
device/shape key separation and cross-layer measurement reuse.

The final downstream capability can arrive later. The source-pinned FlagScale
KERV development substitute is now available via `--kerv-capability`; preparation,
guards and the optional single-Kernel probe are documented in
`calibration/KERV_SILU_MUL.md`. This lets development proceed without treating the
temporary source contract as downstream deployment approval.

An E0 mechanism PASS does not mean the measured fusion Profile is complete or
SROP is effective. The report exposes these as separate readiness flags. The
next protocol stage is the isolated full-Kernel-Profile fixed-model evaluator,
followed by the MCTS-state SROP interface and reveal/update/re-search loop.
