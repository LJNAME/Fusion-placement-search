# KERV SiLU-Mul as the temporary development fusion capability

The final project-team fusion capability is expected later. Development uses the
source-pinned FlagScale `kerv_silu_mul` implementation as an explicit substitute.
The contract can be replaced when the final file arrives. No full operator graph
executor or downstream parallel configuration is part of this adapter.

The inspected FlagScale revision is
`f8e9a62c499d526a6ffa47f533bcf342330b7326`. The capability hashes all 12 Python
files under `KERVRuntimeOptimization`; the core source is checked against the
inspected Git revision. Its binding ID is
`flagos_embodied.kerv_silu_mul.triton`, with the core SHA256 as implementation
version. Source availability and GPU validation remain separate fields.

## Source conditions

The pinned `embodied_ops/kerv_ops.py` establishes:

- `_AUTO_TRITON_OPERATORS` does not enable this operator. `auto` is native here;
  explicit `configure_backend("triton")` and an enabled operator are required
  (`:49`, `:83`).
- Both tensors must share shape/device, be GPU tensors and contiguous; the last
  dimension must be at most 8192 (`:330`). The development adapter additionally
  requires matching BF16/FP16/FP32 dtype and positive dimensions.
- The fused code computes with FP32 temporaries and stores into `gate` (`:122`).
  The public schema also declares the mutated gate/output alias (`:630`).
- Width 11008 takes native PyTorch operations; that path is not fused evidence.

The graph adapter therefore rejects fusion when another consumer or an exported
graph output needs the original gate value. It does not insert an uncharged
clone to bypass this restriction. The unfused alternative stays searchable.
Existing structural memory accounting is unchanged; recording the output alias
does not establish exact in-place peak-memory prediction.

Lower-precision fused math has one final cast, while an unfused SiLU/Mul chain
can round twice. Validation is tolerance-based, not bitwise equality. The current
probe checks both an FP32 mathematical reference and an unfused reference.
Backward execution is outside this development contract.

## Prepare and check without a GPU

From the Fusion repository, using `fusion-dev`:

```powershell
conda run -n fusion-dev python -m calibration.kerv_silu_mul prepare
conda run -n fusion-dev python -m calibration.kerv_silu_mul check
conda run -n fusion-dev python -m cost_model.srop_e0 --kerv-capability data/kerv_silu_mul_proxy_v1/capability.json --out-dir data/srop_e0_kerv_proxy_v1
conda run -n fusion-dev python -m unittest cost_model.tests.test_kerv_capability cost_model.tests.test_srop_e0 -v
```

The default source tree is the sibling
`FlagScale-main/FlagScale-main`; use `--flagscale-root /path/to/FlagScale` on
another host. `prepare` refuses to overwrite an existing capability. `check`
verifies content, source hashes and the reviewed contract without importing
Triton or running a GPU.

Outputs:

- `data/kerv_silu_mul_proxy_v1/capability.json`: development authority, source
  hashes, implementation binding, guards and unvalidated GPU/latency status.
- `data/kerv_silu_mul_proxy_v1/applicability.json`: graph-level eligibility,
  rejection reasons and unresolved runtime layout requirements.
- `data/srop_e0_kerv_proxy_v1/report.json`: E0 inventory using the source binding
  and guarded development registry. The controlled coupling fixture inside this
  report is still synthetic; its 4.2-ms number is not a measured KERV latency.

The existing logical Transformer graphs do not state boundary tensor layouts.
The strict policy rejects unknown layouts; development inventory may retain a
candidate conditionally, carrying a contiguous-input runtime requirement. A
known noncontiguous layout is always rejected.

| Reference graph | Candidate groups | Conditional on contiguous inputs | Strictly verified layouts |
|---|---:|---:|---:|
| Qwen3 layer, FFN width 3072 | 1 | 1 | 0 |
| Qwen3 28 layers, width 3072 | 28 | 28 | 0 |
| Qwen2.5-3B 36 layers, width 11008 | 36 | 0 | 0 |

The default global registry is unchanged. The temporary registry only admits
the guarded SiLU-Mul multi-node pattern, and retains `planned` status until GPU
and final capability checks. It is usable with development search's
`require_executable=False`; it cannot promote output to downstream-deployable.

## Optional single-Kernel GPU probe

On a GPU host with the matching source tree and a working vendor Triton stack:

```bash
python -m calibration.probe_kerv_silu_mul \
  --capability data/kerv_silu_mul_proxy_v1/capability.json \
  --flagscale-root /path/to/FlagScale \
  --device cuda:0 --shape 1,256,3072 --dtype bf16 \
  --out-dir data/kerv_silu_mul_probe_gpu0
```

The probe forces and checks the Triton path, observes the public function's
single JIT dispatch once, and rejects CPU/native fallback. The observation is
host-dispatch instrumentation, not a GPU profiler trace. Timed iterations use
the original uninstrumented public function. GPU validation has not been run
by preparing the capability or by CPU native tests.

Default counts are 10 warmups and 3 samples of 20 iterations per method. Each
mutating invocation receives restored original gate values. Restore is outside
the Kernel-event interval, but included in the complete profiling wall time.
Fused/unfused block order alternates. Raw event durations, block wall times,
numeric errors, alias checks and runtime/source identities are retained.
The complete probe wall includes both methods and validation; it must not be
silently reused as the cost of a production one-method query.

Existing outputs are never overwritten; failures are saved in `failure.json`
without an accepted latency. A successful `probe.json` is Kernel-level evidence
for that shape/dtype/device only. It is not SROP benefit, full-model performance,
or automatic certification of a downstream fusion capability. Probe labels need
an audited mapping before use in the group-Key Profile.

Direct remeasurement under new group-Key identities is also available as a
prepared alternative to old-label migration. Its 12-slot development draft and
protocol are in `calibration/GROUP_KEY_REMEASUREMENT.md`; it is not an online
profiler or an automatic Profile update path.
