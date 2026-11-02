# Scoped KERV Profile admission

Project review authorized by the user's P2 continuation on 2026-11-02, after
the independently verified `kerv_profile_20261102_111921` receipt. Archive SHA256:
`7c70899d87afa6a64451f31ff53b697f8458868b1ee7e61b44350aadeac36920`.

`measurements/profile.json` and its 16 evidence files are copied unchanged from
that receipt. All four original records remain `conditional_dev`.
`admission.json` records an explicit decision for exactly two measured Fused
records, bound by original record IDs, Keys, contracts and the source bundle
hash. The two Native chain records remain comparison-only.

| Device | Admitted Fused duration, us/call | Native reference, us/call |
|---|---:|---:|
| GPU0 | 7.251179814338683 | 10.316779613494875 |
| GPU1 | 7.232000231742859 | 11.254359483718872 |

The accepted use is `selected_fused_group_graph_batch_proxy`: BF16
`[1,256,3072]`, contiguous tensors, the two exact device UUIDs and observed
implementations/runtimes, 7 rounds of 50 independent calls, 10 warmup replays,
and the recorded graph replay/Event contract. The measured boundary values
are synthetic with reference-model shapes. Acceptance is based on evidence
quality and identity, not whether fusion won the comparison.

This decision does not admit eager latency, singleton costs, communication,
full-plan additive costs, hardware-type sharing or downstream deployment.
The scoped profile refuses positive-byte transfers and non-input singleton
estimates rather than silently filling those with analytic costs.

`measurements/admitted_profile.json` is a derived inspection snapshot. Each
accepted record has a new record ID and retains `source_record_id`,
`admission_decision_id` and `admission_use`. Original Key/contract identities,
samples and durations do not change. Use the explicit decision loader/CLI to
consume these records within their permitted use.

```bash
python -m calibration.kerv_profile_workflow query \
  --profile data/kerv_profile_admission_v1/measurements/profile.json \
  --admission data/kerv_profile_admission_v1/admission.json
python data/kerv_profile_admission_v1/reproduce.py
```

The first command is strict: it must not use `--allow-conditional-dev`.
`strict_cost_queries.json` records two accepted measured costs, no OOD, and
source/derived/decision identity lineage. Without the decision, strict lookup
of the original measurement bundle still fails. `verification.json` records
the CPU replay; no new GPU measurements or MCTS search are run here.

Numeric evidence consists of the source-matched collector assertions with
the pinned BF16 tolerances, alongside dispatch traces and all timing samples.
Raw output tensors were not supplied for an independent local elementwise
replay. This decision is local graph-proxy admission, not a production runtime
or whole-model performance approval.
