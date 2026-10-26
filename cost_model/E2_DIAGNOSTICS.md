# Frozen E2 diagnostics

Run from the repository root with the existing Python environment. These
experiments use recorded BW1000 measurements and run on CPU; no GPU is needed.

```bash
python -m unittest cost_model.tests.test_e2_diagnostics -v
python -m cost_model.e2_diagnostics \
  --phase all \
  --output data/bw1000_e2/e2c_frozen_diagnostics.json
```

`--phase zero` runs only initial screening and full-budget recovery checks.
`--phase one-step` first completes any missing zero-budget tasks and then
evaluates each possible first measurement independently. `--phase all` does
the same, finishing screening for every task before starting the second phase.

The default search configuration is 12 candidates per source, seeds 0/1/7/23,
32 MCTS iterations per seed, and 3 rollouts. Record all results, including zero
initial regret. The frozen 84 configurations include repeated subsets and
overlapping graph workloads; they are not 84 independent workloads.

## Labels and candidate scope

- A task exposes only its frozen initial training IDs. All other labels remain
  hidden, including original training objects withheld by a 16/32-key budget.
- The active measurement pool is the set of timed objects used by this graph,
  minus its initially visible objects. Revealing one object exposes only that
  label to the predictor.
- The original zero-shot contract is unchanged. Task-local `hidden` means
  initially unmeasured and is not a claim of unseen-Shape generalization.
- Candidate pools are the device-exchange-deduplicated union of initial
  predicted and oracle searches. Oracle-derived candidate structures influence
  this diagnostic. It is not an oracle-free adaptive-search experiment.
- The oracle is the structural scheduler using measured operator times and
  fixed calibrated links, not end-to-end GPU timing or a global search optimum.
- Revealing all graph-used objects must recover every fixed candidate's oracle
  cost and ranking. A failure stops the run rather than skipping that task.

## One-measurement diagnostic

Acquisition scores for all unmeasured objects are computed from a masked public
view before inspecting outcomes. For each object, the evaluator then reveals
its actual measurement and reports:

1. Error and coverage of the initial latency prediction.
2. Predicted uncertainty, differential plan impact, and SROP-Proxy score.
3. Regret after replacing only this object's duration.
4. Regret after fitting the predictor with this observation.
5. Changes in other unmeasured predictions and their errors.

The difference between items 3 and 4 diagnoses effects missed by the current
target-only Proxy. It does not train or implement SROP-Posterior. Realized gains
are evaluation labels; they never choose the policy's first measurement.

The diagnostic reports each deterministic policy's actual first choice. Random
is the exact uniform expectation over all one-object choices, not a four-seed
estimate. Negative gains are retained. A rank-tie tolerance of 1e-12 seconds is
used; regret improvements smaller than 1e-9 ms are not counted as signal.

## Checkpoints and reproducibility

Compressed candidate plans and per-task results are saved under
`data/bw1000_e2/diagnostics_checkpoints/` (ignored by Git). Restart the same
command to resume. The checkpoint namespace includes frozen task/data content,
hardware, search settings, and source hashes. A new configuration uses a new
namespace instead of silently reusing incompatible results. Report JSON is
written atomically after each completed task.

For a disposable smoke test, use a separate output path:

```bash
python -m cost_model.e2_diagnostics \
  --limit 1 --candidates 4 --seeds 0 --iterations 8 --rollouts 1 \
  --output data/bw1000_e2/e2c_diagnostics.smoke.json
```

Partial runs are explicitly marked `partial_smoke_test`. The report records
completed counts and execution status. The input suite remains unchanged;
only a completed report establishes that the tasks were actually evaluated.

## Candidate coverage audit

Inspect the exact existing candidate placements after the batch finishes:

```bash
python -m cost_model.audit_e2_candidate_coverage \
  --report data/bw1000_e2/e2c_frozen_diagnostics.json \
  --output data/bw1000_e2/e2c_candidate_coverage.json
```

The audit reports nodes placed away from each plan's majority device and the
layers these nodes belong to. A complete graph does not imply a broad candidate
pool: variations can still be confined to the first few groups. This audit
does not modify the frozen pools or select favorable tasks.
