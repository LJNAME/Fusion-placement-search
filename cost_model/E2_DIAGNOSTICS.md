# Frozen E2 diagnostics

## Shared-engine total-cost comparison (2026-10-27)

```bash
python -m cost_model.e2_total_cost_evaluation --workers 2 --repeats 2 \
  --output data/bw1000_e2/e2c_total_cost.json
python -m cost_model.verify_e2_total_cost \
  --output data/bw1000_e2/e2c_total_cost_verification.json
python cost_model/plot_e2_total_cost.py \
  --report data/bw1000_e2/e2c_total_cost.json \
  --out-dir data/bw1000_e2/total_cost_figures
```

Requires the frozen scoring report, distance-Proxy stopping report and their
candidate/trajectory checkpoints. `--limit 1` with a separate output is a smoke
test. Source, data, protocol and environment hashes namespace resumable runs.

All samplers use the same nearest-shape mean, distance-floor intervals,
analytic-times-360 denominator, fixed executable candidates and recorded
measurement objects. Impact and target-only Proxy now use the same exact
scenario batching engine as Refit. An untimed check on each task's initial
public state compares all scalar/batched baseline candidate rankings and
acquisition values. Full native-stop prefixes of Proxy and Refit must match
their respective frozen controls. Full-pool replays must recover Oracle Top-1
and strict Top-k ranking. These checks are outside timed selection runs.

The clock adds serial CPU fitting/ranking/acquisition to recorded BW1000
profiling wall costs. The shared candidate preparation fee is measured once
per task. Each repeat also measures a common initial fit/rank fee using the
initial-only control, charged identically to all samplers. Raw first-stage CPU
timings remain in checkpoints; only shared initialization is aligned. Later
CPU stages are freshly measured for each method. No oracle-evaluation CPU is
charged, and graph/candidate discovery and existing training-label acquisition
are outside this incremental calibration comparison.

Before the first model decision is ready, hold the public singleton all-device-0
plan. During acquisition, profiling and subsequent refitting, hold the previous
published decision. A newly measured label is counted as completed when its
profiling operation finishes, but its new decision is usable only after fitting
and ranking finish. Do not delay publication by the *next* acquisition, and do
not publish at profiling completion before refitting. Curves separately report
completed labels and labels used by the last published decision.

The primary comparison forces pool replay and evaluates the available decision
at common absolute horizons 1, 2, 4, 8 and 16 seconds; 4 seconds is predeclared
as the primary horizon. A measurement may straddle a budget: its future result
is unavailable at that budget. This is offline deadline replay, not an online
cancellation or latency guarantee. Integrate the right-continuous regret step
curve, including initialization and holding the final decision after completion.

A separate stopping mode truncates each score-based policy at its first
nonpositive score; Random continues to pool exhaustion. These are prefix
replays of the recorded full trajectories, not newly measured stop-only wall
runs. No later observation determines the cutoff. An initial-only control fits
and ranks once with visible training data and takes no new measurements; a
reference-only control keeps the public reference at zero incremental cost.

Random seeds 0/1/7/23 are retained, each strategy/seed has two fresh repeats,
and rotated/reversed run order balances strategy timing. Average runs within
each configuration before averaging the 84 correlated configurations. Report
all, positive-initial-regret, zero-initial-regret, model and graph-scale groups.
No workload is selected for favorable outcomes. Two local CPU workers share
the host. This mixed-clock replay is neither measured end-to-end GPU wall time
nor an Oracle-free search experiment or a blind holdout result.

| Primary forced-replay 4s budget | Mean regret-time AUC / ms |
|---|---:|
| Random | 0.194089 |
| Refit | 0.198836 |
| Impact | 0.211325 |
| Proxy | 0.219457 |
| Uncertainty | 0.223453 |
| Initial only | 0.297957 |
| Reference only | 0.830214 |

Refit is 2.45% worse than Random in the primary descriptive mean: 10 lower,
13 higher and 61 tied configurations. The ten positive-initial-regret
configurations have 0.320696 versus Random's 0.326301 ms (5 lower/5 higher);
the other 74 have 0.182368 versus 0.176223 ms (5 lower/8 higher/61 tied).
These are correlated development configurations and two CPU repeats; no
statistically significant win or loss is claimed. Keep the primary horizon and
all configurations rather than replacing them with the favorable subgroup.

| Score-stop prefix replay | Completion cost / s | Final regret / ms | Nonzero-final configurations |
|---|---:|---:|---:|
| Random (pool exhaustion) | 1.971653 | 0 | 0 |
| Refit | 1.104336 | 0 | 0 |
| Proxy | 0.963113 | 0.093618 | 6 |
| Initial only | 0.455813 | 0.153956 | 10 |

Refit's stopping prefix uses 2.345 objects and 0.118799 s recorded profiling
versus Random's exhausted pool of 12.952 objects and 0.823349 s. Its completion
cost is 43.99% lower under these particular stop rules. This is distinct from
winning a fixed cost budget: Random may obtain a good decision before pool
exhaustion. The primary cost curve measures that decision availability.

The verifier also reports diagnostic clock ablations. On the common 0..16
measurement axis, Refit/Random AUCs remain 0.015125/0.031447 ms. Removing all
CPU and normalizing by each task's full profiling-pool horizon gives
0.007815/0.038140 ms; this axis is not numerically comparable to the 4s axis.
Keeping the same initial fees and 4s horizon but setting only *later* CPU time
to zero gives Refit/Random 0.178858/0.186418 ms. This counterfactual is a cost
diagnostic, not an implementable speedup or a replacement primary experiment.
It motivates cheaper exact evaluation and examination of avoidable measurements
when the initial decision is already optimal. It does not establish that
CPU reduction alone will generalize to new workloads.

Verification rebuilds 18,920 raw trajectory points from 1,512 measured replays,
with 17,408 observed-object prefixes; initial baseline batching compares 7,616
complete rankings and 290,731 candidate cost/confidence entries. Frozen native
Proxy/Refit prefixes match on every repeat. All non-control full-pool replays
recover Oracle Top-1/strict Top-k. Regression tests: 206 total; resume preserves
the report SHA256. Raw checkpoints retain the full clocks and label lineage.

## Exact hypothetical-scenario batching (2026-10-27)

```bash
python -m cost_model.e2_scoring_benchmark --workers 2 --repeats 2 \
  --output data/bw1000_e2/e2c_scoring_benchmark.json
python -m cost_model.verify_e2_scoring_benchmark \
  --output data/bw1000_e2/e2c_scoring_verification.json
python cost_model/plot_e2_scoring_benchmark.py \
  --report data/bw1000_e2/e2c_scoring_benchmark.json \
  --out-dir data/bw1000_e2/scoring_figures
```

Requires the frozen factorial report, its compressed stopped trajectories and
the original candidate checkpoints. Use `--limit 1` with a separate output for
a smoke test. Source, environment, protocol and data hashes namespace resumable
checkpoints. Changing worker count or repeat count starts a separate experiment.

`BatchedPlacementRanker.rank_many` computes the same scheduler recurrence over
independent hypothetical profiles and placement candidates together. Group
order, addition/max operations, per-device availability, directional/shared
link contention, send blocking and confidence propagation are preserved.
The scenario dimension is chunked to at most 32; group matrices target at most
two million cells each. Only identical partitions, zero interference and
identical transfer durations, confidences, link keys and overlap behavior share
the fast path. Other cases retain scalar ranking. Memory and critical-path
outputs remain outside fixed-pool makespan ranking, as in the original batch
ranker.

Refit still builds the same public-feature nearest-neighbor response for each
query and samples the original five labels. Only independent scenario scoring
is batched. `refit_batch_scenarios=False` explicitly selects the original loop;
the default uses batching when the ranker supports it. The analytic cost
denominator, mean predictor, distance uncertainty and zero-score stop stay
fixed. This is an execution optimization, not a new acquisition function.
Rank-call counters still count logical scenarios, not NumPy batches; the
number of evaluated hypothetical observations and candidates is unchanged.

For every frozen configuration, an untimed scalar pass is compared to the old
factorial trajectory, and an untimed batched pass compares every ordered
candidate ID, makespan and confidence exactly, including all hypothetical
observations. Every non-CPU-timing trajectory field must match: measurement
costs, scores, tie diagnostics, choices, metrics and stopping. A separate
full-reveal check recovers all Oracle candidate costs.

Then run fresh scalar and batched trajectories twice each, alternating AB/BA
order. Untimed audits are excluded from these measurements. Summaries take
the median of repeats per configuration and then the mean across configurations;
also report paired speedup medians and every model/graph-scale group. The two
worker processes share a CPU; these timings are a counterbalanced local CPU
replay benchmark, not an isolated-core or end-to-end GPU training result.
The algorithm counter sums predictor fitting, ordinary candidate ranking and
acquisition. Fixed candidate preparation is outside it and recorded separately;
graph/candidate discovery and GPU execution are outside this replay benchmark.
The replay also evaluates fixed Oracle candidates. All 84 configurations remain
correlated development tasks; no blind-test or Oracle-free search claim follows.

| Paired local CPU replay | Scalar scenarios / s | Scenario batch / s | Ratio of means |
|---|---:|---:|---:|
| All 84: fitting + ranking + acquisition | 6.422553 | 0.965615 | 6.65x |
| Layer, 42 configurations | 0.761877 | 0.454469 | 1.68x |
| Full graph, 42 configurations | 12.083228 | 1.476760 | 8.18x |
| All 84: acquisition only | 6.165020 | 0.709259 | 8.69x |

All 84 paired configurations are faster; minimum paired speedup is 1.23x.
The separate fixed candidate preparation averages about 0.614/0.617 s.
Including that preparation and the full replay timer gives 7.037/1.583 s,
about 4.44x; graph/candidate discovery and GPU execution remain excluded.
This does not establish overall SROP efficiency relative to other sampling
strategies: that requires a fresh shared-engine CPU-plus-profiling comparison.

Untimed validation checks 16,061 full ranking lists and 613,385 candidate
cost/confidence entries. All 281 stopped-trajectory points match the frozen
factorial experiment, including scores and stop decisions; 336 clean paired
timed replays match those same semantics. Count-budget AUC remains
0.015124928 ms, and wrong stops remain zero. Regression tests: 202 total.
The original analytic cost denominator is retained; the negative wall-cost
normalization ablation is unchanged.

## Profiling wall-cost normalization (2026-10-27)

```bash
python -m cost_model.e2_cost_normalization --workers 4 \
  --output data/bw1000_e2/e2c_cost_normalization.json
python -m cost_model.verify_e2_cost_normalization \
  --output data/bw1000_e2/e2c_cost_verification.json
python cost_model/plot_e2_cost_normalization.py \
  --report data/bw1000_e2/e2c_cost_normalization.json \
  --out-dir data/bw1000_e2/cost_figures
```

Requires the completed factorial report and its compressed trajectories, plus
the original frozen candidate checkpoints. `--limit 1` with a separate output
path is a smoke test. Resume uses data, controls, protocol and source hashes.

The predeclared cost predictor uses the median visible profiling wall time per
operator family, falling back to the visible global median or, with no wall
labels, analytic time times repetitions. Global median is also evaluated as a
prediction baseline. Wall labels are separate from GPU-duration residuals.
Each active round fits only the masked sampler view: initial training labels
plus legally revealed objects. The actual hidden costs enter evaluation only.
Input preparation, warmup and synchronization are inside the recorded wall
timer; this model is specific to the recorded BW1000 environment and recipe.

The experiment holds distance uncertainty, Refit acquisition numerator, five
hypothetical observations, candidates and zero-score stopping fixed. Both
the denominator and reported next cost use the same public-only prediction.
The analytic denominator remains the default.

| All 84 configurations | Analytic x 360 | Family wall median |
|---|---:|---:|
| Count-budget AUC / ms | 0.015125 | 0.016148 |
| Actual-cost-budget step AUC / ms | 0.007815 | 0.008926 |
| Wrong stops | 0 | 0 |
| Mean measurements | 2.345 | 2.071 |
| Recorded profiling wall time / s | 0.118799 | 0.115698 |

For each paired task the common time horizon is the sum of all its pool wall
costs, independent of policy order. Regret stays at the prior decision during
a measurement, changes only at completion, and is held after an early stop.
Integrate this step curve and divide by that common horizon. This differs
from the older trapezoidal count AUC; compare variants only within one axis.

Forty selection prefixes change. Actual-cost AUC has 4 improvements, 7
regressions and 73 ties: mean AUC worsens 14.21%, despite 11.68% fewer
measurements and 2.61% lower final profiling cost. On the 544 deduplicated
initial training-state/object pairs, wall-cost MAPE changes 100.28% to 92.19%,
but mean absolute log error worsens 1.00582 to 1.02425 and over-2x
underestimates increase 126 to 213. This does not justify changing defaults.
Keep this negative ablation; do not tune it on the exposed evaluation labels.

The audit checks 84 unchanged controls, 258 public-only score denominators
and 539 actual-cost prefixes; each newly executed task also separately checks
full-reveal recovery of all fixed Oracle candidate costs. Regression tests:
200 total. CPU costs are reported separately (new mean 7.738 s; old mean
6.823 s from its earlier run), not a simultaneous benchmark or proof of
end-to-end speedup. Fixed pools still contain Oracle search candidates, and
the 84 correlated development configurations are not independent blind tasks.

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

## Graph-wide coverage control

The original 32-iteration prefix MCTS pools varied only in the first five
groups of complete models. Keep that report and its checkpoints as the
control. Expand exactly those candidates without modifying the frozen tasks:

```bash
python -m cost_model.e2_coverage_diagnostics \
  --phase all --workers 4 \
  --output data/bw1000_e2/e2c_graph_wide_diagnostics.json
python -m cost_model.audit_e2_candidate_coverage \
  --report data/bw1000_e2/e2c_graph_wide_diagnostics.json \
  --output data/bw1000_e2/e2c_graph_wide_coverage.json
python -m cost_model.audit_e2_first_step \
  --report data/bw1000_e2/e2c_graph_wide_diagnostics.json \
  --output data/bw1000_e2/e2c_graph_wide_first_step_audit.json
```

The runner needs the original candidate checkpoints. A fresh clone must first
run the original `e2_diagnostics` command above; checkpoints are Git-ignored.
The suite, data, hardware, task IDs and original candidate IDs are checked
before expansion. Original candidates remain in the pool, including those
discovered by Oracle search. This remains a fixed-pool development diagnostic.

`graph-wide-v1` adds topology-only probes to the executable reference plan:

- Move one local operator motif to the other GPU at every layer.
- Move each fork-to-join linear chain at every occurrence.
- Move one chain at each of six evenly spaced layer anchors.
- Split the graph at the quarter, half and three-quarter layer boundaries.

All proposals are retained after device-exchange deduplication. There is no
latency-based truncation. This tests depth coverage; it does not exhaust the
placement space, search parallelism configurations, or modify MCTS itself.
Candidates, measurement pools, ranking metrics, predictor updates and policy
scoring are otherwise evaluated by the original diagnostic functions.

`--phase zero` runs screening only. `--phase all` completes every zero-budget
task before evaluating each possible first measurement. Both support resume;
independent configurations can run in worker processes. The source hash and
parent report identity isolate new checkpoints from the original experiment.

The one-step diagnostic forcibly reveals every object, even if all policy
scores are zero. Those counterfactual gains are not realized policy gains.
The first-step audit reports both forced measurements and outcomes under the
predefined maximum-score <= 0 stopping rule. Always use the latter when
describing what a policy would actually achieve.

## Full budget trajectories

```bash
python -m cost_model.e2_budget_curves --workers 4 \
  --output data/bw1000_e2/e2c_graph_wide_budget_curves.json
python -m cost_model.verify_e2_budget_curves \
  --report data/bw1000_e2/e2c_graph_wide_budget_curves.json \
  --output data/bw1000_e2/e2c_budget_verification.json
```

This keeps every graph-wide candidate and all 84 tasks unchanged. It uses the
original predictor, acquisition scores and tie-breaking. Every actual reveal
refits the predictor. The three deterministic policies must reproduce their
frozen first selection and first-measurement regret exactly. Random uses
seeds 0/1/7/23; average those runs inside each configuration before averaging
configurations. It is no longer the exact one-step uniform expectation.

The raw run continues to pool exhaustion for an Oracle-recovery check.
Published curves show requested budgets 0/1/2/4/8/16 in two modes:

- `forced`: continue measurements even when the acquisition score is zero.
- `stopping`: retain the decision at the first maximum score <= 0. No future
  label or outcome determines the stop. Random stops only at pool exhaustion.

If a pool is smaller than the requested budget, retain its final decision and
report the actual number of measurements. Normalized regret-budget AUC uses
every integer prefix on a **common 0..16 horizon**, holding decisions constant
after stopping or pool exhaustion. Smaller AUC is better. Do not compare AUCs
normalized over different pool lengths or different profiling-time ranges.

The summary keeps all configurations, initially nonzero-regret configurations,
initially zero-regret configurations, and model groups separately. These are
descriptive groupings, not independent statistical samples. Intermediate harm
is retained, even if a later measurement fixes the decision.

Per-prefix costs include recorded profiling wall time, predictor fitting,
ranking and acquisition calculation. Algorithm time includes computing the
next selection at each prefix to evaluate the stopping rule. Original candidate
generation, Oracle scoring and structure preparation are outside that figure;
preparation is recorded separately. Wall times come from concurrent CPU replay
workers and are not end-to-end GPU execution speedups. Full trajectories are
saved in compressed ignored checkpoints; published JSON contains curves and
complete selection traces. The command resumes completed configurations.

The budget runner uses a NumPy batch of the same makespan recurrence when all
candidates share a fusion partition and interference is zero. It retains
directional/shared link contention and send blocking. Mixed partitions and
nonzero interference fall back to the original scheduler. This internal path
does not compute memory/detail/critical-path outputs used elsewhere; it is
limited to fixed-pool ranking. Tests compare exact costs, sorting, and complete
policy trajectories with the original scheduler.

Optional figure export (Python environment with ReportLab):

```bash
python cost_model/plot_e2_budget_curves.py \
  --report data/bw1000_e2/e2c_graph_wide_budget_curves.json \
  --out-dir data/bw1000_e2/budget_figures
```

The export reads the compressed trajectory checkpoints and contains SVG figures
and a CSV table at **every integer budget**. This exposes temporary regressions
between the six requested reporting budgets. Keep the checkpoints or replay the
full runner before regenerating figures.

## Hypothetical-refit ablation

```bash
python -m cost_model.e2_refit_evaluation --workers 4 \
  --output data/bw1000_e2/e2c_refit_evaluation.json
python -m cost_model.audit_e2_refit_stops \
  --report data/bw1000_e2/e2c_refit_evaluation.json \
  --output data/bw1000_e2/e2c_refit_stop_audit.json
python cost_model/plot_e2_budget_curves.py \
  --report data/bw1000_e2/e2c_refit_evaluation.json \
  --out-dir data/bw1000_e2/refit_figures
```

SROP-Refit keeps the frozen pools, all 84 configurations, nearest-shape mean
predictor, public uncertainty, five quadrature points, analytic profiling-cost
normalization and zero-score stopping. For each hypothetical query outcome,
add its synthetic residual to the visible training rows, recompute the means
of all unmeasured objects, and score the loss of retaining the current Top-1
under those updated means. Average the five losses and divide by the unchanged
cost proxy. Actual reveals use the original full predictor refit.

The numeric standardization and nearest-neighbor identities after adding a
query depend on features, not its hypothetical value. Compile those identities
once per query and reuse them across the five outcomes. Tests compare this
response with literal refitting and with actual revealed-label updates.
Synthetic labels are never persisted or published to the active dataset.

This is an update-aware decision-loss proxy. It does not construct a coherent
joint Bayesian posterior, integrate remaining uncertainties jointly, or satisfy
a Knowledge Gradient guarantee. It should not be described as SROP-Posterior.
It isolates the effect of predictor updates before changing uncertainty or
cost estimation. Baseline policy results and their CPU timings are retained
exactly from the previous completed budget study; new CPU timings come from
this separate replay run.

The stop audit uses hidden outcomes only in the evaluator. It examines the
four prior failures, evaluates helpful single measurements at their initial
and new stopping states, checks the public uncertainty/sample envelope, and
retains negative results. Those labels and post-hoc diagnoses never influence
acquisition. The main evaluation still reports every frozen configuration.

## Training-only uncertainty and actual stopping

```bash
python -m cost_model.e2_uncertainty_coverage \
  --output data/bw1000_e2/e2c_uncertainty_coverage.json
python -m cost_model.e2_uncertainty_stopping --workers 4 \
  --output data/bw1000_e2/e2c_uncertainty_stopping.json
python -m cost_model.audit_e2_uncertainty_stops \
  --report data/bw1000_e2/e2c_uncertainty_stopping.json \
  --output data/bw1000_e2/e2c_uncertainty_stop_audit.json
python cost_model/plot_e2_uncertainty.py \
  --coverage data/bw1000_e2/e2c_uncertainty_coverage.json \
  --stopping data/bw1000_e2/e2c_uncertainty_stopping.json \
  --out-dir data/bw1000_e2/uncertainty_figures
```

At the same public training state, all mean predictions stay identical.
Candidate pools and acquisition formulas stay fixed. Actual reveals still
update means through the original NN fitter.
Compare the legacy family interval, a strict family interval, and a distance
interval with the strict family interval as a floor. Each strict calibration
fold excludes the complete object and refits feature normalization and the NN
mean on the remaining **visible training labels**. No held-out error tunes the
formula. This is empirical calibration, not a conformal guarantee.

For each family with sufficient calibration folds (otherwise global fallback),
let r be the median positive nearest-neighbor distance in those folds, using
1.0 only if every distance is zero. Define scale(d) = 1 + d/r. The distance
interval is max(family q90, q90(abs_error/scale(d)) * scale(query_distance)).
Nominal coverage and the minimum family support follow the original protocol.

The coverage report separates strict zero-shot model/sequence holdouts from
active initial states. Active coverage deduplicates identical training-ID-set
and object-ID pairs across layers/models; these pairs are still not independent
hardware measurements. Report coverage, width, distance bins, unseen families,
and the central-90% log interval score: 2*q + 20*max(0, abs_log_error-q).
Widening intervals without limit therefore incurs a visible penalty.

The stopping runner uses **the original SROP-Proxy**, not SROP-Refit, to isolate
uncertainty. Only its q90 changes. Five samples, analytic cost normalization,
tie-breaking and maximum-score <= 0 stopping stay unchanged. New trajectories
really stop at the recommendation; their final decisions are held to budget
16 for the common-horizon AUC. There is no forced-budget curve for these new
runs. Separate full-reveal profile checks recover every candidate's Oracle
cost/ranking; these are checks, not claims that a stopped run revealed labels.

Wide intervals exposed subtraction cancellation in exp(x)-1 for very negative
x. The counterfactual profile now accepts the positive multiplier directly in
that numerical range. The ordinary legacy arithmetic, interval widths and
sampling policy are unchanged. A regression test covers this wide-interval case.

## Frozen interval-by-update factorial

```bash
python -m cost_model.e2_factorial_evaluation --workers 4 \
  --output data/bw1000_e2/e2c_factorial_evaluation.json
python -m cost_model.verify_e2_factorial \
  --report data/bw1000_e2/e2c_factorial_evaluation.json \
  --output data/bw1000_e2/e2c_factorial_verification.json
python cost_model/plot_e2_factorial.py \
  --report data/bw1000_e2/e2c_factorial_evaluation.json \
  --out-dir data/bw1000_e2/factorial_figures
```

The cells are legacy interval + Proxy, legacy interval + Refit, distance
interval + Proxy, and distance interval + Refit. The first three are copied
exactly from the completed reports. Only the fourth cell is newly executed,
on every frozen configuration with genuine score-based stopping. The mean
predictor, distance formula, five quantiles, analytic cost proxy, zero-score
stop, candidate pools and measurement objects do not change. No minimum
measurement count or correction based on the four known failures is added.

The report includes all configurations, initial-regret strata and model groups.
The interaction is (distance+Refit - distance+Proxy) - (legacy+Refit -
legacy+Proxy), using paired regret-budget AUCs. It is a descriptive factorial
contrast; repeated configurations do not establish statistical significance.
Separate full-reveal profile checks recover all fixed candidates' Oracle
costs. New timing is measured in this run; control timings are preserved from
their original runs, so this is not a simultaneous CPU speed benchmark.

Regeneration of the dense SVG/CSV figures requires the compressed checkpoints
of all four source experiments. The default protocols remain unchanged while
the experimental combination awaits cost validation and independent data.

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
