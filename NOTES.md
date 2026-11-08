# NOTES — decisions, substitutions, and surprises

Running log of the study. Every deviation from the spec is recorded here
with its reasoning and the evidence that motivated it.

---

## D1. `conformal_quantile`: the spec's `method="higher"` is off by one order statistic

**Spec §4.2** gives:

```python
level = np.ceil((n + 1) * (1 - alpha)) / n
return float(np.quantile(scores, level, method="higher"))
```

and calls `method="higher"` mandatory.

**What we found.** Write `k = ceil((n + 1) * (1 - alpha))`. The theory (Vovk et al.
2005; Lei et al. 2018) wants the **k-th smallest** calibration score. numpy's
virtual index for `method="higher"` is `ceil(q * (n - 1))`; at `q = k / n` that is
`ceil(k - k/n) = k` in 0-based indexing, i.e. the **(k+1)-th smallest**, whenever
`k < n`.

Measured marginal coverage, 200k replicates, uniform scores, α = 0.1:

| n_cal | tight (k-th) | spec's `"higher"` ((k+1)-th) |
|------:|-------------:|-----------------------------:|
| 19    | 0.9007       | 0.9508                       |
| 39    | 0.9006       | 0.9254                       |
| 99    | 0.8984       | 0.9084                       |

`"higher"` is **valid** (never below 1−α) but **conservative**, and the excess is
largest at small n — precisely the regime this paper studies.

**Why this mattered enough to change.** Spec §7.3 (called "the most important
test") requires empirical Split-CP coverage *within 3 binomial SE of 1−α*. That
gate is unsatisfiable under `"higher"` (0.9508 vs 0.9000 at n_cal = 50-ish). So
§4.2 and §7.3 contradict each other, and §7.3 pins the author's actual intent.
Independently: publishing a Split-vs-CV+ comparison with a needlessly
conservative Split baseline would inflate Split's coverage *and* set size and
thereby bias the headline results **in favour of our own thesis**. That is the
one direction of error we cannot afford.

**Decision.** Default `QUANTILE_METHOD_DEFAULT = "tight"`, implemented as the
exact k-th order statistic via `np.partition`. `method="higher"` remains
available and is pinned by a unit test for spec fidelity. The choice is a single
module-level constant and is recorded in every result row
(`quantile_method` column), so the whole study can be re-run the other way.

**Sub-finding.** `np.quantile(s, k / n, method="inverted_cdf")` looks like the
right tight substitute but is **not safe**: `k / n * n` is not always exactly `k`
in binary floating point, and it selects the wrong order statistic in **71 of the
~9000** `(n, alpha)` combinations swept (first failure: n = 35, α = 0.2, k = 29,
returns the 30th). `np.partition` avoids any float level and is exact in all
9000. Pinned by `test_inverted_cdf_shortcut_is_not_safe`.

---

## D2. Spec §7.1's requested counterexample does not exist (honest negative finding)

**Spec §7.1** asks us to "assert that `method="higher"` matters: construct a case
where linear interpolation would under-cover."

**We could not, because no such case exists at the conformal level.** At level
`k / n`, linear interpolation's virtual index `k - k/n` always lies strictly
between the k-th and (k+1)-th order statistics, so linear returns a value **≥ the
tight quantile** and therefore still covers. Swept every
`n ∈ [3, 800) × α ∈ {0.05, 0.1, 0.2}` (2391 cases): **zero** violations.

Measured coverage, 60k replicates, α = 0.1:

| n | `np.quantile(s, 1−α)` (no correction) | linear @ `k/n` | tight | `"higher"` |
|--:|--------------------------------------:|---------------:|------:|-----------:|
| 19  | **0.8612** | 0.9039 | 0.9011 | 0.9499 |
| 39  | **0.8806** | 0.9015 | 0.8996 | 0.9248 |
| 99  | **0.8905** | 0.8995 | 0.8984 | 0.9088 |
| 199 | **0.8942** | 0.8986 | 0.8980 | 0.9029 |
| 499 | **0.8979** | 0.8998 | 0.8996 | 0.9015 |

The genuine under-coverage trap is **omitting the `(n+1)/n` finite-sample
correction** and quantiling at `1 − α` directly — 0.8612 at n = 19. The test was
rewritten to pin that real failure mode
(`test_missing_finite_sample_correction_under_covers`), plus a second test
recording the negative finding
(`test_linear_at_the_conformal_level_is_conservative_not_broken`).

So the interpolation method still matters, but for tightness, not validity. The
paper text must not repeat the common claim that linear interpolation at the
conformal level breaks the guarantee.

---

## D3. SSCV gets an explicit size-0 bin

Spec §4.10 lists SSCV bins `[1], [2-3], [4-6], [7+]`. Empty prediction sets
(size 0) fall in none of them. Folding them into `[1]` would hide an empty-set
failure; dropping them would silently discard test points. `metrics.SSCV_BINS`
therefore carries an explicit `"0"` bin. Empty bins are skipped, never reported
as NaN.

---

## D4. GPU: obtained on the second attempt, after the CUDA wheel was fetched manually

The target machine has an RTX 2060 (6 GB, sm_75) but shipped `torch 2.7.0+cpu`.
The CUDA wheel is only distributed from `download.pytorch.org`, which measured
**60-140 KB/s** here from every mirror tried (pytorch.org, aliyun, tsinghua,
pypi.org) -- i.e. the ~1 Mbit link, not the mirror, was the bottleneck. Two
in-process attempts aborted with `WinError 10054`. The user downloaded
`torch-2.7.0+cu126-cp311-cp311-win_amd64.whl` (2 770 890 365 bytes) out of band;
it was verified (zip integrity OK, 11 024 entries) and installed with
`--force-reinstall --no-deps`.

Confirmed live: `torch 2.7.0+cu126`, CUDA 12.6, `NVIDIA GeForce RTX 2060`,
6.00 GB, sm_75.

**Measured speed-up over the CPU build** (TabPFN, `predict` on 150 points):

| n_train | CPU | GPU | speed-up |
|--------:|----:|----:|---------:|
| 200  | 2.136 s | 0.098 s | 22x |
| 500  | 4.088 s | 0.100 s | 41x |
| 1000 | 7.608 s | 0.137 s | 55x |
| 2000 | --      | 0.200 s | --  |

Every timing in the final study is therefore **GPU** timing; `device_used`
records `cuda` on each row. Peak VRAM at `n_estimators=4`, n_train=500 is
**0.11 GB**, far inside the 6 GB budget, so the OOM/halve-batch path in
`TabPFNWrapper` is never exercised here (it is kept, and unit-tested, for
smaller cards).

TabPFN also refuses CPU inference above 1000 rows by default; we keep
`TABPFN_ALLOW_CPU_LARGE_DATASET=1` and `ignore_pretraining_limits=True` so the
CPU fallback the spec requires still works on a machine without CUDA.

Aside: CPU thread scaling for TabPFN is poor -- 8 threads 2.136 s vs 1 thread
3.821 s, only 1.8x for 8x the threads. On a CPU-only machine, running several
cells in parallel with 1-2 threads each would beat one cell on 8 threads.

---

## D5. **RQ1's central thesis does not survive measurement** (flag loudly, per section 11)

The paper's thesis is: *TabPFN does no gradient training, `fit()` only stores the
in-context set, therefore CV+/jackknife+ are nearly free for it.*

Measured on wdbc (n_train=200, n_test=150, 1 seed, CPU), total fit+predict
seconds and the ratio against the same model's Split-CP cost:

| model | Split | CV+ K=5 | CV+ K=10 | **K5/Split** | **K10/Split** |
|---|---|---|---|---|---|
| tabpfn  | 8.035 s | 41.94 s | 88.64 s | **5.22** | **11.03** |
| xgboost | 0.032 s | 0.168 s | 0.365 s | **5.21** | **11.35** |
| rf      | 0.339 s | 1.733 s | 3.467 s | **5.11** | **10.22** |
| logreg  | 0.010 s | 0.050 s | 0.100 s | **4.95** | **9.86** |

**The cost ratio is ~K for every model, TabPFN included.** There is no free
lunch: CV+ costs TabPFN the same relative multiple it costs a gradient-trained
baseline, and in *absolute* seconds TabPFN is 250-2700x more expensive than the
baselines at every K.

Why the thesis fails. The premise about `fit()` is correct -- TabPFN fit is
0.166 s against 7.87 s of predict, i.e. 2% of cell cost (for XGBoost the split is
the other way round: fit is 85% of cost). But CV+ does not only refit K times, it
also **re-predicts K times**: every fold model must score every test point under
its own context. For a gradient-trained model that second cost is negligible;
for an in-context learner it *is* the cost. The saving on the free half is
exactly cancelled by the K-fold multiplication of the expensive half.

This is a real negative result, not an implementation artifact. The work per cell
is `(n_train + K * n_test) * context`, which is minimal for the algorithm as
specified in section 4.5 -- CV+ *requires* each fold model's prediction at every
test point. Measured cost tracks that formula to within a few percent.

What survives, and what the paper should claim instead:
  * The statistical results (RQ2 stability, RQ4 conservativeness, RQ5 fairness,
    RQ6 crossover) are untouched -- they are about coverage, not seconds.
  * The honest RQ1 finding is: **the cost of CV+ moves from training to
    inference for in-context learners, and the K-fold multiplier survives that
    move.** The interesting question becomes whether the RQ2 variance reduction
    is worth a ~Kx bill, which is exactly what the Pareto figure (F2) shows.
  * `fit_time ~ 0` for TabPFN is confirmed and worth reporting -- it is just not
    sufficient to make CV+ cheap.

Do **not** write "CV+ is nearly free for TabPFN" anywhere in the paper.

---

## D6. Compute budget and experiment scale

Cost scales as `(n_train + K * n_test) * context`, which the measurements match
to within a few percent. On GPU, TabPFN at n_train=500 / n_test=350:

| n_estimators | fit | predict(350) | implied K=10 cell |
|---|---|---|---|
| 1 | 0.081 s | 0.126 s | 2.1 s |
| 2 | 0.088 s | 0.175 s | 2.6 s |
| **4 (default)** | **0.101 s** | **0.243 s** | **3.4 s** |
| 8 | 0.151 s | 0.506 s | 6.6 s |

Note that `fit` is nearly flat in `n_estimators` while `predict` doubles -- an
independent confirmation of the "fit is free, inference is not" story in D5.

Settings for the final run, all recorded in every result row:
  * `n_train = 500`, `test_cap = 300`. Defensible because the paper is explicitly
    about the small-data regime, and 500 is also the spec's own jackknife+
    ceiling. 300 test points give a binomial SE of ~1.7% on coverage.
  * TabPFN `n_estimators = 4`, the library default (GPU makes it affordable).
  * jackknife+ demoted to a dedicated timing sub-study rather than a full
    20-seed x 15-dataset arm: at K = n_train each cell costs ~100x a K=10 cell.

Nothing else is reduced: all 15 datasets, all 4 models, all 20 seeds, all 3
scores, all 3 alphas are retained. An earlier CPU-era run used n_train=200 /
test_cap=150 / `n_estimators=1`; those cells were **deleted**, not merged, because
CPU and GPU timings must not be mixed in an RQ1 table.

---

## D7. Gate 8 caught a real bug: XGBoost rejects a fold that is missing a class

`tests/test_determinism.py::test_missing_class_still_yields_aligned_columns`
failed for `xgboost` with:

    ValueError: Invalid classes inferred from unique values of `y`.
    Expected: [0 1], got [0 2]

XGBoost's sklearn wrapper requires the training labels to be a contiguous
`0..m-1` set. A stratified CV fold that happens to contain no examples of a rare
class produces exactly the gapped label set it rejects, so **every CV+ cell on a
multiclass dataset with a rare class would have crashed**. This is precisely the
failure mode spec section 7.8 exists to catch.

Fix: `GlobalClassAligned.fit` now relabels to a contiguous local range before
fitting (`np.searchsorted(seen_classes_, y)`), and stores `global_of_column_`,
the map from the estimator's own column order back to global label indices.
`predict_proba` scatters through that map. The remap is applied unconditionally
rather than special-cased for one library, because it is a no-op for estimators
that tolerate gaps. Classes absent from a fold keep probability exactly 0.

---

