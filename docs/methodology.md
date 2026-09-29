# Methodology

Why each modelling choice was made, and what the alternative would have cost.

---

## 1. Data quality as a gate, not a report

The 21 checks in `src/etl.py` run before any modelling, and a CRITICAL failure
raises `DataQualityError` rather than logging a warning. Checks fall into four
families:

| Family | Examples |
|---|---|
| Structural | primary-key uniqueness on every table, foreign-key integrity |
| Domain | non-negative measures, `trx ≤ market_trx`, `nrx ≤ trx`, `opens ≤ sends` |
| Completeness | calendar completeness, balanced panel, activity/Rx grain match |
| Business | positive quota, no nulls |

`trx ≤ market_trx` and `nrx ≤ trx` matter most: violating either produces brand
shares above 100% or negative switch rates, which survive right through to a
dashboard and destroy trust in everything else on the page. The suite includes
tests that inject each defect and assert the correct check fires.

## 2. Feature families

The HCP base table is organised the way a brand team frames a prescriber, which
keeps the segmentation explainable to non-technical stakeholders:

| Family | Question | Features |
|---|---|---|
| Value | What do they write today? | `trx_12m`, `revenue_12m`, `market_share` |
| Potential | What could they write? | `market_trx_12m`, `unrealized_opportunity` |
| Momentum | Which way are they moving? | `trx_growth_pct`, `adoption_slope`, `share_delta` |
| Engagement | What do we currently invest? | `call_frequency`, `reach_pct`, channel mix |
| Response | What does that investment return? | `promo_response_index`, `promo_contrast` |

**`promo_response_index`** is the load-bearing feature: a per-prescriber OLS slope
of monthly brand **share** on the 3-month moving average of calls.

Estimating it on share rather than raw TRx is deliberate. A TRx-on-calls slope is
dominated by the prescriber's own category volume, which varies far more than
their brand preference does; dividing that out first roughly doubles the
signal-to-noise ratio. `promo_contrast` (share in called months minus share in
quiet months) is a non-parametric companion that is robust when a prescriber has
little call variation.

Individual-level precision is inherently limited — roughly 24 monthly
observations against binomial share noise — so the per-prescriber estimate is
noisy by construction. It is used as a *clustering input*, where averaging across
hundreds of prescribers recovers the group structure, not as a per-prescriber
verdict.

## 3. Segmentation

**Transform → PCA → K-Means.** Yeo-Johnson handles the heavy right skew in
volume features; standardisation prevents scale dominance; PCA at 90% variance
de-correlates before a distance-based algorithm runs, so no latent factor
(value and potential move together) gets silently double-weighted.

**Choosing k.** The silhouette curve is flat — 0.200 to 0.205 across k = 3…6 —
so its argmax is effectively arbitrary. k is instead chosen on a composite of
normalised silhouette (50%), Calinski-Harabasz (25%) and inverted Davies-Bouldin
(25%), with ties breaking toward the smaller k. The candidate range stops at 6
because a segmentation a field team cannot action is worth nothing regardless of
its index scores.

**Stability.** Twelve bootstrap re-fits on 80% subsamples are scored against the
full-data labelling with the Adjusted Rand Index: mean 0.921. Without this, a
segmentation is one random seed away from being a different segmentation, and
nobody can tell which one is real.

**Naming.** Clusters are matched to business archetypes by scoring each centroid's
cross-cluster percentile profile against archetype weight vectors, then assigning
greedily on descending fit without repeats. The label — not the cluster id — is
what reaches the field report, so it has to be both meaningful and distinct.

**Two outputs, not one.** The behavioural K-Means segments sit alongside a
deterministic Value × Opportunity 2×2 (Priority Invest / Protect / Grow /
Deprioritise). The grid is stable across refresh cycles even when cluster ids
shuffle, which is what makes it safe to embed in an operating rhythm.

## 4. Promotional response

The model is the standard saturating form:

```
share(c) = s0 + a · (1 − exp(−b · c))
```

`a` is the maximum achievable share lift from detailing; `b` controls how fast it
saturates; `c` is the trailing 3-month call average.

**Identification.** Fitted on within-prescriber variation (see README, problem 1).
Cross-sectional fitting recovers the call plan, not the drug's response.

**Bounds and flags.** `b ∈ [0.2, 2.0]` prevents the degenerate near-linear fit
where `a` and `b` trade off freely. Any fit landing on a bound is flagged
`identified=False` and treated as non-responsive.

**No extrapolation.** Each segment's `max_supported_calls` is the highest call
level with ≥ 30 observations. No recommendation exceeds it.

**Attribution.** A within-prescriber fixed-effects OLS of share on calls, email
opens and speaker programmes, with de-meaning done via `np.bincount` rather than
dummy variables (thousands of fixed effects would otherwise dominate the design
matrix). Standard errors use the residual degrees of freedom net of the absorbed
effects.

## 5. Call-plan optimisation

**Marginal economics.** For a prescriber at frequency `c`, the next call yields

```
Δshare  = a · (e^(−b·c) − e^(−b·(c+1)))
ΔTRx    = Δshare × monthly category volume
margin  = ΔTRx × net revenue per TRx × gross margin
ROI     = margin ÷ fully-loaded call cost
```

**Constraints.** Territory capacity (budget-neutral, or physical capacity,
whichever binds), per-prescriber frequency cap = min(compliance cap, evidence
cap), no field calls to no-see prescribers, and — in scenario B — a marginal
ROI ≥ 1 floor.

**Why greedy is exact here.** The objective is separable across prescribers and
concave in frequency, every call unit costs the same, and there is one capacity
constraint per territory. Under those conditions selecting the highest marginal
values until capacity is exhausted is provably optimal. An LP or MIP would return
the same answer far more slowly.

## 6. Territory and rep performance

Raw attainment ranks the territory a rep was handed, not the rep. The scorecard
therefore reports both attainment and a **potential-adjusted rep effect**: the
residual from an OLS of log territory revenue on log category volume, log
prescriber count and high-value prescriber mix (R² = 0.709).

The adjustment is validated against the simulator's hidden `rep_skill_index`:
the residual correlates 0.57 with true skill, while raw attainment correlates far
more weakly. Rep skill and rep call intensity are generated independently in the
simulator precisely so that effort cannot masquerade as effectiveness.

Alignment imbalance is measured with a Gini coefficient over territory potential
and workload — the standard diagnostic for whether the footprint itself, rather
than the reps, is the binding constraint.

## 7. Decline-risk model

**Label.** A >20% drop in brand TRx in the following quarter versus the preceding
quarter.

**Population.** Restricted to prescribers with ≥ 8 baseline TRx: a 100% drop from
1 TRx is noise, and including it inflates AUC while producing a useless field
list.

**Leakage control.** Features are built strictly from months up to the cut-off;
the label is measured entirely after it. The feature builder is the same function
used in production scoring, applied to a truncated mart — so training and serving
features cannot drift apart.

**Model choice.** A scaled, class-balanced logistic regression is benchmarked
against gradient boosting on identical folds. Logistic regression won on holdout
ROC-AUC (0.783 vs 0.777 in the reference run) and ships as coefficients, which a
BI layer can score without a Python runtime. Gradient boosting is better
calibrated (lower Brier); if calibrated probabilities mattered more than
portability, that would be the pick. Both are reported.

**Decision metric.** ROC-AUC is reported alongside top-decile precision and lift,
because the field can only work a short list. Top-decile precision of 36.8% at
2.81× the base rate is what actually determines whether the model is useful.

**Revenue at risk** is an expected loss — `P(decline) × revenue × decline
threshold` — not probability-weighted total revenue. A flagged prescriber is
assumed to lose the threshold share of their book, not all of it.

## 8. What would change with real data

| Simulated here | Real engagement |
|---|---|
| Generated Rx panel | IQVIA Xponent / Symphony, DDD or projected TRx |
| Assumed call costs | Finance-supplied fully-loaded cost per call |
| No payer dimension | Formulary status, prior-auth burden, payer mix by plan |
| No competitor promotion | Competitive share of voice where licensed |
| Contemporaneous effects | Adstock / carryover with decay estimated per channel |
| Observational response | Stepped-wedge or geo-randomised field test |
| One rep per territory | Multi-rep alignments, shared accounts, call-cycle logistics |
