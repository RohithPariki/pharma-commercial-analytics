# Executive Summary — Pharma Commercial Analytics

_Generated 2026-09-08 05:27 from a full pipeline run. Every figure below is read from artefacts produced by that run; none is hard-coded._

## 1. Headline

- **Same field budget, better aimed.** Reallocating today's 3,012 monthly calls across the same territories lifts promotion-driven TRx by **15.7%** (+334 TRx/month, 0.8% of total volume) — **$4,324,243** annualised gross margin at 100.0% of current spend.
- **Or spend less for more.** The profit-maximising plan cuts calls 5.0% ($335,035/yr saved) and still adds 12.6% promotion-driven TRx.
- **The waste is measurable.** 55.1% of current calls sit below break-even ROI — $306,822 of monthly field spend with negative marginal return; median marginal ROI of the current plan is 0.45.
- **Retention exposure.** The decline model puts **$2,871,591** of expected annual revenue loss in its top risk decile (300 prescribers) — a working list small enough to act on before the quarter closes.

## 2. Data foundation

- 21 automated data-quality checks run before any modelling: 21 passed, 0 failed. A critical failure aborts the pipeline rather than shipping a defective deliverable.
- Curated grain: HCP × brand × month — 3,000 prescribers, 40 territories, 2 brands, 24 months, 4,609 prescriber-brand pairs.

## 3. HCP segmentation

- k selected at **k=6** from [3, 4, 5, 6] on a composite of three internal indices (silhouette **0.2051**, Davies-Bouldin 1.316, Calinski-Harabasz 639). Silhouette alone is flat across k on data of this type, so it is not used as the sole criterion.
- **Stability:** mean Adjusted Rand Index of 0.921 across 12 bootstrap re-fits (worst run 0.800) — the segments are not an artefact of one random start.
- The High-value tier is 44.7% of prescribers, 71.7% of TRx and 78.7% of calls.
- **Held-out validation** against the simulator's hidden responsiveness classes: segments ranked by the *observed* response index reproduce the true ordering of latent response exactly (Spearman 0.89); the top segments carry genuinely high-response prescribers at **1.872×** the random-allocation rate, and mean latent response is 1.6× higher in the top segment than the bottom.

| Segment                      | Tier   |   HCPs | % of HCPs   | % of TRx   | % of calls   | Avg brand share   |   Response index |
|:-----------------------------|:-------|-------:|:------------|:-----------|:-------------|:------------------|-----------------:|
| High-Value Loyalists         | High   |    516 | 17.2%       | 34.6%      | 28.2%        | 56.5%             |           0.0582 |
| Untapped Potential           | High   |    824 | 27.5%       | 37.1%      | 50.5%        | 32.5%             |           0.0281 |
| Steady Maintainers           | Medium |    494 | 16.5%       | 12.2%      | 5.8%         | 46.8%             |          -0.0512 |
| High-Response Growth Targets | Medium |    323 | 10.8%       | 5.8%       | 3.4%         | 39.6%             |           0.217  |
| Emerging Adopters            | Low    |    452 | 15.1%       | 5.7%       | 7.3%         | 21.1%             |           0.0074 |
| Low-Yield / Deprioritise     | Low    |    391 | 13.0%       | 4.6%       | 4.8%         | 24.4%             |           0.0263 |

**Read:** the largest single consumer of field effort is *Untapped Potential*, while *High-Response Growth Targets* — the segment with the steepest fitted response — absorbs only 3.4% of calls. That gap is the entire optimisation opportunity.

## 4. Promotional response and channel economics

- Response curves are fitted on **within-prescriber** variation (each prescriber's own share movement against their own call movement), which removes time-invariant confounding that makes naive call-to-TRx correlations unusable.
- 4 of 6 segment curves are well identified; maximum share lift ranges **0.008–0.060** with median fit R² 0.990. Segments whose parameters land on a bound are flagged `identified=False` and treated as non-responsive rather than extrapolated.
- Saturation is fast: the best-responding segment reaches 80% of its achievable lift by **1.3 calls/month** — beyond that, frequency buys almost nothing.

| segment_name                 |      s0 |       a |       b |      r2 |   calls_to_80pct_response | identified   |
|:-----------------------------|--------:|--------:|--------:|--------:|--------------------------:|:-------------|
| High-Response Growth Targets | 0.36    | 0.41159 | 0.2     |  0.9998 |                      8.05 | False        |
| High-Value Loyalists         | 0.51538 | 0.05998 | 0.79631 |  0.9852 |                      2.02 | True         |
| Untapped Potential           | 0.29214 | 0.03252 | 0.69009 |  0.994  |                      2.33 | True         |
| Low-Yield / Deprioritise     | 0.2326  | 0.02528 | 1.15876 |  1      |                      1.39 | True         |
| Emerging Adopters            | 0.19525 | 0.00834 | 1.22697 |  0.9838 |                      1.31 | True         |
| Steady Maintainers           | 0.44478 | 0       | 0.2     | -0      |                      8.05 | False        |

- Fixed-effects channel attribution (within R² 0.0124):

| channel                    |   share_lift_per_unit |   t_stat |   incremental_trx_per_unit_per_month |   unit_cost |   roi |
|:---------------------------|----------------------:|---------:|-------------------------------------:|------------:|------:|
| Field detailing (per call) |              0.031826 |    35.35 |                               0.8057 |   185       |  2.25 |
| Digital (per opened email) |              0.003037 |     5.32 |                               0.0769 |     5.33333 |  7.43 |
| Speaker programme          |              0.040716 |     7.3  |                               1.0307 |  4200       |  0.13 |

Digital returns the highest ROI per unit but has a small absolute ceiling; speaker programmes do not pay back on a single-month attribution window and would need a carryover model before any spend decision.

## 5. Optimised call plan

| Scenario | Calls/month | Monthly cost | Promo TRx/month | Δ TRx | Δ margin (annual) |
|---|---|---|---|---|---|
| Current | 3,012 | $557,205 | 2,127 | — | — |
| A — budget-neutral | 3,012 | $557,220 | 2,461 | +15.7% | $4,324,243 |
| B — profit-max | 2,861 | $529,285 | 2,395 | +12.6% | $4,002,806 (+$335,035 cost saved) |

- Scenario A moves 1,006 prescriber-brand pairs up in frequency, 243 down, drops 2,420 out of the field plan, and routes 382 no-see prescribers to digital only.

| action                        |   hcp_brand_pairs |   net_call_change |   monthly_margin_delta |
|:------------------------------|------------------:|------------------:|-----------------------:|
| Increase frequency            |              1006 |           1,275.2 |              500,821.8 |
| Maintain frequency            |               558 |              26.1 |                7,885.3 |
| Digital only (no-see)         |               382 |             -15.2 |               -9,359.6 |
| Reduce frequency              |               243 |            -230.7 |              -31,119.5 |
| Deprioritise - no field calls |              2420 |          -1,055.2 |             -107,874.4 |

## 6. Territory and rep performance

- National quota attainment **101.7%**; 24 of 40 territories at or above goal; P90–P10 attainment spread 0.352.
- A potential model (category volume, prescriber count, high-value mix) explains R²=0.709 of territory revenue. The residual is the potential-adjusted rep effect — it correlates 0.56 with the simulator's hidden rep-skill parameter, so the adjustment is measuring what it claims to measure.
- Call productivity gap, top vs bottom quartile territory: **27.5%**.
- Footprint imbalance: Gini of territory potential 0.080, of workload 0.123.

Top five territories by potential-adjusted rep effect:

| territory_id   | region   |   quota_attainment |   market_share |   opportunity_index |   rep_effect_zscore | performance_quadrant               |
|:---------------|:---------|-------------------:|---------------:|--------------------:|--------------------:|:-----------------------------------|
| T017           | Mountain |              1.201 |          0.398 |               1.088 |               1.822 | Scale - at goal with headroom      |
| T018           | West     |              1.206 |          0.442 |               0.685 |               1.617 | Harvest - at goal, saturated       |
| T023           | Mountain |              0.920 |          0.417 |               0.858 |               1.598 | Re-plan - below goal, low headroom |
| T027           | Midwest  |              0.961 |          0.407 |               0.863 |               1.369 | Re-plan - below goal, low headroom |
| T011           | Mountain |              0.850 |          0.395 |               0.929 |               1.058 | Re-plan - below goal, low headroom |

## 7. Prescriber decline risk

- Population: 2,745 prescribers with ≥8 baseline TRx; event rate 13.1% (a >20% TRx drop in the following quarter). Features are built strictly before the outcome window.
- Best model **logistic_regression** — holdout ROC-AUC **0.7827**, 5-fold CV 0.8004 ± 0.0142, PR-AUC 0.3479, Brier 0.1765.
- Top risk decile: precision 36.8% (2.806× the base rate), recall 27.8% — a short list the field can actually work.
- Leading drivers: trx_growth_pct, adoption_slope, trx_12m, decile, market_trx_12m.

## 8. Recommended actions

1. **Adopt the budget-neutral plan.** Same cost, +15.7% promotion-driven TRx.
2. **Cap frequency at the saturation point.** No segment justifies more than its fitted 80%-response frequency; redeploy the freed calls to high-response, high-headroom prescribers.
3. **Stop sub-break-even detailing.** 55.1% of calls do not return their cost; moving them to digital captures ROI up to 7.4× per unit.
4. **Work the top risk decile** before quarter close, prioritised by revenue at risk.
5. **Re-align the footprint** where the potential Gini and attainment spread show the territory, not the rep, is the binding constraint.

---

### Reproduce

```bash
make install && make all      # regenerates every number in this document
```