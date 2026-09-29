# Interview notes

The questions this project invites, and the answers — with the numbers they
depend on. Every figure here comes from `reports/`; re-run `make all` before an
interview so what you quote matches what the repo prints.

---

## The 60-second version

> A two-brand field force makes about 3,000 calls a month. I built a pipeline
> that estimates each prescriber's promotional response curve, then reallocates
> that fixed call budget to where the marginal return is highest. Holding spend
> flat, it lifts promotion-driven prescriptions 15.7%, worth about $4.3M in
> annual gross margin. The reason the gap is that large is that the legacy call
> plan targets prescriber *volume*, while response is driven by prescriber
> *responsiveness* — and the two are only weakly correlated. The segment with the
> steepest response curve gets 3.4% of the call budget today.

## The 10-second version

> Segmentation and call-plan optimisation for a pharma field force. Same budget,
> better targeting, +15.7% promotional TRx and $4.3M annual margin — the key
> insight being that the current plan is built on volume rather than responsiveness.

---

## Questions you should expect

### "Isn't this just correlation? More calls go to bigger prescribers."

This is the question, and it's the one the project is built around. A
cross-sectional regression of TRx on calls measures the call plan, not the drug.

The response curves are fitted on the panel within-transformation:

```
adjusted_share[i,t] = share[i,t] − mean_share[i] + segment_mean_share
```

Subtracting each prescriber's own mean removes every time-invariant confounder —
specialty, affiliation, baseline brand affinity, payer mix, practice size. What
remains is month-to-month movement in a prescriber's own share against
month-to-month movement in their own call frequency. Channel attribution runs
the same identification as an explicit fixed-effects OLS (n ≈ 110,580, field
detailing t = 35.4).

**Follow-up: "What does that still not fix?"** Time-varying confounders. If a rep
calls more precisely *because* a prescriber started rising, the within-estimator
still absorbs that as a call effect. Only a randomised or stepped-wedge field
test identifies this cleanly. That's stated in the README limitations.

### "Where does the 15.7% come from? That sounds high."

Two things to separate. It's 15.7% of *promotion-driven* TRx, not of total TRx —
promotion-driven volume is 2,127 of the monthly total, so the uplift is about
+335 TRx a month, roughly 0.9% of total volume. Quoting it against total volume
would be the dishonest version of the same number.

The mechanism is the misallocation: 55.1% of current calls sit below break-even
ROI, $306,823 of monthly spend at a median marginal ROI of 0.45. Moving that
spend up the marginal-value ranking is where the lift comes from.

### "Why greedy? Wouldn't an LP be more rigorous?"

Greedy is *exactly* optimal here, not an approximation. The objective is separable
across prescribers, each prescriber's response curve is concave (so marginal
value strictly decreases with frequency), every call unit costs the same, and
there's one capacity constraint per territory. Under those conditions, taking the
highest marginal values until capacity is exhausted is the optimum. An LP returns
the same answer more slowly. If I added cross-prescriber constraints — call-cycle
routing, multi-rep shared accounts — that argument breaks and I'd need a MIP.

### "How do you know the segmentation is real and not just K-Means finding shapes in noise?"

Three separate checks:

1. **Stability**: 12 bootstrap re-fits on 80% subsamples, scored against the
   full-data labelling by Adjusted Rand Index — mean 0.921. Without this a
   segmentation is one random seed away from being a different segmentation.
2. **Held-out validation**: the simulator assigns each prescriber a hidden
   responsiveness class the models never see. Segment ordering by observed
   response matches the true latent ordering at Spearman 0.886, with 1.87×
   enrichment of high responders in the priority segments over random.
3. **Honest reporting of what's weak**: silhouette is 0.205 and essentially flat
   across k = 3…6, which I say out loud rather than presenting k as well
   separated. That's why k is chosen on a composite of silhouette,
   Calinski-Harabasz and Davies-Bouldin, capped at 6 because a segmentation the
   field can't action is worthless regardless of its index scores.

### "Walk me through a bug you found."

The best one: the optimiser was recommending call frequencies far beyond anything
in the data. Two causes.

First, the saturating curve `share(c) = s0 + a·(1 − e^(−b·c))` has an
identification failure mode — when the observed call range is short the curve is
locally linear and `a` and `b` trade off freely. The fit drifted to `a = 0.6` at
the parameter bound with `b → 0`, which is a model claiming unbounded returns to
frequency. Fixed by bounding `b` away from zero and flagging any fit that lands
on a bound as `identified=False`; two of six segments are flagged and treated as
non-responsive.

Second, even a well-behaved curve invites extrapolation. Each segment now carries
`max_supported_calls` — the highest call level with at least 30 observations —
and no recommendation exceeds it. **This lowered the headline uplift**, which is
the point: the earlier number was a modelling artefact.

### "Why is the data synthetic? Doesn't that make it a toy?"

Prescriber-level Rx data is licensed (IQVIA, Symphony) and can't be published, so
the alternative was no project. The simulator is the substantive part of the
choice rather than a shortcut:

- Each prescriber has a hidden responsiveness class with its own `alpha`
  (maximum achievable lift) and `beta` (saturation rate).
- The historical call plan is allocated on decile alone and is deliberately blind
  to responsiveness — reproducing the misallocation real legacy plans contain.
  If I'd generated the call plan from responsiveness, the optimiser would have
  found nothing, which is the tell that a simulation is circular.
- Rep *skill* and rep *call intensity* are independent, so effort can't be
  mistaken for effectiveness.

Ground truth is written to a held-out file, which makes the pipeline
**falsifiable** — the segmentation recovers true responsiveness ordering at 0.886
and the rep-effect residual correlates 0.57 with the simulator's hidden skill
parameter. Every module downstream of `generate_data.py` reads a fixed schema, so
swapping in warehouse extracts changes one file.

### "0.783 AUC isn't great."

Agreed, and AUC isn't the metric that decides whether this ships. The field can
only work a short list, so the number that matters is top-decile precision:
36.8%, which is 2.81× the 13.1% base rate, capturing 27.8% of all decliners in
10% of the population. That's an actionable retention list.

I also benchmarked gradient boosting on identical folds — it lost on holdout AUC
(0.777 vs 0.783) but is better calibrated on Brier score. I shipped logistic
regression because it exports as coefficients a BI layer can score without a
Python runtime. If calibrated probabilities mattered more than portability, GBM
is the pick. Both are reported.

### "How did you prevent leakage?"

Temporal split: features are built strictly from months up to the cut-off, the
label is measured entirely after it. Critically, the feature builder is the *same
function* used in production scoring, applied to a truncated mart — so training
and serving features can't drift apart. The population is also restricted to
prescribers with ≥ 8 baseline TRx, because a 100% drop from 1 TRx is noise that
inflates AUC while producing a useless field list.

### "What would you do differently with more time?"

- **Adstock/carryover.** Effects are contemporaneous, which is why speaker
  programmes look uneconomic at ROI 0.13. Almost certainly an artefact of a
  single-month attribution window — I wouldn't let anyone cut that budget on this
  evidence.
- **A field experiment.** Stepped-wedge on call frequency in a subset of
  territories is the only clean identification of response.
- **Payer and formulary dimensions** — material share drivers that are entirely
  absent here.
- **Hierarchical response model.** Partial pooling would give per-prescriber
  curves with sensible shrinkage instead of segment-level curves applied
  uniformly.

### "What's the weakest part of the project?"

The two unidentified segment curves. I treat them as non-responsive, which is the
conservative reading but not necessarily the true one — it may just be that those
prescribers have too little call variation to learn from. The correct fix is
deliberate frequency variation in those segments to generate the data, which is
an experiment, not a model change.

---

## Numbers worth having memorised

| | |
|---|---|
| Scale | 3,000 prescribers · 40 territories · 2 brands · 24 months · 110,688 mart rows |
| Budget-neutral uplift | +15.7% promo TRx · +$4.32M annual gross margin |
| Profit-max | −5.0% calls · +12.6% promo TRx · $335k saved |
| Waste | 55.1% of calls below break-even · median marginal ROI 0.45 |
| Segmentation | k=6 · silhouette 0.205 · stability ARI 0.921 · Spearman 0.886 vs truth |
| Key gap | Steepest-response segment = 12.9% of value tier but 3.4% of calls |
| Attribution | Detailing ROI 2.25 (t=35.4) · digital 7.43 (t=5.3) · speaker 0.13 |
| Risk model | Holdout AUC 0.783 · CV 0.800±0.014 · top-decile precision 36.8% (2.81×) |
| Territory | Attainment 101.7% · 24/40 above quota · potential R² 0.709 · rep effect r=0.57 |
| Engineering | 21 data-quality checks · 40 tests · ~25s full pipeline |

## Things not to say

- Don't imply the data is real. Lead with "I simulated it, and here's why that's
  the interesting part" — it's a stronger answer than a hedge that gets caught.
- Don't quote +15.7% without the denominator. It's promotion-driven TRx.
- Don't call 0.783 AUC strong. Pivot to top-decile precision, which is the
  decision-relevant metric.
- Don't defend the speaker-programme ROI. Volunteer it as a known limitation of
  contemporaneous attribution.
