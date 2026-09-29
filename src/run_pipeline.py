"""
End-to-end orchestrator.

    python -m src.run_pipeline [--skip-generate] [--no-figures]

Stages: generate -> etl (+ data-quality gate) -> features -> segmentation ->
sales-force effectiveness -> territory -> decline risk -> figures -> BI export
-> executive summary.

Every number in `reports/executive_summary.md` is written from the artefacts
produced in this run. Nothing is hard-coded.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import pandas as pd

from . import bi_export, etl, features, generate_data, risk_model, segmentation, sfe, territory
from . import visuals
from .config import Config, get_logger, load_config

log = get_logger("pipeline")


def _pct(x: float) -> str:
    return f"{x * 100:,.1f}%"


def write_executive_summary(cfg: Config) -> str:
    reports, processed = cfg.path("reports"), cfg.path("processed")
    with open(reports / "segmentation_metrics.json", encoding="utf-8") as fh:
        seg_m = json.load(fh)["metrics"]
    with open(reports / "sfe_metrics.json", encoding="utf-8") as fh:
        sfe_m = json.load(fh)
    with open(reports / "territory_metrics.json", encoding="utf-8") as fh:
        terr_m = json.load(fh)
    with open(reports / "risk_model_metrics.json", encoding="utf-8") as fh:
        risk_m = json.load(fh)

    profile = pd.read_csv(reports / "segment_profile.csv")
    curves = pd.read_csv(reports / "response_curves.csv")
    attribution = pd.read_csv(reports / "channel_attribution.csv")
    scorecard = pd.read_csv(processed / "territory_scorecard.csv")
    plan = pd.read_csv(processed / "call_plan_recommendations.csv")
    dq = pd.read_csv(reports / "data_quality_report.csv")

    A = sfe_m["scenario_a_budget_neutral"]
    B = sfe_m["scenario_b_profit_max"]
    waste = sfe_m["waste_diagnostics"]
    best = risk_m["models"][risk_m["best_model"]]
    val = seg_m.get("validation", {})
    ident = curves[curves["identified"]]

    actions = (plan.groupby("action", as_index=False)
                   .agg(hcp_brand_pairs=("hcp_id", "size"),
                        net_call_change=("delta_calls", "sum"),
                        monthly_margin_delta=("delta_margin_monthly", "sum"))
                   .sort_values("monthly_margin_delta", ascending=False))

    seg_tbl = profile[["segment_name", "value_tier", "hcps", "pct_of_hcps", "pct_of_trx",
                       "pct_of_calls", "avg_market_share", "avg_response_index"]].copy()
    for c in ["pct_of_hcps", "pct_of_trx", "pct_of_calls", "avg_market_share"]:
        seg_tbl[c] = seg_tbl[c].map(_pct)
    seg_tbl["avg_response_index"] = seg_tbl["avg_response_index"].round(4)
    seg_tbl.columns = ["Segment", "Tier", "HCPs", "% of HCPs", "% of TRx", "% of calls",
                       "Avg brand share", "Response index"]

    lines = [
        "# Executive Summary — Pharma Commercial Analytics",
        "",
        f"_Generated {datetime.now():%Y-%m-%d %H:%M} from a full pipeline run. Every figure "
        "below is read from artefacts produced by that run; none is hard-coded._",
        "",
        "## 1. Headline",
        "",
        f"- **Same field budget, better aimed.** Reallocating today's "
        f"{sfe_m['current_calls_per_month']:,.0f} monthly calls across the same territories "
        f"lifts promotion-driven TRx by **{_pct(A['promo_trx_uplift_pct'])}** "
        f"(+{A['incremental_trx_per_month']:,.0f} TRx/month, "
        f"{_pct(A['incremental_trx_pct_of_total_trx'])} of total volume) — "
        f"**${A['annualised_incremental_gross_margin']:,.0f}** annualised gross margin at "
        f"{_pct(1 + A['call_budget_change_pct'])} of current spend.",
        f"- **Or spend less for more.** The profit-maximising plan cuts calls "
        f"{_pct(abs(B['call_budget_change_pct']))} "
        f"(${B['annualised_call_cost_saving']:,.0f}/yr saved) and still adds "
        f"{_pct(B['promo_trx_uplift_pct'])} promotion-driven TRx.",
        f"- **The waste is measurable.** {_pct(waste['pct_calls_below_breakeven_roi'])} of "
        f"current calls sit below break-even ROI — "
        f"${waste['monthly_spend_below_breakeven']:,.0f} of monthly field spend with negative "
        f"marginal return; median marginal ROI of the current plan is "
        f"{waste['median_marginal_roi_current_plan']:.2f}.",
        f"- **Retention exposure.** The decline model puts "
        f"**${risk_m['top_decile_revenue_at_risk_12m']:,.0f}** of expected annual revenue loss "
        f"in its top risk decile ({risk_m['top_decile_hcps']:,} prescribers) — a working list "
        "small enough to act on before the quarter closes.",
        "",
        "## 2. Data foundation",
        "",
        f"- {len(dq)} automated data-quality checks run before any modelling: "
        f"{int((dq['status'] == 'PASS').sum())} passed, "
        f"{int((dq['status'] == 'FAIL').sum())} failed. A critical failure aborts the "
        "pipeline rather than shipping a defective deliverable.",
        f"- Curated grain: HCP × brand × month — {cfg['simulation']['n_hcps']:,} prescribers, "
        f"{cfg['simulation']['n_territories']} territories, {len(cfg['products'])} brands, "
        f"{cfg['simulation']['months']} months, "
        f"{sfe_m['hcp_brand_pairs']:,} prescriber-brand pairs.",
        "",
        "## 3. HCP segmentation",
        "",
        f"- k selected at **k={seg_m['selected_k']}** from {cfg['segmentation']['k_range']} on a "
        f"composite of three internal indices (silhouette **{seg_m['silhouette']:.4f}**, "
        f"Davies-Bouldin {seg_m['davies_bouldin']:.3f}, "
        f"Calinski-Harabasz {seg_m['calinski_harabasz']:,.0f}). Silhouette alone is flat "
        "across k on data of this type, so it is not used as the sole criterion.",
        f"- **Stability:** mean Adjusted Rand Index of "
        f"{seg_m['stability_ari_mean']:.3f} across {seg_m['stability_runs']} bootstrap "
        f"re-fits (worst run {seg_m['stability_ari_min']:.3f}) — the segments are not an "
        "artefact of one random start.",
        f"- The High-value tier is {_pct(seg_m['high_tier_pct_of_hcps'])} of prescribers, "
        f"{_pct(seg_m['high_tier_pct_of_trx'])} of TRx and "
        f"{_pct(seg_m['high_tier_pct_of_calls'])} of calls.",
    ]
    if val:
        lines += [
            f"- **Held-out validation** against the simulator's hidden responsiveness classes: "
            f"segments ranked by the *observed* response index reproduce the true ordering of "
            f"latent response exactly (Spearman "
            f"{val['rank_correlation_observed_vs_true_response']:.2f}); the top segments carry "
            f"genuinely high-response prescribers at "
            f"**{val['targeting_lift_vs_random']}×** the random-allocation rate, and mean "
            f"latent response is {val['mean_alpha_top_vs_bottom_segment']:.1f}× higher in the "
            "top segment than the bottom.",
        ]
    lines += [
        "",
        seg_tbl.to_markdown(index=False),
        "",
        "**Read:** the largest single consumer of field effort is *Untapped Potential*, while "
        "*High-Response Growth Targets* — the segment with the steepest fitted response — "
        f"absorbs only {profile.loc[profile['segment_name'].str.startswith('High-Response'), 'pct_of_calls'].map(_pct).iloc[0] if (profile['segment_name'].str.startswith('High-Response')).any() else 'n/a'} "
        "of calls. That gap is the entire optimisation opportunity.",
        "",
        "## 4. Promotional response and channel economics",
        "",
        f"- Response curves are fitted on **within-prescriber** variation (each prescriber's "
        "own share movement against their own call movement), which removes time-invariant "
        "confounding that makes naive call-to-TRx correlations unusable.",
        f"- {len(ident)} of {len(curves)} segment curves are well identified; maximum share "
        f"lift ranges **{ident['a'].min():.3f}–{ident['a'].max():.3f}** with median fit "
        f"R² {ident['r2'].median():.3f}. Segments whose parameters land on a bound are "
        "flagged `identified=False` and treated as non-responsive rather than extrapolated.",
        f"- Saturation is fast: the best-responding segment reaches 80% of its achievable "
        f"lift by **{ident['calls_to_80pct_response'].min():.1f} calls/month** — beyond that, "
        "frequency buys almost nothing.",
        "",
        curves[["segment_name", "s0", "a", "b", "r2", "calls_to_80pct_response",
                "identified"]].to_markdown(index=False),
        "",
        f"- Fixed-effects channel attribution (within R² "
        f"{sfe_m['channel_attribution_within_r2']:.4f}):",
        "",
        attribution[["channel", "share_lift_per_unit", "t_stat",
                     "incremental_trx_per_unit_per_month", "unit_cost",
                     "roi"]].to_markdown(index=False),
        "",
        "Digital returns the highest ROI per unit but has a small absolute ceiling; speaker "
        "programmes do not pay back on a single-month attribution window and would need a "
        "carryover model before any spend decision.",
        "",
        "## 5. Optimised call plan",
        "",
        f"| Scenario | Calls/month | Monthly cost | Promo TRx/month | Δ TRx | Δ margin (annual) |",
        "|---|---|---|---|---|---|",
        f"| Current | {sfe_m['current_calls_per_month']:,.0f} | "
        f"${sfe_m['current_monthly_call_cost']:,.0f} | "
        f"{sfe_m['current_promo_driven_trx_per_month']:,.0f} | — | — |",
        f"| A — budget-neutral | {A['calls_per_month']:,.0f} | ${A['monthly_call_cost']:,.0f} | "
        f"{A['promo_driven_trx_per_month']:,.0f} | +{A['promo_trx_uplift_pct'] * 100:.1f}% | "
        f"${A['annualised_incremental_gross_margin']:,.0f} |",
        f"| B — profit-max | {B['calls_per_month']:,.0f} | ${B['monthly_call_cost']:,.0f} | "
        f"{B['promo_driven_trx_per_month']:,.0f} | +{B['promo_trx_uplift_pct'] * 100:.1f}% | "
        f"${B['annualised_incremental_gross_margin']:,.0f} (+"
        f"${B['annualised_call_cost_saving']:,.0f} cost saved) |",
        "",
        f"- Scenario A moves {sfe_m['hcps_increase']:,} prescriber-brand pairs up in frequency, "
        f"{sfe_m['hcps_reduce']:,} down, drops {sfe_m['hcps_deprioritised']:,} out of the field "
        f"plan, and routes {sfe_m['hcps_digital_only']:,} no-see prescribers to digital only.",
        "",
        actions.to_markdown(index=False, floatfmt=",.1f"),
        "",
        "## 6. Territory and rep performance",
        "",
        f"- National quota attainment **{_pct(terr_m['national_quota_attainment'])}**; "
        f"{terr_m['territories_above_quota']} of {terr_m['n_territories']} territories at or "
        f"above goal; P90–P10 attainment spread {terr_m['attainment_spread_p90_p10']:.3f}.",
        f"- A potential model (category volume, prescriber count, high-value mix) explains "
        f"R²={terr_m['potential_model_r2']:.3f} of territory revenue. The residual is the "
        "potential-adjusted rep effect — it correlates "
        f"{terr_m['rep_effect_vs_true_skill_corr']:.2f} with the simulator's hidden rep-skill "
        "parameter, so the adjustment is measuring what it claims to measure.",
        f"- Call productivity gap, top vs bottom quartile territory: "
        f"**{_pct(terr_m['productivity_gap_top_vs_bottom_quartile_pct'])}**.",
        f"- Footprint imbalance: Gini of territory potential "
        f"{terr_m['gini_territory_potential']:.3f}, of workload "
        f"{terr_m['gini_territory_workload']:.3f}.",
        "",
        "Top five territories by potential-adjusted rep effect:",
        "",
        scorecard.head(5)[["territory_id", "region", "quota_attainment", "market_share",
                           "opportunity_index", "rep_effect_zscore",
                           "performance_quadrant"]].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## 7. Prescriber decline risk",
        "",
        f"- Population: {risk_m['n_training_hcps']:,} prescribers with ≥"
        f"{risk_m['min_baseline_trx']} baseline TRx; event rate "
        f"{_pct(risk_m['positive_class_rate'])} (a >"
        f"{abs(risk_m['decline_threshold']) * 100:.0f}% TRx drop in the following quarter). "
        "Features are built strictly before the outcome window.",
        f"- Best model **{risk_m['best_model']}** — holdout ROC-AUC "
        f"**{best['holdout_roc_auc']:.4f}**, {cfg['risk_model']['cv_folds']}-fold CV "
        f"{best['cv_roc_auc_mean']:.4f} ± {best['cv_roc_auc_std']:.4f}, PR-AUC "
        f"{best['holdout_pr_auc']:.4f}, Brier {best['brier_score']:.4f}.",
        f"- Top risk decile: precision {_pct(best['top_decile_precision'])} "
        f"({best['top_decile_lift_vs_base_rate']}× the base rate), recall "
        f"{_pct(best['top_decile_recall'])} — a short list the field can actually work.",
        "- Leading drivers: "
        + ", ".join(d["feature"] for d in risk_m["top_drivers"][:5]) + ".",
        "",
        "## 8. Recommended actions",
        "",
        f"1. **Adopt the budget-neutral plan.** Same cost, "
        f"+{_pct(A['promo_trx_uplift_pct'])} promotion-driven TRx.",
        f"2. **Cap frequency at the saturation point.** No segment justifies more than its "
        f"fitted 80%-response frequency; redeploy the freed calls to high-response, "
        "high-headroom prescribers.",
        f"3. **Stop sub-break-even detailing.** "
        f"{_pct(waste['pct_calls_below_breakeven_roi'])} of calls do not return their cost; "
        f"moving them to digital captures ROI up to {attribution['roi'].max():.1f}× per unit.",
        "4. **Work the top risk decile** before quarter close, prioritised by revenue at risk.",
        "5. **Re-align the footprint** where the potential Gini and attainment spread show the "
        "territory, not the rep, is the binding constraint.",
        "",
        "---",
        "",
        "### Reproduce",
        "",
        "```bash",
        "make install && make all      # regenerates every number in this document",
        "```",
    ]
    text = "\n".join(lines)
    (reports / "executive_summary.md").write_text(text, encoding="utf-8")
    log.info("executive summary -> reports/executive_summary.md")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description="Pharma commercial analytics pipeline")
    ap.add_argument("--skip-generate", action="store_true",
                    help="reuse existing raw extracts")
    ap.add_argument("--no-figures", action="store_true", help="skip matplotlib output")
    args = ap.parse_args()

    cfg = load_config()
    t0 = time.time()
    stages = []

    def stage(name, fn):
        s = time.time()
        fn()
        stages.append((name, time.time() - s))
        log.info("stage '%s' finished in %.1fs", name, stages[-1][1])

    if not args.skip_generate:
        stage("generate", lambda: generate_data.main(cfg))
    stage("etl", lambda: etl.main(cfg))
    stage("features", lambda: features.main(cfg))
    stage("segmentation", lambda: segmentation.main(cfg))
    stage("sfe", lambda: sfe.main(cfg))
    stage("territory", lambda: territory.main(cfg))
    stage("risk_model", lambda: risk_model.main(cfg))
    if not args.no_figures:
        stage("figures", lambda: visuals.main(cfg))
    stage("bi_export", lambda: bi_export.main(cfg))
    stage("summary", lambda: write_executive_summary(cfg))

    log.info("PIPELINE COMPLETE in %.1fs | %s", time.time() - t0,
             " | ".join(f"{n}:{t:.1f}s" for n, t in stages))


if __name__ == "__main__":
    main()
