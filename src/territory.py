"""
Territory and representative performance.

Raw attainment ranks reps by the territory they were handed, not by how well
they worked it. This module therefore reports two things side by side:

  * ATTAINMENT  - revenue vs quota, market share, growth, call productivity.
  * REP EFFECT  - the residual from an OLS regression of territory revenue on
                  territory market potential (category TRx, prescriber count,
                  decile mix). A positive residual means the rep beat what the
                  territory alone predicts. This is the fair comparison.

It also quantifies alignment imbalance with a Gini coefficient over territory
potential - the standard diagnostic for whether the field footprint itself, not
the reps, is the problem.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import Config, get_logger, load_config

log = get_logger("territory")


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def build_scorecard(mart: pd.DataFrame, sales: pd.DataFrame, dim_territory: pd.DataFrame,
                    segments: pd.DataFrame, plan: pd.DataFrame, cfg: Config
                    ) -> tuple[pd.DataFrame, dict]:
    window = cfg["features"]["analysis_window_months"]
    last_idx = int(mart["month_index"].max())
    w_start = last_idx - window + 1
    cur = mart[mart["month_index"] >= w_start]
    prior = mart[(mart["month_index"] >= w_start - window) & (mart["month_index"] < w_start)]

    base = cur.groupby("territory_id", as_index=False).agg(
        trx_12m=("trx", "sum"), nrx_12m=("nrx", "sum"),
        market_trx_12m=("market_trx", "sum"), revenue_12m=("revenue", "sum"),
        calls_12m=("calls", "sum"), samples_12m=("samples_dropped", "sum"),
        speaker_12m=("speaker_programs", "sum"), hcps_covered=("hcp_id", "nunique"))
    prior_agg = prior.groupby("territory_id", as_index=False).agg(
        trx_prior_12m=("trx", "sum"), revenue_prior_12m=("revenue", "sum"))
    sc = base.merge(prior_agg, on="territory_id", how="left")

    months = sorted(mart["month"].unique())[-window:]
    q = (sales[sales["month"].isin(months)].groupby("territory_id", as_index=False)
              .agg(quota_revenue_12m=("quota_revenue", "sum"),
                   actual_revenue_12m=("revenue", "sum")))
    sc = sc.merge(q, on="territory_id", how="left")

    seg_terr = segments.groupby("territory_id", as_index=False).agg(
        hcps_total=("hcp_id", "size"),
        high_value_hcps=("value_tier", lambda s: int((s == "High").sum())),
        opportunity_revenue=("opportunity_revenue", "sum"),
        avg_response_index=("promo_response_index", "mean"))
    sc = sc.merge(seg_terr, on="territory_id", how="left")

    plan_terr = plan.groupby("territory_id", as_index=False).agg(
        current_calls_monthly=("current_calls_monthly", "sum"),
        recommended_calls_monthly=("recommended_calls_monthly", "sum"),
        projected_incremental_trx_monthly=("delta_trx_monthly", "sum"),
        projected_incremental_margin_monthly=("delta_margin_monthly", "sum"))
    sc = sc.merge(plan_terr, on="territory_id", how="left")
    sc = sc.merge(dim_territory, on="territory_id", how="left")

    # ---- KPIs -----------------------------------------------------------
    sc["quota_attainment"] = sc["actual_revenue_12m"] / sc["quota_revenue_12m"]
    sc["market_share"] = sc["trx_12m"] / sc["market_trx_12m"]
    sc["yoy_growth_pct"] = np.where(sc["trx_prior_12m"] > 0,
                                    (sc["trx_12m"] - sc["trx_prior_12m"])
                                    / sc["trx_prior_12m"], 0.0)
    sc["trx_per_call"] = sc["trx_12m"] / sc["calls_12m"].replace(0, np.nan)
    sc["revenue_per_call"] = sc["revenue_12m"] / sc["calls_12m"].replace(0, np.nan)
    sc["calls_per_hcp_per_month"] = sc["calls_12m"] / sc["hcps_total"] / window
    sc["coverage_pct"] = sc["hcps_covered"] / sc["hcps_total"]
    sc["nrx_ratio"] = sc["nrx_12m"] / sc["trx_12m"].replace(0, np.nan)
    sc["opportunity_index"] = (sc["opportunity_revenue"]
                               / sc["opportunity_revenue"].mean()).round(3)

    # ---- potential-adjusted rep effect ----------------------------------
    X = np.column_stack([
        np.ones(len(sc)),
        np.log1p(sc["market_trx_12m"].to_numpy(float)),
        np.log1p(sc["hcps_total"].to_numpy(float)),
        sc["high_value_hcps"].to_numpy(float) / sc["hcps_total"].to_numpy(float),
    ])
    y = np.log1p(sc["revenue_12m"].to_numpy(float))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    resid = y - pred
    r2 = 1 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())
    sc["expected_revenue_12m"] = np.expm1(pred).round(2)
    sc["rep_effect_residual"] = resid.round(4)
    sc["rep_effect_zscore"] = ((resid - resid.mean()) / resid.std(ddof=0)).round(3)
    sc["rep_effect_percentile"] = (sc["rep_effect_zscore"].rank(pct=True) * 100).round(1)

    sc["performance_quadrant"] = np.select(
        [(sc["quota_attainment"] >= 1) & (sc["opportunity_index"] >= 1),
         (sc["quota_attainment"] >= 1) & (sc["opportunity_index"] < 1),
         (sc["quota_attainment"] < 1) & (sc["opportunity_index"] >= 1)],
        ["Scale - at goal with headroom", "Harvest - at goal, saturated",
         "Turnaround - below goal with headroom"],
        # Below goal AND low headroom is usually a goal-setting or alignment problem,
        # not a rep problem - the rep-effect z-score is the column to read here.
        default="Re-plan - below goal, low headroom")

    sc = sc.sort_values("rep_effect_zscore", ascending=False).reset_index(drop=True)
    sc["performance_rank"] = np.arange(1, len(sc) + 1)

    # ---- footprint diagnostics ------------------------------------------
    corr_skill = float(np.corrcoef(sc["rep_effect_zscore"],
                                   sc["rep_skill_index"])[0, 1]) if "rep_skill_index" in sc else np.nan
    summary = {
        "n_territories": int(len(sc)),
        "national_quota_attainment": round(float(sc["actual_revenue_12m"].sum()
                                                 / sc["quota_revenue_12m"].sum()), 4),
        "territories_above_quota": int((sc["quota_attainment"] >= 1).sum()),
        "national_market_share": round(float(sc["trx_12m"].sum()
                                             / sc["market_trx_12m"].sum()), 4),
        "attainment_spread_p90_p10": round(float(sc["quota_attainment"].quantile(0.9)
                                                 - sc["quota_attainment"].quantile(0.1)), 4),
        "gini_territory_potential": round(gini(sc["market_trx_12m"].to_numpy()), 4),
        "gini_territory_workload": round(gini(sc["calls_12m"].to_numpy()), 4),
        "potential_model_r2": round(r2, 4),
        "rep_effect_vs_true_skill_corr": round(corr_skill, 4),
        "top_quartile_trx_per_call": round(float(sc["trx_per_call"].quantile(0.75)), 3),
        "bottom_quartile_trx_per_call": round(float(sc["trx_per_call"].quantile(0.25)), 3),
    }
    summary["productivity_gap_top_vs_bottom_quartile_pct"] = round(
        summary["top_quartile_trx_per_call"] / summary["bottom_quartile_trx_per_call"] - 1, 4)
    log.info("territory scorecard: national attainment %.1f%%, potential model R2=%.3f",
             100 * summary["national_quota_attainment"], r2)
    return sc, summary


def monthly_trend(mart: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    trend = mart.groupby(["month", "territory_id", "brand"], as_index=False).agg(
        trx=("trx", "sum"), nrx=("nrx", "sum"), market_trx=("market_trx", "sum"),
        revenue=("revenue", "sum"), calls=("calls", "sum"))
    trend["market_share"] = trend["trx"] / trend["market_trx"].replace(0, np.nan)
    return trend


def main(cfg: Config | None = None) -> tuple[pd.DataFrame, dict]:
    cfg = cfg or load_config()
    processed, raw, reports = cfg.path("processed"), cfg.path("raw"), cfg.path("reports")
    mart_pq = processed / "monthly_mart.parquet"
    mart = pd.read_parquet(mart_pq) if mart_pq.exists() else pd.read_csv(
        processed / "monthly_mart.csv")
    sales = pd.read_csv(raw / "fact_sales.csv")
    dim_territory = pd.read_csv(raw / "dim_territory.csv")
    segments = pd.read_csv(processed / "hcp_segments.csv")
    plan = pd.read_csv(processed / "call_plan_recommendations.csv")

    sc, summary = build_scorecard(mart, sales, dim_territory, segments, plan, cfg)
    sc.to_csv(processed / "territory_scorecard.csv", index=False)
    monthly_trend(mart, sales).to_csv(processed / "territory_monthly_trend.csv", index=False)
    with open(reports / "territory_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return sc, summary


if __name__ == "__main__":
    main()
