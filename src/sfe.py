"""
Sales-Force Effectiveness (SFE).

This is where the segmentation is converted into money. Four steps:

  1. RESPONSE CURVES - for each behavioural segment, fit the saturating promotional
     response function

         share(c) = s0 + a * (1 - exp(-b * c))

     where c is the trailing 3-month call average. `a` is the maximum achievable
     share lift from detailing and `b` controls how fast it saturates. Fitted by
     non-linear least squares on observation-weighted call-level bucket means.

  2. MULTICHANNEL ATTRIBUTION - a within-HCP (fixed-effects) OLS of brand share on
     calls, email opens and speaker programmes. De-meaning by prescriber removes
     time-invariant prescriber quality, which is the main confounder in naive
     call-to-TRx correlations.

  3. MARGINAL ECONOMICS - for every HCP x brand, the incremental TRx from the
     next call is the derivative of the fitted curve evaluated at the current
     frequency, scaled by that prescriber's monthly category volume. Multiply by
     net revenue and gross margin, divide by fully-loaded call cost -> marginal ROI.

  4. BUDGET-NEUTRAL OPTIMISER - greedy allocation of call units in descending
     marginal value, subject to (a) a territory capacity constraint no larger than
     the current call load, (b) a per-HCP frequency cap, (c) an ROI >= 1 floor and
     (d) a no-see compliance rule. The result is a call plan that costs no more
     than today's and returns more.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit

from .config import Config, get_logger, load_config

log = get_logger("sfe")

MIN_OBS_PER_BUCKET = 30
# Parameter bounds for the response curve: [baseline share, max lift, saturation rate].
# `b` is bounded away from zero because a near-linear fit makes `a` unidentifiable.
B_LOWER = [0.0, 0.0, 0.20]
B_UPPER = [0.95, 0.50, 2.00]


def response_fn(c: np.ndarray, s0: float, a: float, b: float) -> np.ndarray:
    return s0 + a * (1.0 - np.exp(-b * c))


# --------------------------------------------------------------------- curves
def fit_response_curves(panel: pd.DataFrame, group_col: str = "segment_name") -> pd.DataFrame:
    """Fit share(c) per segment using WITHIN-prescriber variation only.

    Fitting on raw cross-sectional bucket means does not identify `a` and `b`
    separately - prescribers who receive four calls differ from those who receive
    none in ways that have nothing to do with detailing, and the optimiser then
    drifts to the parameter bounds. The fix is the panel within-transformation:

        adjusted_share[i,t] = share[i,t] - mean_share[i] + segment_mean_share

    Subtracting each prescriber's own mean removes every time-invariant confounder
    (specialty, affiliation, baseline brand affinity, payer mix); adding the
    segment mean back restores an interpretable level. What remains is month-to-
    month movement in a prescriber's own share against month-to-month movement in
    their own call frequency, which is the quantity the call plan can act on.
    """
    panel = panel.copy()
    panel["market_share"] = panel["market_share"].fillna(0.0)
    panel["call_bucket"] = panel["calls_ma3"].round(0).clip(0, 4)
    pair_key = panel["hcp_id"].astype(str) + "|" + panel["product_id"].astype(str)
    panel["_pair_mean_share"] = panel.groupby(pair_key)["market_share"].transform("mean")

    rows = []
    for seg, g in panel.groupby(group_col, sort=False):
        adjusted = g["market_share"] - g["_pair_mean_share"] + g["market_share"].mean()
        buckets = (pd.DataFrame({"call_bucket": g["call_bucket"], "share": adjusted})
                     .groupby("call_bucket")
                     .agg(share=("share", "mean"), n=("share", "size")).reset_index())
        buckets = buckets[buckets["n"] >= MIN_OBS_PER_BUCKET]
        if len(buckets) < 3:
            log.warning("segment %s: insufficient call variation, flat curve assumed", seg)
            rows.append({group_col: seg, "s0": float(g["market_share"].mean()), "a": 0.0,
                         "b": 0.5, "r2": 0.0, "n_obs": int(len(g)), "n_call_levels": len(buckets),
                         "max_supported_calls": int(buckets["call_bucket"].max())
                         if len(buckets) else 0,
                         "calls_to_80pct_response": np.nan, "identified": False})
            continue

        x = buckets["call_bucket"].to_numpy(float)
        y = buckets["share"].to_numpy(float)
        w = buckets["n"].to_numpy(float)
        try:
            with warnings.catch_warnings():
                # The covariance matrix is not used - only the point estimates are,
                # and they are separately checked against the parameter bounds.
                warnings.simplefilter("ignore", OptimizeWarning)
                popt, _ = curve_fit(
                    response_fn, x, y, p0=[y.min(), 0.12, 0.6], sigma=1.0 / np.sqrt(w),
                    absolute_sigma=False, bounds=(B_LOWER, B_UPPER), maxfev=30000)
        except RuntimeError:  # pragma: no cover
            popt = [float(y.mean()), 0.0, 0.5]
        s0, a, b = (float(v) for v in popt)
        resid = y - response_fn(x, s0, a, b)
        ss_res = float(np.sum(w * resid ** 2))
        ss_tot = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
        at_bound = bool(np.isclose(a, B_UPPER[1], atol=1e-4)
                        or np.isclose(b, B_LOWER[2], atol=1e-4)
                        or np.isclose(b, B_UPPER[2], atol=1e-4))
        rows.append({
            group_col: seg, "s0": round(s0, 5), "a": round(a, 5), "b": round(b, 5),
            "r2": round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else 0.0,
            "n_obs": int(len(g)), "n_call_levels": int(len(buckets)),
            # Highest call frequency with enough observations to fit on. Recommendations
            # are never allowed past this point: a curve that is linear across its
            # observed range will happily promise unlimited lift if extrapolated.
            "max_supported_calls": int(buckets["call_bucket"].max()),
            "calls_to_80pct_response": round(float(-np.log(0.2) / b), 2) if b > 0 else np.nan,
            "identified": not at_bound,
        })
    curves = pd.DataFrame(rows).sort_values("a", ascending=False).reset_index(drop=True)
    log.info("fitted %d response curves | max lift a=%.4f | min a=%.4f | %d well-identified",
             len(curves), curves["a"].max(), curves["a"].min(), int(curves["identified"].sum()))
    return curves


# ---------------------------------------------------------------- attribution
def channel_attribution(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Within-HCP fixed-effects OLS of share on promotional channels."""
    df = panel[["hcp_id", "product_id", "market_share", "calls_ma3",
                "emails_opened", "speaker_programs", "market_trx",
                "net_revenue_per_trx"]].dropna()
    key = df["hcp_id"] + "|" + df["product_id"]
    X_cols = ["calls_ma3", "emails_opened", "speaker_programs"]
    Xd = df[X_cols].to_numpy(float)
    yd = df["market_share"].to_numpy(float)

    # de-mean within prescriber x brand
    codes, _ = pd.factorize(key)
    n_groups = codes.max() + 1
    counts = np.bincount(codes, minlength=n_groups).astype(float)
    for j in range(Xd.shape[1]):
        Xd[:, j] -= (np.bincount(codes, weights=Xd[:, j], minlength=n_groups) / counts)[codes]
    yd = yd - (np.bincount(codes, weights=yd, minlength=n_groups) / counts)[codes]

    beta, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
    resid = yd - Xd @ beta
    dof = max(len(yd) - Xd.shape[1] - n_groups, 1)
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.pinv(Xd.T @ Xd)
    se = np.sqrt(np.diag(xtx_inv) * sigma2)
    r2_within = 1 - float(resid @ resid) / float(yd @ yd)

    econ = cfg["economics"]
    unit_cost = {"calls_ma3": econ["cost_per_call"],
                 "emails_opened": econ["cost_per_email"] / 0.3,   # cost per *opened* email
                 "speaker_programs": econ["cost_per_speaker_program"]}
    mean_market_trx = float(df["market_trx"].mean())
    mean_rev = float(df["net_revenue_per_trx"].mean())

    out = []
    for name, coef, err in zip(X_cols, beta, se):
        inc_trx = coef * mean_market_trx                      # incremental TRx per unit/month
        inc_rev = inc_trx * mean_rev * econ["gross_margin"]
        out.append({
            "channel": {"calls_ma3": "Field detailing (per call)",
                        "emails_opened": "Digital (per opened email)",
                        "speaker_programs": "Speaker programme"}[name],
            "share_lift_per_unit": round(float(coef), 6),
            "std_error": round(float(err), 6),
            "t_stat": round(float(coef / err), 2) if err else np.nan,
            "incremental_trx_per_unit_per_month": round(float(inc_trx), 4),
            "gross_margin_per_unit": round(float(inc_rev), 2),
            "unit_cost": unit_cost[name],
            "roi": round(float(inc_rev / unit_cost[name]), 2),
        })
    res = pd.DataFrame(out)
    res.attrs["r2_within"] = round(r2_within, 4)
    res.attrs["n_obs"] = int(len(yd))
    log.info("channel attribution fitted on %s obs (within R2=%.4f)", f"{len(yd):,}", r2_within)
    return res


# ------------------------------------------------------------------ economics
def build_marginal_economics(abt_hp: pd.DataFrame, segments: pd.DataFrame,
                             curves: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    econ, tgt = cfg["economics"], cfg["targeting"]
    window = cfg["features"]["analysis_window_months"]

    df = abt_hp.merge(segments[["hcp_id", "segment_name", "value_tier",
                                "strategic_quadrant"]], on="hcp_id", how="left")
    df = df.merge(curves[["segment_name", "a", "b", "max_supported_calls", "identified"]],
                  on="segment_name", how="left")
    df[["a", "b"]] = df[["a", "b"]].fillna({"a": 0.0, "b": 0.5})
    df["max_supported_calls"] = df["max_supported_calls"].fillna(1).astype(int)
    # Frequency cap = the tighter of the compliance cap and the evidence cap.
    df["call_cap"] = np.minimum(df["max_supported_calls"],
                                tgt["max_calls_per_hcp_per_month"]).astype(int)

    df["current_calls_monthly"] = df["calls_12m"] / window
    df["monthly_market_trx"] = df["market_trx_12m"] / window
    df["value_per_trx"] = df["net_revenue_per_trx"] * econ["gross_margin"]

    max_calls = tgt["max_calls_per_hcp_per_month"]
    for c in range(max_calls):
        delta_share = df["a"] * (np.exp(-df["b"] * c) - np.exp(-df["b"] * (c + 1)))
        inc_trx = delta_share * df["monthly_market_trx"]
        df[f"mtrx_{c + 1}"] = inc_trx
        df[f"mroi_{c + 1}"] = inc_trx * df["value_per_trx"] / econ["cost_per_call"]
    return df


def _allocate(df: pd.DataFrame, capacity: pd.Series, roi_floor: float,
              max_calls: int) -> pd.DataFrame:
    """Greedy marginal-value allocation of call units under a territory capacity.

    Because each prescriber's response curve is concave, marginal value is
    strictly decreasing in frequency. For a separable concave objective with unit
    costs and a single knapsack constraint, greedy selection of the highest
    marginal values is exactly optimal - no heuristic gap.
    """
    units = []
    for c in range(1, max_calls + 1):
        units.append(pd.DataFrame({
            "row": df.index,
            "territory_id": df["territory_id"].to_numpy(),
            "unit": c,
            "inc_trx": df[f"mtrx_{c}"].to_numpy(),
            "roi": df[f"mroi_{c}"].to_numpy(),
            "value": (df[f"mtrx_{c}"] * df["value_per_trx"]).to_numpy(),
            "eligible": ((df["no_see_flag"].to_numpy() == 0)
                         & (df["call_cap"].to_numpy() >= c)),
        }))
    u = pd.concat(units, ignore_index=True)
    u = u[u["eligible"] & (u["roi"] >= roi_floor)]
    u = u.sort_values(["territory_id", "value"], ascending=[True, False])
    u["rank_in_terr"] = u.groupby("territory_id").cumcount() + 1
    u["cap"] = u["territory_id"].map(capacity)
    chosen = u[u["rank_in_terr"] <= u["cap"]]
    return chosen.groupby("row").agg(calls=("unit", "max"))


def _score_plan(df: pd.DataFrame, calls: pd.Series) -> pd.Series:
    """Total promotion-driven TRx implied by a call plan, on the fitted curves."""
    return df["a"] * (1 - np.exp(-df["b"] * calls)) * df["monthly_market_trx"]


def optimise_call_plan(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Two decision scenarios, because 'optimal' depends on the objective.

    SCENARIO A - Budget-neutral reallocation. Spend exactly today's number of
        calls in each territory, but on the prescribers where the fitted curves
        say they earn the most. No ROI floor: the answer to "same money, better
        aimed". This is the primary recommendation.

    SCENARIO B - Profit-maximising plan. Drop every call whose marginal ROI is
        below break-even and cap at physical field capacity. Usually fewer calls,
        higher margin, and it quantifies how much of the current plan is
        value-destroying.
    """
    econ, tgt = cfg["economics"], cfg["targeting"]
    max_calls = tgt["max_calls_per_hcp_per_month"]

    current_by_terr = df.groupby("territory_id")["current_calls_monthly"].sum()
    cap_neutral = current_by_terr.round().astype(int)
    cap_profit = np.minimum(current_by_terr, econ["calls_per_rep_per_month"]).round().astype(int)

    alloc_a = _allocate(df, cap_neutral, roi_floor=0.0, max_calls=max_calls)
    alloc_b = _allocate(df, cap_profit, roi_floor=1.0, max_calls=max_calls)

    plan = df.copy()
    plan["recommended_calls_monthly"] = (
        alloc_a["calls"].reindex(plan.index).fillna(0).astype(int))
    plan["profitmax_calls_monthly"] = (
        alloc_b["calls"].reindex(plan.index).fillna(0).astype(int))

    cur_c = plan["current_calls_monthly"]
    plan["current_plan_trx_uplift"] = _score_plan(plan, cur_c)
    plan["recommended_plan_trx_uplift"] = _score_plan(plan, plan["recommended_calls_monthly"])
    plan["profitmax_plan_trx_uplift"] = _score_plan(plan, plan["profitmax_calls_monthly"])

    plan["delta_calls"] = plan["recommended_calls_monthly"] - cur_c
    plan["delta_trx_monthly"] = (plan["recommended_plan_trx_uplift"]
                                 - plan["current_plan_trx_uplift"])
    plan["delta_margin_monthly"] = plan["delta_trx_monthly"] * plan["value_per_trx"]

    plan["action"] = np.select(
        [plan["no_see_flag"] == 1,
         plan["recommended_calls_monthly"] == 0,
         plan["delta_calls"] >= 0.5,
         plan["delta_calls"] <= -0.5],
        ["Digital only (no-see)", "Deprioritise - no field calls",
         "Increase frequency", "Reduce frequency"],
        default="Maintain frequency")

    window_trx = float(plan["trx_12m"].sum()) / cfg["features"]["analysis_window_months"]
    cost = econ["cost_per_call"]

    def scenario(calls_col: str, uplift_col: str) -> dict:
        calls = float(plan[calls_col].sum())
        uplift = float(plan[uplift_col].sum())
        cur_calls = float(cur_c.sum())
        cur_uplift = float(plan["current_plan_trx_uplift"].sum())
        margin_delta = (uplift - cur_uplift) * float(plan["value_per_trx"].mean())
        margin_delta = float(((plan[uplift_col] - plan["current_plan_trx_uplift"])
                              * plan["value_per_trx"]).sum())
        return {
            "calls_per_month": round(calls, 1),
            "call_budget_change_pct": round((calls - cur_calls) / cur_calls, 4),
            "monthly_call_cost": round(calls * cost, 2),
            "promo_driven_trx_per_month": round(uplift, 1),
            "incremental_trx_per_month": round(uplift - cur_uplift, 1),
            "promo_trx_uplift_pct": round((uplift - cur_uplift) / cur_uplift, 4),
            "incremental_trx_pct_of_total_trx": round((uplift - cur_uplift) / window_trx, 4),
            "incremental_gross_margin_per_month": round(margin_delta, 2),
            "annualised_incremental_gross_margin": round(margin_delta * 12, 2),
            "annualised_call_cost_saving": round((cur_calls - calls) * cost * 12, 2),
        }

    summary = {
        "current_calls_per_month": round(float(cur_c.sum()), 1),
        "current_monthly_call_cost": round(float(cur_c.sum()) * cost, 2),
        "current_promo_driven_trx_per_month": round(
            float(plan["current_plan_trx_uplift"].sum()), 1),
        "scenario_a_budget_neutral": scenario("recommended_calls_monthly",
                                              "recommended_plan_trx_uplift"),
        "scenario_b_profit_max": scenario("profitmax_calls_monthly",
                                          "profitmax_plan_trx_uplift"),
        "hcps_increase": int((plan["action"] == "Increase frequency").sum()),
        "hcps_reduce": int((plan["action"] == "Reduce frequency").sum()),
        "hcps_deprioritised": int((plan["action"] == "Deprioritise - no field calls").sum()),
        "hcps_digital_only": int((plan["action"] == "Digital only (no-see)").sum()),
        "hcp_brand_pairs": int(len(plan)),
    }
    a = summary["scenario_a_budget_neutral"]
    log.info("scenario A: %+.1f%% calls -> %+.1f%% promo TRx | scenario B: %+.1f%% calls -> %+.1f%% promo TRx",
             100 * a["call_budget_change_pct"], 100 * a["promo_trx_uplift_pct"],
             100 * summary["scenario_b_profit_max"]["call_budget_change_pct"],
             100 * summary["scenario_b_profit_max"]["promo_trx_uplift_pct"])
    return plan, summary


def waste_diagnostics(plan: pd.DataFrame, cfg: Config) -> dict:
    econ = cfg["economics"]
    cur = plan["current_calls_monthly"]
    # A call is "unprofitable" if the marginal ROI at the current frequency < 1
    b, a = plan["b"], plan["a"]
    marg_trx = a * (np.exp(-b * (cur - 1).clip(lower=0)) - np.exp(-b * cur.clip(lower=0)))
    marg_roi = marg_trx * plan["monthly_market_trx"] * plan["value_per_trx"] / econ["cost_per_call"]
    unprofitable = (marg_roi < 1) & (cur > 0)
    wasted_calls = float(cur[unprofitable].sum())
    total_calls = float(cur.sum())
    low_tier = plan["value_tier"].eq("Low")
    return {
        "pct_calls_below_breakeven_roi": round(wasted_calls / total_calls, 4),
        "monthly_spend_below_breakeven": round(wasted_calls * econ["cost_per_call"], 2),
        "pct_calls_to_low_value_tier": round(float(cur[low_tier].sum()) / total_calls, 4),
        "pct_calls_to_no_see_hcps": round(
            float(cur[plan["no_see_flag"] == 1].sum()) / total_calls, 4),
        "median_marginal_roi_current_plan": round(float(marg_roi[cur > 0].median()), 3),
    }


def main(cfg: Config | None = None) -> tuple[pd.DataFrame, dict]:
    cfg = cfg or load_config()
    processed, reports = cfg.path("processed"), cfg.path("reports")

    mart_pq = processed / "monthly_mart.parquet"
    mart = pd.read_parquet(mart_pq) if mart_pq.exists() else pd.read_csv(
        processed / "monthly_mart.csv")
    segments = pd.read_csv(processed / "hcp_segments.csv")
    abt_hp = pd.read_csv(processed / "abt_hcp_product.csv")

    panel = mart.merge(segments[["hcp_id", "segment_name"]], on="hcp_id", how="left")
    curves = fit_response_curves(panel)
    attribution = channel_attribution(panel, cfg)

    econ_df = build_marginal_economics(abt_hp, segments, curves, cfg)
    plan, summary = optimise_call_plan(econ_df, cfg)
    summary["waste_diagnostics"] = waste_diagnostics(plan, cfg)
    summary["channel_attribution_within_r2"] = attribution.attrs["r2_within"]

    keep = ["hcp_id", "product_id", "brand", "territory_id", "specialty", "decile",
            "segment_name", "value_tier", "strategic_quadrant", "no_see_flag", "call_cap",
            "trx_12m", "market_trx_12m", "market_share", "unrealized_opportunity",
            "current_calls_monthly", "recommended_calls_monthly",
            "profitmax_calls_monthly", "delta_calls", "delta_trx_monthly",
            "delta_margin_monthly", "action", "a", "b"]
    plan_out = plan[keep].rename(columns={"a": "segment_max_share_lift",
                                          "b": "segment_saturation_rate"})
    plan_out.to_csv(processed / "call_plan_recommendations.csv", index=False)
    curves.to_csv(reports / "response_curves.csv", index=False)
    attribution.to_csv(reports / "channel_attribution.csv", index=False)
    with open(reports / "sfe_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return plan_out, summary


if __name__ == "__main__":
    main()
