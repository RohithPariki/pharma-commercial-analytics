"""
Feature engineering.

Builds the HCP Analytical Base Table (ABT) - one row per prescriber - from the
curated monthly mart. Features are grouped into five families that mirror how a
commercial analytics team actually frames a prescriber:

    VALUE        what they write today          (trx_12m, revenue_12m, market_share)
    POTENTIAL    what they could write          (market_trx_12m, unrealized_opportunity)
    MOMENTUM     which way they are moving      (trx_growth_pct, adoption_slope, share_delta)
    ENGAGEMENT   how we currently invest        (call_frequency, reach_pct, channel mix)
    RESPONSE     what our investment returns    (promo_response_index)

`promo_response_index` is an ordinary-least-squares slope of monthly TRx on the
3-month moving average of calls, computed per prescriber and scaled by their own
mean TRx. It is the observable proxy for the hidden responsiveness class and is
what turns a descriptive segmentation into an actionable one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, get_logger, load_config

log = get_logger("features")


def _ols_slope(df: pd.DataFrame, key: list[str], x: str, y: str, out: str) -> pd.DataFrame:
    """Vectorised per-group OLS slope: cov(x, y) / var(x)."""
    g = df.groupby(key, sort=False)
    stats = g.agg(_mx=(x, "mean"), _my=(y, "mean"), _n=(x, "size")).reset_index()
    prod = df.assign(_xy=df[x] * df[y], _xx=df[x] * df[x])
    m2 = prod.groupby(key, sort=False).agg(_mxy=("_xy", "mean"), _mxx=("_xx", "mean")).reset_index()
    stats = stats.merge(m2, on=key)
    cov = stats["_mxy"] - stats["_mx"] * stats["_my"]
    var = stats["_mxx"] - stats["_mx"] ** 2
    stats[out] = np.where(var > 1e-9, cov / var.where(var > 1e-9, np.nan), 0.0)
    stats[out] = stats[out].fillna(0.0)
    return stats[key + [out]]


def build_hcp_product_abt(mart: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Brand-level base table (HCP x product) - the grain used for call planning."""
    window = cfg["features"]["analysis_window_months"]
    recent = cfg["features"]["recent_window_months"]
    last_idx = int(mart["month_index"].max())
    w_start = last_idx - window + 1

    cur = mart[mart["month_index"] >= w_start]
    prior = mart[(mart["month_index"] >= w_start - window) & (mart["month_index"] < w_start)]

    keys = ["hcp_id", "product_id"]
    abt = cur.groupby(keys, as_index=False).agg(
        brand=("brand", "first"),
        therapy_area=("therapy_area", "first"),
        territory_id=("territory_id", "first"),
        specialty=("specialty", "first"),
        decile=("decile", "first"),
        is_kol=("is_kol", "first"),
        no_see_flag=("no_see_flag", "first"),
        net_revenue_per_trx=("net_revenue_per_trx", "first"),
        trx_12m=("trx", "sum"),
        nrx_12m=("nrx", "sum"),
        market_trx_12m=("market_trx", "sum"),
        revenue_12m=("revenue", "sum"),
        calls_12m=("calls", "sum"),
        samples_12m=("samples_dropped", "sum"),
        emails_opened_12m=("emails_opened", "sum"),
        emails_sent_12m=("emails_sent", "sum"),
        speaker_12m=("speaker_programs", "sum"),
        months_active=("trx", lambda s: int((s > 0).sum())),
        months_reached=("calls", lambda s: int((s > 0).sum())),
        trx_std=("trx", "std"),
        trx_mean=("trx", "mean"),
    )

    prior_agg = prior.groupby(keys, as_index=False).agg(
        trx_prior_12m=("trx", "sum"), market_trx_prior_12m=("market_trx", "sum"),
        calls_prior_12m=("calls", "sum"))
    abt = abt.merge(prior_agg, on=keys, how="left").fillna(
        {"trx_prior_12m": 0, "market_trx_prior_12m": 0, "calls_prior_12m": 0})

    # --- recent vs preceding quarter -------------------------------------
    r_last = mart[mart["month_index"] > last_idx - recent]
    r_prev = mart[(mart["month_index"] > last_idx - 2 * recent)
                  & (mart["month_index"] <= last_idx - recent)]
    abt = abt.merge(r_last.groupby(keys, as_index=False).agg(
        trx_last3m=("trx", "sum"), market_trx_last3m=("market_trx", "sum"),
        calls_last3m=("calls", "sum")), on=keys, how="left")
    abt = abt.merge(r_prev.groupby(keys, as_index=False).agg(
        trx_prev3m=("trx", "sum"), market_trx_prev3m=("market_trx", "sum")),
        on=keys, how="left")
    abt = abt.fillna({"trx_last3m": 0, "trx_prev3m": 0, "market_trx_last3m": 0,
                      "market_trx_prev3m": 0, "calls_last3m": 0})

    # --- derived ratios ---------------------------------------------------
    abt["market_share"] = abt["trx_12m"] / abt["market_trx_12m"].replace(0, np.nan)
    share_last = abt["trx_last3m"] / abt["market_trx_last3m"].replace(0, np.nan)
    share_prev = abt["trx_prev3m"] / abt["market_trx_prev3m"].replace(0, np.nan)
    abt["share_delta"] = (share_last - share_prev).fillna(0.0)
    abt["trx_growth_pct"] = np.where(
        abt["trx_prev3m"] > 0, (abt["trx_last3m"] - abt["trx_prev3m"]) / abt["trx_prev3m"], 0.0)
    abt["trx_growth_pct"] = np.clip(abt["trx_growth_pct"], -1.5, 3.0)
    abt["nrx_ratio"] = (abt["nrx_12m"] / abt["trx_12m"].replace(0, np.nan)).fillna(0.0)
    abt["call_frequency"] = abt["calls_12m"] / window
    abt["reach_pct"] = abt["months_reached"] / window
    abt["email_open_rate"] = (abt["emails_opened_12m"]
                              / abt["emails_sent_12m"].replace(0, np.nan)).fillna(0.0)
    abt["trx_volatility"] = (abt["trx_std"] / abt["trx_mean"].replace(0, np.nan)).fillna(0.0)
    abt["market_share"] = abt["market_share"].fillna(0.0)

    ceiling = cfg["targeting"]["target_share_ceiling"]
    abt["unrealized_opportunity"] = (abt["market_trx_12m"]
                                     * (ceiling - abt["market_share"]).clip(lower=0))
    abt["opportunity_revenue"] = abt["unrealized_opportunity"] * abt["net_revenue_per_trx"]

    # --- slopes -----------------------------------------------------------
    slope_src = mart[mart["month_index"] >= w_start][keys + ["month_index", "trx", "calls_ma3"]]
    abt = abt.merge(_ols_slope(slope_src, keys, "month_index", "trx", "adoption_slope"),
                    on=keys, how="left")
    abt["adoption_slope"] = (abt["adoption_slope"]
                             / abt["trx_mean"].replace(0, np.nan)).fillna(0.0).clip(-1, 1)

    # Promotional response is estimated on brand SHARE, not raw TRx: share removes
    # the prescriber's own category volume, which is the dominant source of noise
    # in a TRx-on-calls regression. Full history is used (not just the trailing
    # window) because slope precision is driven by the number of call observations.
    resp_src = mart[keys + ["calls_ma3", "market_share"]].copy()
    resp_src["market_share"] = resp_src["market_share"].fillna(0.0)
    abt = abt.merge(_ols_slope(resp_src, keys, "calls_ma3", "market_share",
                               "promo_response_index"), on=keys, how="left")
    abt["promo_response_index"] = abt["promo_response_index"].fillna(0.0).clip(-0.6, 0.6)

    # Complementary non-parametric contrast: share in called months vs quiet months.
    resp_src["_called"] = (resp_src["calls_ma3"] >= 1).astype(int)
    contrast = (resp_src.groupby(keys + ["_called"])["market_share"].mean()
                        .unstack("_called"))
    contrast = contrast.reindex(columns=[0, 1])
    contrast["promo_contrast"] = (contrast[1] - contrast[0])
    abt = abt.merge(contrast[["promo_contrast"]].reset_index(), on=keys, how="left")
    abt["promo_contrast"] = abt["promo_contrast"].fillna(0.0).clip(-0.6, 0.6)

    # --- recency ----------------------------------------------------------
    last_call = (mart[mart["calls"] > 0].groupby(keys, as_index=False)["month_index"].max()
                 .rename(columns={"month_index": "_last_call_idx"}))
    abt = abt.merge(last_call, on=keys, how="left")
    abt["months_since_last_call"] = (last_idx - abt["_last_call_idx"]).fillna(window + 1)
    last_rx = (mart[mart["trx"] > 0].groupby(keys, as_index=False)["month_index"].max()
               .rename(columns={"month_index": "_last_rx_idx"}))
    abt = abt.merge(last_rx, on=keys, how="left")
    abt["months_since_last_rx"] = (last_idx - abt["_last_rx_idx"]).fillna(window + 1)

    return abt.drop(columns=["_last_call_idx", "_last_rx_idx", "trx_std", "trx_mean"])


def _decile_score(s: pd.Series, ascending: bool = True) -> pd.Series:
    """1-5 quintile score, robust to heavy ties (large zero mass)."""
    ranked = s.rank(method="first", ascending=ascending)
    return pd.qcut(ranked, 5, labels=[1, 2, 3, 4, 5]).astype(int)


def build_hcp_abt(hcp_product_abt: pd.DataFrame, mart: pd.DataFrame,
                  cfg: Config) -> pd.DataFrame:
    """Portfolio-level base table (one row per HCP) used for segmentation."""
    weights = hcp_product_abt["trx_12m"].clip(lower=0.0001)
    df = hcp_product_abt.assign(_w=weights,
                                _w_resp=weights * hcp_product_abt["promo_response_index"],
                                _w_contrast=weights * hcp_product_abt["promo_contrast"],
                                _w_slope=weights * hcp_product_abt["adoption_slope"],
                                _w_share=weights * hcp_product_abt["market_share"],
                                _w_delta=weights * hcp_product_abt["share_delta"])

    abt = df.groupby("hcp_id", as_index=False).agg(
        specialty=("specialty", "first"),
        territory_id=("territory_id", "first"),
        decile=("decile", "first"),
        is_kol=("is_kol", "first"),
        no_see_flag=("no_see_flag", "first"),
        n_brands_prescribed=("trx_12m", lambda s: int((s > 0).sum())),
        trx_12m=("trx_12m", "sum"),
        nrx_12m=("nrx_12m", "sum"),
        market_trx_12m=("market_trx_12m", "sum"),
        revenue_12m=("revenue_12m", "sum"),
        trx_prior_12m=("trx_prior_12m", "sum"),
        trx_last3m=("trx_last3m", "sum"),
        trx_prev3m=("trx_prev3m", "sum"),
        calls_12m=("calls_12m", "sum"),
        calls_last3m=("calls_last3m", "sum"),
        samples_12m=("samples_12m", "sum"),
        emails_opened_12m=("emails_opened_12m", "sum"),
        speaker_12m=("speaker_12m", "sum"),
        months_active=("months_active", "max"),
        months_reached=("months_reached", "max"),
        months_since_last_call=("months_since_last_call", "min"),
        months_since_last_rx=("months_since_last_rx", "min"),
        unrealized_opportunity=("unrealized_opportunity", "sum"),
        opportunity_revenue=("opportunity_revenue", "sum"),
        trx_volatility=("trx_volatility", "mean"),
        email_open_rate=("email_open_rate", "mean"),
        _w=("_w", "sum"), _w_resp=("_w_resp", "sum"), _w_slope=("_w_slope", "sum"),
        _w_contrast=("_w_contrast", "sum"),
        _w_share=("_w_share", "sum"), _w_delta=("_w_delta", "sum"),
    )

    window = cfg["features"]["analysis_window_months"]
    abt["market_share"] = abt["_w_share"] / abt["_w"]
    abt["share_delta"] = abt["_w_delta"] / abt["_w"]
    abt["promo_response_index"] = abt["_w_resp"] / abt["_w"]
    abt["promo_contrast"] = abt["_w_contrast"] / abt["_w"]
    abt["adoption_slope"] = abt["_w_slope"] / abt["_w"]
    abt = abt.drop(columns=["_w", "_w_resp", "_w_slope", "_w_share", "_w_delta",
                            "_w_contrast"])

    abt["nrx_ratio"] = (abt["nrx_12m"] / abt["trx_12m"].replace(0, np.nan)).fillna(0.0)
    abt["call_frequency"] = abt["calls_12m"] / window
    abt["reach_pct"] = abt["months_reached"] / window
    abt["trx_growth_pct"] = np.clip(
        np.where(abt["trx_prev3m"] > 0,
                 (abt["trx_last3m"] - abt["trx_prev3m"]) / abt["trx_prev3m"], 0.0), -1.5, 3.0)
    abt["yoy_growth_pct"] = np.clip(
        np.where(abt["trx_prior_12m"] > 0,
                 (abt["trx_12m"] - abt["trx_prior_12m"]) / abt["trx_prior_12m"], 0.0), -1.5, 3.0)
    abt["revenue_per_call"] = (abt["revenue_12m"]
                               / abt["calls_12m"].replace(0, np.nan)).fillna(0.0)

    # --- RFM (Recency of last Rx, Frequency of active months, Monetary TRx) --
    abt["r_score"] = _decile_score(abt["months_since_last_rx"], ascending=False)
    abt["f_score"] = _decile_score(abt["months_active"].astype(float))
    abt["m_score"] = _decile_score(abt["trx_12m"].astype(float))
    abt["rfm_score"] = abt["r_score"] + abt["f_score"] + abt["m_score"]
    abt["rfm_cell"] = (abt["r_score"].astype(str) + abt["f_score"].astype(str)
                       + abt["m_score"].astype(str))

    log.info("HCP ABT built: %s prescribers x %s features", f"{len(abt):,}", abt.shape[1])
    return abt


def main(cfg: Config | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = cfg or load_config()
    processed = cfg.path("processed")
    mart_pq, mart_csv = processed / "monthly_mart.parquet", processed / "monthly_mart.csv"
    mart = pd.read_parquet(mart_pq) if mart_pq.exists() else pd.read_csv(mart_csv)

    hcp_product_abt = build_hcp_product_abt(mart, cfg)
    hcp_abt = build_hcp_abt(hcp_product_abt, mart, cfg)
    hcp_product_abt.to_csv(processed / "abt_hcp_product.csv", index=False)
    hcp_abt.to_csv(processed / "abt_hcp.csv", index=False)
    return hcp_abt, hcp_product_abt


if __name__ == "__main__":
    main()
