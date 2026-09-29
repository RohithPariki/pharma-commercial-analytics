"""
Power BI export layer.

Emits a clean star schema to `powerbi/data/`. Dimensions are conformed and
deduplicated, facts carry only keys and additive measures, and every analytical
output (segment, risk band, call-plan action) is attached to the HCP dimension
so the report can slice without extra relationships.

Relationships expected in the model (all single-direction, one-to-many):

    DimDate[MonthKey]        -> FactRx[MonthKey], FactActivity[MonthKey], FactSales[MonthKey]
    DimHCP[HCPKey]           -> FactRx[HCPKey], FactActivity[HCPKey], FactCallPlan[HCPKey]
    DimProduct[ProductKey]   -> FactRx, FactActivity, FactSales, FactCallPlan
    DimTerritory[TerritoryKey] -> DimHCP[TerritoryKey], FactSales[TerritoryKey]
"""

from __future__ import annotations

import pandas as pd

from .config import Config, get_logger, load_config

log = get_logger("bi_export")


def main(cfg: Config | None = None) -> dict[str, pd.DataFrame]:
    cfg = cfg or load_config()
    raw, processed, out = cfg.path("raw"), cfg.path("processed"), cfg.path("powerbi")

    dim_date = pd.read_csv(raw / "dim_date.csv")
    dim_product = pd.read_csv(raw / "dim_product.csv")
    dim_hcp = pd.read_csv(raw / "dim_hcp.csv")
    fact_rx = pd.read_csv(raw / "fact_rx.csv")
    fact_activity = pd.read_csv(raw / "fact_activity.csv")
    fact_sales = pd.read_csv(raw / "fact_sales.csv")

    segments = pd.read_csv(processed / "hcp_segments.csv")
    risk = pd.read_csv(processed / "hcp_risk_scores.csv")
    plan = pd.read_csv(processed / "call_plan_recommendations.csv")
    scorecard = pd.read_csv(processed / "territory_scorecard.csv")

    dim_hcp_bi = dim_hcp.merge(
        segments[["hcp_id", "segment_name", "value_tier", "strategic_quadrant",
                  "trx_12m", "revenue_12m", "market_share", "unrealized_opportunity",
                  "opportunity_revenue", "promo_response_index", "rfm_score",
                  "call_frequency", "trx_growth_pct"]], on="hcp_id", how="left")
    dim_hcp_bi = dim_hcp_bi.merge(
        risk[["hcp_id", "decline_risk_score", "risk_band", "risk_decile",
              "revenue_at_risk"]], on="hcp_id", how="left")
    dim_hcp_bi = dim_hcp_bi.drop(columns=["monthly_market_volume"], errors="ignore")

    tables = {
        "DimDate": dim_date.rename(columns={"month": "MonthKey"}),
        "DimProduct": dim_product.rename(columns={"product_id": "ProductKey"}),
        "DimTerritory": scorecard.rename(columns={"territory_id": "TerritoryKey"}),
        "DimHCP": dim_hcp_bi.rename(columns={"hcp_id": "HCPKey",
                                             "territory_id": "TerritoryKey"}),
        "FactRx": fact_rx.rename(columns={"hcp_id": "HCPKey", "product_id": "ProductKey",
                                          "month": "MonthKey"}),
        "FactActivity": fact_activity.rename(columns={"hcp_id": "HCPKey",
                                                      "product_id": "ProductKey",
                                                      "month": "MonthKey"}),
        "FactSales": fact_sales.rename(columns={"territory_id": "TerritoryKey",
                                                "product_id": "ProductKey",
                                                "month": "MonthKey"}),
        "FactCallPlan": plan.rename(columns={"hcp_id": "HCPKey", "product_id": "ProductKey",
                                             "territory_id": "TerritoryKey"}),
    }
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
        log.info("%-14s -> %8s rows x %2d cols", name, f"{len(df):,}", df.shape[1])
    return tables


if __name__ == "__main__":
    main()
