"""
ETL layer.

Raw CSV extracts -> validated -> curated monthly fact mart (Parquet).

The data-quality gate runs *before* any modelling. Every check returns a row in
`reports/data_quality_report.csv`; a failed CRITICAL check aborts the pipeline
rather than letting a silent defect propagate into a segmentation deliverable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, get_logger, load_config

log = get_logger("etl")

RAW_TABLES = ["dim_date", "dim_product", "dim_territory", "dim_hcp",
              "fact_rx", "fact_activity", "fact_sales"]


class DataQualityError(RuntimeError):
    pass


@dataclass
class DataQualityReport:
    rows: list[dict] = field(default_factory=list)

    def check(self, table: str, name: str, passed: bool, severity: str,
              detail: str = "", observed: object = "") -> None:
        self.rows.append({
            "table": table, "check": name, "severity": severity,
            "status": "PASS" if passed else "FAIL",
            "observed": observed, "detail": detail,
        })
        level = log.info if passed else (log.error if severity == "CRITICAL" else log.warning)
        level("[%s] %-34s %-4s %s", table, name, "PASS" if passed else "FAIL", observed)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    @property
    def critical_failures(self) -> list[dict]:
        return [r for r in self.rows if r["status"] == "FAIL" and r["severity"] == "CRITICAL"]


def load_raw(raw_dir: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    for name in RAW_TABLES:
        path = raw_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing raw extract: {path} - run generate_data first")
        tables[name] = pd.read_csv(path)
    return tables


def validate(tables: dict[str, pd.DataFrame]) -> DataQualityReport:
    r = DataQualityReport()
    hcp, terr, prod = tables["dim_hcp"], tables["dim_territory"], tables["dim_product"]
    rx, act, sales, date = (tables["fact_rx"], tables["fact_activity"],
                            tables["fact_sales"], tables["dim_date"])

    # --- uniqueness -------------------------------------------------------
    for name, df, keys in [
        ("dim_hcp", hcp, ["hcp_id"]),
        ("dim_territory", terr, ["territory_id"]),
        ("dim_product", prod, ["product_id"]),
        ("fact_rx", rx, ["hcp_id", "product_id", "month"]),
        ("fact_activity", act, ["hcp_id", "product_id", "month"]),
        ("fact_sales", sales, ["territory_id", "product_id", "month"]),
    ]:
        dupes = int(df.duplicated(subset=keys).sum())
        r.check(name, "primary_key_unique", dupes == 0, "CRITICAL",
                f"key={keys}", f"{dupes} duplicates")

    # --- referential integrity -------------------------------------------
    orphan_hcp = int((~rx["hcp_id"].isin(hcp["hcp_id"])).sum())
    r.check("fact_rx", "fk_hcp_id", orphan_hcp == 0, "CRITICAL", observed=f"{orphan_hcp} orphans")
    orphan_terr = int((~hcp["territory_id"].isin(terr["territory_id"])).sum())
    r.check("dim_hcp", "fk_territory_id", orphan_terr == 0, "CRITICAL",
            observed=f"{orphan_terr} orphans")
    orphan_prod = int((~rx["product_id"].isin(prod["product_id"])).sum())
    r.check("fact_rx", "fk_product_id", orphan_prod == 0, "CRITICAL",
            observed=f"{orphan_prod} orphans")

    # --- nulls ------------------------------------------------------------
    for name, df in [("dim_hcp", hcp), ("fact_rx", rx), ("fact_activity", act),
                     ("fact_sales", sales)]:
        nulls = int(df.isna().sum().sum())
        r.check(name, "no_null_values", nulls == 0, "WARNING", observed=f"{nulls} nulls")

    # --- domain / business rules -----------------------------------------
    neg = int((rx[["trx", "nrx", "market_trx"]] < 0).to_numpy().sum())
    r.check("fact_rx", "non_negative_measures", neg == 0, "CRITICAL", observed=f"{neg} rows")

    over_market = int((rx["trx"] > rx["market_trx"]).sum())
    r.check("fact_rx", "trx_lte_market_trx", over_market == 0, "CRITICAL",
            "brand TRx cannot exceed category TRx", f"{over_market} rows")

    nrx_gt_trx = int((rx["nrx"] > rx["trx"]).sum())
    r.check("fact_rx", "nrx_lte_trx", nrx_gt_trx == 0, "CRITICAL", observed=f"{nrx_gt_trx} rows")

    opened_gt_sent = int((act["emails_opened"] > act["emails_sent"]).sum())
    r.check("fact_activity", "opens_lte_sends", opened_gt_sent == 0, "CRITICAL",
            observed=f"{opened_gt_sent} rows")

    # --- completeness / grain --------------------------------------------
    expected_months = set(date["month"])
    missing_months = expected_months - set(rx["month"])
    r.check("fact_rx", "calendar_complete", not missing_months, "CRITICAL",
            observed=f"{len(missing_months)} missing months")

    panel = rx.groupby(["hcp_id", "product_id"])["month"].nunique()
    ragged = int((panel != len(expected_months)).sum())
    r.check("fact_rx", "balanced_panel", ragged == 0, "WARNING",
            "every HCP-product must span the full window", f"{ragged} ragged series")

    coverage = act.merge(rx[["hcp_id", "product_id", "month"]],
                         on=["hcp_id", "product_id", "month"], how="outer", indicator=True)
    unmatched = int((coverage["_merge"] != "both").sum())
    r.check("fact_activity", "activity_rx_grain_match", unmatched == 0, "CRITICAL",
            observed=f"{unmatched} unmatched rows")

    zero_quota = int((sales["quota_revenue"] <= 0).sum())
    r.check("fact_sales", "positive_quota", zero_quota == 0, "WARNING",
            observed=f"{zero_quota} rows")
    return r


def build_monthly_mart(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per HCP x product x month, fully conformed and enriched."""
    rx, act = tables["fact_rx"], tables["fact_activity"]
    hcp, prod, date = tables["dim_hcp"], tables["dim_product"], tables["dim_date"]

    mart = rx.merge(act, on=["hcp_id", "product_id", "month"], how="inner")
    mart = mart.merge(
        hcp[["hcp_id", "specialty", "affiliation_type", "territory_id", "decile",
             "is_kol", "no_see_flag", "commercial_payer_pct"]], on="hcp_id", how="left")
    mart = mart.merge(
        prod[["product_id", "brand", "therapy_area", "lifecycle", "net_revenue_per_trx"]],
        on="product_id", how="left")
    mart = mart.merge(date[["month", "month_index", "year", "quarter", "year_quarter"]],
                      on="month", how="left")

    mart["revenue"] = mart["trx"] * mart["net_revenue_per_trx"]
    mart["market_share"] = (mart["trx"] / mart["market_trx"].replace(0, np.nan)).astype(float)
    mart["promo_cost"] = mart["calls"] * 185.0  # overwritten by economics in sfe module
    mart = mart.sort_values(["hcp_id", "product_id", "month_index"]).reset_index(drop=True)

    grp = mart.groupby(["hcp_id", "product_id"], sort=False)
    mart["calls_ma3"] = grp["calls"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    mart["trx_ma3"] = grp["trx"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    mart["trx_mom_growth"] = grp["trx"].transform(lambda s: s.pct_change()).replace(
        [np.inf, -np.inf], np.nan)
    return mart


def main(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    tables = load_raw(cfg.path("raw"))
    report = validate(tables)
    report.to_frame().to_csv(cfg.path("reports") / "data_quality_report.csv", index=False)

    if report.critical_failures:
        raise DataQualityError(
            f"{len(report.critical_failures)} critical data-quality checks failed; "
            "see reports/data_quality_report.csv")

    mart = build_monthly_mart(tables)
    out = cfg.path("processed") / "monthly_mart.parquet"
    try:
        mart.to_parquet(out, index=False)
    except Exception:  # pragma: no cover - environments without pyarrow
        out = out.with_suffix(".csv")
        mart.to_csv(out, index=False)
    log.info("curated mart -> %s (%s rows)", out.name, f"{len(mart):,}")
    return mart


if __name__ == "__main__":
    main()
