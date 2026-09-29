"""Static figure pack (matplotlib only, no notebook required)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import Config, get_logger, load_config  # noqa: E402

log = get_logger("visuals")

PALETTE = ["#1b4965", "#5fa8d3", "#62b6cb", "#cae9ff", "#e07a5f", "#f2cc8f",
           "#81b29a", "#3d405b"]
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "figure.autolayout": True,
})


def _save(fig, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("figure -> %s", path.name)


def plot_k_selection(diagnostics: dict, out: Path) -> None:
    diag = pd.DataFrame(diagnostics["k_diagnostics"])
    fig, ax1 = plt.subplots(figsize=(6.2, 3.6))
    ax1.plot(diag["k"], diag["inertia"], marker="o", color=PALETTE[0], label="Inertia")
    ax1.set_xlabel("k (clusters)")
    ax1.set_ylabel("Inertia", color=PALETTE[0])
    ax2 = ax1.twinx()
    ax2.plot(diag["k"], diag["silhouette"], marker="s", color=PALETTE[4], label="Silhouette")
    ax2.set_ylabel("Silhouette", color=PALETTE[4])
    ax2.grid(False)
    best = diagnostics["metrics"]["selected_k"]
    ax1.axvline(best, color="grey", ls=":", lw=1)
    ax1.set_title(f"Cluster count selection (selected k={best})")
    _save(fig, out / "01_k_selection.png")


def plot_segment_scatter(seg: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for i, (name, g) in enumerate(seg.groupby("segment_name")):
        ax.scatter(g["pca_1"], g["pca_2"], s=7, alpha=0.55,
                   color=PALETTE[i % len(PALETTE)], label=f"{name} (n={len(g):,})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("HCP behavioural segments (PCA projection)")
    ax.legend(fontsize=6.5, markerscale=1.8, loc="best", frameon=False)
    _save(fig, out / "02_segment_pca.png")


def plot_segment_value(summary: pd.DataFrame, out: Path) -> None:
    s = summary.sort_values("pct_of_trx", ascending=True)
    y = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.barh(y - 0.2, s["pct_of_hcps"] * 100, height=0.38, color=PALETTE[1],
            label="% of prescribers")
    ax.barh(y + 0.2, s["pct_of_trx"] * 100, height=0.38, color=PALETTE[0],
            label="% of TRx")
    ax.set_yticks(y)
    ax.set_yticklabels(s["segment_name"], fontsize=7.5)
    ax.set_xlabel("Percent")
    ax.set_title("Value concentration by segment")
    ax.legend(fontsize=7, frameon=False)
    _save(fig, out / "03_segment_value_concentration.png")


def plot_response_curves(curves: pd.DataFrame, out: Path) -> None:
    c = np.linspace(0, 4, 120)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for i, row in curves.iterrows():
        y = row["s0"] + row["a"] * (1 - np.exp(-row["b"] * c))
        ax.plot(c, y * 100, color=PALETTE[i % len(PALETTE)], lw=1.8,
                label=f"{row['segment_name']} (a={row['a']:.3f})")
    ax.set_xlabel("Calls per month (3-month moving average)")
    ax.set_ylabel("Brand share (%)")
    ax.set_title("Fitted promotional response curves by segment")
    ax.legend(fontsize=6.5, frameon=False)
    _save(fig, out / "04_response_curves.png")


def plot_call_reallocation(plan: pd.DataFrame, out: Path) -> None:
    g = plan.groupby("segment_name", as_index=False).agg(
        current=("current_calls_monthly", "sum"),
        recommended=("recommended_calls_monthly", "sum"))
    g = g.sort_values("current", ascending=True)
    y = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.barh(y - 0.2, g["current"], height=0.38, color=PALETTE[3], label="Current plan")
    ax.barh(y + 0.2, g["recommended"], height=0.38, color=PALETTE[0], label="Optimised plan")
    ax.set_yticks(y)
    ax.set_yticklabels(g["segment_name"], fontsize=7.5)
    ax.set_xlabel("Calls per month")
    ax.set_title("Call reallocation: current vs optimised")
    ax.legend(fontsize=7, frameon=False)
    _save(fig, out / "05_call_reallocation.png")


def plot_territory_quadrant(sc: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    sizes = 12 + 180 * (sc["revenue_12m"] / sc["revenue_12m"].max())
    sctr = ax.scatter(sc["opportunity_index"], sc["quota_attainment"] * 100, s=sizes,
                      c=sc["rep_effect_zscore"], cmap="RdYlBu", edgecolor="white", lw=0.5)
    ax.axhline(100, color="grey", ls=":", lw=1)
    ax.axvline(1.0, color="grey", ls=":", lw=1)
    ax.set_xlabel("Opportunity index (1.0 = national average headroom)")
    ax.set_ylabel("Quota attainment (%)")
    ax.set_title("Territory performance vs opportunity")
    fig.colorbar(sctr, ax=ax, label="Potential-adjusted rep effect (z)")
    _save(fig, out / "06_territory_quadrant.png")


def plot_risk(scores: pd.DataFrame, out: Path) -> None:
    g = scores.groupby("risk_decile", as_index=False).agg(
        revenue_at_risk=("revenue_at_risk", "sum"), hcps=("hcp_id", "size"))
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.bar(g["risk_decile"], g["revenue_at_risk"] / 1000, color=PALETTE[4])
    ax.set_xlabel("Decline-risk decile (10 = highest risk)")
    ax.set_ylabel("Revenue at risk ($000, 12m)")
    ax.set_title("Concentration of revenue at risk")
    _save(fig, out / "07_revenue_at_risk.png")


def plot_share_trend(mart: pd.DataFrame, out: Path) -> None:
    t = mart.groupby(["month", "brand"], as_index=False).agg(
        trx=("trx", "sum"), market_trx=("market_trx", "sum"))
    t["share"] = t["trx"] / t["market_trx"] * 100
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    for i, (brand, g) in enumerate(t.groupby("brand")):
        ax.plot(g["month"], g["share"], marker="o", ms=3, lw=1.6,
                color=PALETTE[i], label=brand)
    ax.set_xticks(ax.get_xticks()[::3])
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.set_ylabel("Brand share of category TRx (%)")
    ax.set_title("Brand share trend")
    ax.legend(fontsize=7, frameon=False)
    _save(fig, out / "08_brand_share_trend.png")


def main(cfg: Config | None = None) -> None:
    cfg = cfg or load_config()
    processed, reports, figs = cfg.path("processed"), cfg.path("reports"), cfg.path("figures")
    mart_pq = processed / "monthly_mart.parquet"
    mart = pd.read_parquet(mart_pq) if mart_pq.exists() else pd.read_csv(
        processed / "monthly_mart.csv")

    seg = pd.read_csv(processed / "hcp_segments.csv")
    summary = pd.read_csv(reports / "segment_profile.csv")
    curves = pd.read_csv(reports / "response_curves.csv")
    plan = pd.read_csv(processed / "call_plan_recommendations.csv")
    sc = pd.read_csv(processed / "territory_scorecard.csv")
    scores = pd.read_csv(processed / "hcp_risk_scores.csv")
    with open(reports / "segmentation_metrics.json", encoding="utf-8") as fh:
        diagnostics = json.load(fh)

    plot_k_selection(diagnostics, figs)
    plot_segment_scatter(seg, figs)
    plot_segment_value(summary, figs)
    plot_response_curves(curves, figs)
    plot_call_reallocation(plan, figs)
    plot_territory_quadrant(sc, figs)
    plot_risk(scores, figs)
    plot_share_trend(mart, figs)


if __name__ == "__main__":
    main()
