"""
HCP segmentation.

Two complementary views are produced, because a single K-Means label is not a
call plan:

  1. BEHAVIOURAL SEGMENTS - unsupervised K-Means over value, potential, momentum,
     engagement and response features. k is selected empirically over a candidate
     range using silhouette, Davies-Bouldin and Calinski-Harabasz rather than
     being asserted. Clusters are then auto-named from their centroid percentile
     profile so the output is readable by a brand team.

  2. STRATEGIC GRID - a deterministic Value x Opportunity 2x2 (Protect / Grow /
     Maintain / Deprioritise). This is the layer sales leadership actually acts
     on, and it stays stable across refresh cycles even if cluster ids shuffle.

The segmentation is validated against the held-out responsiveness ground truth:
we measure how much of the promotionally-responsive prescriber base lands in the
priority segments versus a random-allocation baseline.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

from .config import Config, get_logger, load_config

log = get_logger("segmentation")

SKEWED = {"trx_12m", "market_trx_12m", "unrealized_opportunity", "revenue_12m"}


def build_matrix(abt: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, list[str]]:
    X = abt[features].copy()
    for col in X.columns:
        if col in SKEWED:
            X[col] = np.log1p(X[col].clip(lower=0))
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True))
    return X.to_numpy(dtype=float), list(X.columns)


def _minmax(s: pd.Series, higher_is_better: bool = True) -> pd.Series:
    rng = s.max() - s.min()
    if rng == 0:
        return pd.Series(0.5, index=s.index)
    z = (s - s.min()) / rng
    return z if higher_is_better else 1 - z


def select_k(X: np.ndarray, cfg: Config, seed: int) -> tuple[int, pd.DataFrame]:
    """Choose k on a composite of three internal indices, not one.

    On real prescriber data the silhouette curve is close to flat, so picking its
    argmax is effectively arbitrary. The composite averages min-max normalised
    silhouette (up), Calinski-Harabasz (up) and Davies-Bouldin (down); ties break
    toward the smaller k, because a segmentation the brand team cannot action is
    worth nothing regardless of its index scores.
    """
    rows = []
    for k in cfg["segmentation"]["k_range"]:
        km = KMeans(n_clusters=k, n_init=cfg["segmentation"]["n_init"],
                    max_iter=cfg["segmentation"]["max_iter"], random_state=seed)
        labels = km.fit_predict(X)
        rows.append({
            "k": k,
            "inertia": round(float(km.inertia_), 2),
            "silhouette": round(float(silhouette_score(X, labels)), 4),
            "davies_bouldin": round(float(davies_bouldin_score(X, labels)), 4),
            "calinski_harabasz": round(float(calinski_harabasz_score(X, labels)), 1),
        })
        log.info("k=%d  silhouette=%.4f  db=%.4f  ch=%.0f", k, rows[-1]["silhouette"],
                 rows[-1]["davies_bouldin"], rows[-1]["calinski_harabasz"])

    diag = pd.DataFrame(rows)
    diag["composite_score"] = (
        _minmax(diag["silhouette"]) * 0.5
        + _minmax(diag["calinski_harabasz"]) * 0.25
        + _minmax(diag["davies_bouldin"], higher_is_better=False) * 0.25).round(4)
    best_k = int(diag.sort_values(["composite_score", "k"],
                                  ascending=[False, True])["k"].iloc[0])
    log.info("selected k=%d on composite index (silhouette range %.3f-%.3f)",
             best_k, diag["silhouette"].min(), diag["silhouette"].max())
    return best_k, diag


def cluster_stability(X: np.ndarray, k: int, cfg: Config) -> dict:
    """Bootstrap stability: refit on subsamples, score agreement with the full fit."""
    seg_cfg = cfg["segmentation"]
    runs, frac = seg_cfg["stability_runs"], seg_cfg["stability_subsample"]
    base = KMeans(n_clusters=k, n_init=seg_cfg["n_init"], random_state=cfg.seed).fit(X)
    base_labels = base.labels_
    rng = np.random.default_rng(cfg.seed)
    scores = []
    for i in range(runs):
        idx = rng.choice(len(X), size=int(frac * len(X)), replace=False)
        km = KMeans(n_clusters=k, n_init=10, random_state=cfg.seed + i + 1).fit(X[idx])
        scores.append(adjusted_rand_score(base_labels, km.predict(X)))
    return {"stability_ari_mean": round(float(np.mean(scores)), 4),
            "stability_ari_min": round(float(np.min(scores)), 4),
            "stability_runs": runs}


ARCHETYPES: list[tuple[str, dict[str, float]]] = [
    # name                             centroid-percentile weights (sum of |w| = 1)
    ("High-Value Loyalists",           {"revenue_12m": 0.45, "market_share": 0.35,
                                        "unrealized_opportunity": -0.20}),
    ("High-Response Growth Targets",   {"promo_response_index": 0.45,
                                        "unrealized_opportunity": 0.30,
                                        "trx_growth_pct": 0.25}),
    ("Untapped Potential",             {"unrealized_opportunity": 0.45,
                                        "market_share": -0.35, "trx_12m": 0.20}),
    ("At-Risk Value",                  {"revenue_12m": 0.40, "trx_growth_pct": -0.40,
                                        "market_share": 0.20}),
    ("Emerging Adopters",              {"trx_growth_pct": 0.45, "nrx_ratio": 0.35,
                                        "revenue_12m": -0.20}),
    ("Steady Maintainers",             {"market_share": 0.35, "trx_growth_pct": 0.15,
                                        "unrealized_opportunity": -0.25,
                                        "call_frequency": 0.25}),
    ("Low-Yield / Deprioritise",       {"revenue_12m": -0.40,
                                        "promo_response_index": -0.35,
                                        "unrealized_opportunity": -0.25}),
]


def name_clusters(profile: pd.DataFrame) -> dict[int, str]:
    """Assign each cluster its best-fitting archetype, without repeats.

    Centroids are converted to cross-cluster percentile ranks so the scoring is
    scale-free, then matched greedily on descending fit. Greedy assignment keeps
    labels distinct and stable, which matters because the label - not the cluster
    id - is what appears on the field report.
    """
    pct = profile.rank(pct=True)
    scores = []
    for cid in profile.index:
        for name, weights in ARCHETYPES:
            fit = sum(w * (pct.loc[cid, f] - 0.5) for f, w in weights.items()
                      if f in pct.columns)
            scores.append((float(fit), int(cid), name))
    scores.sort(reverse=True)

    names: dict[int, str] = {}
    used: set[str] = set()
    for _, cid, name in scores:
        if cid in names or name in used:
            continue
        names[cid] = name
        used.add(name)
    for cid in profile.index:                       # safety net, never expected
        names.setdefault(int(cid), f"Segment {cid}")
    return names


def strategic_grid(abt: pd.DataFrame) -> pd.Series:
    """Deterministic Value x Opportunity 2x2, split at the median of each axis."""
    v_hi = abt["revenue_12m"] >= abt["revenue_12m"].median()
    o_hi = abt["opportunity_revenue"] >= abt["opportunity_revenue"].median()
    return pd.Series(
        np.select(
            [v_hi & o_hi, v_hi & ~o_hi, ~v_hi & o_hi],
            ["Priority Invest", "Protect", "Grow"],
            default="Deprioritise"),
        index=abt.index, name="strategic_quadrant")


def validate_against_truth(seg: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """Score the segmentation against the held-out responsiveness ground truth.

    The test is deliberately the operational one: rank segments by the *observed*
    response index (what the model can see), take the segments that make up the
    top ~40% of prescribers, and ask how enriched they are in truly high-response
    prescribers relative to random allocation.
    """
    df = seg.merge(truth, on="hcp_id", how="inner")
    if df.empty:
        return {}

    ranked = (df.groupby("segment_name")
                .agg(n=("hcp_id", "size"),
                     observed_response=("promo_response_index", "mean"),
                     mean_alpha=("alpha", "mean"),
                     pct_high_responder=("resp_class", lambda s: float((s == "High").mean())))
                .sort_values("observed_response", ascending=False))
    ranked["cum_share_of_hcps"] = ranked["n"].cumsum() / ranked["n"].sum()

    priority = list(ranked.index[:max(int((ranked["cum_share_of_hcps"] <= 0.45).sum()), 1)])
    in_priority = df["segment_name"].isin(priority)
    base = float((df["resp_class"] == "High").mean())
    captured = float((df.loc[in_priority, "resp_class"] == "High").mean())

    return {
        "priority_segments_by_observed_response": priority,
        "high_responder_base_rate": round(base, 4),
        "high_responder_rate_in_priority_segments": round(captured, 4),
        "targeting_lift_vs_random": round(captured / base, 3) if base else None,
        "mean_alpha_top_vs_bottom_segment": round(
            float(ranked["mean_alpha"].iloc[0] / ranked["mean_alpha"].iloc[-1]), 3),
        "segment_alpha_spread": round(float(ranked["mean_alpha"].max()
                                            - ranked["mean_alpha"].min()), 4),
        "rank_correlation_observed_vs_true_response": round(
            float(ranked["observed_response"].corr(ranked["mean_alpha"], method="spearman")), 3),
        "segment_truth_profile": ranked.round(4).reset_index().to_dict("records"),
    }


def run(abt: pd.DataFrame, cfg: Config, truth: pd.DataFrame | None = None
        ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    features = cfg["segmentation"]["features"]
    X_raw, cols = build_matrix(abt, features)

    pipe = Pipeline([
        ("yeo", PowerTransformer(method="yeo-johnson", standardize=False)),
        ("scale", StandardScaler()),
    ])
    Z = pipe.fit_transform(X_raw)

    # Engineered features are correlated by construction (value and potential move
    # together). PCA de-correlates before distance-based clustering so no latent
    # factor is silently double-weighted.
    var = cfg["segmentation"].get("pca_variance", 0)
    if var:
        pca_pre = PCA(n_components=var, random_state=cfg.seed)
        X = pca_pre.fit_transform(Z)
        log.info("PCA pre-reduction: %d -> %d dims (%.1f%% variance retained)",
                 Z.shape[1], X.shape[1], 100 * pca_pre.explained_variance_ratio_.sum())
    else:
        X = Z

    best_k, diag = select_k(X, cfg, cfg.seed)
    km = KMeans(n_clusters=best_k, n_init=cfg["segmentation"]["n_init"],
                max_iter=cfg["segmentation"]["max_iter"], random_state=cfg.seed)
    labels = km.fit_predict(X)

    seg = abt.copy()
    seg["cluster_id"] = labels

    profile_cols = ["trx_12m", "revenue_12m", "market_share", "market_trx_12m",
                    "unrealized_opportunity", "promo_response_index", "trx_growth_pct",
                    "call_frequency", "nrx_ratio", "rfm_score", "adoption_slope"]
    profile = seg.groupby("cluster_id")[profile_cols].mean()
    seg["segment_name"] = seg["cluster_id"].map(name_clusters(profile))

    # Value tier: clusters ordered by mean revenue, split into thirds
    order = profile["revenue_12m"].rank(ascending=False, method="first")
    tier_map = {cid: ("High" if r <= best_k / 3 else "Medium" if r <= 2 * best_k / 3 else "Low")
                for cid, r in order.items()}
    seg["value_tier"] = seg["cluster_id"].map(tier_map)
    seg["strategic_quadrant"] = strategic_grid(seg)

    pca = PCA(n_components=2, random_state=cfg.seed)
    comp = pca.fit_transform(Z)
    seg["pca_1"], seg["pca_2"] = comp[:, 0], comp[:, 1]

    summary = (seg.groupby(["cluster_id", "segment_name", "value_tier"], as_index=False)
                  .agg(hcps=("hcp_id", "size"),
                       trx_12m=("trx_12m", "sum"),
                       revenue_12m=("revenue_12m", "sum"),
                       avg_market_share=("market_share", "mean"),
                       avg_response_index=("promo_response_index", "mean"),
                       avg_growth_pct=("trx_growth_pct", "mean"),
                       avg_call_frequency=("call_frequency", "mean"),
                       opportunity_revenue=("opportunity_revenue", "sum"))
                  .sort_values("revenue_12m", ascending=False))
    summary["pct_of_hcps"] = (summary["hcps"] / summary["hcps"].sum()).round(4)
    summary["pct_of_trx"] = (summary["trx_12m"] / summary["trx_12m"].sum()).round(4)
    summary["pct_of_calls"] = (
        seg.groupby("cluster_id")["calls_12m"].sum().reindex(summary["cluster_id"]).to_numpy()
        / seg["calls_12m"].sum()).round(4)

    top_tier = summary[summary["value_tier"] == "High"]
    metrics = {
        "selected_k": best_k,
        "silhouette": float(diag.loc[diag["k"] == best_k, "silhouette"].iloc[0]),
        "davies_bouldin": float(diag.loc[diag["k"] == best_k, "davies_bouldin"].iloc[0]),
        "calinski_harabasz": float(diag.loc[diag["k"] == best_k, "calinski_harabasz"].iloc[0]),
        "n_hcps": int(len(seg)),
        "n_features": len(cols),
        "pca_variance_explained_2d": round(float(pca.explained_variance_ratio_.sum()), 4),
        "high_tier_pct_of_hcps": round(float(top_tier["hcps"].sum() / len(seg)), 4),
        "high_tier_pct_of_trx": round(float(top_tier["trx_12m"].sum()
                                            / seg["trx_12m"].sum()), 4),
        "high_tier_pct_of_calls": round(float(top_tier["pct_of_calls"].sum()), 4),
    }
    metrics.update(cluster_stability(X, best_k, cfg))
    if truth is not None:
        metrics["validation"] = validate_against_truth(seg, truth)

    seg["_diag"] = None
    return seg.drop(columns=["_diag"]), summary, {"metrics": metrics,
                                                  "k_diagnostics": diag.to_dict("records")}


def main(cfg: Config | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cfg = cfg or load_config()
    abt = pd.read_csv(cfg.path("processed") / "abt_hcp.csv")
    truth_path = cfg.path("raw") / "ground_truth_responsiveness.csv"
    truth = pd.read_csv(truth_path) if truth_path.exists() else None

    seg, summary, diagnostics = run(abt, cfg, truth)
    seg.to_csv(cfg.path("processed") / "hcp_segments.csv", index=False)
    summary.to_csv(cfg.path("reports") / "segment_profile.csv", index=False)
    with open(cfg.path("reports") / "segmentation_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(diagnostics, fh, indent=2)
    log.info("segmentation complete: k=%s silhouette=%.4f",
             diagnostics["metrics"]["selected_k"], diagnostics["metrics"]["silhouette"])
    return seg, summary, diagnostics


if __name__ == "__main__":
    main()
