"""
Prescriber decline-risk model.

Business question: which currently-valuable prescribers will materially reduce
their brand writing next quarter, while there is still time to intervene?

Design notes that matter more than the algorithm choice:

  * TEMPORAL SPLIT - features are built only from data up to a cut-off month;
    the label is measured strictly after it. No feature can see its own outcome.
  * POPULATION - restricted to prescribers with a meaningful baseline
    (>= `min_baseline_trx` in the pre-period), because a 100% drop from 1 TRx is
    noise, not churn.
  * BENCHMARKING - a scaled logistic regression (interpretable, shippable to a
    BI layer as coefficients) is compared against gradient boosting on identical
    folds. Both are reported; neither is assumed to win.
  * DECISION METRIC - ROC-AUC plus precision/recall in the top risk decile,
    because the field can only action a short list.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import Config, get_logger, load_config
from .features import build_hcp_abt, build_hcp_product_abt

log = get_logger("risk_model")

MIN_BASELINE_TRX = 8
FEATURES = ["trx_12m", "market_trx_12m", "market_share", "share_delta", "trx_growth_pct",
            "yoy_growth_pct", "nrx_ratio", "adoption_slope", "promo_response_index",
            "call_frequency", "reach_pct", "months_since_last_call", "months_since_last_rx",
            "trx_volatility", "rfm_score", "decile", "n_brands_prescribed",
            "revenue_per_call", "email_open_rate"]


def build_training_frame(mart: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    horizon = cfg["features"]["recent_window_months"]
    last_idx = int(mart["month_index"].max())
    cutoff = last_idx - horizon

    hist = mart[mart["month_index"] <= cutoff]
    abt = build_hcp_abt(build_hcp_product_abt(hist, cfg), hist, cfg)

    pre = (mart[(mart["month_index"] > cutoff - horizon) & (mart["month_index"] <= cutoff)]
           .groupby("hcp_id", as_index=False)["trx"].sum().rename(columns={"trx": "trx_pre"}))
    post = (mart[mart["month_index"] > cutoff]
            .groupby("hcp_id", as_index=False)["trx"].sum().rename(columns={"trx": "trx_post"}))

    df = abt.merge(pre, on="hcp_id", how="left").merge(post, on="hcp_id", how="left")
    df[["trx_pre", "trx_post"]] = df[["trx_pre", "trx_post"]].fillna(0)
    df = df[df["trx_pre"] >= MIN_BASELINE_TRX].copy()
    df["future_change_pct"] = (df["trx_post"] - df["trx_pre"]) / df["trx_pre"]
    df["decline_flag"] = (df["future_change_pct"] < cfg["risk_model"]["decline_threshold"]).astype(int)
    log.info("training frame: %s HCPs, decline rate %.3f",
             f"{len(df):,}", df["decline_flag"].mean())
    return df


def _top_decile_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    k = max(int(0.1 * len(proba)), 1)
    idx = np.argsort(-proba)[:k]
    flagged = np.zeros_like(y_true)
    flagged[idx] = 1
    base = float(y_true.mean())
    prec = float(precision_score(y_true, flagged, zero_division=0))
    return {
        "top_decile_precision": round(prec, 4),
        "top_decile_recall": round(float(recall_score(y_true, flagged, zero_division=0)), 4),
        "top_decile_lift_vs_base_rate": round(prec / base, 3) if base else None,
    }


def train(df: pd.DataFrame, cfg: Config) -> tuple[dict, pd.DataFrame, Pipeline]:
    X = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    y = df["decline_flag"].to_numpy()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg["risk_model"]["test_size"], stratify=y, random_state=cfg.seed)

    models = {
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=0.6, class_weight="balanced",
                                       random_state=cfg.seed)),
        ]),
        "gradient_boosting": Pipeline([
            ("clf", GradientBoostingClassifier(n_estimators=260, learning_rate=0.055,
                                               max_depth=3, subsample=0.85,
                                               random_state=cfg.seed)),
        ]),
    }

    cv = StratifiedKFold(n_splits=cfg["risk_model"]["cv_folds"], shuffle=True,
                         random_state=cfg.seed)
    results, fitted = {}, {}
    for name, pipe in models.items():
        cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc")
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        results[name] = {
            "cv_roc_auc_mean": round(float(cv_auc.mean()), 4),
            "cv_roc_auc_std": round(float(cv_auc.std()), 4),
            "holdout_roc_auc": round(float(roc_auc_score(y_te, proba)), 4),
            "holdout_pr_auc": round(float(average_precision_score(y_te, proba)), 4),
            "holdout_f1": round(float(f1_score(y_te, pred, zero_division=0)), 4),
            "brier_score": round(float(brier_score_loss(y_te, proba)), 4),
            **_top_decile_metrics(y_te, proba),
        }
        fitted[name] = pipe
        log.info("%-20s cv_auc=%.4f  holdout_auc=%.4f", name,
                 results[name]["cv_roc_auc_mean"], results[name]["holdout_roc_auc"])

    best = max(results, key=lambda m: results[m]["holdout_roc_auc"])
    best_pipe = fitted[best]

    if best == "logistic_regression":
        coefs = best_pipe.named_steps["clf"].coef_[0]
        imp = pd.DataFrame({"feature": FEATURES, "coefficient": coefs.round(4),
                            "abs_effect": np.abs(coefs).round(4)})
    else:
        imp = pd.DataFrame({"feature": FEATURES,
                            "coefficient": best_pipe.named_steps["clf"].feature_importances_.round(5),
                            "abs_effect": best_pipe.named_steps["clf"].feature_importances_.round(5)})
    imp = imp.sort_values("abs_effect", ascending=False).reset_index(drop=True)

    metrics = {
        "n_training_hcps": int(len(df)),
        "positive_class_rate": round(float(y.mean()), 4),
        "decline_threshold": cfg["risk_model"]["decline_threshold"],
        "min_baseline_trx": MIN_BASELINE_TRX,
        "n_features": len(FEATURES),
        "best_model": best,
        "models": results,
        "top_drivers": imp.head(8).to_dict("records"),
    }
    return metrics, imp, best_pipe


def score_current_book(pipe: Pipeline, cfg: Config, mart: pd.DataFrame) -> pd.DataFrame:
    """Score every prescriber on the latest available data for field action."""
    abt = build_hcp_abt(build_hcp_product_abt(mart, cfg), mart, cfg)
    X = abt[FEATURES].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    abt["decline_risk_score"] = pipe.predict_proba(X)[:, 1].round(4)
    abt["risk_decile"] = pd.qcut(abt["decline_risk_score"].rank(method="first"),
                                 10, labels=range(1, 11)).astype(int)
    abt["risk_band"] = pd.cut(abt["decline_risk_score"], [-0.01, 0.25, 0.5, 0.75, 1.01],
                              labels=["Low", "Moderate", "Elevated", "High"])
    # Expected revenue loss, not probability-weighted total revenue: a flagged
    # prescriber is assumed to lose the threshold share of their book, not all of it.
    loss_rate = abs(cfg["risk_model"]["decline_threshold"])
    abt["revenue_at_risk"] = (abt["decline_risk_score"] * abt["revenue_12m"]
                              * loss_rate).round(2)
    return abt[["hcp_id", "territory_id", "decile", "trx_12m", "revenue_12m",
                "decline_risk_score", "risk_decile", "risk_band", "revenue_at_risk"]]


def main(cfg: Config | None = None) -> tuple[dict, pd.DataFrame]:
    cfg = cfg or load_config()
    processed, reports = cfg.path("processed"), cfg.path("reports")
    mart_pq = processed / "monthly_mart.parquet"
    mart = pd.read_parquet(mart_pq) if mart_pq.exists() else pd.read_csv(
        processed / "monthly_mart.csv")

    train_df = build_training_frame(mart, cfg)
    metrics, importance, pipe = train(train_df, cfg)
    scores = score_current_book(pipe, cfg, mart)

    top_decile = scores[scores["risk_decile"] == 10]
    metrics["expected_revenue_loss_next_quarter"] = round(
        float(scores["revenue_at_risk"].sum()) / 4, 2)
    metrics["top_decile_revenue_at_risk_12m"] = round(
        float(top_decile["revenue_at_risk"].sum()), 2)
    metrics["top_decile_hcps"] = int(len(top_decile))
    metrics["high_risk_hcps"] = int((scores["risk_band"] == "High").sum())

    scores.to_csv(processed / "hcp_risk_scores.csv", index=False)
    importance.to_csv(reports / "risk_model_drivers.csv", index=False)
    with open(reports / "risk_model_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    return metrics, scores


if __name__ == "__main__":
    main()
