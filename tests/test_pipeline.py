"""
Test suite.

The suite is built around the properties that would actually break a commercial
deliverable: a broken grain, a leaking label, an optimiser that violates the
capacity it was given, a metric that silently changes definition. Statistical
quality is asserted with loose thresholds; exact values are not pinned, because
a test that fails on a library upgrade teaches nothing.

A small fixture dataset is generated once per session so the suite runs in
seconds without touching the full pipeline output.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src import etl, features, segmentation, sfe, territory
from src.config import Config, load_config
from src.generate_data import PharmaDataSimulator


# --------------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def cfg() -> Config:
    raw = copy.deepcopy(load_config().raw)
    raw["simulation"].update(n_hcps=400, n_territories=8, n_reps=8, months=18)
    raw["segmentation"].update(k_range=[3, 4], n_init=5, stability_runs=3)
    raw["risk_model"].update(cv_folds=3)
    return Config(raw)


@pytest.fixture(scope="session")
def tables(cfg: Config) -> dict[str, pd.DataFrame]:
    sim = PharmaDataSimulator(cfg)
    dim_date = sim.build_dim_date()
    dim_product = sim.build_dim_product()
    dim_territory = sim.build_dim_territory()
    dim_hcp = sim.build_dim_hcp(dim_territory)
    fact_rx, fact_activity = sim.simulate(dim_hcp, dim_product, dim_territory)
    fact_sales = sim.build_fact_sales(fact_rx, dim_hcp, dim_product)
    return {
        "dim_date": dim_date, "dim_product": dim_product,
        "dim_territory": dim_territory,
        "dim_hcp": dim_hcp.drop(columns=[c for c in dim_hcp.columns if c.startswith("_")]),
        "_dim_hcp_full": dim_hcp,
        "fact_rx": fact_rx, "fact_activity": fact_activity, "fact_sales": fact_sales,
    }


@pytest.fixture(scope="session")
def mart(tables, cfg) -> pd.DataFrame:
    return etl.build_monthly_mart({k: v for k, v in tables.items()
                                   if not k.startswith("_")})


@pytest.fixture(scope="session")
def abts(mart, cfg):
    hp = features.build_hcp_product_abt(mart, cfg)
    return features.build_hcp_abt(hp, mart, cfg), hp


@pytest.fixture(scope="session")
def seg_output(abts, tables, cfg):
    abt, _ = abts
    truth = tables["_dim_hcp_full"][["hcp_id", "_resp_class", "_alpha"]].rename(
        columns={"_resp_class": "resp_class", "_alpha": "alpha"})
    return segmentation.run(abt, cfg, truth)


# ------------------------------------------------------------- data generation
class TestGeneration:
    def test_business_rules_hold_by_construction(self, tables):
        rx = tables["fact_rx"]
        assert (rx["trx"] <= rx["market_trx"]).all(), "brand TRx exceeded category TRx"
        assert (rx["nrx"] <= rx["trx"]).all(), "NRx exceeded TRx"
        assert (rx[["trx", "nrx", "market_trx"]] >= 0).to_numpy().all()

    def test_panel_is_balanced(self, tables, cfg):
        rx = tables["fact_rx"]
        counts = rx.groupby(["hcp_id", "product_id"])["month"].nunique()
        assert counts.eq(cfg["simulation"]["months"]).all()

    def test_no_see_hcps_are_effectively_unreachable(self, tables):
        act = tables["fact_activity"].merge(
            tables["dim_hcp"][["hcp_id", "no_see_flag"]], on="hcp_id")
        no_see = act[act["no_see_flag"] == 1]["calls"].mean()
        seeing = act[act["no_see_flag"] == 0]["calls"].mean()
        assert no_see < 0.25 * seeing

    def test_response_signal_is_present_for_high_responders(self, tables, mart):
        truth = tables["_dim_hcp_full"][["hcp_id", "_resp_class"]].rename(
            columns={"_resp_class": "resp_class"})
        df = mart.merge(truth, on="hcp_id")
        high = df[df["resp_class"] == "High"]
        called = high.loc[high["calls_ma3"] >= 1, "market_share"].mean()
        quiet = high.loc[high["calls_ma3"] < 1, "market_share"].mean()
        assert called > quiet, "generator produced no promotional response to recover"

    def test_reproducible_from_seed(self, cfg):
        a = PharmaDataSimulator(cfg).build_dim_hcp(
            PharmaDataSimulator(cfg).build_dim_territory())
        b = PharmaDataSimulator(cfg).build_dim_hcp(
            PharmaDataSimulator(cfg).build_dim_territory())
        pd.testing.assert_frame_equal(a, b)


# ------------------------------------------------------------------ data quality
class TestDataQuality:
    def test_clean_data_passes_every_check(self, tables):
        report = etl.validate({k: v for k, v in tables.items() if not k.startswith("_")})
        assert not report.critical_failures
        assert (report.to_frame()["status"] == "PASS").all()

    def test_injected_defect_is_caught(self, tables):
        broken = {k: v.copy() for k, v in tables.items() if not k.startswith("_")}
        broken["fact_rx"].loc[0, "trx"] = broken["fact_rx"].loc[0, "market_trx"] + 5
        report = etl.validate(broken)
        failed = {r["check"] for r in report.critical_failures}
        assert "trx_lte_market_trx" in failed

    def test_orphan_key_is_caught(self, tables):
        broken = {k: v.copy() for k, v in tables.items() if not k.startswith("_")}
        broken["fact_rx"].loc[0, "hcp_id"] = "H99999"
        report = etl.validate(broken)
        assert "fk_hcp_id" in {r["check"] for r in report.critical_failures}

    def test_pipeline_aborts_on_critical_failure(self, tables, cfg, monkeypatch, tmp_path):
        broken = {k: v.copy() for k, v in tables.items() if not k.startswith("_")}
        broken["fact_rx"].loc[0, "nrx"] = broken["fact_rx"].loc[0, "trx"] + 1
        monkeypatch.setattr(etl, "load_raw", lambda _: broken)
        monkeypatch.setattr(Config, "path", lambda self, key: tmp_path)
        with pytest.raises(etl.DataQualityError):
            etl.main(cfg)


# ---------------------------------------------------------------------- mart
class TestMart:
    def test_grain_is_unique(self, mart):
        assert not mart.duplicated(subset=["hcp_id", "product_id", "month"]).any()

    def test_moving_average_respects_series_boundaries(self, mart):
        first = mart.sort_values("month_index").groupby(["hcp_id", "product_id"]).first()
        assert np.allclose(first["calls_ma3"], first["calls"]), \
            "first month's MA3 leaked across prescribers"

    def test_share_is_bounded(self, mart):
        share = mart["market_share"].dropna()
        assert share.between(0, 1).all()


# ------------------------------------------------------------------- features
class TestFeatures:
    def test_one_row_per_hcp(self, abts, tables):
        abt, _ = abts
        assert len(abt) == len(tables["dim_hcp"])
        assert abt["hcp_id"].is_unique

    def test_no_nulls_or_infinities(self, abts):
        abt, _ = abts
        numeric = abt.select_dtypes(include=[np.number])
        assert not numeric.isna().to_numpy().any()
        assert np.isfinite(numeric.to_numpy()).all()

    def test_volume_reconciles_with_the_mart(self, abts, mart, cfg):
        abt, _ = abts
        window = cfg["features"]["analysis_window_months"]
        last = mart["month_index"].max()
        expected = mart.loc[mart["month_index"] >= last - window + 1, "trx"].sum()
        assert abt["trx_12m"].sum() == expected

    def test_opportunity_is_never_negative(self, abts):
        abt, _ = abts
        assert (abt["unrealized_opportunity"] >= 0).all()

    def test_rfm_scores_are_in_range(self, abts):
        abt, _ = abts
        assert abt["rfm_score"].between(3, 15).all()

    def test_response_index_tracks_latent_responsiveness(self, abts, tables):
        abt, _ = abts
        truth = tables["_dim_hcp_full"][["hcp_id", "_alpha"]].rename(
            columns={"_alpha": "alpha"})
        df = abt.merge(truth, on="hcp_id")
        assert df["promo_response_index"].corr(df["alpha"]) > 0.10


# --------------------------------------------------------------- segmentation
class TestSegmentation:
    def test_every_hcp_is_assigned_exactly_one_segment(self, seg_output, abts):
        seg, _, _ = seg_output
        abt, _ = abts
        assert len(seg) == len(abt)
        assert seg["segment_name"].notna().all()
        assert seg["cluster_id"].notna().all()

    def test_segment_names_are_unique_per_cluster(self, seg_output):
        seg, _, _ = seg_output
        mapping = seg.groupby("cluster_id")["segment_name"].nunique()
        assert (mapping == 1).all()
        assert seg["segment_name"].nunique() == seg["cluster_id"].nunique()

    def test_summary_shares_sum_to_one(self, seg_output):
        _, summary, _ = seg_output
        assert summary["pct_of_hcps"].sum() == pytest.approx(1.0, abs=1e-3)
        assert summary["pct_of_trx"].sum() == pytest.approx(1.0, abs=1e-3)

    def test_selected_k_is_inside_the_candidate_range(self, seg_output, cfg):
        _, _, diag = seg_output
        assert diag["metrics"]["selected_k"] in cfg["segmentation"]["k_range"]

    def test_clusters_are_separated_and_stable(self, seg_output):
        _, _, diag = seg_output
        assert diag["metrics"]["silhouette"] > 0.10
        assert diag["metrics"]["stability_ari_mean"] > 0.5

    def test_strategic_grid_covers_all_four_quadrants(self, seg_output):
        seg, _, _ = seg_output
        assert set(seg["strategic_quadrant"]) == {
            "Priority Invest", "Protect", "Grow", "Deprioritise"}

    def test_segments_order_true_responsiveness_correctly(self, seg_output):
        _, _, diag = seg_output
        val = diag["metrics"].get("validation", {})
        assert val, "ground-truth validation did not run"
        assert val["rank_correlation_observed_vs_true_response"] > 0.5


# ------------------------------------------------------------------------ SFE
@pytest.fixture(scope="session")
def fitted(mart, seg_output, abts, cfg):
    seg, _, _ = seg_output
    _, hp_abt = abts
    panel = mart.merge(seg[["hcp_id", "segment_name"]], on="hcp_id", how="left")
    curves = sfe.fit_response_curves(panel)
    econ = sfe.build_marginal_economics(hp_abt, seg, curves, cfg)
    plan, summary = sfe.optimise_call_plan(econ, cfg)
    return curves, plan, summary


@pytest.fixture(scope="session")
def scorecard(mart, tables, seg_output, fitted, cfg):
    seg, _, _ = seg_output
    _, plan, _ = fitted
    return territory.build_scorecard(mart, tables["fact_sales"],
                                     tables["dim_territory"], seg, plan, cfg)


class TestSalesForceEffectiveness:
    def test_response_fn_is_monotone_and_saturating(self):
        c = np.linspace(0, 6, 60)
        y = sfe.response_fn(c, 0.3, 0.15, 0.6)
        assert np.all(np.diff(y) >= 0)
        assert y[-1] < 0.3 + 0.15 + 1e-9
        d = np.diff(y)
        assert np.all(np.diff(d) <= 1e-12), "marginal returns must be diminishing"

    def test_curve_parameters_are_in_bounds(self, fitted):
        curves, _, _ = fitted
        assert (curves["a"].between(sfe.B_LOWER[1], sfe.B_UPPER[1])).all()
        assert (curves["b"].between(sfe.B_LOWER[2], sfe.B_UPPER[2])).all()

    def test_every_segment_gets_a_curve(self, fitted, seg_output):
        curves, _, _ = fitted
        seg, _, _ = seg_output
        assert set(curves["segment_name"]) == set(seg["segment_name"])

    def test_optimiser_respects_territory_capacity(self, fitted):
        _, plan, _ = fitted
        current = plan.groupby("territory_id")["current_calls_monthly"].sum()
        recommended = plan.groupby("territory_id")["recommended_calls_monthly"].sum()
        # budget-neutral: never more than today's load (allow 1 call of rounding)
        assert (recommended <= current.round() + 1).all()

    def test_optimiser_never_exceeds_the_evidence_cap(self, fitted):
        _, plan, _ = fitted
        assert (plan["recommended_calls_monthly"] <= plan["call_cap"]).all()
        assert (plan["profitmax_calls_monthly"] <= plan["call_cap"]).all()

    def test_no_see_prescribers_get_no_field_calls(self, fitted):
        _, plan, _ = fitted
        assert plan.loc[plan["no_see_flag"] == 1, "recommended_calls_monthly"].sum() == 0

    def test_reallocation_never_loses_promotional_trx(self, fitted):
        """Greedy allocation on a concave objective cannot do worse than today."""
        _, _, summary = fitted
        assert summary["scenario_a_budget_neutral"]["incremental_trx_per_month"] >= 0

    def test_profit_max_never_spends_more_than_budget_neutral(self, fitted):
        _, _, summary = fitted
        assert (summary["scenario_b_profit_max"]["calls_per_month"]
                <= summary["scenario_a_budget_neutral"]["calls_per_month"] + 1)

    def test_actions_are_consistent_with_call_deltas(self, fitted):
        _, plan, _ = fitted
        increased = plan[plan["action"] == "Increase frequency"]
        assert (increased["delta_calls"] > 0).all()
        dropped = plan[plan["action"] == "Deprioritise - no field calls"]
        assert (dropped["recommended_calls_monthly"] == 0).all()

    def test_channel_attribution_recovers_a_positive_call_effect(self, mart, seg_output, cfg):
        seg, _, _ = seg_output
        panel = mart.merge(seg[["hcp_id", "segment_name"]], on="hcp_id", how="left")
        attr = sfe.channel_attribution(panel, cfg)
        detail = attr[attr["channel"].str.startswith("Field")].iloc[0]
        assert detail["share_lift_per_unit"] > 0
        assert detail["t_stat"] > 2


# ------------------------------------------------------------------ territory
class TestTerritory:
    def test_one_row_per_territory(self, scorecard, tables):
        sc, _ = scorecard
        assert len(sc) == len(tables["dim_territory"])
        assert sc["territory_id"].is_unique

    def test_attainment_and_share_are_plausible(self, scorecard):
        sc, _ = scorecard
        assert sc["quota_attainment"].between(0.4, 2.0).all()
        assert sc["market_share"].between(0, 1).all()

    def test_rep_effect_is_centred(self, scorecard):
        sc, _ = scorecard
        assert abs(sc["rep_effect_zscore"].mean()) < 0.05

    def test_gini_bounds(self):
        assert territory.gini(np.ones(10)) == pytest.approx(0.0, abs=1e-9)
        assert territory.gini(np.array([0, 0, 0, 100.0])) > 0.6
        assert territory.gini(np.array([])) == 0.0

    def test_potential_model_explains_most_variance(self, scorecard):
        _, summary = scorecard
        assert summary["potential_model_r2"] > 0.4
