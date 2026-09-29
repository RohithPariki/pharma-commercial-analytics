"""
Synthetic data generator for a pharmaceutical commercial dataset.

The generator is not a random-number dump: it encodes the causal structure that
real commercial analytics teams try to recover from data.

  1. Every HCP has a hidden *promotional responsiveness class* (High / Moderate /
     Low / Non-responder) with its own saturating call-response curve
     `lift = alpha * (1 - exp(-beta * calls))`.
  2. The *historical* call plan is allocated on decile (market volume) alone -
     the way most legacy call plans are built - and is deliberately blind to
     responsiveness. This creates genuine, recoverable misallocation: high-alpha
     prescribers are under-called and non-responders are over-called.
  3. Brand share is bounded, promotion is subject to diminishing returns, and
     both TRx and market TRx carry seasonality, lifecycle drift and noise.

Downstream modules never see alpha, beta or the responsiveness class - they are
written to a held-out ground-truth file used only for validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, get_logger, load_config

log = get_logger("generate_data")

FIRST_NAMES = ["Aarav", "Priya", "Daniel", "Sofia", "Marcus", "Elena", "Rohan", "Grace",
               "Julian", "Nadia", "Omar", "Clara", "Victor", "Ingrid", "Hassan", "Maya",
               "Peter", "Lucia", "Samuel", "Reena", "Tobias", "Anika", "Felix", "Noor"]
LAST_NAMES = ["Whitfield", "Marchetti", "Okonkwo", "Lindqvist", "Ramachandran", "Beaumont",
              "Kowalski", "Nakamura", "Delgado", "Ferreira", "Haddad", "Novak", "Sorensen",
              "Bhattacharya", "Castellano", "Vandermeer", "Ashworth", "Petrov", "Malhotra",
              "Quinn", "Baptiste", "Eriksen", "Moreau", "Sandoval"]
AFFILIATIONS = ["Academic Medical Center", "Community Hospital", "Private Clinic",
                "Integrated Delivery Network", "Group Practice"]
REGIONS = ["Northeast", "Southeast", "Midwest", "South Central", "Mountain", "West"]


class PharmaDataSimulator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.sim = cfg["simulation"]
        self.months = pd.period_range(self.sim["start_month"],
                                      periods=self.sim["months"], freq="M")

    # ------------------------------------------------------------------ dims
    def build_dim_date(self) -> pd.DataFrame:
        df = pd.DataFrame({"month": self.months.astype(str)})
        df["month_index"] = np.arange(len(df))
        ts = self.months.to_timestamp()
        df["month_start"] = ts.strftime("%Y-%m-%d")
        df["year"] = ts.year
        df["quarter"] = "Q" + ts.quarter.astype(str)
        df["year_quarter"] = df["year"].astype(str) + "-" + df["quarter"]
        df["month_name"] = ts.strftime("%b")
        df["month_of_year"] = ts.month
        return df

    def build_dim_product(self) -> pd.DataFrame:
        return pd.DataFrame(self.cfg["products"])

    def build_dim_territory(self) -> pd.DataFrame:
        n_terr = self.sim["n_territories"]
        regions = [REGIONS[i % len(REGIONS)] for i in range(n_terr)]
        rep_ids = [f"R{i + 1:03d}" for i in range(n_terr)]
        rep_names = [f"{self.rng.choice(FIRST_NAMES)} {self.rng.choice(LAST_NAMES)}"
                     for _ in range(n_terr)]
        return pd.DataFrame({
            "territory_id": [f"T{i + 1:03d}" for i in range(n_terr)],
            "territory_name": [f"{regions[i]}-{i + 1:02d}" for i in range(n_terr)],
            "region": regions,
            "rep_id": rep_ids,
            "rep_name": rep_names,
            # Execution quality (how well a call lands) and call capacity (how many
            # calls get made) are deliberately independent - conflating effort with
            # effectiveness is the classic error in rep scorecards.
            "rep_skill_index": np.round(self.rng.normal(1.0, 0.18, n_terr).clip(0.55, 1.5), 3),
            "rep_call_intensity": np.round(self.rng.normal(1.0, 0.22, n_terr).clip(0.5, 1.6), 3),
            "rep_tenure_months": self.rng.integers(4, 132, n_terr),
        })

    def build_dim_hcp(self, dim_territory: pd.DataFrame) -> pd.DataFrame:
        n = self.sim["n_hcps"]
        spec_cfg = self.cfg["specialties"]
        specs = list(spec_cfg)
        probs = np.array([spec_cfg[s]["weight"] for s in specs], dtype=float)
        probs = probs / probs.sum()
        specialty = self.rng.choice(specs, size=n, p=probs)

        resp_cfg = self.cfg["responsiveness"]
        classes = list(resp_cfg)
        rprobs = np.array([resp_cfg[c]["share"] for c in classes], dtype=float)
        rprobs = rprobs / rprobs.sum()
        resp_class = self.rng.choice(classes, size=n, p=rprobs)

        base_volume = np.array([spec_cfg[s]["base_volume"] for s in specialty], dtype=float)
        # Lognormal heterogeneity -> classic long-tailed prescriber distribution
        volume_multiplier = self.rng.lognormal(mean=-0.16, sigma=0.62, size=n)
        hcp_volume = np.round(base_volume * volume_multiplier, 2).clip(2, None)

        hcp = pd.DataFrame({
            "hcp_id": [f"H{i + 1:05d}" for i in range(n)],
            "hcp_name": [f"Dr. {self.rng.choice(FIRST_NAMES)} {self.rng.choice(LAST_NAMES)}"
                         for _ in range(n)],
            "specialty": specialty,
            "affiliation_type": self.rng.choice(
                AFFILIATIONS, size=n, p=[0.12, 0.24, 0.30, 0.18, 0.16]),
            "territory_id": self.rng.choice(dim_territory["territory_id"], size=n),
            "years_in_practice": self.rng.integers(2, 38, n),
            "is_kol": (self.rng.random(n) < 0.06).astype(int),
            "commercial_payer_pct": np.round(self.rng.beta(6, 4, n), 3),
            "no_see_flag": (self.rng.random(n) < 0.08).astype(int),
            "monthly_market_volume": hcp_volume,
        })
        # Decile 10 = highest market volume (standard pharma convention)
        hcp["decile"] = pd.qcut(hcp["monthly_market_volume"].rank(method="first"),
                                10, labels=range(1, 11)).astype(int)

        # ---- hidden ground truth ------------------------------------------
        hcp["_resp_class"] = resp_class
        hcp["_alpha"] = [resp_cfg[c]["alpha"] for c in resp_class]
        hcp["_beta"] = [resp_cfg[c]["beta"] for c in resp_class]
        hcp["_alpha"] *= self.rng.normal(1.0, 0.12, n).clip(0.5, 1.6)
        hcp["_base_affinity"] = self.rng.beta(2.6, 6.0, n).clip(0.01, 0.55)
        return hcp

    # ----------------------------------------------------------------- facts
    def _eligible_pairs(self, dim_hcp: pd.DataFrame, dim_product: pd.DataFrame) -> pd.DataFrame:
        spec_cfg = self.cfg["specialties"]
        rows = []
        for prod in dim_product.itertuples():
            eligible = dim_hcp["specialty"].map(lambda s: prod.therapy_area in spec_cfg[s]["ta"])
            sub = dim_hcp.loc[eligible, ["hcp_id", "territory_id", "decile",
                                         "monthly_market_volume", "no_see_flag",
                                         "_alpha", "_beta", "_base_affinity"]].copy()
            sub["product_id"] = prod.product_id
            sub["lifecycle"] = prod.lifecycle
            rows.append(sub)
        pairs = pd.concat(rows, ignore_index=True)
        # Multi-TA prescribers split their category volume between therapy areas
        share_of_volume = pairs.groupby("hcp_id")["product_id"].transform("size")
        pairs["pair_market_volume"] = pairs["monthly_market_volume"] / np.sqrt(share_of_volume)
        return pairs

    def _historical_call_lambda(self, pairs: pd.DataFrame,
                                dim_territory: pd.DataFrame) -> np.ndarray:
        """Legacy call plan: decile-driven, responsiveness-blind, noisy."""
        decile = pairs["decile"].to_numpy()
        lam = 0.10 + 2.00 * (decile / 10.0) ** 2.2                 # 0.1 - 2.1 calls/month
        lam *= self.rng.normal(1.0, 0.28, len(pairs)).clip(0.25, 2.0)
        intensity = dim_territory.set_index("territory_id")["rep_call_intensity"]
        lam *= pairs["territory_id"].map(intensity).to_numpy()
        lam *= np.where(pairs["no_see_flag"].to_numpy() == 1, 0.05, 1.0)
        return lam.clip(0, self.cfg["targeting"]["max_calls_per_hcp_per_month"])

    def simulate(self, dim_hcp: pd.DataFrame, dim_product: pd.DataFrame,
                 dim_territory: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        pairs = self._eligible_pairs(dim_hcp, dim_product)
        n_pairs = len(pairs)
        n_months = self.sim["months"]
        log.info("Simulating %s HCP-product pairs over %s months", f"{n_pairs:,}", n_months)

        call_lambda = self._historical_call_lambda(pairs, dim_territory)
        skill = dim_territory.set_index("territory_id")["rep_skill_index"]
        skill_vec = pairs["territory_id"].map(skill).to_numpy()
        # Execution quality shifts baseline share as well as call response: a strong
        # rep holds share through access, follow-up and speaker logistics.
        execution_effect = 0.10 * (skill_vec - 1.0)

        alpha = pairs["_alpha"].to_numpy()
        beta = pairs["_beta"].to_numpy()
        affinity = pairs["_base_affinity"].to_numpy()
        volume = pairs["pair_market_volume"].to_numpy()
        lifecycle = pairs["lifecycle"].to_numpy()
        max_calls = self.cfg["targeting"]["max_calls_per_hcp_per_month"]

        # Lifecycle drift on baseline share: growth brand gains, mature brand erodes
        drift_per_month = np.where(lifecycle == "growth", 0.0055, -0.0011)

        call_hist = np.zeros((3, n_pairs))
        email_hist = np.zeros((3, n_pairs))
        rx_records, act_records = [], []

        month_labels = self.months.astype(str)
        for m in range(n_months):
            seasonal_rx = 1.0 + 0.075 * np.sin(2 * np.pi * (m + 2) / 12)
            seasonal_field = 1.0 + 0.10 * np.sin(2 * np.pi * (m + 9) / 12)  # holiday dips

            calls = self.rng.poisson(call_lambda * seasonal_field).clip(0, max_calls)
            emails = self.rng.poisson(np.where(pairs["no_see_flag"] == 1, 3.0, 1.6))
            open_rate = np.where(pairs["no_see_flag"] == 1, 0.18, 0.31)
            email_opens = self.rng.binomial(emails, open_rate)
            samples = np.where(calls > 0, self.rng.poisson(1.4 * calls), 0)
            speaker = (self.rng.random(n_pairs) < 0.010 * (pairs["decile"].to_numpy() / 10)).astype(int)

            call_hist = np.vstack([call_hist[1:], calls])
            email_hist = np.vstack([email_hist[1:], email_opens])
            calls_ma3 = call_hist.mean(axis=0)
            opens_ma3 = email_hist.mean(axis=0)

            promo_lift = alpha * skill_vec * (1 - np.exp(-beta * calls_ma3))
            digital_lift = 0.22 * alpha * (1 - np.exp(-0.5 * opens_ma3))
            speaker_lift = 0.035 * speaker
            noise = self.rng.normal(0, 0.028, n_pairs)

            share = np.clip(affinity + execution_effect + drift_per_month * m + promo_lift
                            + digital_lift + speaker_lift + noise, 0.005, 0.88)

            market_trx = self.rng.poisson(volume * seasonal_rx).clip(0, None)
            trx = self.rng.binomial(market_trx, share)
            # New-to-brand share is higher for growth assets and recently promoted HCPs
            nrx_rate = np.clip(0.17 + 0.09 * (lifecycle == "growth") + 0.05 * (calls_ma3 > 1), 0, 0.6)
            nrx = self.rng.binomial(trx, nrx_rate)

            rx_records.append(pd.DataFrame({
                "hcp_id": pairs["hcp_id"].to_numpy(),
                "product_id": pairs["product_id"].to_numpy(),
                "month": month_labels[m],
                "market_trx": market_trx,
                "trx": trx,
                "nrx": nrx,
            }))
            act_records.append(pd.DataFrame({
                "hcp_id": pairs["hcp_id"].to_numpy(),
                "product_id": pairs["product_id"].to_numpy(),
                "month": month_labels[m],
                "calls": calls,
                "samples_dropped": samples,
                "emails_sent": emails,
                "emails_opened": email_opens,
                "speaker_programs": speaker,
            }))

        fact_rx = pd.concat(rx_records, ignore_index=True)
        fact_activity = pd.concat(act_records, ignore_index=True)
        return fact_rx, fact_activity

    def build_fact_sales(self, fact_rx: pd.DataFrame, dim_hcp: pd.DataFrame,
                         dim_product: pd.DataFrame) -> pd.DataFrame:
        rev = dim_product.set_index("product_id")["net_revenue_per_trx"]
        df = fact_rx.merge(dim_hcp[["hcp_id", "territory_id"]], on="hcp_id", how="left")
        df["revenue"] = df["trx"] * df["product_id"].map(rev)
        sales = (df.groupby(["territory_id", "product_id", "month"], as_index=False)
                   .agg(units=("trx", "sum"), revenue=("revenue", "sum")))
        sales = sales.sort_values(["territory_id", "product_id", "month"])
        # Quota = trailing 12-month run rate uplifted by a growth target, with
        # territory-level goal-setting error (the reason attainment varies).
        prior = (sales.groupby(["territory_id", "product_id"])["revenue"]
                      .transform(lambda s: s.shift(12).rolling(3, min_periods=1).mean()))
        fallback = sales.groupby(["territory_id", "product_id"])["revenue"].transform("mean")
        base = prior.fillna(fallback)
        # National stretch target above realised growth, plus territory-level goal
        # setting error. Without the error every territory lands on plan and the
        # scorecard says nothing.
        terr_bias = pd.Series(
            self.rng.normal(1.0, 0.11, sales["territory_id"].nunique()),
            index=sorted(sales["territory_id"].unique())).clip(0.75, 1.30)
        goal_error = (sales["territory_id"].map(terr_bias).to_numpy()
                      * self.rng.normal(1.0, 0.05, len(sales)).clip(0.85, 1.18))
        sales["quota_revenue"] = np.round(base * 1.14 * goal_error, 2)
        sales["revenue"] = np.round(sales["revenue"], 2)
        return sales.reset_index(drop=True)


def main(cfg: Config | None = None) -> dict[str, pd.DataFrame]:
    cfg = cfg or load_config()
    sim = PharmaDataSimulator(cfg)
    raw = cfg.path("raw")

    dim_date = sim.build_dim_date()
    dim_product = sim.build_dim_product()
    dim_territory = sim.build_dim_territory()
    dim_hcp = sim.build_dim_hcp(dim_territory)
    fact_rx, fact_activity = sim.simulate(dim_hcp, dim_product, dim_territory)
    fact_sales = sim.build_fact_sales(fact_rx, dim_hcp, dim_product)

    ground_truth = dim_hcp[["hcp_id", "_resp_class", "_alpha", "_beta", "_base_affinity"]].rename(
        columns=lambda c: c.lstrip("_"))
    dim_hcp_public = dim_hcp.drop(columns=[c for c in dim_hcp.columns if c.startswith("_")])

    tables = {
        "dim_date": dim_date,
        "dim_product": dim_product,
        "dim_territory": dim_territory,
        "dim_hcp": dim_hcp_public,
        "fact_rx": fact_rx,
        "fact_activity": fact_activity,
        "fact_sales": fact_sales,
        "ground_truth_responsiveness": ground_truth,
    }
    for name, df in tables.items():
        df.to_csv(raw / f"{name}.csv", index=False)
        log.info("wrote %-28s %8s rows", name + ".csv", f"{len(df):,}")
    return tables


if __name__ == "__main__":
    main()
