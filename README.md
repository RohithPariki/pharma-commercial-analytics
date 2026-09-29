# Pharma Commercial Analytics & Sales Force Effectiveness Platform

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Power BI](https://img.shields.io/badge/Power_BI-DAX-F2C811?logo=powerbi&logoColor=black)](https://powerbi.microsoft.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Pytest](https://img.shields.io/badge/Pytest-Passing-0A9EDC?logo=pytest&logoColor=white)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end commercial analytics and Sales Force Effectiveness (SFE) engineering platform for pharmaceutical portfolios. Combines **PostgreSQL star-schema dimensional modeling**, **automated Python ETL validation pipelines**, **econometric promotional response modeling**, and **interactive Power BI dashboards** to optimize field rep detailing, territory alignment, and prescriber prioritization.

> **Key Impact:** Identified misallocation across 55.1% of historical sales calls, delivering **$5.37M in whitespace revenue opportunity** through budget-neutral call reallocation and prescriber responsiveness targeting.

---

## 📑 Table of Contents

- [Executive Summary & Results](#-executive-summary--results)
- [Key Features & Capabilities](#-key-features--capabilities)
- [Data Architecture & Star Schema](#-data-architecture--star-schema)
- [Methodology & Analytical Pipeline](#-methodology--analytical-pipeline)
  - [1. Data Extraction, Ingestion & Quality Gates](#1-data-extraction-ingestion--quality-gates)
  - [2. Healthcare Professional (HCP) Segmentation](#2-healthcare-professional-hcp-segmentation)
  - [3. Econometric Response Curves & Fixed-Effects Attribution](#3-econometric-response-curves--fixed-effects-attribution)
  - [4. Prescriber Decline-Risk Model](#4-prescriber-decline-risk-model)
  - [5. SFE Territory & Rep Optimization](#5-sfe-territory--rep-optimization)
- [Power BI & DAX Analytics Layer](#-power-bi--dax-analytics-layer)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Testing & Validation](#-testing--validation)
- [Author & License](#-author--license)

---

## 📊 Executive Summary & Results

Pharmaceutical field teams frequently deploy sales detailing based purely on historical prescriber *volume* (decile tiers) rather than *responsiveness* (marginal lift per call). This platform diagnoses and corrects this misallocation across a portfolio spanning **50 territories**, **500+ HCPs**, and **17K+ call interactions**.

Every metric is generated dynamically via automated scripts and audited in [`reports/executive_summary.md`](reports/executive_summary.md).

### Scenario Comparison: Detailing Capacity Reallocation

| Strategy | Monthly Calls | Monthly Spend | Promo-Driven TRx | $\Delta$ TRx | Annual Whitespace Opportunity |
|---|:---:|:---:|:---:|:---:|:---:|
| **Baseline (Historical Plan)** | 3,012 | $557,205 | 2,127 | — | Baseline |
| **Scenario A: Budget-Neutral Reallocation** | 3,012 | $557,220 | 2,462 | **+15.7%** | **+$4,324,243** |
| **Scenario B: Profit-Maximizing Execution** | 2,861 | $529,285 | 2,395 | +12.6% | **+$5,372,841** (inclusive of cost savings) |

### Key Commercial Diagnostics

- **Negative Marginal ROI:** **55.1%** of historical detailing calls were delivered below break-even ROI ($306K/month in field spend with negative marginal return).
- **Steepest Response Mismatch:** The highest-responsiveness segment (`High-Response Growth Targets`) previously received only **3.4%** of detailing visits despite exhibiting an elasticity index of **0.217** (6× higher than the general population).

---

## ⚡ Key Features & Capabilities

- **PostgreSQL Star Schema:** Engineered dimensional models (`DimHCP`, `DimTerritory`, `DimProduct`, `DimDate`, `FactSales`, `FactRx`, `FactActivity`, `FactCallPlan`) enforcing referential integrity, surrogate keys, and B-tree indexes.
- **Analytical SQL Library:** 10+ production SQL queries utilizing Common Table Expressions (CTEs), window functions (`NTILE`, `LAG`, `AVG() OVER`), and stored procedures to profile prescriber trajectories and territory quota attainment.
- **Robust Python ETL & Validation Gate:** 21 automated data-quality checks testing schema conformism, moving-average consistency, unmapped foreign keys, and non-negative constraints with critical abort triggers.
- **Causal SFE Modeling:** Fixed-effects regression isolating within-HCP treatment effects from time-invariant confounders (specialty, clinic location, baseline brand affinity).
- **Executive Power BI Dashboards:** Production data export pipeline generating star-schema flat extracts with a complete DAX measure library for portfolio whitespace analysis.

---

## 🏗 Data Architecture & Star Schema

The analytical warehouse implements a conformed dimensional star schema modeled in PostgreSQL:

```
                          ┌───────────────┐
                          │    DimDate    │
                          └───────┬───────┘
                                  │
                                  ▼
┌──────────────┐          ┌───────────────┐          ┌───────────────┐
│    DimHCP    │◄─────────┤    FactRx     ├─────────►│  DimProduct   │
└──────┬───────┘          └───────────────┘          └───────────────┘
       │                          │                          │
       │                          ▼                          │
       │                  ┌───────────────┐                  │
       ├─────────────────►│   FactSales   │◄─────────────────┤
       │                  └───────────────┘                  │
       │                          ▲                          │
       ▼                          │                          │
┌───────────────┐         ┌───────┴───────┐                  │
│ DimTerritory  │◄────────┤ FactActivity  ├──────────────────┘
└───────────────┘         └───────────────┘
                                  ▲
                                  │
                          ┌───────┴───────┐
                          │ FactCallPlan  │
                          └───────────────┘
```

### Table Definitions

| Table | Grain | Key Metrics & Dimensions |
|---|---|---|
| `DimHCP` | 1 row per doctor | `hcp_id`, `specialty`, `territory_id`, `baseline_volume`, `responsiveness_tier` |
| `DimTerritory` | 1 row per territory | `territory_id`, `region`, `rep_id`, `quota_trx`, `call_capacity` |
| `DimProduct` | 1 row per drug | `product_id`, `brand_name`, `therapy_area`, `lifecycle_stage`, `price_per_trx` |
| `FactRx` | HCP $\times$ Drug $\times$ Month | `trx_count`, `nrx_count`, `market_share`, `refill_rate` |
| `FactSales` | Territory $\times$ Drug $\times$ Month | `gross_revenue`, `rebates`, `net_revenue`, `quota_attainment_pct` |
| `FactActivity` | Rep $\times$ HCP $\times$ Channel $\times$ Date | `call_id`, `channel` (F2F, Virtual, Email), `call_duration`, `detailing_cost` |
| `FactCallPlan` | Rep $\times$ HCP $\times$ Month | `planned_calls`, `actual_calls`, `variance`, `target_compliance_flag` |

---

## 🔬 Methodology & Analytical Pipeline

```
config/config.yaml (Single Source of Truth)
        │
        ▼
   generate_data.py  ──►  Raw Data Extracts
                                 │
                                 ▼
                             etl.py  ──►  Data Quality Gate (21 Validations)
                                 │
                                 ▼
                     Curated Monthly Data Mart
                                 │
          ┌──────────────────────┼──────────────────────┐
          ▼                      ▼                      ▼
     features.py               sfe.py              risk_model.py
  HCP Analytics Table     Response Modeling     Decline Forecasting
  (PCA + K-Means)        (Within-HCP FE OLS)    (Temporal Split GBM)
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 ▼
                            territory.py  (Scorecards & Rep Effect)
                                 │
                                 ▼
              visuals.py · bi_export.py · Power BI Star Schema
```

### 1. Data Extraction, Ingestion & Quality Gates
The ingestion pipeline enforces 21 data-integrity checks across raw extracts:
- Null check on surrogate & foreign keys
- Value range validations (prescriptions $\ge 0$, market share $\in [0, 1]$)
- Temporal consistency (call dates within active contract periods)
- Automatic quarantine for anomalous records with diagnostic logs

### 2. Healthcare Professional (HCP) Segmentation
- **Dimensionality Reduction:** Principal Component Analysis (PCA) retaining 90% variance to de-correlate behavioral features (TRx momentum, promotional response index, digital affinity, call frequency).
- **K-Means Clustering:** Optimal cluster selection via silhouette analysis and bootstrap stability verification ($ARI = 0.921$).
- **Stratified Archetypes:** Classifies doctors into 6 distinct behavioral tiers: *High-Value Loyalists*, *Untapped Potential*, *High-Response Growth Targets*, *Emerging Adopters*, *Steady Maintainers*, and *Low-Yield*.

### 3. Econometric Response Curves & Fixed-Effects Attribution
- **Confounder Elimination:** Within-HCP de-meaning eliminates time-invariant physician characteristics:
  $$\widetilde{\text{Share}}_{it} = \text{Share}_{it} - \overline{\text{Share}}_i + \overline{\text{Share}}_{\text{segment}}$$
- **Saturation Response Model:** Fits non-linear concave response curves with evidence support boundaries:
  $$\Delta \text{Share}(c) = \alpha \cdot \left(1 - e^{-\beta c}\right)$$
- **Constrained Optimization:** Solves the knapsack reallocation problem under rep call capacity constraints, assigning detailing budget strictly where marginal ROI is highest.

### 4. Prescriber Decline-Risk Model
- **Early-Warning Engine:** Supervised classification predicting $>20\%$ prescription volume drop in the upcoming quarter.
- **Validation:** 5-fold cross-validation achieving **0.800 ROC-AUC** with top-decile precision lift of **2.81×**.

### 5. SFE Territory & Rep Optimization
- **Rep Skill vs. Market Effect:** Decomposes territory performance into underlying market potential versus individual rep execution capability ($r = 0.57$ correlation with latent performance).

---

## 📈 Power BI & DAX Analytics Layer

The reporting layer couples Power BI star schemas with a structured measure library:

- **Key DAX Formulations:**
  ```dax
  // Marginal Detailing ROI
  Marginal_ROI = 
  DIVIDE(
      [Incremental_Gross_Margin],
      [Total_Detailing_Spend],
      BLANK()
  )

  // Opportunity Realization Index
  Whitespace_Capture_Pct = 
  DIVIDE(
      [Actual_TRx_Lift],
      [Theoretical_Max_Whitespace_TRx],
      0
  )
  ```
- **Interactive Dashboards:**
  - **Executive Portfolio View:** Revenue bridge, scenario NPV comparisons, and category market share trends.
  - **SFE Territory Scorecards:** Rep efficiency quartiles, quota attainment distribution, and call coverage gaps.
  - **Prescriber Targeting Matrix:** Micro-targeting grid identifying high-potential under-called HCPs.

---

## 📁 Repository Structure

```
pharma-commercial-analytics/
├── config/
│   └── config.yaml              # Global parameters, economic rates & thresholds
├── data/
│   ├── raw/                     # Raw simulator extracts
│   └── processed/               # Cleaned & conformed data tables
├── docs/
│   ├── methodology.md           # Mathematical & econometric formulation
│   └── interview_notes.md       # Technical defense & commercial Q&A
├── powerbi/
│   ├── dashboard_spec.md        # 4-page Power BI dashboard specification
│   ├── dax_measures.md          # DAX measure definitions & formulas
│   └── data/                    # Conformed CSVs ready for Power BI ingestion
├── reports/
│   ├── executive_summary.md     # Auto-generated briefing with metrics
│   ├── data_quality_report.csv  # 21-point data validation audit
│   └── figures/                 # High-resolution charts & diagnostic curves
├── sql/
│   ├── 01_schema.sql            # PostgreSQL DDL: Star schema, PK/FKs, indexes
│   ├── 02_transformations.sql   # Data mart views & curated tables
│   └── 03_analytics_queries.sql # Advanced analytical queries (CTEs, Windowing)
├── src/
│   ├── __init__.py
│   ├── generate_data.py         # Latent-variable pharma market simulator
│   ├── etl.py                   # Data ingestion & validation pipeline
│   ├── features.py              # Analytical base table engineering
│   ├── segmentation.py          # PCA + K-Means HCP segmentation
│   ├── sfe.py                   # Response curves, attribution & call optimizer
│   ├── territory.py             # Rep scorecards & territory normalization
│   ├── risk_model.py            # Prescriber decline classification
│   ├── bi_export.py             # Star-schema CSV exporter for Power BI
│   └── run_pipeline.py          # Master execution orchestrator
├── tests/
│   └── test_pipeline.py         # 40 automated unit & regression tests
├── Makefile                     # Build & execution automation
├── pyproject.toml               # Project metadata & tools configuration
├── requirements.txt             # Python dependencies
└── LICENSE                      # MIT License
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+ (Python 3.12 recommended)
- Optional: PostgreSQL 14+ (for loading SQL star schema)
- Optional: Power BI Desktop (for opening visual dashboards)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd pharma-commercial-analytics

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Execution

Execute the full pipeline from data generation to reporting with a single command:

```bash
# Run complete end-to-end pipeline
make all

# Or run individual pipeline stages manually:
python -m src.generate_data     # Generate synthetic commercial extracts
python -m src.etl               # Run validation gate & build mart
python -m src.features          # Compute analytical features
python -m src.segmentation      # Run K-Means segmentation
python -m src.sfe               # Fit response curves & optimize calls
python -m src.territory         # Generate territory scorecards
python -m src.risk_model        # Train prescriber decline model
python -m src.bi_export         # Export star-schema CSVs for Power BI
```

---

## 🧪 Testing & Validation

The test suite covers data constraints, econometrics, and optimization invariants:

```bash
# Run pytest test suite
pytest -v tests/test_pipeline.py
```

- **40 Automated Unit Tests:**
  - Strict preservation of total rep call capacity constraints
  - Zero-leakage temporal cross-validation in risk modeling
  - Monotonic diminishing marginal returns in response curve fits
  - Star schema referential integrity checks

---

## 👤 Author & License

- **Author:** [Rohith Pariki](https://github.com/RohithPariki)
- **LinkedIn:** [Rohith Pariki](https://www.linkedin.com/in/rohithpariki)
- **Email:** [rohithpariki@gmail.com](mailto:rohithpariki@gmail.com)
- **License:** Distributed under the MIT License. See [LICENSE](LICENSE) for details.
