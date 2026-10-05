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

## ⚡ Key Features & Capabilities

- **PostgreSQL Star Schema:** Engineered dimensional models enforcing referential integrity, surrogate keys, and B-tree indexes.
- **Analytical SQL Library:** 10+ production SQL queries utilizing Common Table Expressions (CTEs), window functions, and stored procedures.
- **Robust Python ETL & Validation Gate:** 21 automated data-quality checks testing schema conformism and non-negative constraints.
- **Causal SFE Modeling:** Fixed-effects regression isolating within-HCP treatment effects from time-invariant confounders.
- **Executive Power BI Dashboards:** Production data export pipeline generating star-schema flat extracts with a complete DAX measure library.

---

## 🏗 Data Architecture & Star Schema

The analytical warehouse implements a conformed dimensional star schema modeled in PostgreSQL with `FactRx`, `FactSales`, `FactActivity`, `FactCallPlan`, and corresponding dimension tables for HCP, Territory, Product, and Date.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+ (Python 3.12 recommended)
- Optional: PostgreSQL 14+ 
- Optional: Power BI Desktop

### Installation

```bash
# Clone the repository
git clone https://github.com/RohithPariki/pharma-commercial-analytics.git
cd pharma-commercial-analytics

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Execution

Execute the full pipeline from data generation to reporting with a single command:

```bash
# Run complete end-to-end pipeline
make all
```

---

## 👤 Ownership & License

- **Full Ownership & Author:** [Rohith Pariki](https://github.com/RohithPariki)
- **LinkedIn:** [Rohith Pariki](https://www.linkedin.com/in/rohithpariki)
- **Email:** [rohithpariki@gmail.com](mailto:rohithpariki@gmail.com)
- **License:** Distributed under the MIT License. See [LICENSE](LICENSE) for details.
