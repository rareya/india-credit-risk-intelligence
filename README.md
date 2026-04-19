# India Credit Risk Intelligence

**End-to-end credit risk analytics and decision-support platform for Indian NBFC lending**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange?style=flat-square)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red?style=flat-square)](https://streamlit.io)
[![SQLite](https://img.shields.io/badge/SQLite-3-lightgrey?style=flat-square)](https://sqlite.org)
[![DuckDB](https://img.shields.io/badge/DuckDB-0.9+-yellow?style=flat-square)](https://duckdb.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

> A business-facing analytics platform for credit risk managers to monitor portfolio health, identify risky borrower segments, explain individual model decisions, and simulate underwriting policy changes before rolling them out.
>
> **This is not an ML notebook. It is a decision-support system.**

---

## What This Project Does

This project ingests raw CIBIL-style loan data, processes it through a Medallion Architecture pipeline (Bronze → Silver → Gold), trains an XGBoost probability-of-default model, and delivers an interactive 7-panel Streamlit dashboard plus a 10-query SQL analytics library.

The pipeline processes **51,336 borrowers** across **60+ engineered features**, covering India's retail lending segment. The trained model achieves **AUC 0.8994** using only behavioural signals — no bureau score required.

---

## The Key Finding

> `Credit_Score` achieved **AUC = 0.9998** as a single predictor — statistically impossible. It was derived from the target variable (`Approved_Flag`) in the same upstream CIBIL batch process, making it a circular feature. It was excluded entirely.
>
> A behavioural model on **15 raw signals** achieves **AUC = 0.8994** with no credit score required.
>
> The single strongest predictor: **recent credit enquiries (`enq_L6m`)**. Borrowers applying to 4+ lenders in 6 months default at **4× the baseline rate** — a real-time stress signal that bureau scores miss by weeks or months.

See [`docs/LEAKAGE_ANALYSIS.md`](docs/LEAKAGE_ANALYSIS.md) for the full analysis, detection method, and implications.

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Portfolio size | 51,336 borrowers |
| Default rate | **26.0%** (vs 22% national NBFC average) |
| Model AUC | **0.8994** — XGBoost, 15 behavioural features |
| Precision (Model B) | **69.1%** — false alarm rate 31% |
| Recall (Model B) | **71.0%** — catches 7 in 10 risky borrowers |
| F1 Score | **0.700** |
| CV AUC | **0.8921 ± 0.0038** — 5-fold stratified, stable |
| False alarm rate improved | **41% → 31%** via `scale_pos_weight` tuning |
| Estimated NPA reduction | **18–24%** across 3 data-driven policies |

---

![Dashboard Overview](assets/screenshots/dashboard/DB_Page1.png)

![Dashboard Overview](assets/screenshots/dashboard/DB_Page2.png)

![Dashboard Overview](assets/screenshots/dashboard/DB_Page3.png)

## The Dataset

Two independent, publicly available Kaggle data sources are joined on `PROSPECTID`.

| Source | What It Contains | Time Period | Why We Use It |
|--------|-----------------|-------------|---------------|
| **Internal Bank Dataset** (`case_study1.xlsx`) | Tradeline features: total loans, active loans, missed payments, loan type breakdown (gold, home, personal, auto), enquiry counts, credit history age | Static snapshot | Only source for granular loan-type behavioural signals. Contains the `enq_L6m` feature — the strongest predictor in the model. |
| **External CIBIL Dataset** (`case_study2.xlsx`) | Bureau features + target variable: `Credit_Score`, `Approved_Flag` (P1–P4), age, gender, education, income, delinquency counts, DPD history | Static snapshot | Ground truth for the target variable. `Approved_Flag` encodes risk grade: P1 = safest, P4 = riskiest. |

### Target Variable

`Approved_Flag` is encoded as binary `default_risk`:

| Flag | Risk Level | CIBIL Score (avg) | Delinquencies (avg) | `default_risk` |
|------|-----------|-------------------|---------------------|----------------|
| P1 | Lowest — safest borrowers | ~780 | ~0.1 | 0 (Safe) |
| P2 | Low risk | ~720 | ~0.4 | 0 (Safe) |
| P3 | High risk | ~640 | ~1.8 | 1 (Risky) |
| P4 | Highest — riskiest borrowers | ~580 | ~4.2 | 1 (Risky) |

**Encoding verified:** P3/P4 borrowers have consistently lower CIBIL scores and higher delinquency counts than P1/P2 — confirmed in `transform_silver.py` sanity check and `investigate_data.py` output.

### Key Tables Produced

| Table | Records | Description |
|-------|---------|-------------|
| `bronze_internal_bank.parquet` | 51,336 | Raw tradeline features, metadata-tagged |
| `bronze_cibil_external.parquet` | 51,336 | Raw bureau features + target, metadata-tagged |
| `silver_master.parquet` | 51,336 | Joined, cleaned, 60+ features engineered |
| `fact_credit_risk` (DuckDB) | 51,336 | Star schema fact table — all measures |
| `credit_risk.db` (SQLite) | 51,336 rows · 3 tables | Queryable by any SQL tool |
| `credit_risk_powerbi_input.csv` | 51,336 | Scored, business-ready for Power BI |
| `credit_risk_model_v2.pkl` | — | Trained XGBoost Model B |

---

## Data Pipeline

```
Raw Kaggle files (xlsx / csv)
         │
         ▼
 ┌───────────────┐
 │    BRONZE     │  ingest_kaggle.py
 │  (Parquet)    │  Raw data preserved as-is. Source metadata tagged.
 │               │  Join integrity validated on PROSPECTID.
 └──────┬────────┘
        │
        ▼
 ┌───────────────┐
 │    SILVER     │  transform_silver.py
 │  (Parquet)    │  Sentinel value handling (-99999).
 │               │  Target encoding. 12 derived features.
 │               │  Data quality validation (5 checks).
 └──────┬────────┘
        │
        ├──────────────────────────────┐
        ▼                              ▼
 ┌───────────────┐            ┌──────────────────┐
 │     GOLD      │            │   ML PIPELINE    │
 │  (DuckDB)     │            │                  │
 │  build_gold   │            │  run_ml_model.py │  XGBoost + SHAP
 │  Star schema  │            │  improve_        │  Class imbalance
 │  4 dim tables │            │  precision.py    │  handling
 │  5 views      │            │                  │  Model B tuning
 └──────┬────────┘            └────────┬─────────┘
        │                              │
        ▼                              │
 ┌───────────────┐                     │
 │    SQLite     │◄────────────────────┘
 │  credit_risk  │  create_database.py
 │  .db          │  3 tables · 51,336 rows
 └──────┬────────┘
        │
        ▼
 ┌──────────────────────────────────────┐
 │   STREAMLIT DASHBOARD (7 panels)    │
 │   + Power BI CSV export             │
 └──────────────────────────────────────┘
```

### Pipeline Phases

| Phase | Script | What Happens | Output |
|-------|--------|-------------|--------|
| Bronze | `ingest_kaggle.py` | Reads 8 raw files (xlsx/csv), tags with source metadata, validates PROSPECTID join integrity | 4 Parquet files in `data/bronze/parquet/` |
| Silver | `transform_silver.py` | Handles -99999 sentinels, encodes `Approved_Flag` → `default_risk`, engineers 12 derived features, validates 5 data quality checks | `silver_master.parquet` — 51,336 × 60+ cols |
| Gold | `build_gold.py` | Builds DuckDB star schema: `dim_borrower`, `dim_credit`, `dim_loan_portfolio`, `fact_credit_risk` + 5 analytical views | `credit_risk.duckdb` + Parquet exports |
| Analytics | `run_analytics.py` | 6 modules: CIBIL threshold analysis, Gini coefficient, KMeans segmentation (k=4), early warning signals, delinquency deep dive, gold loan analysis | 7 analytical Parquet files |
| ML Model | `run_ml_model.py` | XGBoost classifier, 5-fold CV, SHAP TreeExplainer, leakage detection, model comparison | `credit_risk_model.pkl`, SHAP parquets, metrics JSON |
| Model B | `improve_precision.py` | 3 precision strategies: threshold tuning, `scale_pos_weight` sweep, feature engineering (4 interaction terms) | `credit_risk_model_v2.pkl`, precision charts |
| Database | `create_database.py` | Exports silver parquet → SQLite with 3 tables | `credit_risk.db` |

---

## Dashboard — 7 Panels

```
streamlit run app.py
```

| Panel | Business Question | Key Visual |
|-------|-----------------|------------|
| **01 · Portfolio Overview** | How healthy is our loan book? | Risk band donut · Income vs default rate (dual axis) · EL stat bar |
| **02 · Risk Segmentation** | Which segments drive NPA? | Default rate bar with baseline · Segment leaderboard · Lift ratio |
| **03 · Model Performance** | Can we trust this model? | ROC curve · PD separation histogram · Threshold sensitivity table |
| **04 · SHAP Explainability** | Why was this borrower flagged? | Global SHAP bar · Individual scorer with reason codes |
| **05 · Policy Simulator** | What if we tighten the rules? | Before/after bar chart · RAAE ratio · EL reduction |
| **06 · Risk Monitoring** | Who needs attention this week? | Watchlist table · Delinquency funnel · Concentration alerts |
| **07 · SQL Query Library** | SQL proof of every ML finding | Live query runner · 10 queries with charts · Export CSV |

**Panel 05 (Policy Simulator) is the key differentiator.** Adjust enquiry caps, income floors, and PD thresholds — see the real-time impact on approval rate, defaults prevented, and expected loss in crores before any policy is rolled out.

**Panel 07 (SQL Query Library) bridges ML and business.** Every SHAP finding has a SQL equivalent that any analyst can run without Python. Query 03 proves the `enq_L6m` finding in 12 lines of SQL.

---

## Analytical Methods

| Method | Formula | What It Measures | Why This Method |
|--------|---------|-----------------|-----------------|
| XGBoost | Gradient boosted decision trees | Probability of default (0–1) | Handles class imbalance via `scale_pos_weight`; native SHAP compatibility; best-in-class on tabular financial data |
| SHAP | Shapley Additive Explanations | Per-feature contribution per prediction | Theoretically grounded (game theory); globally consistent; satisfies RBI explainability requirement |
| scale_pos_weight tuning | Sweep 0.3×–2.0× natural ratio | Precision/recall tradeoff | Direct loss function adjustment; more principled than SMOTE (no synthetic data) |
| Threshold sensitivity | Sweep 0.25–0.70 in 0.05 steps | Approval rate vs recall tradeoff | Shows credit team the business consequence of each operating point |
| 5-fold Stratified CV | AUC on held-out fold | Model stability | Manual loop (NaN-proof); maintains class distribution in each fold |
| KMeans (k=4) | On: delinquency, active ratio, credit history, income, total loans | Borrower risk profiles | Silhouette tested k=3–5; k=4 gives best score and policy-relevant groupings |
| Gini Coefficient | `(2·Σ(i·v)) / (n·Σv) − (n+1)/n` | Inequality of creditworthiness | Scale-independent; standard metric; directly comparable across income tiers |
| Expected Loss | `PD × LGD × EAD` | Portfolio credit risk in ₹ Crore | Standard banking risk metric; LGD=45% (industry standard for unsecured India loans) |

See [`docs/TECHNICAL_REFERENCE.md`](docs/TECHNICAL_REFERENCE.md) for complete formula derivations, worked examples, and rationale for each method over alternatives.

---

## Feature Engineering

15 raw features + 4 engineered interaction terms:

| Feature Group | Features | Business Meaning |
|--------------|---------|-----------------|
| Delinquency Behaviour | `num_times_delinquent`, `num_times_60p_dpd`, `delinquency_score`, `missed_payment_ratio` | Payment history signals — strongest default predictors after enquiries |
| Loan Portfolio | `Total_TL`, `active_loan_ratio`, `loan_type_diversity`, `Age_Oldest_TL` | Credit experience and leverage signals |
| Credit Seeking | `enq_L6m`, `enq_L12m`, `tot_enq` | **Real-time stress signal.** enq_L6m = #1 SHAP feature (importance 1.183) |
| Demographics | `AGE`, `NETMONTHLYINCOME` | Capacity and lifecycle signals |
| India-Specific | `Gold_TL`, `Home_TL` | Gold loans = last-resort collateral borrowing in India |
| Engineered | `enq_per_credit_year`, `delinquency_rate`, `enq_acceleration`, `severe_delinquency_ratio` | Interaction terms: normalise for credit history length and loan count |

---

## SQL Query Library (10 Business Queries)

Run all 10 at once:
```bash
python src/analytics/run_queries.py
```

Or run live in the dashboard: **Panel 07 → select query → view result + chart**.

| Query | Business Question | SHAP Connection |
|-------|-----------------|-----------------|
| `01_portfolio_health.sql` | How risky is our overall loan book? | Portfolio baseline |
| `02_default_by_segment.sql` | Which segments drive NPA? | Risk segment analysis |
| `03_enquiry_default_correlation.sql` | Does enquiry behaviour predict default? | **Proves enq_L6m (SHAP #1) in SQL** |
| `04_delinquency_funnel.sql` | Where do borrowers fall off? | Collections intervention timing |
| `05_credit_history_vs_default.sql` | Does credit history protect against default? | Age_Oldest_TL (SHAP #3, protective) |
| `06_high_risk_identification.sql` | Which profiles trigger auto-review? | 3 policy rules as SQL flags |
| `07_income_default_analysis.sql` | Income vs behaviour: what predicts more? | NETMONTHLYINCOME (SHAP #12) vs enq_L6m |
| `08_gold_loan_analysis.sql` | Are gold loan borrowers higher risk? | Gold_TL (SHAP #10) — India-specific |
| `09_early_intervention_candidates.sql` | Who should collections call this week? | Urgency score = weighted SHAP features |
| `10_policy_impact_simulation.sql` | What NPA reduction do the 3 policies give? | SQL equivalent of Policy Simulator |

---

## Three Credit Policy Recommendations

Based on SHAP analysis of 51,336 borrowers, validated in SQL Query 06 and 10:

| Policy | Rule | Default Rate in Segment | Estimated NPA Reduction |
|--------|------|------------------------|------------------------|
| **Enquiry Auto-Flag** | Reject if enq_L6m ≥ 4 | ~45% | 8–11% |
| **Thin-File Pricing** | Premium if Age_Oldest_TL < 24 months | ~34% | 6–9% |
| **30-DPD Intervention** | Collections outreach at first missed payment | 65% escalation rate | 5–8% |
| **Combined (non-additive)** | All three applied together | — | **18–24%** |

*All estimates assume 70% true positive action rate. Validate with a 90-day controlled pilot before full rollout.*

---

## Project Structure

```
india-credit-risk-intelligence/
│
├── app.py                              ← Entrypoint: streamlit run app.py
├── requirements.txt
├── README.md
├── CHANGELOG.md
├── Dockerfile                          ← Docker deployment
├── docker-compose.yml
│
├── .streamlit/
│   └── config.toml                     ← Streamlit Cloud settings
│
├── src/
│   ├── ingestion/
│   │   └── ingest_kaggle.py            ← Bronze layer — raw file ingestion
│   ├── transformation/
│   │   └── transform_silver.py         ← Silver layer — cleaning + feature engineering
│   ├── modeling/
│   │   └── build_gold.py               ← Gold layer — DuckDB star schema
│   ├── analytics/
│   │   ├── run_ml_model.py             ← XGBoost + SHAP pipeline
│   │   ├── improve_precision.py        ← Model B — 3 precision strategies
│   │   ├── run_analytics.py            ← Gini, KMeans, EWS, gold loan modules
│   │   ├── portfolio_metrics.py        ← EL, risk bands, segment contribution
│   │   ├── policy_simulator.py         ← What-if underwriting engine
│   │   └── run_queries.py              ← Execute all 10 SQL queries
│   ├── data/
│   │   ├── create_database.py          ← Parquet → SQLite
│   │   └── export_powerbi_dataset.py   ← Scored CSV for Power BI
│   ├── utils/
│   │   └── config.py                   ← Central configuration
│   └── visualization/
│       └── build_dashboard.py          ← Streamlit dashboard (7 panels)
│
├── sql/queries/                        ← 10 business SQL queries
│   ├── 01_portfolio_health.sql
│   ├── 02_default_by_segment.sql
│   ├── 03_enquiry_default_correlation.sql
│   ├── 04_delinquency_funnel.sql
│   ├── 05_credit_history_vs_default.sql
│   ├── 06_high_risk_identification.sql
│   ├── 07_income_default_analysis.sql
│   ├── 08_gold_loan_analysis.sql
│   ├── 09_early_intervention_candidates.sql
│   └── 10_policy_impact_simulation.sql
│
├── docs/
│   ├── PROBLEM_STATEMENT.md            ← The three business gaps this solves
│   ├── BUSINESS_CASE.md                ← Who uses this, what decisions it supports
│   ├── METRICS_DEFINITION.md           ← PD, EL, AUC, DPD, NPA glossary
│   ├── MODEL_CARD.md                   ← Architecture, performance, limitations, biases
│   ├── LEAKAGE_ANALYSIS.md             ← Leakage detection and resolution
│   └── TECHNICAL_REFERENCE.md         ← Full formula and method documentation
│
├── powerbi/
│   └── credit_risk_dashboard.pbix      ← Power BI dashboard file
│
└── data/                               ← Not committed (gitignored)
    ├── bronze/kaggle_cibil/            ← Place raw Kaggle files here
    ├── silver/                         ← Generated by transform_silver.py
    ├── gold/                           ← Generated by build_gold.py
    ├── processed/                      ← Model .pkl files
    ├── powerbi/                        ← Scored CSV for Power BI
    └── credit_risk.db                  ← Generated by create_database.py
```

---

## Setup

### Prerequisites

| Requirement | Why |
|-------------|-----|
| Python 3.10+ | Required by pandas 2.0, DuckDB, XGBoost |
| pip | Package installer |
| ~500MB disk | Data files and model artefacts |

No API keys, credentials, or cloud accounts required. All data sources are from Kaggle.

### Step 1 — Clone and install

```bash
git clone https://github.com/rareya/india-credit-risk-intelligence.git
cd india-credit-risk-intelligence
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2 — Download the dataset

Download from Kaggle and place files in `data/bronze/kaggle_cibil/`:

```
data/bronze/kaggle_cibil/
├── case_study1.xlsx                  ← Internal Bank Dataset
├── case_study2.xlsx                  ← External CIBIL Dataset (contains target)
├── External_Cibil_Dataset.xlsx       ← Duplicate of case_study2
├── Internal_Bank_Dataset.xlsx        ← Duplicate of case_study1
├── Features_Target_Description.xlsx  ← Data dictionary
├── train_modified.csv                ← Loan application data
├── test_modified.csv
└── Unseen_Dataset.xlsx
```

### Step 3 — Run the pipeline

```bash
# Bronze — ingest raw files
python src/ingestion/ingest_kaggle.py

# Silver — clean, encode, engineer features
python src/transformation/transform_silver.py

# Gold — DuckDB star schema
python src/modeling/build_gold.py

# Analytics — Gini, KMeans, EWS, gold loan
python src/analytics/run_analytics.py

# ML — XGBoost + SHAP (produces credit_risk_model.pkl)
python src/analytics/run_ml_model.py

# Model B — precision improvement
python src/analytics/improve_precision.py

# SQLite database — for dashboard queries
python src/data/create_database.py

# Power BI export — scored CSV
python src/data/export_powerbi_dataset.py

# Launch dashboard
streamlit run app.py
```

### Step 4 — Run with Docker (optional)

```bash
# Run the full pipeline first (Steps above), then:
docker-compose up --build
# Dashboard at http://localhost:8501
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| ML Model | XGBoost 1.7+, scikit-learn | Gradient boosted classifier, class imbalance handling |
| Explainability | SHAP 0.42+ | TreeExplainer, global + individual feature attribution |
| Data Processing | pandas 2.0+, numpy, pyarrow | Feature engineering, parquet I/O |
| Analytical DB | DuckDB 0.9+ | Gold layer — star schema, columnar OLAP queries |
| Operational DB | SQLite | Dashboard SQL queries, portable, zero-infrastructure |
| Dashboard | Streamlit 1.25+, Plotly | 7-panel interactive platform |
| Deployment | Docker, Streamlit Cloud | Containerised + free hosted demo |
| Language | Python 3.10+ | End-to-end single language |

---

## Why India Context Matters

- **`Gold_TL` (Gold loans)** are a uniquely Indian signal — borrowers pledging jewellery are often at last-resort financing stage. No Western credit dataset has this feature.
- **`enq_L6m` (Recent enquiries)** captures real-time financial desperation. Bureau scores are updated monthly; a borrower applying to 5 lenders this week looks identical to a stable borrower on last month's score.
- **Thin-file borrowers** (~29% of this portfolio) have under 24 months of credit history. Behavioural signals are the only viable predictor for this segment.
- **RBI Fair Practices Code** legally requires plain-English rejection reasons for every credit decision. SHAP reason codes directly satisfy this regulatory requirement.
- **Seasonal patterns** — Diwali and agricultural harvest cycles drive predictable default fluctuations that purely data-driven models miss without domain context.

---

## Documentation

| Document | Content |
|----------|---------|
| [`docs/PROBLEM_STATEMENT.md`](docs/PROBLEM_STATEMENT.md) | The three business gaps — lagging scores, no EWS, no explainability |
| [`docs/BUSINESS_CASE.md`](docs/BUSINESS_CASE.md) | Who uses this platform and how, expected business impact |
| [`docs/METRICS_DEFINITION.md`](docs/METRICS_DEFINITION.md) | PD, EL, AUC, DPD, NPA, RAAE — plain-English glossary |
| [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) | Full model documentation: architecture, performance, limitations, biases |
| [`docs/LEAKAGE_ANALYSIS.md`](docs/LEAKAGE_ANALYSIS.md) | How Credit_Score AUC=0.9998 was detected and excluded |
| [`docs/TECHNICAL_REFERENCE.md`](docs/TECHNICAL_REFERENCE.md) | Every dataset, formula, method, and design decision |
| [`CHANGELOG.md`](CHANGELOG.md) | Version history: v1.0 ingestion → v1.1 model → v1.2 dashboard |

---

*Built by Aarya Patankar — Data Analysis Project, 2026.*
*Dataset: CIBIL-style Indian borrower records (anonymised/synthetic). MIT License.*