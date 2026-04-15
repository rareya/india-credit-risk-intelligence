# India Credit Risk Intelligence

**End-to-end credit risk analytics and decision-support platform for Indian lending portfolios**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange?style=flat-square)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red?style=flat-square)](https://streamlit.io)
[![SQLite](https://img.shields.io/badge/SQLite-3-lightgrey?style=flat-square)](https://sqlite.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

> A business-facing analytics platform for credit risk managers to monitor portfolio health, explain individual model decisions, and simulate underwriting policy changes before rolling them out.

---

## Results at a Glance

| Metric | Value |
|--------|-------|
| Portfolio size | 51,336 borrowers |
| Model AUC | **0.8994** (XGBoost, 15 behavioural features) |
| Precision (Model B) | **69.1%** — 31% false alarm rate |
| Recall (Model B) | **71.0%** — catches 7 in 10 risky borrowers |
| F1 Score | **0.700** |
| Estimated NPA reduction (3 policies) | **18–24%** |
| False alarm rate improvement | **41% → 31%** via class weight tuning |

---

## The Key Finding

> `Credit_Score` achieves **AUC = 0.9998** on a single feature — statistically impossible. It was derived from the target variable during preprocessing (data leakage). A behavioural model using **15 raw signals achieves AUC = 0.8994** with no credit score required.
>
> The single strongest predictor: **recent credit enquiries (`enq_L6m`)**. Borrowers applying to 4+ lenders in 6 months default at **4× the baseline rate** — a real-time stress signal invisible to any bureau score.

See [`docs/LEAKAGE_ANALYSIS.md`](docs/LEAKAGE_ANALYSIS.md) for full analysis.

---

## Dashboard — 6 Business Panels

```
streamlit run app.py
```

| Panel | Business Question |
|-------|------------------|
| **1 · Executive Overview** | How healthy is our current loan book? KPIs + Expected Loss + Risk Distribution |
| **2 · Borrower Segmentation** | Which segments are driving NPA? Interactive leaderboard + heatmap |
| **3 · Model Performance** | Can we trust this model? ROC + PD Distribution + **Threshold Sensitivity Table** |
| **4 · Explainable Decisions** | Why was this borrower flagged? SHAP waterfall + Plain-English Risk Card |
| **5 · Policy Simulator** | What if we tighten the enquiry rule? Real-time what-if analysis |
| **6 · Risk Monitoring** | Who needs attention this week? Watchlist + Concentration Alerts |

**Panel 5 (Policy Simulator) is the key differentiator** — allows credit teams to adjust underwriting rules and see the real-time impact on approval rate, defaults prevented, and expected loss before any policy is rolled out.

---

## Data Pipeline

```
Raw Kaggle files (xlsx/csv)
        │
        ▼
 ┌─────────────┐
 │   BRONZE    │  ingest_kaggle.py
 │  (Parquet)  │  Raw data preserved as-is. Metadata tagged.
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │   SILVER    │  transform_silver.py
 │  (Parquet)  │  Cleaning, feature engineering, target encoding.
 └──────┬──────┘  Join internal bank + CIBIL on PROSPECTID.
        │
        ▼
 ┌─────────────┐
 │    GOLD     │  build_gold.py
 │  (DuckDB)   │  Star schema: 4 tables, 5 analytical views.
 └──────┬──────┘
        │
        ├──────────────────────────┐
        ▼                          ▼
 ┌─────────────┐          ┌───────────────┐
 │  ML MODEL   │          │    SQLite DB   │
 │ (XGBoost)   │          │  (3 tables)    │
 │ run_ml_     │          │ create_        │
 │ model.py    │          │ database.py    │
 └──────┬──────┘          └───────┬────────┘
        │                          │
        └──────────┬───────────────┘
                   ▼
          ┌─────────────────┐
          │   STREAMLIT     │
          │   DASHBOARD     │
          │   (6 panels)    │
          └─────────────────┘
```

---

## Technical Decisions

**Why XGBoost over logistic regression:**
XGBoost handles class imbalance via `scale_pos_weight`, is natively compatible with SHAP for regulatory explainability, and consistently outperforms linear models on tabular financial data with interaction effects (e.g. the enq_L6m × Age_Oldest_TL interaction term).

**Why class weight tuning over SMOTE:**
SMOTE (synthetic oversampling) creates artificial minority class samples that may not represent real borrower behaviour. Tuning `scale_pos_weight` directly adjusts the loss function, which is both more principled and more interpretable to a credit risk team.

**Why SHAP over LIME:**
SHAP values are theoretically grounded (Shapley values from game theory) and globally consistent — the same feature has the same importance interpretation across all predictions. LIME is locally faithful but not globally coherent, which makes it unsuitable for regulatory audit trails.

**Why SQLite over a full analytical database:**
The 10 business SQL queries answer specific portfolio questions. SQLite provides a portable, zero-infrastructure database that any analyst can query with standard tools, without requiring a cloud database setup. For production, the same queries run unchanged on PostgreSQL.

---

## Project Structure

```
india-credit-risk-intelligence/
│
├── app.py                              ← Single entrypoint: streamlit run app.py
├── requirements.txt
├── README.md
├── CHANGELOG.md
│
├── src/
│   ├── analytics/
│   │   ├── run_ml_model.py             ← XGBoost + SHAP pipeline
│   │   ├── improve_precision.py        ← Model B tuning (3 strategies)
│   │   ├── run_analytics.py            ← 6 analytical modules (Gini, KMeans, EWS)
│   │   ├── portfolio_metrics.py        ← EL, risk bands, segment contribution
│   │   ├── policy_simulator.py         ← What-if policy engine
│   │   └── run_queries.py              ← Execute 10 SQL queries
│   ├── data/
│   │   ├── create_database.py          ← Parquet → SQLite
│   │   └── export_powerbi_dataset.py   ← Scored CSV for Power BI
│   ├── ingestion/
│   │   └── ingest_kaggle.py            ← Bronze layer
│   ├── modeling/
│   │   └── build_gold.py               ← DuckDB star schema
│   ├── transformation/
│   │   └── transform_silver.py         ← Silver layer
│   ├── utils/
│   │   └── config.py                   ← Central configuration
│   └── visualization/
│       └── build_dashboard.py          ← Streamlit dashboard (6 panels)
│
├── sql/queries/                        ← 10 business SQL queries
│   ├── 01_portfolio_health.sql
│   ├── 03_enquiry_default_correlation.sql
│   ├── 06_high_risk_identification.sql
│   ├── 09_early_intervention_candidates.sql
│   ├── 10_policy_impact_simulation.sql
│   └── ...
│
├── docs/
│   ├── PROBLEM_STATEMENT.md
│   ├── BUSINESS_CASE.md
│   ├── METRICS_DEFINITION.md
│   ├── MODEL_CARD.md                   ← Model documentation + limitations
│   └── LEAKAGE_ANALYSIS.md             ← Data leakage analysis and resolution
│
└── data/                               ← Not committed — see Setup below
    ├── credit_risk.db                  ← Generated by create_database.py
    └── bronze/kaggle_cibil/            ← Place Kaggle files here
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/rareya/india-credit-risk-intelligence.git
cd india-credit-risk-intelligence
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download the dataset

This project uses the [CIBIL Indian Credit Risk dataset from Kaggle](https://www.kaggle.com/datasets).

Place the raw files in `data/bronze/kaggle_cibil/`:
```
data/bronze/kaggle_cibil/
├── case_study1.xlsx
├── case_study2.xlsx
├── External_Cibil_Dataset.xlsx
├── Internal_Bank_Dataset.xlsx
├── Features_Target_Description.xlsx
├── train_modified.csv
├── test_modified.csv
└── Unseen_Dataset.xlsx
```

### 3. Run the pipeline

```bash
# Step 1: Ingest raw data → Bronze Parquet
python src/ingestion/ingest_kaggle.py

# Step 2: Clean + feature engineer → Silver Parquet
python src/transformation/transform_silver.py

# Step 3: Build star schema → Gold DuckDB
python src/modeling/build_gold.py

# Step 4: Run analytical modules
python src/analytics/run_analytics.py

# Step 5: Train XGBoost model + SHAP
python src/analytics/run_ml_model.py

# Step 6: Improve precision (Model B)
python src/analytics/improve_precision.py

# Step 7: Build SQLite database for dashboard
python src/data/create_database.py

# Step 8: Launch dashboard
streamlit run app.py
```

---

## SQL Query Library (10 Business Queries)

Run all queries at once:
```bash
python src/analytics/run_queries.py
```

| Query | Business Question |
|-------|------------------|
| `01_portfolio_health.sql` | Portfolio-level KPIs |
| `02_default_by_segment.sql` | Which segments drive NPA? |
| `03_enquiry_default_correlation.sql` | Does enquiry behaviour predict default? |
| `04_delinquency_funnel.sql` | Where do borrowers fall off? |
| `05_credit_history_vs_default.sql` | Does history length protect against default? |
| `06_high_risk_identification.sql` | Which profiles trigger auto-review? |
| `07_income_default_analysis.sql` | Income vs behaviour: what predicts more? |
| `08_gold_loan_analysis.sql` | Are gold loan borrowers higher risk? |
| `09_early_intervention_candidates.sql` | Who should collections call this week? |
| `10_policy_impact_simulation.sql` | What NPA reduction do the 3 policies achieve? |

---

## Three Credit Policy Recommendations

Based on SHAP analysis of 51,336 borrowers:

| Policy | Rule | Default Rate in Segment | Estimated NPA Reduction |
|--------|------|------------------------|------------------------|
| **Enquiry Auto-Flag** | Flag ≥4 enquiries in 6m | ~45% | 8–11% |
| **Thin-File Pricing** | Premium for <2yr history | ~34% | 6–9% |
| **30-DPD Intervention** | Outreach at first miss | 65% escalation rate | 5–8% |
| **Combined** | All three (non-additive) | — | **18–24%** |

*All estimates require validation via a 90-day controlled pilot before full policy rollout.*

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | XGBoost, scikit-learn |
| Explainability | SHAP |
| Data layer | pandas, numpy, pyarrow, DuckDB |
| Database | SQLite |
| Dashboard | Streamlit + Plotly |
| Language | Python 3.10+ |

---

## Documentation

| Document | Content |
|----------|---------|
| [`docs/PROBLEM_STATEMENT.md`](docs/PROBLEM_STATEMENT.md) | The three business gaps this platform solves |
| [`docs/BUSINESS_CASE.md`](docs/BUSINESS_CASE.md) | Who uses this, what decisions it supports |
| [`docs/METRICS_DEFINITION.md`](docs/METRICS_DEFINITION.md) | PD, EL, AUC, DPD, NPA — plain-English glossary |
| [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) | Model architecture, performance, limitations, biases |
| [`docs/LEAKAGE_ANALYSIS.md`](docs/LEAKAGE_ANALYSIS.md) | How data leakage was detected and resolved |
| [`CHANGELOG.md`](CHANGELOG.md) | Version history and technical decisions |

---

## Why India Context Matters

- **Gold loans (`Gold_TL`)** are a uniquely Indian stress signal — borrowers pledging jewellery often face financial difficulty
- **Enquiry acceleration** captures desperation that bureau scores miss entirely
- **Thin-file borrowers** (~29% of portfolio) have insufficient formal credit history — behavioural signals compensate
- **Seasonal default patterns** (agricultural cycles, Diwali credit peaks) affect Indian risk differently from Western markets
- **RBI Fair Practices Code** requires plain-English rejection reasons — SHAP reason codes address this directly

---

*Built by Aarya Patankar — Data Analysis Project, 2026.*
*Dataset: CIBIL-style Indian borrower records (anonymised/synthetic). MIT License.*