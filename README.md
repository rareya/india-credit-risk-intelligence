# India Credit Risk Intelligence

**End-to-end credit risk analytics and decision-support platform for Indian NBFC lending**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange?style=flat-square)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red?style=flat-square)](https://streamlit.io)
[![SQLite](https://img.shields.io/badge/SQLite-3-lightgrey?style=flat-square)](https://sqlite.org)
[![DuckDB](https://img.shields.io/badge/DuckDB-Analytical-yellow?style=flat-square)](https://duckdb.org)
[![CI](https://img.shields.io/github/actions/workflow/status/rareya/india-credit-risk-intelligence/ci.yml?label=CI&style=flat-square)](https://github.com/rareya/india-credit-risk-intelligence/actions)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

> A business-facing analytics platform for credit risk managers to monitor portfolio health, identify risky borrower segments, explain model decisions, and simulate underwriting policy changes before rolling them out.
>
> **This is not an ML notebook. It is an end-to-end credit-risk decision-support system.**

---

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [The Key Finding](#the-key-finding)
- [Key Metrics](#key-metrics)
- [Datasets Used](#datasets-used)
  - [Target Variable](#target-variable)
  - [Key Tables and Artifacts Produced](#key-tables-and-artifacts-produced)
- [Data Pipeline](#data-pipeline)
  - [Pipeline Phases](#pipeline-phases)
- [Dashboard — 7 Panels](#dashboard--7-panels)
- [Analytical Methods](#analytical-methods)
- [Feature Engineering](#feature-engineering)
- [SQL Query Library](#sql-query-library)
- [Three Credit Policy Recommendations](#three-credit-policy-recommendations)
- [Model Governance and Reproducibility](#model-governance-and-reproducibility)
- [Project Structure](#project-structure)
- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Step 1 — Clone and Install](#step-1--clone-and-install)
  - [Step 2 — Download the Dataset](#step-2--download-the-dataset)
  - [Step 3 — Run the Pipeline](#step-3--run-the-pipeline)
  - [Step 4 — Launch the Dashboard](#step-4--launch-the-dashboard)
- [Tech Stack](#tech-stack)
- [Why the India Context Matters](#why-the-india-context-matters)
- [Documentation](#documentation)
- [Limitations](#limitations)

---

## What This Project Does

Credit-risk projects often stop after producing a model score. This project connects the **data pipeline, machine learning model, SQL analytical layer, explainability layer, and business decision layer** into one reproducible workflow.

The platform processes **51,336 borrower records** from two independent, publicly available Kaggle sources joined on `PROSPECTID`. The data moves through a **Bronze → Silver → Gold** architecture, where the pipeline performs ingestion, data-quality validation, target construction, feature engineering, and analytical modelling.

The project then:

- investigates and removes target leakage;
- develops and evaluates XGBoost models;
- uses cross-validation and held-out evaluation for model assessment;
- freezes a governed **Conservative XGBoost** model configuration;
- scores the full portfolio and persists the predictions;
- exposes repeatable business questions through SQL;
- provides SHAP-based explainability;
- simulates underwriting-policy changes;
- delivers the final results through a 7-panel Streamlit dashboard.

The final model artifact is **`conservative_xgb_v1`**, with a frozen 30-feature configuration.

### The central business question

> **How can a lender use borrower behaviour to identify high-risk applicants earlier, make decisions more consistently, explain those decisions, and quantify the impact of changing underwriting policy?**

The goal is therefore not only to build a classifier, but to turn the model into a usable **credit-risk intelligence workflow**.

> **Target-definition note:** `default_risk` is a P1/P2 vs P3/P4 **high-risk classification proxy**, not an observed loan-default event label.

---

## The Key Finding

### 1. A major target-leakage problem was discovered

During model development, `Credit_Score` produced an **AUC of 0.9998 by itself** when predicting the project target.

That result was far too strong to accept at face value.

The investigation indicated that `Credit_Score` was generated upstream in the same CIBIL-style scoring ecosystem as `Approved_Flag`, the source risk classification. This created a circular relationship between the feature and the modelling target.

The project therefore treated this as a **feature-provenance and model-governance problem**, not as a modelling success.

```text
Credit_Score only
        │
        ▼
   AUC = 0.9998
        │
        ▼
LEAKAGE INVESTIGATION
        │
        ▼
Credit_Score excluded
        │
        ▼
Behavioural feature set
        │
        ▼
Model selection + evaluation
        │
        ▼
Final governed model
```

### 2. Why the leakage mattered

If the feature had been retained:

1. the model would appear artificially close to perfect;
2. downstream performance claims would be misleading;
3. the model would largely reproduce an upstream score rather than discover independent behavioural signal;
4. the dashboard and policy simulator would give a false sense of model quality.

The project therefore excluded `Credit_Score` from the governed feature set and moved to behavioural features such as delinquency, recent credit enquiries, credit-history length, loan exposure, income, and loan-type behaviour.

### 3. The strongest genuine behavioural signal

Once the leaking feature was removed, **recent credit enquiries (`enq_L6m`)** emerged as the strongest model driver in the model analysis.

The documented portfolio analysis found that borrowers with **4+ enquiries in the previous six months** had approximately **4× the baseline high-risk/default-proxy rate**.

This makes enquiry behaviour an important early-stress signal because it captures recent credit-seeking activity rather than only established repayment history.

### 4. What the leakage finding unlocked

The leakage investigation changed the modelling question from:

> “How accurately can the model reproduce the bureau score?”

to:

> “Which borrower behaviours contain genuine signal about high-risk borrowers?”

That change in framing is one of the main analytical outcomes of the project.

See [`docs/LEAKAGE_ANALYSIS.md`](docs/LEAKAGE_ANALYSIS.md) for the detailed investigation.

---

## Key Metrics

| Metric | Value |
|---|---:|
| Portfolio size | **51,336 borrowers** |
| Gold fact table | **51,336 rows × 40 columns** |
| Silver engineered dataset | **60+ features** |
| High-risk classification rate | **26.0%** |
| Held-out training split | **41,068 borrowers (80%)** |
| Held-out test split | **10,268 borrowers (20%)** |
| Leakage feature AUC (`Credit_Score` only) | **0.9998** |
| Model A AUC | **0.8985** |
| Model B benchmark AUC | **0.8994** |
| Model B Precision | **69.1%** |
| Model B Recall | **71.0%** |
| Model B F1 | **0.700** |
| Model A false-alarm rate | **41.0%** |
| Model B false-alarm rate | **31.0%** |
| False-alarm improvement | **41% → 31%** |
| 5-fold stratified CV AUC | **0.8921 ± 0.0038** |
| Top SHAP feature | **`enq_L6m`** |
| Top SHAP importance | **1.183** |
| Risk rate for 4+ enquiries | **~4× baseline** |
| Estimated combined policy impact | **18–24%** |

### Held-out confusion matrix from the documented Model B benchmark

```text
                    Predicted Safe    Predicted Risky
Actual Safe              6,074              848
Actual Risky               773            2,573
```

### Top SHAP drivers

| Rank | Feature | Mean |SHAP| | Direction |
|---:|---|---:|---|
| 1 | `enq_L6m` | **1.183** | Higher = riskier |
| 2 | `num_times_delinquent` | **0.654** | Higher = riskier |
| 3 | `Age_Oldest_TL` | **0.460** | Higher = less risky |
| 4 | `Total_TL` | **0.242** | Higher = less risky |
| 5 | `delinquency_score` | **0.210** | Higher = riskier |

> **Evaluation note:** the 0.8994 / 69.1% / 71.0% / 0.700 benchmark is from the documented held-out Model B evaluation. The final `conservative_xgb_v1` artifact is the frozen model that is subsequently fit on all 51,336 Gold rows for downstream scoring. Its full-data predictions must not be reported as unbiased generalization performance.

---

## Datasets Used

Two independent, publicly available Kaggle data sources are joined on `PROSPECTID`.

| Source | What It Contains | Time Period | Why We Use It |
|---|---|---|---|
| **Internal Bank Dataset** (`case_study1.xlsx`) | Tradeline features: total loans, active loans, missed payments, loan-type breakdown (gold, home, personal, auto), enquiry counts, credit history age | Static snapshot | Primary source for granular loan-behaviour signals, including `enq_L6m`. |
| **External CIBIL Dataset** (`case_study2.xlsx`) | Bureau features + target variable: `Credit_Score`, `Approved_Flag` (P1–P4), age, gender, education, income, delinquency counts, DPD history | Static snapshot | Provides the source risk classification and bureau attributes. |

The pipeline also retains a separate loan-application source as part of the Bronze/Silver workflow.

### Dataset join

```text
Internal Bank Dataset
        │
        │ PROSPECTID
        │
        ├──────────────────┐
        │                  │
        ▼                  ▼
External CIBIL      Loan Applications
Dataset              (separate pipeline input)
        │
        ▼
   Silver master
        │
        ▼
   Gold fact table
```

---

## Target Variable

The source target is `Approved_Flag`, an ordinal risk classification:

| Flag | Risk Level | Approx. CIBIL Score | Avg. Delinquencies | `default_risk` |
|---|---|---:|---:|---:|
| P1 | Lowest — safest | ~780 | ~0.1 | 0 |
| P2 | Low risk | ~720 | ~0.4 | 0 |
| P3 | High risk | ~640 | ~1.8 | 1 |
| P4 | Highest risk | ~580 | ~4.2 | 1 |

The project maps the grades as:

```text
P1 / P2  →  0  →  lower-risk class
P3 / P4  →  1  →  higher-risk class
```

The resulting `default_risk` variable is used throughout the modelling pipeline, but it is explicitly documented as a **high-risk classification proxy**, not an observed default event.

---

## Key Tables and Artifacts Produced

| Artifact | Records / Shape | Purpose |
|---|---:|---|
| `bronze_internal_bank.parquet` | 51,336 | Raw Internal Bank source |
| `bronze_cibil_external.parquet` | 51,336 | Raw CIBIL source |
| `bronze_loan_applications.parquet` | Separate source | Loan-application data retained in pipeline |
| `silver_master.parquet` | 51,336 × 60+ | Cleaned, joined, engineered dataset |
| `fact_credit_risk.parquet` | 51,336 × 40 | Gold fact table used by ML/analytics |
| `credit_risk.duckdb` | Gold analytical DB | Analytical / OLAP layer |
| `credit_risk.db` | SQLite | Portable dashboard SQL layer |
| `predictions.parquet` | 51,336 | Full-portfolio final-model predictions |
| `prediction_metadata.json` | Metadata | Model/provenance/scoring information |
| `conservative_xgb_v1.joblib` | Final artifact | Frozen final XGBoost model |
| `conservative_xgb_v1_metadata.json` | Metadata | Final model configuration/provenance |
| `conservative_xgb_v1_manifest.json` | Manifest | Artifact identity + checksum |
| Power BI export | 51,336 | Business-ready reporting dataset |

---

## Data Pipeline

```text
                 RAW INPUTS
                     │
                     ▼
        ┌────────────────────────┐
        │        BRONZE          │
        │ raw Parquet + metadata │
        └───────────┬────────────┘
                    │
                    ▼
        ┌────────────────────────┐
        │         SILVER         │
        │ clean + validate +     │
        │ engineer features      │
        └───────────┬────────────┘
                    │
                    ▼
        ┌────────────────────────┐
        │          GOLD          │
        │ fact_credit_risk       │
        │ DuckDB / Parquet       │
        └───────────┬────────────┘
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
     ANALYTICS             MODELING
          │                   │
          │             leakage detection
          │             model comparison
          │             CV / held-out eval
          │             SHAP / thresholding
          │                   │
          └─────────┬─────────┘
                    ▼
            FINAL MODEL BUILD
          conservative_xgb_v1
                    │
                    ▼
          PERSISTED PREDICTIONS
                    │
                    ▼
              SQL / DASHBOARD
```

### Pipeline Phases

| Phase | Component | What Happens | Output |
|---|---|---|---|
| **Bronze** | `ingest_kaggle.py` | Loads source files, tags provenance, validates source structure and joins | Bronze Parquet datasets |
| **Silver** | `transform_silver.py` | Cleans sentinels, preserves missingness, normalizes categoricals, encodes target, engineers features, validates data | `silver_master.parquet` |
| **Gold** | `build_gold.py` | Builds the governed analytical fact layer and DuckDB representation | `fact_credit_risk.parquet`, DuckDB |
| **Analytics** | `src/analytics/` | Segmentation, Gini, delinquency, early-warning, loan analysis, and ML diagnostics | Analytical artifacts |
| **Model development** | ML experiments | Leakage tests, model comparison, CV, held-out evaluation, SHAP, threshold analysis | Metrics / explainability artifacts |
| **Final model** | `train_final_model.py` | Validates the frozen configuration and fits the selected model on all Gold rows | `.joblib` + metadata + manifest |
| **Scoring** | Prediction pipeline | Scores the full portfolio and records provenance | `predictions.parquet` |
| **SQL** | `sql/queries/` | Encodes repeatable business questions | SQL query library |
| **Dashboard** | `build_dashboard.py` | Presents the analytical outputs and policy tools | Streamlit application |

### Final-model rule

`train_final_model.py` does **not** select or tune the model. It validates the Gold dataset, validates the frozen feature/model configuration, builds the exact selected Conservative XGBoost, fits it on all 51,336 Gold rows, saves the final model, and writes authoritative metadata and a checksum manifest.

---

## Dashboard — 7 Panels

Run:

```bash
streamlit run app.py
```

| Panel | Business Question | Key Output |
|---|---|---|
| **01 · Portfolio Overview** | How healthy is the portfolio? | Portfolio KPIs, risk distribution, expected-loss indicators |
| **02 · Risk Segmentation** | Which borrower groups drive risk? | Segment risk, lift and contribution |
| **03 · Model Performance** | How well does the model separate risk? | ROC, threshold analysis, confusion matrix, score distribution |
| **04 · SHAP Explainability** | Why was a borrower flagged? | Global SHAP drivers + borrower-level reason codes |
| **05 · Policy Simulator** | What happens if underwriting policy changes? | Approval, risk and expected-loss impact |
| **06 · Risk Monitoring** | Who requires attention? | Monitoring views, watchlists, delinquency and concentration signals |
| **07 · SQL Query Library** | Can the findings be reproduced in SQL? | Live SQL execution, results, charts and CSV export |

### Dashboard design principle

The dashboard is a **presentation layer**.

It does not retrain the model on page load or create a second competing scoring pipeline. It consumes persisted model outputs and the analytical SQL/data layer.

---

## Analytical Methods

| Method | What It Measures / Does | Why It Is Used |
|---|---|---|
| **XGBoost** | Nonlinear tabular classification | Strong fit for structured borrower data |
| **5-fold Stratified CV** | Model stability | Preserves class proportions across folds |
| **Held-out evaluation** | Generalization performance | Keeps performance claims separate from final fitting |
| **SHAP TreeExplainer** | Global and local feature attribution | Makes model behaviour inspectable |
| **Threshold sensitivity** | Precision/recall and operating-point trade-offs | Credit decisions are not automatically optimal at 0.50 |
| **`scale_pos_weight` analysis** | Class-imbalance trade-off | Studies precision vs recall |
| **KMeans (k=4)** | Borrower segmentation | Produces policy-relevant behavioural groups |
| **Gini coefficient** | Distribution / inequality analysis | Compares concentration across groups |
| **Expected Loss** | `PD × LGD × EAD` | Converts risk scores into monetary impact proxies |
| **Statistical segment analysis** | Group-level relationships | Checks whether model findings are visible in portfolio data |

See [`TECHNICAL_REFERENCE.md`](TECHNICAL_REFERENCE.md) for detailed formulas and methodological rationale.

---

## Feature Engineering

| Feature Group | Examples | Business Meaning |
|---|---|---|
| Delinquency behaviour | `num_times_delinquent`, `num_times_60p_dpd`, `delinquency_score`, `missed_payment_ratio` | Repayment stress and severity |
| Loan portfolio | `Total_TL`, `active_loan_ratio`, `loan_type_diversity` | Existing exposure and borrowing structure |
| Credit seeking | `enq_L6m`, `enq_L12m`, `tot_enq` | Recent credit-shopping / financial stress |
| Credit history | `Age_Oldest_TL`, `credit_history_months` | Thin-file vs established borrowers |
| Capacity | `AGE`, `NETMONTHLYINCOME` | Borrower lifecycle / income proxy |
| India-specific lending | `Gold_TL`, `Home_TL`, other loan-type counts | Loan-type behaviour relevant to the Indian portfolio |
| Derived interactions | `enq_per_credit_year`, `delinquency_rate`, `enq_acceleration`, `severe_delinquency_ratio` | Normalize and combine behavioural signals |

The final production configuration freezes **30 governed features**, split across numeric and categorical inputs.

---

## SQL Query Library

The SQL layer is designed so a credit-risk analyst can investigate the portfolio without opening the modelling notebook.

| Query | Business Question |
|---|---|
| `01_portfolio_health.sql` | How risky is the overall loan book? |
| `02_default_by_segment.sql` | Which borrower segments drive risk? |
| `03_enquiry_default_correlation.sql` | Does enquiry behaviour relate to risk? |
| `04_delinquency_funnel.sql` | Where do borrowers move through delinquency stages? |
| `05_credit_history_vs_default.sql` | Does credit-history length relate to risk? |
| `06_high_risk_identification.sql` | Which profiles should trigger review? |
| `07_income_default_analysis.sql` | How does income interact with risk? |
| `08_gold_loan_analysis.sql` | What is the risk profile of gold-loan borrowers? |
| `09_early_intervention_candidates.sql` | Who should be prioritised for intervention? |
| `10_policy_impact_simulation.sql` | What happens under alternative policy rules? |

The Streamlit SQL page reads these query files from the repository and executes them against the analytical database.

---

## Three Credit Policy Recommendations

| Policy | Rule | Observed / estimated impact |
|---|---|---|
| **Enquiry Auto-Flag** | Review/restrict borrowers with `enq_L6m ≥ 4` | Segment risk around **~45%**; estimated **8–11%** NPA reduction proxy |
| **Thin-File Pricing** | Review/pricing action when `Age_Oldest_TL < 24 months` | Segment risk around **~34%**; estimated **6–9%** NPA reduction proxy |
| **30-DPD Intervention** | Early collections contact at the first missed-payment signal | Historical escalation around **65%**; estimated **5–8%** NPA reduction proxy |
| **Combined** | Apply all three | Estimated **18–24%** combined reduction proxy; non-additive |

> These are **simulation / analytical estimates**, not production claims. A controlled pilot is required before deployment.

---

## Model Governance and Reproducibility

### Frozen model contract

`src/modeling/final_model_config.py` is the single source of truth for:

- model name and version;
- target definition;
- 30 frozen features;
- numeric/categorical feature types;
- preprocessing;
- XGBoost hyperparameters;
- excluded features; and
- configuration validation.

The selected model is:

```text
Conservative XGBoost
conservative_xgb_v1
```

### Final model artifact

The final model build produces:

```text
data/gold/exports/analytics/ml/final_model/
├── conservative_xgb_v1.joblib
├── conservative_xgb_v1_metadata.json
└── conservative_xgb_v1_manifest.json
```

The manifest includes the model artifact checksum for provenance.

### Persisted scoring

The full-portfolio model outputs are stored separately:

```text
src/data/model/predictions/
├── predictions.parquet
└── prediction_metadata.json
```

This separation prevents the Streamlit dashboard from becoming a second modelling pipeline.

### CI

GitHub Actions runs on pushes and pull requests and checks dependency installation, Python compilation, SQL-file integrity, and the test suite when present.

### Streamlit configuration

`.streamlit/config.toml` provides dashboard runtime configuration.

---

## Project Structure

```text
india-credit-risk-intelligence/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── .streamlit/
│   └── config.toml
│
├── assets/
│   └── screenshots/
│
├── data/
│   ├── bronze/
│   ├── silver/
│   ├── gold/
│   │   └── exports/
│   │       └── analytics/ml/final_model/
│   └── model/
│       └── predictions/
│
├── docs/
│   ├── CHANGELOG.md
│   ├── MODEL_CARD.md
│   ├── LEAKAGE_ANALYSIS.md
│   └── ...
│
├── sql/
│   └── queries/
│
├── src/
│   ├── ingestion/
│   ├── transformation/
│   ├── modeling/
│   ├── analytics/
│   ├── data/
│   └── visualization/
│
├── app.py
├── requirements.txt
├── PROBLEM_STATEMENT.md
└── TECHNICAL_REFERENCE.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- Git
- The primary Kaggle source files

No API keys or cloud credentials are required for the core local workflow.

### Step 1 — Clone and Install

```bash
git clone https://github.com/rareya/india-credit-risk-intelligence.git
cd india-credit-risk-intelligence

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Step 2 — Download the Dataset

Place the primary files in the configured Bronze input directory:

```text
data/bronze/kaggle_cibil/
├── case_study1.xlsx
├── case_study2.xlsx
├── Features_Target_Description.xlsx
├── train_modified.csv
├── test_modified.csv
└── Unseen_Dataset.xlsx
```

### Step 3 — Run the Pipeline

```bash
# Bronze
python src/ingestion/ingest_kaggle.py

# Silver
python src/transformation/transform_silver.py

# Gold
python src/modeling/build_gold.py

# Analytics
python src/analytics/run_analytics.py

# Final frozen model
python src/modeling/train_final_model.py
```

Then refresh the scoring / SQL layer as configured by the repository.

### Step 4 — Launch the Dashboard

```bash
streamlit run app.py
```

### Local CI checks

```bash
python -m compileall -q src app.py
```

If a `tests/` directory is present:

```bash
pytest -q
```

### Important evaluation rule

Do **not** evaluate the final full-data model on the same rows it was trained on and report those numbers as generalization performance. Use the held-out evaluation and cross-validation artifacts for model-quality claims.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.10+ | End-to-end implementation |
| Data processing | pandas, NumPy, PyArrow | Cleaning, feature engineering, Parquet I/O |
| ML | XGBoost, scikit-learn | Classification / risk scoring |
| Explainability | SHAP | Global and local feature attribution |
| Analytical DB | DuckDB | Gold analytical layer |
| SQL layer | SQL + SQLite / DuckDB | Repeatable business questions |
| Dashboard | Streamlit + Plotly | Interactive decision-support UI |
| Reporting | Power BI export | Downstream reporting |
| CI | GitHub Actions | Automated repository checks |
| Configuration | `.streamlit/config.toml` | Dashboard runtime settings |

---

## Why the India Context Matters

### Gold loans

`Gold_TL` and related loan-type variables capture borrowing patterns that are especially relevant in the Indian lending context.

### Recent enquiries

`enq_L6m` captures recent credit-seeking behaviour and emerged as the strongest SHAP driver in the documented model analysis.

### Thin-file borrowers

Short credit-history borrowers are an important monitoring segment. Behavioural signals become particularly important when traditional credit history is limited.

### Behaviour-first modelling

The leakage finding is central to the India context: the project deliberately avoids depending on a potentially circular bureau score and instead investigates borrower behaviour that can be independently explained and validated.

---

## Documentation

| Document | Purpose |
|---|---|
| [`PROBLEM_STATEMENT.md`](PROBLEM_STATEMENT.md) | Business context, project scope and objectives |
| [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) | Model architecture, evaluation, intended use, limitations and biases |
| [`docs/LEAKAGE_ANALYSIS.md`](docs/LEAKAGE_ANALYSIS.md) | Detailed leakage investigation and remediation |
| [`docs/CHANGELOG.md`](docs/CHANGELOG.md) | Development history |
| [`TECHNICAL_REFERENCE.md`](TECHNICAL_REFERENCE.md) | Formulas, methods and technical design decisions |

---

## Limitations

This is a portfolio analytics and decision-support project, not a production credit-origination system.

Important limitations:

1. **Target definition:** `default_risk` is a P1/P2 vs P3/P4 high-risk classification proxy, not an observed loan-default event.
2. **Static data:** the source portfolio is a historical/static snapshot rather than a continuously refreshed lending population.
3. **Synthetic/anonymised context:** the data is CIBIL-style and should be validated against live production populations before deployment.
4. **Generalization:** the final full-data model artifact is not itself a held-out evaluation artifact.
5. **Explainability:** SHAP identifies model contribution; it does not establish causal effects.
6. **Expected loss:** LGD and EAD are simplified analytical assumptions.
7. **Policy simulation:** NPA/loss-reduction figures are scenario estimates and require controlled validation.

---

## License

MIT License.

---

**Built by Aarya Patankar — 2026.**

*India Credit Risk Intelligence: from data quality and leakage detection to governed ML, SQL validation, and business-facing credit decisions.*
