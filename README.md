# India Credit Risk Intelligence

**End-to-end credit risk analytics, machine learning, SQL analytics, and decision-support platform for an Indian retail lending portfolio.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange?style=flat-square)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red?style=flat-square)](https://streamlit.io)
[![DuckDB](https://img.shields.io/badge/DuckDB-Analytical-yellow?style=flat-square)](https://duckdb.org)
[![SQLite](https://img.shields.io/badge/SQLite-Operational-lightgrey?style=flat-square)](https://sqlite.org)
[![CI](https://img.shields.io/github/actions/workflow/status/rareya/india-credit-risk-intelligence/ci.yml?label=CI&style=flat-square)](https://github.com/rareya/india-credit-risk-intelligence/actions)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

> A business-facing credit risk intelligence platform that takes borrower data from ingestion through governed ML scoring, persisted predictions, SQL analytics, explainability, and underwriting policy simulation.
>
> **This is not just an ML notebook. It is an end-to-end decision-support system.**

---

## What this project demonstrates

This project was built to answer a practical credit-risk question:

> **Can borrower behaviour be turned into a reliable probability-of-default signal and then connected to the decisions a lending team actually makes?**

The platform processes **51,336 borrowers** through a Bronze → Silver → Gold data pipeline, applies explicit data-quality and leakage controls, trains a governed **Conservative XGBoost** model, persists its portfolio predictions, exposes business questions through SQL, and delivers the results through a 7-panel Streamlit dashboard.

The final ML artifact is versioned as **`conservative_xgb_v1`** with a frozen 30-feature governance contract. fileciteturn53file0

---

## The most important ML finding: target leakage

One of the strongest parts of the project was discovering that a bureau score-like variable created an implausibly powerful model signal.

A feature based on the upstream credit-score information produced near-perfect predictive performance when used naively. Investigation showed that the feature was not a clean independent predictor for the intended task and was therefore excluded from the governed production feature set.

The important lesson was not simply **"remove a feature"**. The project treats leakage as a model-governance problem:

```text
Raw features
    ↓
Leakage / provenance investigation
    ↓
Approved feature registry
    ↓
Frozen conservative feature set
    ↓
Model selection
    ↓
Final model artifact
```

The final model configuration explicitly defines the target, the 30 allowed features, feature types, preprocessing, hyperparameters, and excluded features. fileciteturn53file0

See:

- [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md)
- [`docs/LEAKAGE_ANALYSIS.md`](docs/LEAKAGE_ANALYSIS.md)
- [`TECHNICAL_REFERENCE.md`](TECHNICAL_REFERENCE.md)

---

## Final ML architecture

```text
                    DATA LAYER
                        │
      Raw borrower / bureau source files
                        │
                        ▼
                 ┌─────────────┐
                 │   BRONZE    │
                 │ raw + tagged│
                 └──────┬──────┘
                        │
                        ▼
                 ┌─────────────┐
                 │   SILVER    │
                 │ cleaned +   │
                 │ engineered  │
                 └──────┬──────┘
                        │
                        ▼
                 ┌─────────────┐
                 │    GOLD     │
                 │ fact table  │
                 └──────┬──────┘
                        │
          ┌─────────────┴─────────────┐
          │                           │
          ▼                           ▼
   MODEL GOVERNANCE             ANALYTICAL LAYER
          │                           │
  frozen feature set            DuckDB / SQL
  leakage controls                  │
  evaluation                         │
          │                           │
          ▼                           │
  Conservative XGBoost               │
  conservative_xgb_v1                │
          │                           │
          ▼                           │
  persisted predictions              │
          │                           │
          └─────────────┬─────────────┘
                        ▼
                Decision-support layer
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
    Streamlit dashboard          SQL Query Library
          │                           │
          ▼                           ▼
    portfolio / risk /         business validation /
    explainability / policy    monitoring / audit
```

### Why the final model is refit on all Gold data

Model selection and honest generalization evaluation happen before the final build. The final artifact is then refit on 100% of the Gold dataset for downstream scoring.

This distinction is deliberate:

| Artifact | Purpose |
|---|---|
| Held-out evaluation | Honest generalization metrics |
| Cross-validation | Stability / robustness evidence |
| `conservative_xgb_v1.joblib` | Final deployable model artifact |
| `predictions.parquet` | Full-portfolio downstream scoring |

The project metadata explicitly warns that the full-data predictions must **not** be presented as unbiased generalization metrics. fileciteturn54file0

---

## Dataset

The project operates on a 51,336-row Indian retail credit portfolio assembled from the project input sources.

### Target

`default_risk` is a binary modelling target representing lower-risk versus higher-risk borrower groups derived from the source risk classification.

### Feature governance

The final model uses **30 frozen features** spanning:

- repayment and delinquency behaviour
- enquiry behaviour
- existing loan portfolio structure
- credit-history duration
- income
- basic borrower attributes
- secured / unsecured and loan-type indicators

The selected feature set is maintained centrally in [`src/modeling/final_model_config.py`](src/modeling/final_model_config.py), rather than being redefined independently across modelling scripts. fileciteturn53file0

---

## Data pipeline

```text
Raw source files
      │
      ▼
BRONZE
- ingestion
- source metadata
- join integrity
      │
      ▼
SILVER
- sentinel / missing-value handling
- target encoding
- feature engineering
- data-quality validation
      │
      ▼
GOLD
- governed analytical fact table
- downstream ML source
      │
      ├──────────────────────────┐
      ▼                          ▼
 ML / MODELING               ANALYTICS
      │                          │
      │                          ├─ segmentation
      │                          ├─ delinquency analysis
      │                          ├─ early-warning analysis
      │                          └─ business SQL
      ▼
FINAL MODEL
conservative_xgb_v1
      │
      ▼
PERSISTED PREDICTIONS
      │
      ▼
SQL + STREAMLIT
```

The repository keeps data preparation, model construction, final-model training, scoring, and dashboard presentation as separate stages rather than combining them into one notebook.

---

## Model governance and evaluation

The model pipeline includes more than model fitting.

### 1. Leakage control

Potentially contaminated or semantically unsafe features are investigated before entering the frozen production feature set.

### 2. Frozen model contract

`final_model_config.py` is the single source of truth for:

- model identity and version
- target definition
- frozen features
- numeric/categorical feature types
- preprocessing
- XGBoost hyperparameters
- excluded features
- configuration validation

The module explicitly separates model definition from model selection, calibration, SHAP analysis, threshold analysis, and final artifact creation. fileciteturn53file0

### 3. Cross-validation

Stratified cross-validation is used to evaluate model stability while preserving class proportions.

### 4. Held-out evaluation

A held-out evaluation split is retained for the performance claims that represent generalization.

### 5. Threshold analysis

The project treats the operating threshold as a **business decision**, not an automatic 0.50 default. Threshold analysis is used to study approval rate, precision, recall, false alarms, and risk capture.

---

## Explainability with SHAP

The project uses SHAP to answer:

> **Why did the model assign this borrower a high or low risk score?**

The explainability layer supports both:

- global feature importance
- borrower-level reason codes / feature contribution analysis

A major design principle is that **SHAP explains the model's behaviour; it does not prove causality**. Business conclusions are therefore cross-checked against portfolio-level SQL analysis.

---

## Persisted prediction layer

The final model writes its full-portfolio predictions to:

```text
src/data/model/predictions/
├── predictions.parquet
└── prediction_metadata.json
```

The prediction metadata records the model version, feature-set name, number of rows scored, prediction artifact hash, model artifact hash, scoring scope, and scoring timestamp. The current artifact covers all **51,336 Gold rows**. fileciteturn34file0

This matters because the dashboard does **not** retrain or silently rescore borrowers on every page load. The dashboard consumes persisted analytical outputs.

---

## SQL analytical layer

The SQL layer exists as an independent business-facing analytical interface.

The goal is simple:

> A credit-risk analyst should be able to investigate the portfolio without opening the modelling notebook.

The repository contains historical/descriptive SQL plus model-prediction analysis.

### Examples

| Query | Business Question |
|---|---|
| `01_portfolio_health.sql` | How risky is the overall loan book? |
| `02_default_by_segment.sql` | Which borrower segments drive default? |
| `03_enquiry_default_correlation.sql` | Does recent credit-enquiry behaviour correlate with default? |
| `04_delinquency_funnel.sql` | Where do borrowers move through delinquency stages? |
| `05_credit_history_vs_default.sql` | Does credit-history length relate to default? |
| `06_high_risk_identification.sql` | Which profiles should trigger review? |
| `07_income_default_analysis.sql` | How does income interact with observed default behaviour? |
| `08_gold_loan_analysis.sql` | What is the observed risk profile of gold-loan borrowers? |
| `09_early_intervention_candidates.sql` | Who should be prioritised for intervention? |
| `10_policy_impact_simulation.sql` | What changes under alternative policy rules? |

The Streamlit SQL Query Library loads the SQL files directly from the repository so the dashboard remains a presentation layer over the analytical queries. 

For example, `01_portfolio_health.sql` is a portfolio-level aggregate query returning borrower count, defaults, default rate, income, loan-portfolio, delinquency, and enquiry KPIs. fileciteturn57file0

---

## Streamlit dashboard — 7 panels

Launch with:

```bash
streamlit run app.py
```

| Panel | Purpose |
|---|---|
| **01 · Portfolio Overview** | Portfolio KPIs, risk-band distribution, income/default relationships, expected-loss indicators |
| **02 · Risk Segmentation** | Segment-level default rates, contribution, and lift |
| **03 · Model Performance** | ROC analysis, predicted-PD separation, confusion matrix, threshold sensitivity |
| **04 · SHAP Explainability** | Global risk drivers and borrower-level reason codes |
| **05 · Policy Simulator** | What-if underwriting rules and expected-loss / approval impacts |
| **06 · Risk Monitoring** | Watchlists, delinquency signals, and concentration monitoring |
| **07 · SQL Query Library** | Live SQL inspection, execution, results, charts, and CSV export |

The dashboard consumes persisted model outputs and analytical tables rather than retraining the model inside the UI.

---

## Policy simulation

A particularly important part of the platform is the policy simulator.

Instead of stopping at:

> “The model has an AUC of X.”

it asks:

> “What happens to approvals, defaults, false alarms, and expected loss if the credit team changes the policy?”

The simulator can vary underwriting constraints such as:

- maximum recent enquiries
- delinquency limits
- 60+ DPD limits
- minimum credit-history duration
- maximum active-loan ratio
- minimum income
- maximum predicted PD

This connects ML output to a measurable business decision.

---

## Expected loss

The project uses the standard credit-risk relationship:

```text
Expected Loss = PD × LGD × EAD
```

where:

- **PD** = predicted probability of default
- **LGD** = loss given default assumption
- **EAD** = exposure at default proxy

The dashboard uses this to translate model risk scores into portfolio-level monetary impact rather than reporting model metrics in isolation.

---

## Engineering / reproducibility

The repository is structured so that model code, analytical SQL, dashboard code, and deployment configuration are separate.

### CI

GitHub Actions validates the repository on pushes and pull requests by:

- installing dependencies
- compiling the Python source tree
- validating the SQL directory and SQL files
- running `pytest` when a test suite is present

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml). fileciteturn51file0

### Streamlit configuration

The repository includes `.streamlit/config.toml` for headless dashboard execution and runtime configuration. fileciteturn52file0

### Reproducible dependencies

The repository tracks a pinned Python environment in `requirements.txt`, including the analytical database dependency used by the project. fileciteturn56file0

---

## Project structure

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
├── data/
│   ├── bronze/
│   ├── silver/
│   ├── gold/
│   └── model/
│       └── predictions/
│
├── docs/
│   ├── MODEL_CARD.md
│   ├── LEAKAGE_ANALYSIS.md
│   ├── CHANGELOG.md
│   └── ...
│
├── sql/
│   └── queries/
│
├── src/
│   ├── data/
│   ├── modeling/
│   ├── analytics/
│   └── visualization/
│
├── assets/
│   └── screenshots/
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

### Install

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

### Run the dashboard

```bash
streamlit run app.py
```

### Run CI checks locally

```bash
python -m compileall -q src app.py
```

If a `tests/` directory is present:

```bash
pytest -q
```

### Important note on model evaluation

Do not treat the full-data final-model predictions as a test-set performance estimate. Use the held-out evaluation artifacts and cross-validation outputs for generalization claims.

---

## Documentation

| Document | Purpose |
|---|---|
| [`PROBLEM_STATEMENT.md`](PROBLEM_STATEMENT.md) | Business problem, scope, and project objective |
| [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) | Final model purpose, assumptions, evaluation, and limitations |
| [`docs/LEAKAGE_ANALYSIS.md`](docs/LEAKAGE_ANALYSIS.md) | Leakage investigation and remediation |
| [`docs/CHANGELOG.md`](docs/CHANGELOG.md) | Project evolution and changes |
| [`TECHNICAL_REFERENCE.md`](TECHNICAL_REFERENCE.md) | Detailed methods, formulas, and technical rationale |

---

## Key design decisions

### Why XGBoost?

The final model is a tree-based gradient boosting classifier suited to structured/tabular borrower data and capable of modelling nonlinear feature interactions without requiring deep-learning-scale datasets.

### Why not simply maximize AUC?

Credit decisions have asymmetric business costs. A model with strong ranking ability can still produce an undesirable operating point. The project therefore evaluates threshold behaviour, false positives, false negatives, and policy consequences.

### Why persist predictions?

Separating scoring from presentation prevents the dashboard from becoming a second modelling pipeline. The model is run once, predictions are persisted, and downstream analytics consume those outputs.

### Why SQL as well as Python?

Python is appropriate for modelling and experimentation; SQL is appropriate for repeatable portfolio analysis and operational business questions. Keeping both makes the analytical layer usable by a broader credit-risk team.

### Why explainability?

A credit-risk platform needs to answer not only **what score was produced**, but also **which variables drove the decision and how those signals behave in the portfolio**.

---

## Limitations

This is an analytical and decision-support project, not a production credit-origination system.

Important limitations include:

- the source data represent a static historical portfolio rather than a continuously refreshed lending population;
- full-data final-model predictions are not unbiased estimates of out-of-sample performance;
- portfolio-level SHAP and SQL associations do not establish causal relationships;
- expected-loss estimates depend on simplifying EAD/LGD assumptions;
- real deployment would require stronger monitoring for population drift, feature availability, policy changes, fairness, and operational controls.

These limitations are part of the intended governance story rather than something hidden from the user.

---

## License

MIT License.

---

## Author

**Aarya R.**

Built as an end-to-end ML + analytics portfolio project focused on translating credit-risk modelling into a usable decision-support workflow.
