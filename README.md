# India Credit Risk Intelligence

**Business Analytics & Credit Portfolio Intelligence for Indian Lending**

> End-to-end credit risk analytics and decision-support platform for Indian lending portfolios using Python, XGBoost, SHAP, SQL, and Streamlit.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange?style=flat-square)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red?style=flat-square)](https://streamlit.io)
[![SQLite](https://img.shields.io/badge/SQLite-3-lightgrey?style=flat-square)](https://sqlite.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## What This Platform Does

A **business-facing analytics platform** for credit risk managers to:
1. Monitor portfolio health and expected loss in real time
2. Identify which borrower segments are driving NPA
3. Explain individual model decisions in plain English
4. Simulate underwriting policy changes before rolling them out

**This is not an ML notebook. It is a decision-support system.**

---

## The Key Finding

> The `Credit_Score` column achieves **AUC = 0.9998** — statistically impossible. It was derived from the target variable during preprocessing (data leakage). A behavioural model using **15 raw signals achieves AUC = 0.8994** with no credit score required.
>
> The single strongest predictor: **recent credit enquiries (enq_L6m)**. Borrowers applying to 4+ lenders in 6 months default at **4× the baseline rate** — a real-time stress signal invisible to any bureau score.

---

## Dashboard — 6 Business Panels

```
streamlit run app.py
```

| Panel | Business Question |
|-------|------------------|
| **1 · Executive Overview** | How healthy is our current loan book? KPIs + Expected Loss + Risk Distribution |
| **2 · Borrower Segmentation** | Which segments are driving NPA? Leaderboard + heatmap |
| **3 · Model Performance** | Can we trust this model? ROC + PD Distribution + **Threshold Sensitivity Table** |
| **4 · Explainable Decisions** | Why was this borrower flagged? SHAP + Plain-English Risk Card |
| **5 · Policy Simulator** | What if we tighten the enquiry rule? Real-time what-if analysis |
| **6 · Risk Monitoring** | Who needs attention this week? Watchlist + Concentration Alerts |

**Panel 5 (Policy Simulator) is the key differentiator** — allows credit teams to simulate underwriting rule changes and see impact on approval rate, defaults prevented, and expected loss before any policy is rolled out.

---

## Project Structure

```
india-credit-risk-intelligence/
│
├── app.py                          ← Single entrypoint
├── requirements.txt
├── README.md
│
├── src/
│   ├── analytics/
│   │   ├── run_ml_model.py         ← XGBoost + SHAP pipeline
│   │   ├── improve_precision.py    ← Model B tuning
│   │   ├── portfolio_metrics.py    ← EL, risk bands, segment contribution
│   │   ├── policy_simulator.py     ← What-if policy engine
│   │   └── run_queries.py          ← Execute SQL queries
│   ├── data/
│   │   └── create_database.py      ← Parquet → SQLite
│   ├── explainability/
│   │   └── decision_reasoning.py   ← SHAP → plain English
│   ├── utils/
│   │   └── config.py               ← Central configuration
│   └── visualization/
│       └── build_dashboard.py      ← Streamlit dashboard
│
├── sql/                            ← 10 business SQL queries
│   ├── 01_portfolio_health.sql
│   ├── 06_high_risk_identification.sql
│   ├── 09_early_intervention_candidates.sql
│   ├── 10_policy_impact_simulation.sql
│   └── ...
│
├── docs/
│   ├── PROBLEM_STATEMENT.md
│   ├── BUSINESS_CASE.md
│   └── METRICS_DEFINITION.md
│
├── assets/
│   └── screenshots/
│
└── data/
    ├── credit_risk.db              ← SQLite (3 tables, 51,336 rows)
    ├── gold/exports/ml/            ← Model outputs
    └── silver/                     ← Feature-engineered parquet (gitignored)
```

---

## Key Metrics at a Glance

| Metric | Value |
|--------|-------|
| Portfolio Size | 51,336 borrowers |
| Default Rate | 26.0% (vs 22% national avg) |
| Model AUC | **0.8994** |
| Precision (Model B) | 69.1% |
| Recall (Model B) | 71.0% |
| F1 Score | 0.700 |
| Expected NPA Reduction (3 policies) | **18-24%** |

**Model B selected over original:** `scale_pos_weight` tuned 2.85 → 1.43. Reduces false alarm rate from 41% → 31% while maintaining 71% recall.

---

## SQL Query Library (10 Business Queries)

| Query | Answers |
|-------|---------|
| `01_portfolio_health.sql` | Portfolio KPIs |
| `03_enquiry_default_correlation.sql` | Proves #1 SHAP finding in SQL |
| `06_high_risk_identification.sql` | Policy rule performance |
| `09_early_intervention_candidates.sql` | Ranked watchlist for collections |
| `10_policy_impact_simulation.sql` | NPA reduction simulation |

---

## Three Credit Policy Recommendations

Based on SHAP + segment analysis of 51,336 borrowers:

**Policy 1 — Enquiry Auto-Flag** · Flag 4+ enquiries in 6m · ~45% default rate · 8-11% NPA reduction

**Policy 2 — Thin-File Pricing** · Premium for <2yr history · ~34% default rate · 6-9% NPA reduction

**Policy 3 — 30-DPD Intervention** · Outreach at first miss · 65% escalation rate · 5-8% NPA reduction

**Combined estimated impact: 18-24% NPA reduction** (validate with 90-day pilot)

---

## Business Documents

| Document | Purpose |
|----------|---------|
| [`docs/BUSINESS_CASE.md`](docs/BUSINESS_CASE.md) | Who uses this, what decisions it supports, expected impact |
| [`docs/PROBLEM_STATEMENT.md`](docs/PROBLEM_STATEMENT.md) | The three business gaps this platform solves |
| [`docs/METRICS_DEFINITION.md`](docs/METRICS_DEFINITION.md) | PD, EL, AUC, DPD, NPA — plain-English glossary |
| [`reports/business_recommendation.pdf`](reports/business_recommendation.pdf) | 3-policy recommendation with data evidence |

---

## Setup

```bash
# Clone
git clone https://github.com/rareya/india-credit-risk-intelligence.git
cd india-credit-risk-intelligence

# Install
pip install -r requirements.txt

# Run ML pipeline
python src/analytics/run_ml_model.py
python src/analytics/improve_precision.py

# Build SQLite database
python src/data/create_database.py

# Launch dashboard
streamlit run app.py
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | XGBoost, scikit-learn |
| Explainability | SHAP |
| Data | pandas, numpy, pyarrow |
| Database | SQLite |
| Dashboard | Streamlit + Plotly |
| Language | Python 3.10+ |

---

## GitHub Topics

`credit-risk` · `credit-risk-analytics` · `business-analytics` · `data-analytics` · `risk-management` · `fintech` · `streamlit` · `xgboost` · `shap` · `sql` · `india` · `portfolio-analytics`

---

*Built by Aarya Patankar — Data Analysis Project, 2026. Dataset: CIBIL-style Indian borrower records (anonymised/synthetic). MIT License.*