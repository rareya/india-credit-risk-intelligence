# Changelog
## India Credit Risk Intelligence

All notable changes to this project are documented here.

---

## [1.2.0] — 2026-03 — Model B + Dashboard

### Added
- `src/analytics/improve_precision.py` — three-strategy precision improvement layer
  - Strategy 1: Threshold tuning sweep (0.1 → 0.9)
  - Strategy 2: `scale_pos_weight` tuning (2.85 → 1.43)
  - Strategy 3: Feature engineering — 4 interaction terms (enq_per_credit_year, delinquency_rate, enq_acceleration, severe_delinquency_ratio)
- `src/visualization/build_dashboard.py` — 6-panel Streamlit dashboard
  - Panel 1: Executive portfolio overview (KPIs, EL, risk distribution)
  - Panel 2: Borrower segmentation (interactive dimension selector)
  - Panel 3: Model performance (ROC, PD distribution, threshold sensitivity table)
  - Panel 4: Explainable decisions (SHAP, individual risk card with plain-English reasons)
  - Panel 5: Policy simulator (real-time what-if underwriting analysis)
  - Panel 6: Risk monitoring (watchlist, delinquency funnel, concentration alerts)
- `src/analytics/policy_simulator.py` — what-if policy engine with `PolicyConfig` dataclass
- `src/analytics/portfolio_metrics.py` — Expected Loss, risk bands, segment contribution
- `src/analytics/decision_reasoning.py` — SHAP → plain-English reason codes
- `src/data/export_powerbi_dataset.py` — scored CSV export for Power BI
- `app.py` — single entrypoint for Streamlit dashboard
- `docs/MODEL_CARD.md` — full model documentation including limitations and biases
- `docs/LEAKAGE_ANALYSIS.md` — analysis of Credit_Score data leakage finding
- `sql/queries/` — 10 business SQL queries

### Changed
- Model B (`scale_pos_weight = 1.43`) selected over original (`2.85`)
- Precision improved from 59.3% → 69.1% (−10pp false alarm rate)
- Recall adjusted from 83.5% → 71.0% (acceptable tradeoff for business deployment)
- F1 improved from 0.6935 → 0.7003

### Key Finding
`Credit_Score` achieves AUC = 0.9998 — identified as data leakage (derived from target variable). Excluded from all models. See `docs/LEAKAGE_ANALYSIS.md`.

---

## [1.1.0] — 2026-02 — XGBoost Model + SHAP

### Added
- `src/analytics/run_ml_model.py` — XGBoost binary classifier
  - 5-fold stratified cross-validation (manual loop, NaN-proof)
  - SHAP TreeExplainer for global and individual explainability
  - `credit_score_myth_proof()` — automated leakage detection
  - ROC curve, confusion matrix, classification report
  - Model serialisation to `.pkl` + metadata JSON
- `src/analytics/run_analytics.py` — 6 analytical modules
  - Credit score threshold analysis (650-cliff detection)
  - Gini coefficient for credit inequality
  - KMeans borrower segmentation (4 risk profiles)
  - Early warning signal ranking (point-biserial correlation)
  - Risky vs safe borrower comparison
  - Gold loan analysis (India-specific insight)
- `src/modeling/build_gold.py` — DuckDB star schema
  - 4 dimension/fact tables: dim_borrower, dim_credit, dim_loan_portfolio, fact_credit_risk
  - 5 analytical views

### Key Finding
`enq_L6m` is the #1 SHAP feature (importance 1.183). Borrowers with 4+ enquiries in 6 months default at 4× baseline — a real-time signal invisible to bureau scores.

---

## [1.0.0] — 2026-01 — Data Pipeline

### Added
- `src/ingestion/ingest_kaggle.py` — Bronze layer ingestion
  - Internal bank dataset (tradeline features)
  - External CIBIL dataset (bureau features + target)
  - Loan applications dataset
  - Data dictionary
  - Join integrity validation
- `src/transformation/transform_silver.py` — Silver layer transformation
  - Sentinel value handling (-99999 → meaningful substitutes)
  - Target encoding: `Approved_Flag` P1/P2/P3/P4 → binary `default_risk`
  - Feature engineering (12 derived features)
  - Data quality validation
- `explore_data.py` — initial EDA script
- `investigate_data.py` — target variable investigation
- `sql/` — 10 business SQL queries
- `src/data/create_database.py` — Parquet → SQLite conversion
- `requirements.txt` — pinned dependencies

### Data
- 51,336 CIBIL-style Indian borrower records
- 26.0% default rate (vs 22% national NBFC average)
- Joined from 2 source files on PROSPECTID (internal bank + external CIBIL)