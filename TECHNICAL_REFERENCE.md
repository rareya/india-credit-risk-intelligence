# Technical Reference
## India Credit Risk Intelligence Platform

This document explains every dataset, preprocessing step, calculation, formula, and analytical method used in this project. It is written for someone reviewing this work — whether a recruiter, interviewer, or technical evaluator.

---

## Table of Contents

1. [The Dataset](#1-the-dataset)
2. [Data Pipeline — All Phases](#2-data-pipeline--all-phases)
3. [Target Variable — Encoding and Verification](#3-target-variable--encoding-and-verification)
4. [Feature Engineering — Every Derived Column](#4-feature-engineering--every-derived-column)
5. [Model — XGBoost Configuration and Tuning](#5-model--xgboost-configuration-and-tuning)
6. [Evaluation Metrics — Formulas and Business Meaning](#6-evaluation-metrics--formulas-and-business-meaning)
7. [SHAP Explainability — What It Computes](#7-shap-explainability--what-it-computes)
8. [Expected Loss — Formula and Assumptions](#8-expected-loss--formula-and-assumptions)
9. [Analytical Modules](#9-analytical-modules)
10. [SQL Query Library — What Each Query Computes](#10-sql-query-library--what-each-query-computes)
11. [Dashboard Panels — What Each Visual Means](#11-dashboard-panels--what-each-visual-means)
12. [Why These Methods and Not Others](#12-why-these-methods-and-not-others)

---

## 1. The Dataset

### 1.1 What Is CIBIL?

CIBIL (Credit Information Bureau India Limited) is India's primary credit bureau, equivalent to FICO in the US. It collects credit information from banks and NBFCs (Non-Banking Financial Companies) and produces credit scores in the range 300–900. A score below 650 is typically the rejection threshold at most Indian lenders.

### 1.2 Data Sources

Two independent Kaggle datasets are joined on `PROSPECTID` — a unique borrower identifier present in both files.

| Source | File | What It Contains | Why We Use It | Collection Method |
|--------|------|-----------------|---------------|-------------------|
| **Internal Bank Dataset** | `case_study1.xlsx` | Tradeline features: total loans opened, active loans, closed loans, missed payments, loan type breakdown (gold, home, personal, auto, credit card, consumer), enquiry counts (6m, 12m, lifetime), credit history age (oldest and newest tradeline) | Only source for granular behavioural loan signals. Contains `enq_L6m` — the #1 SHAP feature. | Kaggle download |
| **External CIBIL Dataset** | `case_study2.xlsx` | Bureau features + target: `Credit_Score` (300–900), `Approved_Flag` (P1–P4), age, gender, marital status, education, monthly income, delinquency counts, DPD history (30+ and 60+), missed payments, credit card and personal loan utilisation | Ground truth for the target variable. CIBIL bureau data. | Kaggle download |

### 1.3 What Each Dataset Contains

#### Internal Bank Dataset — Column Reference

| Column | Type | Meaning |
|--------|------|---------|
| `PROSPECTID` | int | Unique borrower identifier — join key |
| `Total_TL` | int | Total loan accounts ever opened |
| `Tot_Active_TL` | int | Currently active loan accounts |
| `Tot_Closed_TL` | int | Closed (repaid) loan accounts |
| `Tot_Missed_Pmnt` | int | Total missed payments across all loans |
| `Gold_TL` | int | Gold loan accounts — pledged jewellery as collateral |
| `Home_TL` | int | Home loan accounts |
| `PL_TL` | int | Personal loan accounts |
| `CC_TL` | int | Credit card accounts |
| `Auto_TL` | int | Auto loan accounts |
| `Consumer_TL` | int | Consumer loan accounts |
| `Secured_TL` | int | Secured loan accounts |
| `Unsecured_TL` | int | Unsecured loan accounts |
| `Age_Oldest_TL` | int | Months since oldest loan opened — credit history length |
| `Age_Newest_TL` | int | Months since most recent loan opened |
| `Total_TL_opened_L6M` | int | New loans opened in last 6 months |
| `enq_L6m` | int | Credit enquiries in last 6 months — **#1 SHAP feature** |
| `enq_L12m` | int | Credit enquiries in last 12 months |
| `tot_enq` | int | Lifetime total credit enquiries |
| `CC_utilization` | float | Credit card utilisation ratio. -99999 = no credit card. |
| `PL_utilization` | float | Personal loan utilisation. -99999 = no personal loan. |

#### External CIBIL Dataset — Column Reference

| Column | Type | Meaning |
|--------|------|---------|
| `PROSPECTID` | int | Unique borrower identifier — join key |
| `Approved_Flag` | string | Target variable: P1 (safest) to P4 (riskiest) |
| `Credit_Score` | int | CIBIL bureau score 300–900. **Excluded: data leakage (AUC 0.9998)** |
| `AGE` | int | Borrower age in years |
| `GENDER` | string | M / F |
| `MARITALSTATUS` | string | Married / Single / Other |
| `EDUCATION` | string | SSC / 12TH / GRADUATE / POST_GRADUATE / PROFESSIONAL |
| `NETMONTHLYINCOME` | float | Net monthly income in INR |
| `num_times_delinquent` | int | Total delinquency events across all loans |
| `num_times_30p_dpd` | int | Times 30+ days past due |
| `num_times_60p_dpd` | int | Times 60+ days past due |
| `Tot_Missed_Pmnt` | int | Total missed payments |
| `last_prod_enq2` | string | Most recent product enquired for |
| `first_prod_enq2` | string | First product enquired for |
| `Time_With_Curr_Empr` | int | Months with current employer |

### 1.4 Sentinel Values

The value `-99999` appears throughout the dataset. It means "not applicable" — not missing. Examples:

- `CC_utilization = -99999` → borrower has no credit card (not a missing value)
- `max_deliq_level = -99999` → borrower has never been delinquent

**Handling strategy:** Create binary flag columns (`has_cc`, `has_pl`) first, then replace -99999 with 0 for utilisation columns and 9999 for time-since columns. This preserves the "never happened" signal rather than treating it as missing data.

---

## 2. Data Pipeline — All Phases

### 2.1 Bronze Layer — `ingest_kaggle.py`

**What it does:** Reads all 8 raw files, adds two metadata columns (`_source`, `_ingested_at`), saves each as Parquet. No transformation.

**Why Parquet and not CSV:** Parquet is typed (no silent type coercion on re-read), compressed (~5–10× smaller than CSV), and columnar (fast for analytical reads of specific columns). It is the industry standard for data lake storage.

**Join integrity check:** After ingesting, `validate_join_integrity()` confirms that all `PROSPECTID` values in the bank dataset exist in the CIBIL dataset. A failed join would produce a smaller-than-expected silver dataset.

### 2.2 Silver Layer — `transform_silver.py`

**What it does:** Produces `silver_master.parquet` — the single clean dataset used by all downstream steps.

Steps in order:

1. **Load bronze parquets** — drops `_source` and `_ingested_at` metadata columns
2. **Encode target variable** — see Section 3
3. **Handle sentinel values** — see Section 1.4
4. **Clean categorical columns** — standardise casing (EDUCATION, GENDER, MARITALSTATUS), consolidate rare categories
5. **Inner join** — `df_cibil.merge(df_bank, on="PROSPECTID", how="inner")` — only borrowers present in both datasets are retained
6. **Engineer features** — see Section 4
7. **Validate** — 5 data quality checks: null counts in key columns, CIBIL score range 300–900, no duplicate PROSPECTID, binary target, CIBIL direction check (risky borrowers must have lower mean CIBIL than safe borrowers)

### 2.3 Gold Layer — `build_gold.py`

**What it does:** Builds a star schema in DuckDB from `silver_master.parquet`.

| Table | Type | Contents |
|-------|------|---------|
| `dim_borrower` | Dimension | Demographics: age, gender, income, education, employer tenure |
| `dim_credit` | Dimension | Bureau profile: CIBIL score, band, delinquency history, enquiry counts |
| `dim_loan_portfolio` | Dimension | Loan composition: counts by type, flags, active ratio |
| `fact_credit_risk` | Fact | One row per borrower — all measures, all dimension keys |
| `v_risk_by_cibil_band` | View | Default rate aggregated by CIBIL score band |
| `v_risk_by_income` | View | Default rate by income tier |
| `v_risk_by_age` | View | Default rate by age band |
| `v_gold_loan_analysis` | View | Default rate for gold vs non-gold loan holders |
| `v_risk_by_education` | View | Default rate by education level |

**Why DuckDB and not SQLite for Gold:** DuckDB is columnar (optimised for analytical aggregations). SQLite is row-oriented (optimised for transactional lookups). For the Gold layer we run `GROUP BY` and window functions — DuckDB is 10–100× faster. SQLite is used for the operational dashboard layer where we need portability.

### 2.4 ML Pipeline — `run_ml_model.py` and `improve_precision.py`

See Sections 5 and 6 for full model documentation.

### 2.5 SQLite Layer — `create_database.py`

**What it does:** Converts `silver_master.parquet` into three SQLite tables for dashboard use.

| Table | Contents |
|-------|---------|
| `borrowers` | Full feature set — all silver columns |
| `delinquency_summary` | Delinquency-specific columns only |
| `risk_segments` | Derived segment labels (extreme/high/standard) |

**Why SQLite for the dashboard:** The dashboard needs to run SQL queries. SQLite is file-based (no server), portable (committed to git), and supported by every SQL tool. All 10 queries run unchanged on PostgreSQL if a production migration is needed.

---

## 3. Target Variable — Encoding and Verification

### 3.1 What Approved_Flag Means

`Approved_Flag` (P1–P4) is the CIBIL risk grade assigned to each borrower. The mapping to risk level was determined empirically by examining the data:

| Flag | Meaning | Evidence (from investigate_data.py) |
|------|---------|-------------------------------------|
| P1 | Lowest risk — best borrowers | Avg CIBIL ~780, avg delinquencies ~0.1, avg missed payments ~0.3 |
| P2 | Low risk | Avg CIBIL ~720, avg delinquencies ~0.4 |
| P3 | High risk | Avg CIBIL ~640, avg delinquencies ~1.8 |
| P4 | Highest risk — worst borrowers | Avg CIBIL ~580, avg delinquencies ~4.2, avg missed payments ~8.1 |

### 3.2 Binary Encoding

```python
# Risky = P3 or P4 (high risk) → default_risk = 1
# Safe  = P1 or P2 (low risk)  → default_risk = 0
df["default_risk"] = df["Approved_Flag"].isin(["P3", "P4"]).astype(int)
```

**Class distribution:** 26.0% risky (P3/P4), 74.0% safe (P1/P2).

### 3.3 Verification

After encoding, `transform_silver.py` runs an automatic sanity check:

```python
risky_cibil = df[df["default_risk"]==1]["Credit_Score"].mean()
safe_cibil  = df[df["default_risk"]==0]["Credit_Score"].mean()
assert risky_cibil < safe_cibil, "Encoding direction error"
```

If this assertion fails, the encoding is wrong. It passes: risky borrowers (P3/P4) have consistently lower CIBIL scores than safe borrowers (P1/P2).

---

## 4. Feature Engineering — Every Derived Column

All derived features are created in `transform_silver.py`. Every formula is documented here.

| Feature | Formula | Business Meaning |
|---------|---------|-----------------|
| `active_loan_ratio` | `Tot_Active_TL / Total_TL` (0 if Total_TL=0) | Fraction of ever-opened loans still active. High = heavily loaded. Low = paid off most loans. |
| `missed_payment_ratio` | `Tot_Missed_Pmnt / Total_TL` (0 if Total_TL=0) | Missed payments per loan. Normalises for borrowers with many loans. |
| `credit_history_months` | `Age_Oldest_TL` | Alias — months since oldest loan opened. |
| `score_per_income_lakh` | `Credit_Score / (NETMONTHLYINCOME / 100000)` | CIBIL score per lakh of monthly income. Combines creditworthiness with capacity. |
| `income_tier` | `pd.cut(NETMONTHLYINCOME, [0,15k,30k,60k,100k,∞])` | Indian-context brackets: very_low / low / middle / upper_middle / high |
| `delinquency_score` | `(num_times_delinquent × 2) + (num_times_30p_dpd × 3) + (num_times_60p_dpd × 5)` | Weighted composite. 60+ DPD is weighted 5× because it signals near-default and has high 90-DPD transition probability. |
| `recently_active` | `1 if Total_TL_opened_L6M > 0 else 0` | Boolean: did this borrower open a new loan in the last 6 months? |
| `loan_type_diversity` | `count of {Auto_TL, CC_TL, Consumer_TL, Gold_TL, Home_TL, PL_TL} where > 0` | Number of distinct loan types used. Higher = more experienced borrower. |
| `cibil_band` | `pd.cut(Credit_Score, [0,549,649,699,749,900])` | Standard Indian credit score interpretation: poor / fair / good / very_good / excellent |
| `risk_band` | Mapped from `Approved_Flag`: P1/P2=Low Risk, P3=Medium Risk, P4=High Risk | Categorical risk label for Power BI segmentation |
| `enq_per_credit_year` | `enq_L6m / (Age_Oldest_TL / 12 + 1)` | Enquiries normalised by years of credit history. High = recent desperation relative to track record. |
| `delinquency_rate` | `num_times_delinquent / (Total_TL + 1)` | Delinquencies per loan account. Normalises for borrowers with many accounts. |
| `enq_acceleration` | `max(0, enq_L6m − enq_L12m / 2)` | If enq_L6m >> half of enq_L12m, the borrower is accelerating their credit search. Positive = more recent enquiries than historical pace. |
| `severe_delinquency_ratio` | `num_times_60p_dpd / (num_times_delinquent + 1)` | Fraction of all delinquency events that were serious (60+ DPD). |

---

## 5. Model — XGBoost Configuration and Tuning

### 5.1 Model A — Baseline (`run_ml_model.py`)

```python
xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=2.85,   # natural class ratio: n_negative / n_positive
    random_state=42,
    eval_metric="logloss",
)
```

**Why these hyperparameters:**
- `max_depth=4` — Shallow trees prevent overfitting. Deeper trees capture more interactions but memorise noise.
- `learning_rate=0.05` — Slow learning with more trees generalises better than fast learning with fewer trees.
- `subsample=0.8`, `colsample_bytree=0.8` — Stochastic sampling reduces variance. Standard practice for XGBoost.
- `scale_pos_weight=2.85` — Natural class ratio (74/26 ≈ 2.85). Makes XGBoost penalise missing a risky borrower 2.85× more than flagging a safe one.

**Training split:** 80/20 stratified train/test. Stratified ensures both splits have the same 26% default rate.

**Cross-validation:** 5-fold stratified, manual loop (not `cross_val_score`). The manual loop is NaN-proof — `cross_val_score` with XGBoost on Windows can return `nan` due to parallel pickling issues. Each fold trains on 80% of training data and evaluates on the held-out 20%.

### 5.2 Model B — Precision Improvement (`improve_precision.py`)

Model A's `scale_pos_weight=2.85` gives recall 83.5% but precision 59.3% — 41% of flagged borrowers are actually safe. For a lender, every false positive is a creditworthy customer turned away.

**Three strategies applied:**

**Strategy 1: Threshold tuning**
Sweep decision threshold from 0.10 to 0.90 in 0.01 steps. For each threshold, compute precision, recall, F1. The default 0.50 is arbitrary — the optimal threshold is rarely 0.50.

**Strategy 2: `scale_pos_weight` tuning**
Sweep `scale_pos_weight` from 0.3× to 2.0× the natural ratio (0.86 to 5.70). For each weight, train a full model and evaluate on test set. Find the weight where F1 is maximised while recall stays ≥ 0.70.

**Strategy 3: Feature engineering**
Add 4 interaction terms (see Section 4) that SHAP analysis suggested would capture compound risk signals.

**Model B selection:**
- `scale_pos_weight = 1.43` (from weight sweep)
- `threshold = 0.50` (held at default — precision-maximised threshold 0.62 dropped recall to 60%, missing too many risky borrowers)
- Result: Precision 69.1% (+9.8pp), Recall 71.0% (−12.5pp), F1 0.700 (+0.006)

**Why we accepted lower recall:** Reducing false alarms from 41% to 31% means 10 fewer safe borrowers rejected per 100 flags. At portfolio scale (51,336 borrowers), that is ~5,000 fewer incorrect rejections — meaningful revenue preservation.

---

## 6. Evaluation Metrics — Formulas and Business Meaning

### AUC — Area Under the ROC Curve

```
AUC = probability that model ranks a random risky borrower
      higher than a random safe borrower
```

Interpretation:
- 0.50 = random, no skill
- 0.70 = acceptable
- 0.80 = good
- **0.8994 = strong** (this model)

AUC is threshold-independent — it measures the model's ranking ability across all possible thresholds. This is why it is the primary evaluation metric for credit risk models.

### Precision

```
Precision = TP / (TP + FP)
          = true risky flagged / all flagged
```

Business meaning: Of every 100 applications rejected, how many were genuinely risky? Model B: 69.1 out of 100 flagged are actually risky.

### Recall (Sensitivity)

```
Recall = TP / (TP + FN)
       = true risky flagged / all actual risky
```

Business meaning: Of every 100 risky borrowers in the portfolio, how many does the model catch? Model B: 71 out of 100 risky borrowers are flagged.

### F1 Score

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

The harmonic mean of precision and recall. Preferred over accuracy for imbalanced datasets because accuracy is dominated by the majority class. A model that flags nobody as risky has 74% accuracy but 0% recall.

### RAAE — Risk-Adjusted Approval Efficiency

```
RAAE = Defaults Prevented / (Safe Borrowers Rejected + 1)
```

A policy-level metric. Higher = better. A policy that prevents 500 defaults while rejecting 100 safe borrowers has RAAE = 5.0. Used in the Policy Simulator to compare underwriting rules.

---

## 7. SHAP Explainability — What It Computes

### What SHAP Is

SHAP (SHapley Additive exPlanations) decomposes each model prediction into additive contributions from each feature. Formally:

```
f(x) = φ₀ + Σ φᵢ(xᵢ)

where:
  f(x)    = model's predicted PD for this borrower
  φ₀      = base value (average prediction across training data)
  φᵢ(xᵢ) = SHAP value for feature i — how much feature i pushed
             the prediction above or below the base value
```

**Why SHAP over built-in feature importance:** XGBoost's native feature importance (gain, split count) does not account for feature interactions and is not additive — you cannot sum feature importances to reconstruct any individual prediction. SHAP values are exact and additive.

### Global Feature Importance

`Mean |SHAP value|` across all test set predictions. The top 5:

| Rank | Feature | Mean |SHAP| | Direction |
|------|---------|------------|-----------|
| 1 | `enq_L6m` | 1.183 | Higher = more risky |
| 2 | `num_times_delinquent` | 0.654 | Higher = more risky |
| 3 | `Age_Oldest_TL` | 0.460 | Higher = less risky (protective) |
| 4 | `Total_TL` | 0.242 | Higher = less risky (protective) |
| 5 | `delinquency_score` | 0.210 | Higher = more risky |

### RBI Compliance

Under the RBI Fair Practices Code, every credit rejection must be explainable to the borrower. The dashboard's individual scorer converts SHAP values to plain English: "Applied to 6 lenders in 6 months — financial stress signal (increases default probability by 23.4pp)."

---

## 8. Expected Loss — Formula and Assumptions

### Formula

```
EL = PD × LGD × EAD

where:
  PD  = Probability of Default (XGBoost model output, 0–1)
  LGD = Loss Given Default = 0.45 (45%)
  EAD = Exposure at Default = NETMONTHLYINCOME × 12 (proxy)
```

### Assumption Justification

**LGD = 45%:** Industry standard for unsecured retail lending in India. Secured loans (home, gold collateral) have lower LGD (~25–30%) because collateral can be recovered. This portfolio is predominantly unsecured.

**EAD = 12 × monthly income:** A proxy because actual loan amounts are not in the dataset. Monthly income × 12 approximates annual income, which is a common EAD proxy for personal loans in Indian NBFCs where loan amounts are often 1–3 months of income.

**Limitation:** This is an analytical proxy. Production EL requires actual outstanding loan amounts and collateral values. Results should be treated as directional estimates, not regulatory capital calculations.

---

## 9. Analytical Modules

### 9.1 KMeans Borrower Segmentation

**Input features:** `Credit_Score`, `delinquency_score`, `active_loan_ratio`, `credit_history_months`, `NETMONTHLYINCOME`, `Total_TL`

**Preprocessing:** StandardScaler (mean=0, std=1). Required because income (₹0–500,000) would dominate clustering without scaling.

**k selection:** Silhouette score tested for k=3, 4, 5. k=4 gives highest silhouette and produces policy-relevant groupings: Safe Established / Low Risk / Moderate Risk / Stressed Borrower / High Risk.

**Cluster assignment:** `km.fit_predict(X_scaled)`. Clusters are sorted by average CIBIL score descending and named accordingly.

### 9.2 Gini Coefficient

```python
def gini(values):
    values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values)) / (n * np.sum(values)) - (n + 1) / n
```

Applied to CIBIL scores across income tiers. Output: 0 = perfect equality (all borrowers have identical scores), 1 = perfect inequality (one borrower has all the score, others have 300).

**Interpretation:** The Gini measures inequality of creditworthiness. A Gini of 0.35 means moderate spread — some borrowers have very high scores, others very low, but most are in the middle.

### 9.3 Early Warning Signals

`run_analytics.py` computes point-biserial correlation between each numeric feature and `default_risk`. Features with p-value < 0.05 are ranked by absolute correlation. This produces a ranking of features by their individual predictive power, independent of the XGBoost model — useful for validating that the model's SHAP rankings make intuitive statistical sense.

---

## 10. SQL Query Library — What Each Query Computes

Every query runs against `credit_risk.db` (SQLite). The `borrowers` table contains all silver master columns for all 51,336 borrowers.

### Query 01 — Portfolio Health

**Returns:** One-row KPI summary. `COUNT(*)`, `SUM(default_risk)`, `AVG(default_risk)*100` as default rate, average income, average delinquencies, average enquiries.

**Business use:** Monday morning dashboard for the credit risk manager.

### Query 03 — Enquiry–Default Correlation

**Key computation:**
```sql
ROUND(AVG(default_risk) / (SELECT AVG(default_risk) FROM borrowers), 2) AS lift_over_baseline
```

Lift = default rate in this enquiry band / overall default rate. A band with lift=4.0 means borrowers in that band default at 4× the baseline. This is the SQL proof of the `enq_L6m` SHAP finding — no model required.

### Query 04 — Delinquency Funnel

Uses a CTE (`WITH funnel AS`) to compute four portfolio stages in a single pass:
- Total portfolio: `COUNT(*)`
- Ever 30+ DPD: `SUM(CASE WHEN num_times_delinquent > 0 THEN 1 ELSE 0 END)`
- Ever 60+ DPD: `SUM(CASE WHEN num_times_60p_dpd > 0 THEN 1 ELSE 0 END)`
- Confirmed default: `SUM(default_risk)`

Transition rates are computed in the UNION ALL rows using the CTE values.

### Query 09 — Early Intervention Candidates

**Urgency score formula:**
```sql
ROUND(
    (enq_L6m * 0.30) +
    (num_times_delinquent * 0.25) +
    (missed_payment_ratio * 0.25) +
    (active_loan_ratio * 0.20),
    3
) AS urgency_score
```

Weights are proportional to SHAP feature importances from the XGBoost model. The query filters `default_risk = 0` — these are current borrowers who haven't defaulted yet but are showing warning signals.

### Query 10 — Policy Impact Simulation

**NPA after policies formula:**
```sql
ROUND(
    (SUM(default_risk)
        - SUM(caught_by_any_policy * default_risk) * 0.70
        - SUM(saved_by_policy_3) * 0.20
    ) * 100.0 / COUNT(*), 2
) AS estimated_npa_after_policies
```

Assumptions: 70% true positive action rate (bank acts on 70% of flagged borrowers), 20% early intervention success rate. These match the assumptions in `policy_simulator.py`.

---

## 11. Dashboard Panels — What Each Visual Means

### Panel 01 — Portfolio Overview

| Visual | What It Shows | Business Meaning |
|--------|--------------|-----------------|
| 6-KPI strip | Total borrowers, default rate, approval rate, avg PD, high-risk count, expected loss | Monday morning health check — is the portfolio trending toward or away from the 22% national benchmark? |
| Risk band donut | Low/Medium/High Risk split by predicted PD | Distribution of model scores — a healthy portfolio has most borrowers in Low Risk |
| Income vs default (dual axis) | Default rate (bars) + % of total defaults (line) by income band | Shows both severity (rate) and materiality (contribution). A segment with 41% default rate but only 5% of borrowers contributes less to total NPA than a segment with 30% rate and 40% of borrowers. |
| EL stat bar | Total EAD, EL in ₹ Cr, EL rate %, excess NPA, NPA reduction estimate, model AUC | Single-row summary for senior management |

### Panel 03 — Model Performance

| Visual | What It Shows | Business Meaning |
|--------|--------------|-----------------|
| ROC curve | True positive rate vs false positive rate at every threshold | The further the curve bows toward top-left, the better the model discriminates risky from safe. AUC = area under this curve. |
| PD separation histogram | Distribution of predicted PD for safe vs risky borrowers overlaid | Good separation = two distinct peaks. Poor separation = heavily overlapping distributions. |
| Threshold sensitivity table | Approval %, recall %, precision %, false alarm %, F1 for thresholds 0.25–0.70 | The key business decision tool. Credit manager picks the row that best matches their risk appetite. |

### Panel 04 — SHAP Explainability

| Visual | What It Shows | Business Meaning |
|--------|--------------|-----------------|
| Global SHAP bar | Mean |SHAP value| for each feature, top 15 | Which features drive the model overall. Red = risk-increasing, gold = moderate, blue = protective or weak |
| Individual scorer | Heuristic PD for a manually entered borrower profile | Demonstrates the reason-code system. Shows a credit officer how to justify a decision to a borrower or regulator. |

### Panel 05 — Policy Simulator

| Visual | What It Shows | Business Meaning |
|--------|--------------|-----------------|
| 6-KPI strip (live) | Real-time approval rate, defaults prevented, safe rejected, EL after policy | Updated instantly as sliders change. Shows the consequence of every policy adjustment before it's implemented. |
| Before/after bar | Absolute counts for baseline vs policy | Visual comparison of what changes |
| RAAE ratio | Defaults prevented / safe rejected | Efficiency metric: how many defaults do you prevent per safe borrower you turn away? |

---

## 12. Why These Methods and Not Others

### 12.1 Why XGBoost over Logistic Regression?

**What we chose:** XGBoost (gradient boosted trees)

**Why:**
- Handles non-linear interactions natively (e.g. `enq_L6m` × `Age_Oldest_TL` compound risk)
- Class imbalance via `scale_pos_weight` — direct loss function adjustment
- Native SHAP compatibility (exact TreeExplainer, not approximation)
- Consistently outperforms linear models on tabular financial data

**Alternative: Logistic Regression** — Interpretable coefficients but assumes linear relationships. Credit risk features have strong non-linear interactions (e.g. high enquiries are very risky for thin-file borrowers but less concerning for established ones).

### 12.2 Why scale_pos_weight Tuning over SMOTE?

**What we chose:** Sweep `scale_pos_weight` from 0.3×–2.0× natural ratio

**Why:** `scale_pos_weight` adjusts the loss function — the model learns that missing a risky borrower costs more than wrongly flagging a safe one. This is principled and does not modify the training data.

**Alternative: SMOTE** — Generates synthetic minority class samples by interpolating between existing risky borrowers. Problem: synthetic borrowers may not represent real financial behaviour. A synthetic borrower with `enq_L6m=3` and `missed_payment_ratio=0.6` might be statistically plausible but financially unrealistic.

### 12.3 Why SHAP over LIME?

**What we chose:** SHAP TreeExplainer

**Why:** SHAP values are globally consistent — if `enq_L6m` has SHAP importance 1.183 for one prediction, that same contribution magnitude means the same thing for another prediction. LIME (Local Interpretable Model-Agnostic Explanations) is locally faithful but not globally coherent. Two identical borrowers can receive different LIME explanations depending on random sampling.

For a regulatory audit trail where every rejection must be documented, global consistency is mandatory. A credit officer cannot defend a decision if the explanation changes on re-run.

### 12.4 Why DuckDB for Gold, SQLite for Dashboard?

**Gold layer (DuckDB):** The star schema runs analytical queries (`GROUP BY`, window functions, JOINs across 4 tables). DuckDB is columnar and OLAP-optimised — 10–100× faster than SQLite for these workloads.

**Dashboard layer (SQLite):** Portability. The `credit_risk.db` file can be committed to git, opened in DB Browser for SQLite, queried from Excel via ODBC, or run in any SQL tool without server setup. All 10 queries use ANSI SQL and run on PostgreSQL unchanged.

### 12.5 Why Medallion Architecture?

**What we chose:** Bronze (raw) → Silver (clean) → Gold (analytics-ready)

**Why:** Each layer serves a different consumer. Bronze is for data engineers who need to debug ingestion. Silver is for data scientists who need clean features. Gold is for business analysts who need query-optimised tables. If a cleaning rule changes, Silver can be regenerated from Bronze without re-ingesting. If the Gold schema changes, it regenerates from Silver without re-cleaning.

**Alternative: Single-stage ETL** — Process raw data directly into final tables. Simpler but fragile. Any bug requires full re-ingestion from the source.

### 12.6 Why Star Schema?

**What we chose:** `fact_credit_risk` + 4 dimension tables

**Why:** Star schemas are the foundation of business intelligence and data warehousing. Every major BI tool (Power BI, Tableau, Looker) is optimised for star schema queries. A query like "default rate by income tier and education level" is a straightforward JOIN with filters — no aggregation logic required in the BI tool.

**Alternative: Flat denormalised table** — One wide table with everything. Simpler but introduces data redundancy and makes it impossible to add new dimensions without restructuring the entire table.

---

*This document covers every dataset, preprocessing step, calculation, formula, and method in the India Credit Risk Intelligence Platform. Every number shown in the dashboard is computed from source data using the methods described above. No values are hardcoded or fabricated.*

*Built by Aarya Patankar, 2026. Dataset: CIBIL-style synthetic/anonymised Indian borrower records.*