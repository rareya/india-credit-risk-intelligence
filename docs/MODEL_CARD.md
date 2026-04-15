# Model Card
## India Credit Risk Intelligence — XGBoost Default Prediction Model

---

### Model Overview

| Field | Value |
|-------|-------|
| **Model type** | XGBoost binary classifier (Model B) |
| **Version** | v2 (`credit_risk_model_v2.pkl`) |
| **Task** | Probability of Default (PD) prediction |
| **Target** | `default_risk` (1 = high risk P1/P2, 0 = safe P3/P4) |
| **Training data** | 41,068 borrowers (80% of 51,336) |
| **Test data** | 10,268 borrowers (20% held-out) |
| **Decision threshold** | 0.50 |

---

### Intended Use

**Primary use case:** Credit risk scoring for Indian retail lending — NBFC and bank loan applications.

**Intended users:**
- Credit risk analysts using the Streamlit dashboard
- Credit officers reviewing individual loan applications
- Collections teams prioritising early intervention outreach
- Senior management running policy impact simulations

**Out-of-scope uses:**
- This model should not be used as the sole basis for credit decisions without human review
- Not validated for markets outside India
- Not designed for corporate/SME lending (optimised for retail borrowers)
- Should not replace regulatory-mandated credit assessment processes

---

### Model Architecture

**Algorithm:** XGBoost (`XGBClassifier`)

**Hyperparameters (Model B):**
```
n_estimators:      300
max_depth:         4
learning_rate:     0.05
subsample:         0.8
colsample_bytree:  0.8
scale_pos_weight:  1.43   (tuned from natural ratio of 2.85)
decision threshold: 0.50
```

**Why XGBoost over alternatives:**
- Handles class imbalance via `scale_pos_weight` natively
- Built-in SHAP compatibility for regulatory explainability
- Robust to missing values (handled via median imputation here)
- Faster inference than neural networks for real-time scoring
- Strong performance on tabular financial data (industry standard)

---

### Features

**15 raw input features + 4 engineered interaction terms:**

| Feature | Type | Business meaning |
|---------|------|-----------------|
| `enq_L6m` | int | Credit enquiries in last 6 months |
| `enq_L12m` | int | Credit enquiries in last 12 months |
| `tot_enq` | int | Total lifetime enquiries |
| `num_times_delinquent` | int | Total missed payments ever |
| `num_times_60p_dpd` | int | Serious 60+ DPD delinquency count |
| `delinquency_score` | float | Weighted delinquency severity |
| `missed_payment_ratio` | float | Missed payments / total loans |
| `Age_Oldest_TL` | int | Credit history length (months) |
| `Total_TL` | int | Total loan accounts ever |
| `active_loan_ratio` | float | Active loans / total loans |
| `loan_type_diversity` | int | Count of distinct loan types |
| `AGE` | int | Borrower age |
| `NETMONTHLYINCOME` | float | Monthly income (INR) |
| `Gold_TL` | int | Gold loan accounts |
| `Home_TL` | int | Home loan accounts |
| `enq_per_credit_year` | float | *Engineered:* enquiries per year of history |
| `delinquency_rate` | float | *Engineered:* delinquencies per loan |
| `enq_acceleration` | float | *Engineered:* recent vs historical enquiry pace |
| `severe_delinquency_ratio` | float | *Engineered:* 60+ DPD fraction of all misses |

**Excluded features:**
- `Credit_Score` — excluded due to target leakage (AUC = 0.9998 on one feature). See `docs/LEAKAGE_ANALYSIS.md`.

---

### Performance Metrics

**Test set (10,268 held-out borrowers):**

| Metric | Model A (original) | Model B (deployed) |
|--------|-------------------|--------------------|
| AUC | 0.8985 | **0.8994** |
| Precision | 0.5931 | **0.6907** |
| Recall | 0.8346 | **0.7102** |
| F1 | 0.6935 | **0.7003** |
| False alarm rate | 41.0% | **31.0%** |

**Cross-validation (5-fold stratified on training set):**
- CV AUC: 0.8921 ± 0.0038 — stable, no significant overfitting

**Confusion matrix (test set, threshold = 0.50):**
```
                Predicted Safe   Predicted Risky
Actual Safe          6,074              848
Actual Risky           773            2,573
```

**Why Model B over Model A:**
Model A's `scale_pos_weight = 2.85` (natural class ratio) gave high recall (83%) but poor precision (59%) — 4 in 10 flagged borrowers were actually safe. For a lender, rejecting safe borrowers is both a revenue loss and a reputational risk. Model B (`scale_pos_weight = 1.43`) reduces false alarms from 41% to 31% while maintaining 71% recall — acceptable for business deployment.

---

### Feature Importance (SHAP)

Top 5 predictors by mean |SHAP value|:

| Rank | Feature | SHAP Importance | Direction |
|------|---------|----------------|-----------|
| 1 | `enq_L6m` | 1.183 | High = more risky |
| 2 | `num_times_delinquent` | 0.654 | High = more risky |
| 3 | `Age_Oldest_TL` | 0.460 | High = less risky |
| 4 | `Total_TL` | 0.242 | High = less risky |
| 5 | `delinquency_score` | 0.210 | High = more risky |

**Key finding:** Recent credit enquiries (`enq_L6m`) is the strongest predictor — stronger than delinquency history or income. Borrowers applying to 4+ lenders in 6 months default at 4× baseline.

---

### Limitations and Known Biases

**Known limitations:**

1. **New-to-credit borrowers** — borrowers with less than 12 months of credit history have insufficient behavioural signals. The model may underestimate risk for this segment. (~29% of portfolio has <24 month history.)

2. **Income shock events** — sudden job loss or medical emergencies will not appear in features until delinquency begins. The model cannot predict first-time default due to unforeseen events.

3. **Seasonal patterns** — Indian lending has seasonal default variation (agricultural cycles, Diwali credit peaks). This model does not include time-series features and may perform differently across seasons.

4. **EAD and LGD simplification** — Expected Loss calculations use simplified assumptions (LGD = 45%, EAD = 12 × monthly income). These are proxies, not actuals.

5. **Synthetic/anonymised data** — trained on CIBIL-style synthetic data. Performance on live production data may differ. Recommend 90-day controlled pilot before full deployment.

**Potential biases:**

- The model uses `AGE` as a feature. Younger borrowers (18-25) show higher default rates in this dataset. Care should be taken that this does not result in systematic discrimination against young borrowers where the causal factor is credit history length, not age itself.
- `NETMONTHLYINCOME` may systematically disadvantage lower-income borrowers. Policy simulations should test approval rates across income bands before deployment.

---

### Regulatory Notes

**RBI Fair Practices Code compliance:**
- Every prediction is accompanied by SHAP-derived plain-English reason codes (Panel 4 of the dashboard)
- Feature-level explanations are logged per decision
- No protected characteristics (religion, caste, region) are used as features
- Threshold sensitivity table provided for credit officers to understand the decision boundary

**Audit trail:** All SHAP values, model predictions, and feature inputs are saved to `data/gold/exports/ml/` and queryable via SQLite.

---

### How to Use

```python
import pickle
import pandas as pd

with open("data/processed/credit_risk_model_v2.pkl", "rb") as f:
    model = pickle.load(f)

# Prepare features (see feature list above)
X = pd.DataFrame([{
    "enq_L6m": 5, "num_times_delinquent": 3, "Age_Oldest_TL": 18,
    # ... all 15 features
}])

# Predict probability of default
pd_score = model.predict_proba(X)[:, 1][0]
risk_label = "HIGH" if pd_score >= 0.50 else "MEDIUM" if pd_score >= 0.35 else "LOW"
```

---

*Model trained by Aarya Patankar, 2026.*
*Dataset: CIBIL-style synthetic/anonymised Indian borrower records.*
*Not financial advice. All estimates require validation before production deployment.*