# Data Leakage Analysis
## India Credit Risk Intelligence — A Critical Finding

---

### What Happened

During model development, `Credit_Score` achieved **AUC = 0.9998** when used as the sole predictor of `default_risk`. This is statistically impossible in real credit risk modelling — no single bureau feature can predict defaults with near-perfect accuracy.

This is a case of **target leakage**: the feature was almost certainly computed from (or heavily influenced by) the target variable during data preprocessing upstream of this project.

---

### What Leakage Looks Like

```
Normal result:  Credit_Score → AUC 0.70–0.80  (expected for a bureau score)
Leaked result:  Credit_Score → AUC 0.9998      (the feature IS the target, encoded differently)
```

In this dataset, `Credit_Score` appears to be a risk score derived during the same preprocessing pipeline that created `Approved_Flag` (the target variable). Both are outputs of the same upstream CIBIL scoring system — meaning predicting one from the other is circular.

---

### How It Was Detected

The detection code in `src/analytics/run_ml_model.py` (`credit_score_myth_proof` function):

```python
auc1 = roc_auc_score(y_test, m1.predict_proba(X_score_test)[:, 1])

if auc1 > 0.99:
    leakage_flag = " ⚠ LEAKAGE SUSPECTED"
    print("⚠ DATA LEAKAGE WARNING")
    print("Credit_Score AUC = {auc1:.4f} — near-perfect on one feature.")
    print("This strongly suggests Credit_Score was derived FROM default_risk")
```

The threshold of 0.99 AUC on a single feature is the detection heuristic — any real-world credit feature should top out around 0.75–0.80 AUC on its own.

---

### Why This Matters

If `Credit_Score` were used in the model:
1. The model would appear to achieve near-perfect accuracy (AUC ~0.999)
2. In production, when a new borrower arrives without a pre-computed risk score, the model would have no meaningful feature
3. Reported performance would be completely misleading to stakeholders and regulators
4. The model would be useless for the stated goal: identifying at-risk borrowers before their bureau score reflects the risk

This is a common failure mode in Indian NBFC data pipelines, where bureau scores and derived risk flags often originate from the same batch processing job.

---

### How It Was Handled

`Credit_Score` was **excluded from the final model**. The behavioural model uses only 15 raw features:

| Feature Group | Features |
|---------------|----------|
| Delinquency behaviour | `num_times_delinquent`, `num_times_60p_dpd`, `delinquency_score`, `missed_payment_ratio` |
| Loan portfolio | `Total_TL`, `active_loan_ratio`, `loan_type_diversity`, `Age_Oldest_TL` |
| Credit seeking | `enq_L6m`, `enq_L12m`, `tot_enq` |
| Demographics | `AGE`, `NETMONTHLYINCOME` |
| India-specific | `Gold_TL`, `Home_TL` |

None of these features are derived from or circularly related to `default_risk`.

---

### Final Model Comparison

| Model | Features | AUC | Notes |
|-------|----------|-----|-------|
| Credit Score Only | 1 | 0.9998 | ⚠ Leakage — do not trust |
| Behavioural Model | 15 | **0.8994** | ✓ Genuine signal, deployable |
| Combined | 16 | ~0.9998 | Dominated by leaked feature |

The **behavioural model at AUC 0.8994 is the valid result**. This represents strong predictive power achieved entirely from raw transactional signals, with no dependence on a potentially circular bureau score.

---

### The Key Finding This Unlocked

By removing the leaking feature, the analysis could identify which signals genuinely predict default. The result was surprising:

> **Recent credit enquiries (`enq_L6m`) are the strongest predictor — stronger than any delinquency history, income, or age feature.**

Borrowers who apply to 4+ lenders in a 6-month window default at 4× the baseline rate. This signal is not captured by any bureau score because bureau scores are backward-looking (updated monthly at best). `enq_L6m` is a real-time behavioural fingerprint — visible right now, before any payment is missed.

This finding would have been invisible had the leaking `Credit_Score` been left in the model.

---

### Lessons for Production Systems

1. **Always run single-feature AUC checks** before feature selection. AUC > 0.95 on one feature = investigate immediately.
2. **Trace feature provenance** — know whether each column was computed before or after the target was defined.
3. **Exclude scores derived from the target** in any model that aims to replace that score.
4. **The leakage detection code** should be part of any ML pipeline quality gate.

---

*This analysis was performed as part of the India Credit Risk Intelligence project.*
*Dataset: CIBIL-style synthetic/anonymised Indian borrower records.*