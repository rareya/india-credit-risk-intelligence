# Metrics Definition Glossary
## India Credit Risk Intelligence Platform

---

### Core Portfolio Metrics

**Default Rate**
Percentage of borrowers who have defaulted on their loan obligation.
`Default Rate = Defaults / Total Borrowers × 100`
*Benchmark: India NBFC average ~22%. This portfolio: 26%.*

**Approval Rate**
Percentage of applications approved under a given underwriting policy.
`Approval Rate = Approved Applications / Total Applications × 100`
*Varies by policy threshold — see Policy Simulator.*

**High-Risk Exposure %**
Percentage of the portfolio with predicted default probability ≥ 50%.
`High-Risk Exposure = Borrowers with PD ≥ 0.50 / Total × 100`

---

### Risk Metrics

**PD — Probability of Default**
The model's predicted likelihood that a borrower will default within the next 12 months, expressed as a percentage (0–100%).
- Computed by the XGBoost classifier
- Calibrated using 5-fold stratified cross-validation
- AUC = 0.8994 on held-out test set

**EL — Expected Loss**
A standard banking risk metric representing the average loss expected from a portfolio.
`EL = PD × LGD × EAD`

- **PD** = Probability of Default (model output, 0–1)
- **LGD** = Loss Given Default — assumed 45% for unsecured Indian lending (industry standard)
- **EAD** = Exposure at Default — proxied as 12 × monthly income (simplified assumption)

*Note: This is a proxy calculation for analytical purposes. Production EL computation requires actual loan amounts and collateral values.*

**EL Rate %**
Expected Loss as a percentage of total Exposure at Default.
`EL Rate = Total EL / Total EAD × 100`

**Segment Default Contribution %**
The share of total portfolio defaults attributable to a given segment.
`Contribution = Segment Defaults / Total Portfolio Defaults × 100`
Used to identify which segments are disproportionately driving NPA.

---

### Risk Bands

| Band | PD Range | Interpretation |
|------|----------|---------------|
| Low Risk | 0% – 25% | Standard processing. No additional scrutiny. |
| Medium Risk | 25% – 50% | Additional documentation recommended. |
| High Risk | 50% – 100% | Flag for senior review or decline. |

---

### Model Performance Metrics

**AUC — Area Under the ROC Curve**
Measures the model's ability to rank risky borrowers above safe borrowers.
- 0.50 = random (no skill)
- 0.70 = acceptable
- 0.80 = good
- **0.8994 = strong** (this model)

**Precision**
Of all borrowers flagged as risky, what fraction are actually risky?
`Precision = True Positives / (True Positives + False Positives)`
*Model B: 69.1% — 31% false alarm rate.*

**Recall (Sensitivity)**
Of all actual risky borrowers, what fraction does the model catch?
`Recall = True Positives / (True Positives + False Negatives)`
*Model B: 71.0% — catches 7 in 10 risky borrowers.*

**F1 Score**
Harmonic mean of Precision and Recall. Best single metric for imbalanced classification.
`F1 = 2 × (Precision × Recall) / (Precision + Recall)`
*Model B: 0.700*

**PR-AUC — Precision-Recall Area Under Curve**
More informative than ROC-AUC for highly imbalanced datasets (26% default rate).
Measures the tradeoff between precision and recall across all thresholds.

---

### Policy Metrics

**Defaults Prevented**
Number of confirmed-risky borrowers rejected under a given policy.
Compared against a baseline policy (no restrictions).

**Approvals Lost**
Number of safe borrowers incorrectly rejected under a tighter policy.
The business cost of higher precision.

**Risk-Adjusted Approval Efficiency**
`RAAE = (Defaults Prevented) / (Approvals Lost + 1)`
Higher = better. A policy that prevents 500 defaults while losing 100 safe approvals has RAAE = 5.0.

**Threshold Sensitivity**
How key metrics (approval rate, recall, false alarm rate) change as the decision threshold moves from 0.2 to 0.7. Used to select the optimal operating point for the business.

---

### Indian Lending Specific

**DPD — Days Past Due**
Number of days a borrower is overdue on a payment.
- **30 DPD**: Early delinquency — optimal intervention point
- **60+ DPD**: Serious delinquency — recovery rate drops significantly
- **90+ DPD**: NPA classification under RBI guidelines

**NPA — Non-Performing Asset**
An asset (loan) that has stopped generating income because the borrower has defaulted.
Under RBI guidelines: loans where interest or principal is overdue for 90+ days.

**Gold TL — Gold Loan Tradeline**
A loan secured against physical gold jewellery. Common in India as a last-resort collateral financing option. High Gold_TL count combined with high enquiry count is a strong compound stress signal.

**DTI — Debt-to-Income Ratio**
`DTI = Total Monthly Debt Obligations / Gross Monthly Income`
Proxied in this model using active_loan_ratio combined with income features.