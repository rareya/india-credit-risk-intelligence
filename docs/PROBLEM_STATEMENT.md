# Problem Statement

## India Credit Risk Intelligence — Behavioural Default Prediction

---

### Business Context

A mid-size Non-Banking Financial Company (NBFC) in India is facing a
Non-Performing Asset (NPA) rate of **26%** — four percentage points above the
national average of 22%. Every percentage point of NPA above target represents
crores of rupees in provisioning costs, regulatory scrutiny under RBI guidelines,
and lost lending capacity.

The existing credit approval process relies primarily on bureau credit scores.
Three structural problems make this insufficient:

---

### The Three Gaps

**Gap 1 — Credit scores are lagging and opaque**

Credit bureau scores are updated monthly at best. A borrower who began missing
payments three weeks ago looks identical on paper to one who has never missed
a payment. The score is a backward-looking summary — it cannot detect the
behavioural signals that precede default by weeks or months.

Furthermore, in this dataset, the `Credit_Score` column was found to achieve
**AUC = 0.9998** when used to predict `default_risk` — statistically impossible
in real credit risk unless the score was derived from the target variable itself.
This circular derivation is a common data quality failure in Indian NBFC pipelines
and renders the score useless as a standalone predictor.

**Gap 2 — No early warning system**

Once a borrower reaches 60+ days past due (DPD), recovery rates drop
significantly. The portfolio in this dataset shows a 60 DPD → default transition
rate of ~65%. There is no structured mechanism to identify accounts trending
toward delinquency at the 30 DPD stage, when intervention is most cost-effective.

Collections teams currently operate reactively — contacting borrowers after
delinquency is confirmed rather than before the escalation point.

**Gap 3 — No interpretable individual decisions**

When a credit officer needs to justify a rejection to a borrower, to a compliance
team, or to a regulator under the RBI's Fair Practices Code, "the model
said no" is legally and operationally insufficient.

The existing scoring model provides no feature-level explanation for individual
decisions — making it impossible to audit, challenge, or improve specific rejections.

---

### What This Project Builds

An end-to-end behavioural credit risk intelligence system on **51,336 CIBIL-style
Indian borrower records**, addressing each gap directly:

| Gap | Solution Built |
|-----|---------------|
| Lagging credit scores | XGBoost model on 15 raw behavioural features — no bureau score used |
| No early warning | Delinquency funnel + SQL-powered early intervention candidate ranking |
| No interpretability | SHAP values per prediction — feature-level explanation for every decision |

---

### Key Finding

> *Recent credit enquiry behaviour (enq_L6m) is the single strongest predictor of
> default — stronger than delinquency history, income, or age. Borrowers who apply
> to 4+ lenders in a 6-month window default at 4x the baseline rate. This signal
> is not captured by any bureau score — it is a real-time behavioural fingerprint.*

---

### Target Users

| User | How They Use This |
|------|------------------|
| Credit Risk Manager | Portfolio-level NPA monitoring via Streamlit dashboard |
| Credit Officer | Individual loan application decisioning with SHAP explanation |
| Collections Team | Weekly early intervention list from SQL Query 09 |
| Senior Management | Policy impact simulation and NPA reduction estimates |
| Regulator / Auditor | Full audit trail — every feature, every SHAP value, documented |

---

### Scope and Constraints

**In scope:**
- Probability of default prediction (binary classification)
- Feature importance and individual explainability (SHAP)
- Three data-driven credit policy recommendations
- Early intervention candidate identification
- Policy NPA reduction simulation

**Out of scope:**
- Loss Given Default (LGD) estimation
- Exposure at Default (EAD) calculation
- Real-time API deployment (model saved as `.pkl` for integration)
- Regulatory capital calculation (IRBA / Basel III)

**Known limitations:**
- New-to-credit borrowers (<12 months history) have insufficient behavioural data
- Income shock events (job loss, medical emergency) will not appear in features
  until delinquency begins
- All NPA reduction estimates require validation via a 90-day controlled pilot
  before full policy rollout

---

*Built as an intern data analyst project. Dataset is CIBIL-style synthetic/anonymised
Indian borrower data. All findings are analytical — not financial advice.*