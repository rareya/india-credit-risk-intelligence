# Business Case
## India Credit Risk Intelligence Platform

---

### Positioning

**A business-facing analytics platform for credit risk managers to monitor portfolio health, identify risky borrower segments, explain model decisions, and simulate underwriting policy changes.**

This is not a machine learning experiment. It is a decision-support system designed to answer the questions a credit risk team asks every Monday morning.

---

### Who Uses This

| User | How They Use It |
|------|----------------|
| Credit Risk Manager | Portfolio health dashboard, NPA monitoring, segment alerts |
| Credit Officer | Per-borrower risk score + plain-English explanation |
| Collections Team | Weekly early intervention watchlist ranked by urgency |
| Senior Management | Policy simulator — "what if we tighten the enquiry rule?" |
| Compliance / Audit | Full audit trail — SHAP values per decision |

---

### The Three Business Problems

**Problem 1 — NPA above benchmark**
The analysed portfolio shows a 26% default rate — 4pp above India's 22% national NBFC average. Each 1pp excess NPA means crores in provisioning costs and RBI scrutiny.

**Problem 2 — Lagging credit scores**
Bureau scores update monthly. A borrower who began missing payments last week looks safe on paper. This platform captures real-time behavioural signals — recent enquiry acceleration, delinquency patterns — that bureau scores miss entirely.

**Problem 3 — No explainability**
Under RBI's Fair Practices Code, credit rejections must be explainable to the borrower. "The model said no" is legally insufficient. This platform generates plain-English reasons for every decision.

---

### Why India Context Matters

- Gold loans (`Gold_TL`) are a uniquely Indian stress signal — borrowers pledging jewellery are often facing financial difficulty
- Regional NBFC concentration risk differs significantly from Western markets
- India's credit bureau coverage is lower — behavioural signals compensate for thin credit files
- Seasonal default patterns (Diwali, agricultural cycles) affect risk differently than Western credit calendars

---

### Expected Business Impact

Based on simulation across 51,336 borrowers:

| Policy Change | Est. NPA Reduction | Borrowers Affected |
|---------------|-------------------|-------------------|
| Enquiry-based auto-flag (≥4 in 6m) | 8–11% | ~12% of portfolio |
| Thin-file premium pricing (<2yr history) | 6–9% | ~29% of portfolio |
| 30-DPD early intervention programme | 5–8% | ~18% of portfolio |
| **Combined (non-additive)** | **18–24%** | — |

Combined NPA reduction from 26% → ~21% on a portfolio of 51,336 borrowers represents approximately 2,500–3,000 fewer defaults.

---

### Decisions This Platform Supports

1. **Approve or escalate** an individual loan application — with a reason
2. **Identify which segments** to prioritise for collections outreach
3. **Simulate policy changes** before rolling them out — see impact on approval rate, defaults prevented, and expected loss
4. **Monitor portfolio health** weekly — catch concentration risk before it becomes NPA

---

*Platform built on 51,336 CIBIL-style Indian borrower records. All estimates are analytical and should be validated with a 90-day pilot before full deployment.*