# Data Leakage Analysis
## India Credit Risk Intelligence — Two Independent Leakage Findings

---

### Summary

This project has caught target leakage **twice**, using two different methods, at two different stages of the pipeline's evolution:

1. **Original finding (Model B era):** `Credit_Score` achieved AUC = 0.9998 as a single predictor — detected with an ad hoc single-feature AUC check.
2. **Second finding (governance pipeline era):** `cibil_score` / `cibil_band`, when added to the conservative feature set to form an "expanded" candidate model, reproduced the same signature — AUC 0.9998, PR-AUC 0.9995 — this time caught automatically by a formal feature-governance framework built specifically because the first finding happened.

Both features were excluded. The deployed model (`conservative_xgb_v1`) uses neither.

---

### Finding 1 — `Credit_Score` (historical)

During early model development, `Credit_Score` achieved **AUC = 0.9998** when used as the sole predictor of the target. No single bureau feature predicts defaults with near-perfect accuracy in real credit risk modelling — this is the signature of target leakage, not a strong signal.

**Root cause:** `Credit_Score` and the target (`Approved_Flag`, later encoded as `default_risk`) were both outputs of the same upstream CIBIL batch scoring process. Predicting one from the other is circular, not predictive.

**Detection method:** a manual single-feature AUC check in the Model B pipeline (`credit_score_myth_proof`, in an earlier version of `run_ml_model.py`) — any feature scoring above ~0.99 AUC alone was flagged for investigation.

**Resolution:** excluded from the model entirely. The Model B behavioural model (15 raw features, no `Credit_Score`) achieved AUC 0.8994 — the number reported elsewhere in this repo as the "Model B benchmark."

**Note on current schema:** the gold-layer column naming has since changed (this project now uses `cibil_score`, not `Credit_Score`, among other renames). The finding above is preserved as the project's origin story — it's the reason leakage-consciousness became a first-class part of this pipeline's design, not a live bug in the current schema.

---

### Finding 2 — `cibil_score` / `cibil_band` (current governance pipeline)

The redesigned pipeline formalized leakage prevention into an explicit governance framework (`src/modeling/feature_governance.py`, `feature_semantics_audit.py`, `prediction_time_provenance.py`) rather than a one-off check. That framework caught the same underlying problem again, independently, in the new schema.

**What happened:** `feature_governance.py` defines three tiers over the 30+ gold-layer features:

| Tier | Examples | Handling |
|---|---|---|
| **Definite exclusions** | `borrower_id`, `default_risk`, `risk_grade`, `risk_grade_numeric`, `risk_band` | Never eligible for ML — identifiers or directly target-derived |
| **Review features** | `cibil_score`, `cibil_band`, `recent_enquiries_6m`, `recent_enquiries_12m`, `total_enquiries` | Strong signal, provenance not yet confirmed — quarantined pending review |
| **Conservative features** | 30 features, none of the above two tiers | Cleared for the deployed model |

To test whether the review features were safe to include, `run_ml_model.py` trained three models on identical data:

| Model | Features | ROC-AUC | PR-AUC | Accuracy |
|---|---|---|---|---|
| Model 0 — Logistic Baseline | 9 (incl. `cibil_score`) | 0.7042 | 0.4227 | 59.3% |
| **Model 1 — Conservative XGBoost** | 30 (no `cibil_score`/`cibil_band`) | 0.7987 | 0.5895 | 78.5% |
| **Model 2 — Expanded XGBoost** | 30 + `cibil_score` + `cibil_band` | **0.9998** | **0.9995** | **99.4%** |

Model 2's jump from 0.80 to 0.9998 ROC-AUC by adding two features is the same leakage signature as Finding 1 — no genuine behavioural signal produces that jump. A separate sweep of raw gold-layer columns also found **`risk_grade_numeric` alone scores AUC = 1.0** — expected, since `default_risk` is literally derived from `risk_grade` in `build_gold.py` (`P1`/`P2` → 0, `P3`/`P4` → 1, with an explicit consistency check enforcing it). `risk_grade`, `risk_grade_numeric`, and `risk_band` were excluded at the governance-registry stage for exactly this reason, before any model was trained on them.

**Resolution:** `final_model_selection.py` formally rejected the expanded model:

```
Model 2 — Expanded XGBoost: REJECT
"ROC-AUC approximately 0.9998 combined with single-feature AUC of 1.0
for risk_grade_numeric indicates target/proxy leakage."
```

Conservative XGBoost (`conservative_xgb_v1`) was selected as the final model instead — deliberately choosing the *lower*-scoring, leakage-clean model over the artificially high-scoring one.

*(Note: the 0.7987 ROC-AUC for Conservative XGBoost above is from this specific three-way governance comparison run inside `run_ml_model.py` — a different training configuration than the final persisted `conservative_xgb_v1` artifact, which scores 0.8971 ROC-AUC on its own held-out evaluation. See `src/modeling/final_model/model_metadata.json` for the deployed model's real numbers. The two shouldn't be conflated — this comparison run exists only to demonstrate the leakage gap, not to report final model performance.)*

---

### Why `cibil_score` is flagged but not automatically banned

Unlike `risk_grade` (provably, definitionally derived from the target — verifiable directly in `build_gold.py`), `cibil_score` is *not* proven to be circular — it's flagged because it's plausible a real bureau CIBIL score would be this predictive in production, or because in this specific dataset it may have been generated close in time to (or from a shared source with) the outcome label. `feature_governance.py` records the honest distinction:

> "Extremely strong predictive signal. Must confirm that the CIBIL score represents information available before the lending decision."

That's a **provenance question, not a statistical one** — no amount of AUC analysis on this static dataset can confirm whether a production CIBIL pull would happen before or after the credit decision it's meant to inform. The governance framework's answer was conservative: quarantine it, test what including it does (Finding 2, above), see the leakage-shaped result, and exclude it. This is the correct default posture when provenance can't be confirmed from the data alone — treat an unconfirmed pre-decision signal as untrustworthy rather than assume it's fine.

`recent_enquiries_6m`, `recent_enquiries_12m`, and `total_enquiries` went through the same review-tier quarantine but *were* cleared — `prediction_time_provenance.py` confirmed all 30 conservative features (these three included) pass an automated prediction-time availability screen (30/30 PASS, `data/gold/exports/analytics/ml/provenance/prediction_time_provenance_summary.json`). Per that script's own governance language:

> "PASS means all conservative features passed the automated prediction-time provenance screen. This does not constitute independent verification of a production data feed."

That caveat is intentional and should not be dropped in any summary of this project: passing this audit means nothing in the feature's *definition* implies it comes from the future — it is not a substitute for confirming the real production data pipeline's timing.

---

### Governance pipeline, end to end

| Stage | Script | Result |
|---|---|---|
| Feature registry | `feature_governance.py` | 5 definite exclusions, 5 review features, 30 conservative features |
| Semantic direction check | `feature_semantics_audit.py` | PASS — 0 direction mismatches, 0 strong single-feature signals among the 30 conservative features |
| Prediction-time provenance | `prediction_time_provenance.py` | PASS — 30/30 features, 0 failures |
| Model comparison | `run_ml_model.py` | Expanded model rejected (leakage signature), Conservative model selected |
| Final governance decision | `final_model_selection.py` | Writes `final_model_summary.json` — the single script and single file that records whether the model is approved |
| Cross-validation | `cross_validate.py` | 0.8962 ± 0.0042 ROC-AUC, 5-fold |
| Subgroup robustness | `subgroup_robustness.py` | Minimum subgroup AUC ~0.61 (`cibil_band = very_good`) — a real weak spot, not a leakage issue, but worth pairing with this doc in any review |

None of these stages found a reason to include `cibil_score`, `cibil_band`, or any of the definite exclusions in the deployed model.

**On `final_model_summary.json` specifically:** this file is generated by exactly one script — `src/modeling/final_model_selection.py` (writes to `data/gold/exports/analytics/ml/final/final_model_summary.json`). Nothing else in the pipeline touches it. If this file's `decision` field ever looks wrong or stale, the fix is always to re-run that one script — never to hand-edit the JSON, since the next run will overwrite manual edits anyway.

---

### Lessons for production systems (updated)

1. **Automate the single-feature AUC check as a standing governance gate, not a one-off script.** The first leakage finding was caught by hand; the second was caught automatically because the check became infrastructure (`feature_governance.py` + `run_ml_model.py`'s three-way comparison). That's the right direction of travel.
2. **A feature can be legitimately excluded without being provably circular.** `risk_grade` is *definitionally* derived from the target — that's provable from code. `cibil_score` is *suspiciously* predictive with unconfirmed provenance — that's a judgment call under uncertainty, and the conservative default (exclude until confirmed) is the correct one for a lending model.
3. **"PASS" on an automated provenance screen is not the same as verified production timing.** Every summary of this project should preserve that distinction — it's the difference between "the feature's definition doesn't imply future information" and "we confirmed the real data feed."
4. **Track leakage findings as they recur, not just the first one.** A single "we found leakage once and fixed it" narrative undersells a pipeline that's now designed to catch it again — which it did.

---

*This analysis reflects the current `src/modeling/` governance pipeline (feature registry, semantics audit, prediction-time provenance, final model selection) as of the `conservative_xgb_v1` model. The original Model B `Credit_Score` finding is preserved above as the project's origin story.*
