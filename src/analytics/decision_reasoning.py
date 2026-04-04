"""
decision_reasoning.py — SHAP → Business Language Explainer
────────────────────────────────────────────────────────────────────
Converts raw SHAP values into plain-English risk reasons
that a credit officer can read and act on.

This is what makes the model explainable to non-technical users.
────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from typing import Optional

# ── Feature → plain English mapping ──────────────────────────────────────────
FEATURE_LABELS = {
    "enq_L6m":               "Recent credit enquiries (last 6 months)",
    "enq_L12m":              "Credit enquiries in last 12 months",
    "tot_enq":               "Total lifetime credit enquiries",
    "num_times_delinquent":  "Missed payment count",
    "num_times_60p_dpd":     "Serious delinquencies (60+ days overdue)",
    "delinquency_score":     "Delinquency severity score",
    "missed_payment_ratio":  "Fraction of payments missed",
    "Age_Oldest_TL":         "Credit history length",
    "Total_TL":              "Total loan accounts",
    "active_loan_ratio":     "Active loan overextension",
    "loan_type_diversity":   "Loan portfolio diversity",
    "AGE":                   "Borrower age",
    "NETMONTHLYINCOME":      "Monthly income",
    "Gold_TL":               "Gold loan accounts",
    "Home_TL":               "Home loan accounts",
    "enq_per_credit_year":   "Enquiries per year of credit history",
    "delinquency_rate":      "Delinquency rate (misses per loan)",
    "enq_acceleration":      "Acceleration of recent enquiries",
    "severe_delinquency_ratio": "Proportion of severe delinquencies",
}

# ── Risk direction map (positive SHAP = more risky) ───────────────────────────
RISK_INCREASING_FEATURES = {
    "enq_L6m", "enq_L12m", "tot_enq", "num_times_delinquent",
    "num_times_60p_dpd", "delinquency_score", "missed_payment_ratio",
    "active_loan_ratio", "Gold_TL", "enq_per_credit_year",
    "delinquency_rate", "enq_acceleration", "severe_delinquency_ratio",
}

RISK_DECREASING_FEATURES = {
    "Age_Oldest_TL", "Total_TL", "loan_type_diversity",
    "AGE", "NETMONTHLYINCOME", "Home_TL",
}

# ── Business reason templates ─────────────────────────────────────────────────
RISK_TEMPLATES = {
    "enq_L6m": lambda v: f"Applied to {int(v)} lenders in the last 6 months — indicates financial stress",
    "num_times_delinquent": lambda v: f"Missed {int(v)} payment(s) historically — pattern of non-repayment",
    "num_times_60p_dpd": lambda v: f"Has {int(v)} serious 60+ day delinquency event(s) on record",
    "missed_payment_ratio": lambda v: f"Missed {v*100:.0f}% of payments — high missed payment frequency",
    "delinquency_score": lambda v: f"Delinquency severity score of {v:.0f} — above risk threshold",
    "active_loan_ratio": lambda v: f"{v*100:.0f}% of loans are currently active — heavily overextended",
    "Age_Oldest_TL": lambda v: f"Credit history of only {int(v)} months — insufficient track record" if v < 24 else f"Established {int(v)}-month credit history — positive stability signal",
    "NETMONTHLYINCOME": lambda v: f"Monthly income of ₹{v:,.0f} — adequate repayment capacity" if v > 20000 else f"Monthly income of ₹{v:,.0f} — limited repayment buffer",
    "Gold_TL": lambda v: f"Has {int(v)} gold loan(s) — may indicate collateral-backed stress borrowing",
    "enq_acceleration": lambda v: f"Enquiry pace has increased recently — accelerating credit search",
}

SAFE_TEMPLATES = {
    "Age_Oldest_TL": lambda v: f"{int(v)}-month credit history — strong repayment track record",
    "NETMONTHLYINCOME": lambda v: f"Monthly income of ₹{v:,.0f} — good repayment capacity",
    "loan_type_diversity": lambda v: f"Manages {int(v)} loan type(s) — diversified credit experience",
    "Total_TL": lambda v: f"{int(v)} total loan accounts — experienced borrower profile",
}


def explain_borrower(feature_values: dict,
                     shap_values: Optional[dict] = None,
                     predicted_pd: float = 0.5,
                     top_n: int = 5) -> dict:
    """
    Generate a plain-English explanation for a single borrower.

    Args:
        feature_values: dict of feature_name → actual value
        shap_values:    dict of feature_name → SHAP value (optional)
        predicted_pd:   model's predicted probability of default
        top_n:          number of top reasons to return

    Returns:
        dict with risk_level, reasons (list), protective_factors (list),
        summary string, and recommendation.
    """
    # Determine risk level
    if predicted_pd >= 0.60:
        risk_level = "HIGH RISK"
        risk_color = "#DC2626"
    elif predicted_pd >= 0.35:
        risk_level = "MEDIUM RISK"
        risk_color = "#D97706"
    else:
        risk_level = "LOW RISK"
        risk_color = "#16A34A"

    # Build reasons from SHAP if available, else use heuristics
    risk_reasons = []
    protective_factors = []

    if shap_values is not None:
        # Sort by absolute SHAP value
        sorted_features = sorted(shap_values.items(),
                                  key=lambda x: abs(x[1]), reverse=True)
        for feat, shap_val in sorted_features[:top_n * 2]:
            val = feature_values.get(feat, 0)
            label = FEATURE_LABELS.get(feat, feat)

            if shap_val > 0.05:  # pushes toward risky
                template = RISK_TEMPLATES.get(feat)
                reason = template(val) if template else f"{label}: {val} (increases risk)"
                risk_reasons.append({
                    "feature":    feat,
                    "label":      label,
                    "value":      val,
                    "shap":       round(shap_val, 3),
                    "direction":  "risk",
                    "reason":     reason,
                })
            elif shap_val < -0.05:  # pushes toward safe
                template = SAFE_TEMPLATES.get(feat)
                reason = template(val) if template else f"{label}: {val} (reduces risk)"
                protective_factors.append({
                    "feature":    feat,
                    "label":      label,
                    "value":      val,
                    "shap":       round(shap_val, 3),
                    "direction":  "safe",
                    "reason":     reason,
                })
    else:
        # Fallback: heuristic-based reasons (no SHAP)
        enq = feature_values.get("enq_L6m", 0)
        delinq = feature_values.get("num_times_delinquent", 0)
        tl_age = feature_values.get("Age_Oldest_TL", 120)
        income = feature_values.get("NETMONTHLYINCOME", 30000)
        dpd60  = feature_values.get("num_times_60p_dpd", 0)

        if enq >= 4:
            risk_reasons.append({"reason": RISK_TEMPLATES["enq_L6m"](enq), "feature":"enq_L6m","value":enq,"shap":0.9,"direction":"risk","label":"Recent enquiries"})
        if delinq >= 2:
            risk_reasons.append({"reason": RISK_TEMPLATES["num_times_delinquent"](delinq),"feature":"num_times_delinquent","value":delinq,"shap":0.65,"direction":"risk","label":"Missed payments"})
        if dpd60 >= 1:
            risk_reasons.append({"reason": RISK_TEMPLATES["num_times_60p_dpd"](dpd60),"feature":"num_times_60p_dpd","value":dpd60,"shap":0.45,"direction":"risk","label":"60+ DPD"})
        if tl_age >= 48:
            protective_factors.append({"reason": SAFE_TEMPLATES["Age_Oldest_TL"](tl_age),"feature":"Age_Oldest_TL","value":tl_age,"shap":-0.46,"direction":"safe","label":"Credit history"})
        if income >= 25000:
            protective_factors.append({"reason": SAFE_TEMPLATES["NETMONTHLYINCOME"](income),"feature":"NETMONTHLYINCOME","value":income,"shap":-0.20,"direction":"safe","label":"Monthly income"})

    # Trim to top_n
    risk_reasons       = risk_reasons[:top_n]
    protective_factors = protective_factors[:top_n]

    # Build recommendation
    if risk_level == "HIGH RISK":
        recommendation = "Decline or escalate to senior credit committee. Offer to reapply after 6 months if enquiry count reduces."
    elif risk_level == "MEDIUM RISK":
        recommendation = "Proceed with additional documentation. Request 6-month bank statement and proof of income stability."
    else:
        recommendation = "Approve under standard terms. No additional documentation required."

    # One-line summary
    top_reason = risk_reasons[0]["reason"] if risk_reasons else "No significant risk signals detected."
    summary = f"Predicted default probability: {predicted_pd*100:.1f}%. Primary concern: {top_reason}"

    return {
        "predicted_pd":       round(predicted_pd, 4),
        "predicted_pd_pct":   round(predicted_pd * 100, 1),
        "risk_level":         risk_level,
        "risk_color":         risk_color,
        "risk_reasons":       risk_reasons,
        "protective_factors": protective_factors,
        "recommendation":     recommendation,
        "summary":            summary,
    }


def batch_explain(df: pd.DataFrame,
                  pred_proba: np.ndarray,
                  top_n: int = 3) -> pd.DataFrame:
    """
    Generate compact explanations for all borrowers.
    Returns a DataFrame with top reason per borrower.
    Used for the watchlist / early intervention table.
    """
    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        pd_val = pred_proba[i]
        explanation = explain_borrower(row.to_dict(), predicted_pd=pd_val, top_n=1)
        top = explanation["risk_reasons"][0] if explanation["risk_reasons"] else {}
        rows.append({
            "Borrower ID":    row.get("borrower_id", i),
            "Predicted PD %": explanation["predicted_pd_pct"],
            "Risk Level":     explanation["risk_level"],
            "Primary Reason": top.get("reason", "—"),
            "Recommendation": explanation["recommendation"][:60] + "...",
        })
    return pd.DataFrame(rows)