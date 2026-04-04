"""
policy_simulator.py — Underwriting Policy What-If Simulator
────────────────────────────────────────────────────────────────────
Allows credit risk teams to test policy changes and see the
impact on approval rates, expected losses, and default prevention
BEFORE rolling out changes.

This is the key feature that differentiates a DA/BA platform
from a generic ML notebook.
────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

LGD      = 0.45
EAD_MULT = 12


@dataclass
class PolicyConfig:
    """
    Encapsulates a set of underwriting policy rules.
    Default values = current policy (no extra restrictions).
    """
    # Enquiry-based rule (Policy 1)
    max_enq_6m: int = 999              # reject if enq_L6m > this

    # Thin-file rule (Policy 2)
    min_credit_history_months: int = 0 # reject if Age_Oldest_TL < this

    # Delinquency rules
    max_delinquencies: int = 999       # reject if num_times_delinquent > this
    max_60dpd: int = 999               # reject if num_times_60p_dpd > this

    # Income floor
    min_monthly_income: float = 0      # reject if NETMONTHLYINCOME < this

    # Model score cutoff
    max_predicted_pd: float = 1.0     # reject if predicted PD > this (0–1)

    # Active loan cap
    max_active_loan_ratio: float = 1.0 # reject if active_loan_ratio > this

    name: str = "Custom Policy"


# ── Pre-defined policy presets ────────────────────────────────────────────────
CURRENT_POLICY = PolicyConfig(name="Current Policy (No Restrictions)")

CONSERVATIVE_POLICY = PolicyConfig(
    max_enq_6m=3,
    min_credit_history_months=12,
    max_delinquencies=2,
    max_60dpd=0,
    min_monthly_income=10000,
    max_predicted_pd=0.45,
    name="Conservative Policy"
)

MODEL_B_POLICY = PolicyConfig(
    max_enq_6m=4,
    min_credit_history_months=24,
    max_delinquencies=5,
    max_predicted_pd=0.50,
    name="Model B Recommended (3 Policies)"
)

AGGRESSIVE_GROWTH_POLICY = PolicyConfig(
    max_enq_6m=999,
    min_credit_history_months=0,
    max_predicted_pd=0.65,
    name="Growth Mode (Relaxed)"
)


def apply_policy(df: pd.DataFrame,
                 pred_proba: np.ndarray,
                 policy: PolicyConfig) -> pd.Series:
    """
    Apply a policy config to a borrower dataset.
    Returns a boolean Series: True = APPROVED, False = REJECTED.
    """
    n = len(df)
    approved = pd.Series([True] * n, index=df.index)

    # Rule 1: Enquiry cap
    if "enq_L6m" in df.columns:
        approved &= df["enq_L6m"] <= policy.max_enq_6m

    # Rule 2: Credit history floor
    if "Age_Oldest_TL" in df.columns:
        approved &= df["Age_Oldest_TL"] >= policy.min_credit_history_months

    # Rule 3: Delinquency cap
    if "num_times_delinquent" in df.columns:
        approved &= df["num_times_delinquent"] <= policy.max_delinquencies

    # Rule 4: 60+ DPD cap
    if "num_times_60p_dpd" in df.columns:
        approved &= df["num_times_60p_dpd"] <= policy.max_60dpd

    # Rule 5: Income floor
    if "NETMONTHLYINCOME" in df.columns:
        approved &= df["NETMONTHLYINCOME"].fillna(0) >= policy.min_monthly_income

    # Rule 6: Model score cutoff (the most powerful rule)
    approved &= pd.Series(pred_proba <= policy.max_predicted_pd, index=df.index)

    # Rule 7: Active loan ratio cap
    if "active_loan_ratio" in df.columns:
        approved &= df["active_loan_ratio"] <= policy.max_active_loan_ratio

    return approved


def simulate_policy(df: pd.DataFrame,
                    pred_proba: np.ndarray,
                    policy: PolicyConfig,
                    baseline_policy: Optional[PolicyConfig] = None) -> dict:
    """
    Simulate the effect of a policy on the full borrower population.

    Returns a dict with:
      - approval_rate, rejection_rate
      - defaults_in_approved (false negatives)
      - defaults_prevented (risky borrowers correctly rejected)
      - approvals_lost (safe borrowers incorrectly rejected)
      - expected_loss before/after
      - segment impact
    """
    n = len(df)
    actual_defaults = df["default_risk"].values if "default_risk" in df.columns else np.zeros(n)

    # Apply policies
    approved = apply_policy(df, pred_proba, policy)
    if baseline_policy is not None:
        baseline_approved = apply_policy(df, pred_proba, baseline_policy)
    else:
        baseline_approved = pd.Series([True] * n, index=df.index)

    # Core counts
    n_approved   = approved.sum()
    n_rejected   = (~approved).sum()

    # Among approved: how many are actually risky?
    defaults_in_approved = int((actual_defaults[approved]).sum())
    safe_in_approved     = int(((1 - actual_defaults)[approved]).sum())

    # Among rejected: how many are risky (correctly caught)?
    defaults_prevented  = int((actual_defaults[~approved]).sum())
    safe_rejected       = int(((1 - actual_defaults)[~approved]).sum())  # false positives

    # Expected Loss calculation
    ead = (df["NETMONTHLYINCOME"].fillna(df["NETMONTHLYINCOME"].median()) * EAD_MULT
           if "NETMONTHLYINCOME" in df.columns else pd.Series(np.ones(n) * 300000))

    el_all      = (pred_proba * LGD * ead)
    el_approved = el_all[approved].sum()
    el_baseline = el_all[baseline_approved].sum()

    # Baseline stats for comparison
    baseline_approval_rate  = baseline_approved.mean()
    baseline_defaults_in_approved = int((actual_defaults[baseline_approved]).sum())
    baseline_el = el_all[baseline_approved].sum()

    return {
        "policy_name":           policy.name,
        "n_total":               n,

        # Approvals
        "n_approved":            int(n_approved),
        "approval_rate_pct":     round(n_approved / n * 100, 1),
        "n_rejected":            int(n_rejected),
        "rejection_rate_pct":    round(n_rejected / n * 100, 1),

        # Default outcomes
        "defaults_in_approved":  defaults_in_approved,
        "default_rate_in_approved_pct": round(defaults_in_approved / max(n_approved,1) * 100, 1),
        "defaults_prevented":    defaults_prevented,
        "safe_borrowers_rejected": safe_rejected,

        # vs baseline
        "defaults_prevented_vs_baseline": defaults_prevented - (n - baseline_approved.sum()),
        "approvals_lost_vs_baseline": int(baseline_approved.sum()) - int(n_approved),

        # Expected Loss
        "expected_loss_cr":      round(el_approved / 1e7, 2),
        "baseline_el_cr":        round(baseline_el / 1e7, 2),
        "el_reduction_cr":       round((baseline_el - el_approved) / 1e7, 2),
        "el_reduction_pct":      round((baseline_el - el_approved) / max(baseline_el, 1) * 100, 1),
    }


def threshold_sensitivity_table(df: pd.DataFrame,
                                  pred_proba: np.ndarray) -> pd.DataFrame:
    """
    The threshold trade-off table — a key differentiator.
    Shows what happens at each decision threshold.
    """
    actual = df["default_risk"].values if "default_risk" in df.columns else np.zeros(len(df))
    n      = len(df)

    rows = []
    for thresh in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        predicted_risky = pred_proba >= thresh
        approved        = ~predicted_risky

        tp = int(((predicted_risky) & (actual == 1)).sum())  # risky caught
        fp = int(((predicted_risky) & (actual == 0)).sum())  # safe rejected
        fn = int(((~predicted_risky) & (actual == 1)).sum()) # risky missed
        tn = int(((~predicted_risky) & (actual == 0)).sum()) # safe approved

        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-6)

        rows.append({
            "Threshold":            thresh,
            "Approval Rate %":      round(approved.mean() * 100, 1),
            "Defaults Caught %":    round(recall * 100, 1),       # recall
            "False Alarm Rate %":   round(fp / max(fp + tp, 1) * 100, 1),
            "Precision %":          round(precision * 100, 1),
            "F1 Score":             round(f1, 3),
            "Defaults Missed":      fn,
            "Safe Borrowers Rejected": fp,
        })

    return pd.DataFrame(rows)


def compare_policies(df: pd.DataFrame,
                     pred_proba: np.ndarray,
                     policies: list) -> pd.DataFrame:
    """
    Compare multiple PolicyConfig objects side by side.
    Returns a wide comparison DataFrame.
    """
    rows = []
    baseline = policies[0] if policies else CURRENT_POLICY

    for policy in policies:
        result = simulate_policy(df, pred_proba, policy,
                                  baseline_policy=baseline)
        rows.append({
            "Policy":            result["policy_name"],
            "Approval Rate %":   result["approval_rate_pct"],
            "Default Rate in Approved %": result["default_rate_in_approved_pct"],
            "Defaults Prevented": result["defaults_prevented"],
            "Safe Rejected":     result["safe_borrowers_rejected"],
            "Expected Loss (₹ Cr)": result["expected_loss_cr"],
            "EL Reduction (₹ Cr)":  result["el_reduction_cr"],
        })

    return pd.DataFrame(rows)