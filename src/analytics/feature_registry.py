"""
feature_registry.py

Central feature governance for the India Credit Risk ML pipeline.

IMPORTANT:
- TARGET and target-derived fields are NEVER predictive features.
- Conservative features are structurally eligible.
- Review features require provenance confirmation before operational use.
"""

# ============================================================
# TARGET
# ============================================================

TARGET = "default_risk"

ID_COLUMNS = [
    "borrower_id",
]


# ============================================================
# DEFINITE EXCLUSIONS
# ============================================================

EXCLUDED_FEATURES = {
    "default_risk",
    "risk_band",
    "risk_grade",
    "risk_grade_numeric",
    "borrower_id",
}


# ============================================================
# FEATURES REQUIRING PROVENANCE REVIEW
# ============================================================

REVIEW_FEATURES = {
    "cibil_score",
    "cibil_band",
    "recent_enquiries_6m",
    "recent_enquiries_12m",
    "total_enquiries",
}


# ============================================================
# CONSERVATIVE MODEL FEATURES
#
# These correspond to the current 30-feature experiment.
# Review features are included for statistical benchmarking,
# but MUST be cleared by provenance review before operational use.
# ============================================================

CONSERVATIVE_FEATURES = [
    "active_loan_ratio",
    "active_loans",
    "age",
    "auto_loans",
    "closed_loans",
    "credit_card_loans",
    "credit_history_months",
    "delinquency_score",
    "education",
    "gender",
    "gold_loans",
    "has_credit_card",
    "has_gold_loan",
    "has_home_loan",
    "has_personal_loan",
    "home_loans",
    "income_tier",
    "loan_type_diversity",
    "marital_status",
    "missed_payment_ratio",
    "monthly_income_inr",
    "num_times_60p_dpd",
    "num_times_delinquent",
    "personal_loans",
    "recent_enquiries_12m",
    "recent_enquiries_6m",
    "secured_loans",
    "total_enquiries",
    "total_missed_payments",
    "unsecured_loans",
]


# ============================================================
# REVIEW-FREE SUBSET
#
# Useful for a strict "known-safe" experiment.
# ============================================================

PROVISIONALLY_SAFE_FEATURES = [
    f for f in CONSERVATIVE_FEATURES
    if f not in REVIEW_FEATURES
]


def validate_registry(df_columns):
    """
    Validate the registry against an input dataframe.
    """

    columns = set(df_columns)

    missing = [
        f for f in CONSERVATIVE_FEATURES
        if f not in columns
    ]

    if missing:
        raise ValueError(
            "Conservative feature registry contains missing columns:\n"
            + "\n".join(f"  - {x}" for x in missing)
        )

    overlap = set(CONSERVATIVE_FEATURES) & EXCLUDED_FEATURES

    if overlap:
        raise ValueError(
            "CRITICAL: Conservative feature registry contains "
            f"excluded features: {sorted(overlap)}"
        )

    return True