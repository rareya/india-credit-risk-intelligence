"""
feature_governance.py — Credit Risk Feature Governance

Creates a transparent feature registry for ML.

Purpose:
    1. Identify features that are definitely unavailable for prediction.
    2. Identify features that require provenance review.
    3. Define conservative and expanded ML feature sets.
    4. Prevent accidental target leakage in downstream ML.

IMPORTANT:
    This file does NOT train a model.
    It creates the governance contract that the ML pipeline consumes.

Outputs:
    data/gold/exports/analytics/feature_governance.parquet
    data/gold/exports/analytics/ml_feature_registry.parquet
"""

from pathlib import Path
import pandas as pd


GOLD_DIR = Path("data/gold/exports")
ANALYTICS_DIR = "data/model/evaluation" 

FACT_FILE = GOLD_DIR / "fact_credit_risk.parquet"

ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# DEFINITIVE EXCLUSIONS
# ---------------------------------------------------------------------

DEFINITE_EXCLUSIONS = {
    "borrower_id": {
        "status": "EXCLUDED",
        "available_at_decision": "IDENTIFIER",
        "target_derived": False,
        "reason": "Identifier-like field. Must never be used as a predictive feature."
    },

    "default_risk": {
        "status": "EXCLUDED",
        "available_at_decision": "NOT_AVAILABLE",
        "target_derived": True,
        "reason": "Prediction target."
    },

    "risk_grade": {
        "status": "EXCLUDED",
        "available_at_decision": "NOT_AVAILABLE",
        "target_derived": True,
        "reason": "Direct/post-decision risk classification."
    },

    "risk_grade_numeric": {
        "status": "EXCLUDED",
        "available_at_decision": "NOT_AVAILABLE",
        "target_derived": True,
        "reason": "Numeric encoding of target-derived risk grade."
    },

    "risk_band": {
        "status": "EXCLUDED",
        "available_at_decision": "NOT_AVAILABLE",
        "target_derived": True,
        "reason": "Derived from target/risk classification."
    },
}


# ---------------------------------------------------------------------
# FEATURES REQUIRING EXTRA PROVENANCE REVIEW
# ---------------------------------------------------------------------

REVIEW_FEATURES = {
    "cibil_score": {
        "reason": (
            "Extremely strong predictive signal. "
            "Must confirm that the CIBIL score represents information "
            "available before the lending decision."
        )
    },

    "cibil_band": {
        "reason": (
            "Derived from CIBIL score. Include only if the underlying "
            "CIBIL score is confirmed to be pre-decision."
        )
    },

    "recent_enquiries_6m": {
        "reason": (
            "Strong predictive association and meaningful missingness. "
            "Confirm that enquiry information was available at decision time."
        )
    },

    "recent_enquiries_12m": {
        "reason": (
            "Strong predictive association and meaningful missingness. "
            "Confirm that enquiry information was available at decision time."
        )
    },

    "total_enquiries": {
        "reason": (
            "Strong predictive association and meaningful missingness. "
            "Confirm timing and provenance."
        )
    },
}


# ---------------------------------------------------------------------
# BUILD GOVERNANCE REGISTRY
# ---------------------------------------------------------------------

def build_governance_registry(df: pd.DataFrame) -> pd.DataFrame:

    rows = []

    for feature in df.columns:

        if feature in DEFINITE_EXCLUSIONS:

            info = DEFINITE_EXCLUSIONS[feature]

            rows.append({
                "feature": feature,
                "status": info["status"],
                "available_at_decision": info["available_at_decision"],
                "target_derived": info["target_derived"],
                "ml_eligible_conservative": False,
                "ml_eligible_expanded": False,
                "reason": info["reason"],
            })

            continue

        if feature in REVIEW_FEATURES:

            info = REVIEW_FEATURES[feature]

            rows.append({
                "feature": feature,
                "status": "REVIEW",
                "available_at_decision": "REQUIRES_PROVENANCE",
                "target_derived": False,
                "ml_eligible_conservative": False,
                "ml_eligible_expanded": True,
                "reason": info["reason"],
            })

            continue

        rows.append({
            "feature": feature,
            "status": "LIKELY_SAFE",
            "available_at_decision": "LIKELY_PRE_DECISION",
            "target_derived": False,
            "ml_eligible_conservative": True,
            "ml_eligible_expanded": True,
            "reason": (
                "Structurally eligible feature. "
                "Final provenance should still be documented."
            ),
        })

    registry = pd.DataFrame(rows)

    return registry


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("  INDIA CREDIT RISK — FEATURE GOVERNANCE")
    print("=" * 70)

    print("\nLoading Gold fact table...")

    df = pd.read_parquet(FACT_FILE)

    print(
        f"  ✓ Loaded: "
        f"{df.shape[0]:,} rows × {df.shape[1]} columns"
    )

    registry = build_governance_registry(df)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    print("\n" + "━" * 70)
    print("Feature governance summary")
    print("━" * 70)

    print(
        registry["status"]
        .value_counts()
        .rename_axis("classification")
        .to_string()
    )

    # ---------------------------------------------------------------
    # Print exclusions
    # ---------------------------------------------------------------

    print("\nDefinitive exclusions:")

    excluded = registry[
        registry["status"] == "EXCLUDED"
    ]

    for feature in excluded["feature"]:
        print(f"  🔴 {feature}")

    # ---------------------------------------------------------------
    # Print review features
    # ---------------------------------------------------------------

    print("\nFeatures requiring provenance review:")

    review = registry[
        registry["status"] == "REVIEW"
    ]

    for feature in review["feature"]:
        print(f"  🟠 {feature}")

    # ---------------------------------------------------------------
    # Feature sets
    # ---------------------------------------------------------------

    conservative = registry[
        registry["ml_eligible_conservative"]
    ]["feature"].tolist()

    expanded = registry[
        registry["ml_eligible_expanded"]
    ]["feature"].tolist()

    print("\n" + "━" * 70)
    print("ML feature sets")
    print("━" * 70)

    print(
        f"\n  Conservative features: {len(conservative)}"
    )

    for feature in conservative:
        print(f"    ✓ {feature}")

    print(
        f"\n  Expanded features: {len(expanded)}"
    )

    for feature in expanded:
        print(f"    ✓ {feature}")

    # ---------------------------------------------------------------
    # Save governance registry
    # ---------------------------------------------------------------

    governance_path = (
        ANALYTICS_DIR /
        "feature_governance.parquet"
    )

    registry.to_parquet(
        governance_path,
        index=False
    )

    # ---------------------------------------------------------------
    # Save ML registry
    # ---------------------------------------------------------------

    ml_registry = registry[
        [
            "feature",
            "status",
            "ml_eligible_conservative",
            "ml_eligible_expanded",
            "reason",
        ]
    ].copy()

    ml_registry_path = (
        ANALYTICS_DIR /
        "ml_feature_registry.parquet"
    )

    ml_registry.to_parquet(
        ml_registry_path,
        index=False
    )

    print("\n" + "━" * 70)
    print("✓ FEATURE GOVERNANCE COMPLETE")
    print("━" * 70)

    print("\nSaved:")

    print(
        f"  → {governance_path}"
    )

    print(
        f"  → {ml_registry_path}"
    )

    print("\nNext:")
    print(
        "  python src/analytics/run_ml_model.py"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()