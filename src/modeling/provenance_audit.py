"""
provenance_audit.py

Prediction-time provenance audit.

Purpose:
    Determine whether model features were plausibly available
    BEFORE the lending decision.

This script does NOT automatically declare a feature leakage-free.
Historical provenance must be confirmed from source-data documentation.
"""

import json
from pathlib import Path

import pandas as pd

from feature_registry import (
    TARGET,
    EXCLUDED_FEATURES,
    REVIEW_FEATURES,
    CONSERVATIVE_FEATURES,
)

GOLD_DIR = Path("data/gold/exports")
ANALYTICS_DIR = GOLD_DIR / "analytics"
OUTPUT_DIR = 'data\model\evaluation\provenance'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    path = GOLD_DIR / "fact_credit_risk.parquet"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)

    print(f"  ✓ Loaded: {len(df):,} rows × {len(df.columns)} columns")

    return df


def build_provenance_table(df):

    rows = []

    for feature in CONSERVATIVE_FEATURES:

        if feature in REVIEW_FEATURES:
            classification = "REVIEW"
            status = "REQUIRES_DOCUMENTATION"

            if feature == "cibil_score":
                reason = (
                    "High predictive strength. Confirm that CIBIL score "
                    "was available at application/decision time and was not "
                    "constructed using post-decision repayment information."
                )

            elif feature in {
                "recent_enquiries_6m",
                "recent_enquiries_12m",
                "total_enquiries",
            }:
                reason = (
                    "Confirm enquiry information represents bureau information "
                    "available before the lending decision."
                )

            elif feature == "cibil_band":
                reason = (
                    "Derived from CIBIL score. Confirm underlying score "
                    "was available before decision."
                )

            else:
                reason = "Historical provenance requires confirmation."

        else:
            classification = "PROVISIONALLY_SAFE"
            status = "STRUCTURALLY_ELIGIBLE"

            reason = (
                "No target/proxy classification identified by structural "
                "leakage audit. Historical prediction-time availability "
                "should still be documented."
            )

        rows.append({
            "feature": feature,
            "classification": classification,
            "prediction_time_status": status,
            "reason": reason,
            "dtype": str(df[feature].dtype),
            "missing_count": int(df[feature].isna().sum()),
            "missing_pct": round(
                df[feature].isna().mean() * 100, 3
            ),
        })

    return pd.DataFrame(rows)


def build_exclusion_table():

    rows = []

    reasons = {
        "default_risk": "Target variable",
        "risk_band": "Target-derived classification",
        "risk_grade": "Post-decision/target proxy",
        "risk_grade_numeric": "Target-derived numerical proxy",
        "borrower_id": "Identifier",
    }

    for feature in sorted(EXCLUDED_FEATURES):

        rows.append({
            "feature": feature,
            "classification": "EXCLUDED",
            "reason": reasons.get(feature, "Excluded by governance"),
        })

    return pd.DataFrame(rows)


def main():

    print("=" * 70)
    print("  INDIA CREDIT RISK — PREDICTION-TIME PROVENANCE AUDIT")
    print("=" * 70)

    df = load_data()

    print("\nBuilding provenance table...")

    provenance = build_provenance_table(df)
    exclusions = build_exclusion_table()

    provenance.to_parquet(
        OUTPUT_DIR / "feature_provenance.parquet",
        index=False,
    )

    exclusions.to_parquet(
        OUTPUT_DIR / "feature_exclusions.parquet",
        index=False,
    )

    provenance.to_csv(
        OUTPUT_DIR / "feature_provenance.csv",
        index=False,
    )

    print("\nFeature provenance status:")
    print(
        provenance[
            [
                "feature",
                "classification",
                "prediction_time_status",
            ]
        ].to_string(index=False)
    )

    print("\nReview features:")
    print(
        provenance[
            provenance["classification"] == "REVIEW"
        ][
            [
                "feature",
                "reason",
            ]
        ].to_string(index=False)
    )

    metadata = {
        "total_features_reviewed": len(provenance),
        "review_features": sorted(REVIEW_FEATURES),
        "excluded_features": sorted(EXCLUDED_FEATURES),
        "note": (
            "Structural eligibility is not equivalent to historical "
            "prediction-time provenance confirmation."
        ),
    }

    with open(
        OUTPUT_DIR / "provenance_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)

    print("\n✓ Provenance audit complete")

    print(f"\nOutputs:")
    print(f"  → {OUTPUT_DIR / 'feature_provenance.parquet'}")
    print(f"  → {OUTPUT_DIR / 'feature_exclusions.parquet'}")
    print(f"  → {OUTPUT_DIR / 'feature_provenance.csv'}")


if __name__ == "__main__":
    main()