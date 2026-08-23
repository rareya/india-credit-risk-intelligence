"""
prediction_time_provenance.py

INDIA CREDIT RISK — PREDICTION-TIME PROVENANCE AUDIT

Purpose
-------
Audits whether the Conservative ML feature set is plausibly available
at the exact time a borrower is scored.

This is a governance audit.

It does NOT prove causality and it does NOT replace the structural
leakage audit.

The audit checks:

1. Feature exists in the Gold fact table.
2. Feature exists in the conservative ML registry.
3. Feature has an explicit provenance classification.
4. Feature construction does not obviously depend on future outcomes.
5. Feature is plausibly available at prediction time.
6. Features derived from historical borrower behavior are treated as
   historical information, not automatically as leakage.
7. Ambiguous features are flagged for human review rather than silently
   accepted.

IMPORTANT
---------
A PASS means the feature has passed this automated provenance screen.
It does not mean that a real production data feed has been independently
verified.

Outputs
-------
data/gold/exports/analytics/ml/provenance/

    prediction_time_provenance.parquet
    prediction_time_provenance.csv
    prediction_time_provenance_summary.json
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path("data/gold/exports/analytics")
ML_DIR = BASE_DIR / "ml"
OUTPUT_DIR = ML_DIR / "provenance"

GOLD_PATH = Path(
    "data/gold/exports/fact_credit_risk.parquet"
)

REGISTRY_PATH = BASE_DIR / "ml_feature_registry.parquet"

# Existing provenance output, if available.
EXISTING_PROVENANCE_PATH = (
    ML_DIR / "provenance" / "feature_provenance.parquet"
)

TARGET = "default_risk"


# ============================================================
# APPROVED CONSERVATIVE FEATURE SET
# ============================================================

CONSERVATIVE_FEATURES = [
    "delinquency_score",
    "num_times_delinquent",
    "num_times_30p_dpd",
    "num_times_60p_dpd",
    "total_missed_payments",
    "missed_payment_ratio",
    "monthly_income_inr",
    "income_tier",
    "total_loans",
    "active_loans",
    "closed_loans",
    "active_loan_ratio",
    "loan_type_diversity",
    "gold_loans",
    "home_loans",
    "personal_loans",
    "credit_card_loans",
    "auto_loans",
    "secured_loans",
    "unsecured_loans",
    "age",
    "gender",
    "education",
    "marital_status",
    "age_band",
    "credit_history_months",
    "has_gold_loan",
    "has_home_loan",
    "has_personal_loan",
    "has_credit_card",
]


# ============================================================
# FEATURE PROVENANCE POLICY
# ============================================================

# Classification:
#
# HISTORICAL_BEHAVIOR
#     Uses borrower information that should exist before scoring.
#
# CURRENT_APPLICATION
#     Information normally supplied/known during application.
#
# DERIVED_HISTORICAL
#     Derived from historical borrower records.
#
# DEMOGRAPHIC
#     Static/current borrower attributes.
#
# REVIEW
#     Semantically ambiguous and requires human verification.
#
# FAIL
#     Obviously dependent on future outcome information.

FEATURE_POLICY = {

    # -------------------------
    # Historical repayment
    # -------------------------

    "delinquency_score": {
        "category": "DERIVED_HISTORICAL",
        "expected_availability": "Before prediction",
        "future_dependency": False,
        "reason": (
            "Historical repayment/delinquency behavior can be available "
            "before the current credit decision."
        ),
    },

    "num_times_delinquent": {
        "category": "HISTORICAL_BEHAVIOR",
        "expected_availability": "Before prediction",
        "future_dependency": False,
        "reason": (
            "Counts historical delinquency events."
        ),
    },

    "num_times_30p_dpd": {
        "category": "HISTORICAL_BEHAVIOR",
        "expected_availability": "Before prediction",
        "future_dependency": False,
        "reason": (
            "Counts historical 30+ day delinquency events."
        ),
    },

    "num_times_60p_dpd": {
        "category": "HISTORICAL_BEHAVIOR",
        "expected_availability": "Before prediction",
        "future_dependency": False,
        "reason": (
            "Counts historical 60+ day delinquency events."
        ),
    },

    "total_missed_payments": {
        "category": "HISTORICAL_BEHAVIOR",
        "expected_availability": "Before prediction",
        "future_dependency": False,
        "reason": (
            "Historical missed payments are available if calculated "
            "only from records existing before scoring."
        ),
    },

    "missed_payment_ratio": {
        "category": "DERIVED_HISTORICAL",
        "expected_availability": "Before prediction",
        "future_dependency": False,
        "reason": (
            "Derived from historical missed-payment behavior."
        ),
    },

    # -------------------------
    # Income
    # -------------------------

    "monthly_income_inr": {
        "category": "CURRENT_APPLICATION",
        "expected_availability": "At application/scoring",
        "future_dependency": False,
        "reason": (
            "Current borrower income is normally available during "
            "credit assessment."
        ),
    },

    "income_tier": {
        "category": "CURRENT_APPLICATION",
        "expected_availability": "At application/scoring",
        "future_dependency": False,
        "reason": (
            "Derived from current income information."
        ),
    },

    # -------------------------
    # Loan portfolio
    # -------------------------

    "total_loans": {
        "category": "HISTORICAL_BEHAVIOR",
        "expected_availability": "Before prediction",
        "future_dependency": False,
        "reason": (
            "Existing loan count should be available from borrower "
            "credit records."
        ),
    },

    "active_loans": {
        "category": "HISTORICAL_BEHAVIOR",
        "expected_availability": "Before prediction",
        "future_dependency": False,
        "reason": (
            "Current active loans should be known at scoring."
        ),
    },

    "closed_loans": {
        "category": "HISTORICAL_BEHAVIOR",
        "expected_availability": "Before prediction",
        "future_dependency": False,
        "reason": (
            "Previously closed loans are historical borrower information."
        ),
    },

    "active_loan_ratio": {
        "category": "DERIVED_HISTORICAL",
        "expected_availability": "Before prediction",
        "future_dependency": False,
        "reason": (
            "Derived from active and total existing loans."
        ),
    },

    "loan_type_diversity": {
        "category": "DERIVED_HISTORICAL",
        "expected_availability": "Before prediction",
        "future_dependency": False,
        "reason": (
            "Derived from existing borrower loan types."
        ),
    },

    "gold_loans": {
        "category": "HISTORICAL_BEHAVIOR",
        "expected_availability": "Before prediction",
        "future_dependency": False,
        "reason": (
            "Existing gold-loan information should be available "
            "from borrower records."
        ),
    },

    "home_loans": {
        "category": "HISTORICAL_BEHAVIOR",
        "expected_availability": "Before prediction",
        "future_dependency": False,
        "reason": (
            "Existing home-loan information should be available "
            "from borrower records."
        ),
    },

    "personal_loans": {
        "category": "HISTORICAL_BEHAVIOR",
        "expected_availability": "Before prediction",
        "future_dependency": False,
        "reason": (
            "Existing personal-loan information should be available."
        ),
    },

    "credit_card_loans": {
        "category": "HISTORICAL_BEHAVIOR",
        "expected_availability": "Before prediction",
        "future_dependency": False,
        "reason": (
            "Existing credit-card loan information should be available."
        ),
    },

    "auto_loans": {
        "category": "HISTORICAL_BEHAVIOR",
        "expected_availability": "Before prediction",
        "future_dependency": False,
        "reason": (
            "Existing auto-loan information should be available."
        ),
    },

    "secured_loans": {
        "category": "HISTORICAL_BEHAVIOR",
        "expected_availability": "Before prediction",
        "future_dependency": False,
        "reason": (
            "Existing secured-loan information should be available."
        ),
    },

    "unsecured_loans": {
        "category": "HISTORICAL_BEHAVIOR",
        "expected_availability": "Before prediction",
        "future_dependency": False,
        "reason": (
            "Existing unsecured-loan information should be available."
        ),
    },

    # -------------------------
    # Demographics
    # -------------------------

    "age": {
        "category": "DEMOGRAPHIC",
        "expected_availability": "At application/scoring",
        "future_dependency": False,
        "reason": (
            "Age is available from borrower identity/application information."
        ),
    },

    "gender": {
        "category": "DEMOGRAPHIC",
        "expected_availability": "At application/scoring",
        "future_dependency": False,
        "reason": (
            "Gender is treated here as an application-time attribute."
        ),
    },

    "education": {
        "category": "DEMOGRAPHIC",
        "expected_availability": "At application/scoring",
        "future_dependency": False,
        "reason": (
            "Education information can be available during application."
        ),
    },

    "marital_status": {
        "category": "DEMOGRAPHIC",
        "expected_availability": "At application/scoring",
        "future_dependency": False,
        "reason": (
            "Marital status is treated as an application-time attribute."
        ),
    },

    "age_band": {
        "category": "DERIVED_HISTORICAL",
        "expected_availability": "At application/scoring",
        "future_dependency": False,
        "reason": (
            "Derived from age and therefore available at scoring."
        ),
    },

    # -------------------------
    # Credit history
    # -------------------------

    "credit_history_months": {
        "category": "HISTORICAL_BEHAVIOR",
        "expected_availability": "Before prediction",
        "future_dependency": False,
        "reason": (
            "Credit-history duration is historical information."
        ),
    },

    # -------------------------
    # Binary indicators
    # -------------------------

    "has_gold_loan": {
        "category": "HISTORICAL_BEHAVIOR",
        "expected_availability": "Before prediction",
        "future_dependency": False,
        "reason": (
            "Indicator should represent an existing/historical loan."
        ),
    },

    "has_home_loan": {
        "category": "HISTORICAL_BEHAVIOR",
        "expected_availability": "Before prediction",
        "future_dependency": False,
        "reason": (
            "Indicator should represent an existing/historical loan."
        ),
    },

    "has_personal_loan": {
        "category": "HISTORICAL_BEHAVIOR",
        "expected_availability": "Before prediction",
        "future_dependency": False,
        "reason": (
            "Indicator should represent an existing/historical loan."
        ),
    },

    "has_credit_card": {
        "category": "HISTORICAL_BEHAVIOR",
        "expected_availability": "Before prediction",
        "future_dependency": False,
        "reason": (
            "Indicator should represent an existing credit relationship."
        ),
    },
}


# ============================================================
# UTILITY
# ============================================================

def print_section(title):
    print("\n" + "━" * 70)
    print(title)
    print("━" * 70)


# ============================================================
# LOAD GOLD
# ============================================================

def load_gold():
    print_section("Loading Gold fact table")

    if not GOLD_PATH.exists():
        raise FileNotFoundError(
            f"\nGold fact table not found:\n{GOLD_PATH}\n"
        )

    df = pd.read_parquet(GOLD_PATH)

    print(
        f"  ✓ Loaded: {len(df):,} rows × "
        f"{len(df.columns)} columns"
    )

    return df


# ============================================================
# LOAD REGISTRY
# ============================================================

def load_registry():
    print_section("Loading conservative ML registry")

    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(
            f"\nConservative ML feature registry not found:\n"
            f"{REGISTRY_PATH}\n"
        )

    registry = pd.read_parquet(REGISTRY_PATH)

    print(f"  ✓ Registry: {REGISTRY_PATH}")
    print(f"  ✓ Registry rows: {len(registry)}")

    if "feature" not in registry.columns:
        raise ValueError(
            "ML registry does not contain a 'feature' column."
        )

    if "ml_eligible_conservative" in registry.columns:

        mask = (
            registry["ml_eligible_conservative"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )

        features = (
            registry.loc[mask, "feature"]
            .astype(str)
            .tolist()
        )

    else:
        features = registry["feature"].astype(str).tolist()

    features = [
        feature
        for feature in features
        if feature in CONSERVATIVE_FEATURES
    ]

    # Preserve canonical ordering.
    features = [
        feature
        for feature in CONSERVATIVE_FEATURES
        if feature in features
    ]

    if not features:
        raise ValueError(
            "No conservative features found in registry."
        )

    print(
        f"  ✓ Conservative features found: {len(features)}"
    )

    return registry, features


# ============================================================
# TARGET VALIDATION
# ============================================================

def validate_target(df):
    print_section("Validating target")

    if TARGET not in df.columns:
        raise ValueError(
            f"Missing target column: {TARGET}"
        )

    if df[TARGET].isna().any():
        raise ValueError(
            "Target contains missing values."
        )

    values = set(df[TARGET].dropna().unique())

    if not values.issubset({0, 1}):
        raise ValueError(
            f"Target must be binary 0/1. Found: {values}"
        )

    print("  ✓ Target validated")

    print(
        f"  Low risk:  {(df[TARGET] == 0).sum():,}"
    )

    print(
        f"  High risk: {(df[TARGET] == 1).sum():,}"
    )


# ============================================================
# EXISTING PROVENANCE
# ============================================================

def load_existing_provenance():
    """
    Load the existing provenance artifact if available.

    This is supplemental evidence only. The current audit still applies
    its own prediction-time policy.
    """

    if not EXISTING_PROVENANCE_PATH.exists():
        return None

    try:
        return pd.read_parquet(
            EXISTING_PROVENANCE_PATH
        )
    except Exception as exc:
        print(
            f"  ! Existing provenance artifact could not be read: {exc}"
        )
        return None


# ============================================================
# FEATURE AUDIT
# ============================================================

def audit_feature(df, feature, registry, existing_provenance):
    policy = FEATURE_POLICY.get(feature)

    if policy is None:
        return {
            "feature": feature,
            "exists_in_gold": feature in df.columns,
            "exists_in_registry": True,
            "policy_defined": False,
            "feature_category": "UNKNOWN",
            "expected_availability": "UNKNOWN",
            "uses_future_information": None,
            "available_at_prediction": None,
            "provenance_status": "REVIEW",
            "review_required": True,
            "review_reason": (
                "No explicit prediction-time provenance policy "
                "exists for this feature."
            ),
        }

    exists_in_gold = feature in df.columns

    if not exists_in_gold:
        return {
            "feature": feature,
            "exists_in_gold": False,
            "exists_in_registry": True,
            "policy_defined": True,
            "feature_category": policy["category"],
            "expected_availability": policy[
                "expected_availability"
            ],
            "uses_future_information": policy[
                "future_dependency"
            ],
            "available_at_prediction": False,
            "provenance_status": "FAIL",
            "review_required": True,
            "review_reason": (
                "Feature is listed in the conservative registry "
                "but is missing from the Gold fact table."
            ),
        }

    series = df[feature]

    missing_pct = float(
        series.isna().mean() * 100
    )

    # Basic suspicious naming check.
    suspicious_tokens = [
        "future",
        "post",
        "after",
        "outcome",
        "defaulted",
        "recovery",
        "writeoff",
        "charged_off",
        "chargeoff",
    ]

    name_lower = feature.lower()

    suspicious_name = any(
        token in name_lower
        for token in suspicious_tokens
    )

    # Determine status.
    if policy["future_dependency"]:
        status = "FAIL"
        available = False
        review_required = True
        reason = (
            "Feature policy explicitly indicates dependence on "
            "future information."
        )

    elif suspicious_name:
        status = "REVIEW"
        available = None
        review_required = True
        reason = (
            "Feature name contains a token associated with "
            "post-prediction/outcome information. Manual review required."
        )

    else:
        status = "PASS"
        available = True
        review_required = False
        reason = policy["reason"]

    return {
        "feature": feature,
        "exists_in_gold": True,
        "exists_in_registry": True,
        "policy_defined": True,
        "feature_category": policy["category"],
        "expected_availability": policy[
            "expected_availability"
        ],
        "uses_future_information": policy[
            "future_dependency"
        ],
        "available_at_prediction": available,
        "provenance_status": status,
        "review_required": review_required,
        "review_reason": reason,
        "dtype": str(series.dtype),
        "n_rows": int(len(series)),
        "n_missing": int(series.isna().sum()),
        "missing_pct": round(missing_pct, 4),
        "n_unique": int(
            series.nunique(dropna=True)
        ),
    }


# ============================================================
# BUILD AUDIT
# ============================================================

def build_audit(df, registry, features, existing_provenance):

    print_section(
        "Prediction-time provenance audit"
    )

    rows = []

    for feature in features:

        print(
            f"  Auditing: {feature}"
        )

        rows.append(
            audit_feature(
                df=df,
                feature=feature,
                registry=registry,
                existing_provenance=existing_provenance,
            )
        )

    return pd.DataFrame(rows)


# ============================================================
# SUMMARY
# ============================================================

def build_summary(audit):

    total = len(audit)

    pass_count = int(
        (audit["provenance_status"] == "PASS").sum()
    )

    review_count = int(
        (audit["provenance_status"] == "REVIEW").sum()
    )

    fail_count = int(
        (audit["provenance_status"] == "FAIL").sum()
    )

    missing_gold = int(
        (~audit["exists_in_gold"]).sum()
    )

    future_dependency = int(
        (
            audit["uses_future_information"]
            == True
        ).sum()
    )

    if fail_count > 0:
        overall_status = "FAIL"

    elif review_count > 0:
        overall_status = "REVIEW_REQUIRED"

    elif pass_count == total:
        overall_status = "PASS"

    else:
        overall_status = "REVIEW_REQUIRED"

    return {
        "audit_name": "prediction_time_provenance",
        "total_features_audited": total,
        "features_pass": pass_count,
        "features_review": review_count,
        "features_fail": fail_count,
        "features_missing_from_gold": missing_gold,
        "features_with_declared_future_dependency": (
            future_dependency
        ),
        "overall_status": overall_status,
        "governance_interpretation": (
            "PASS means all conservative features passed the "
            "automated prediction-time provenance screen. "
            "This does not constitute independent verification "
            "of a production data feed."
        ),
    }


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(audit, summary):

    print_section(
        "PROVENANCE AUDIT RESULTS"
    )

    print(
        "\n  PASS:"
    )

    passed = audit[
        audit["provenance_status"] == "PASS"
    ]

    if passed.empty:
        print("    None")
    else:
        for feature in passed["feature"]:
            print(f"    ✓ {feature}")

    print(
        "\n  REVIEW REQUIRED:"
    )

    review = audit[
        audit["provenance_status"] == "REVIEW"
    ]

    if review.empty:
        print("    ✓ None")
    else:
        print(
            review[
                [
                    "feature",
                    "review_reason",
                ]
            ].to_string(index=False)
        )

    print(
        "\n  FAIL:"
    )

    failed = audit[
        audit["provenance_status"] == "FAIL"
    ]

    if failed.empty:
        print("    ✓ None")
    else:
        print(
            failed[
                [
                    "feature",
                    "review_reason",
                ]
            ].to_string(index=False)
        )

    print("\n  Summary:")
    print(
        f"    Total features: {summary['total_features_audited']}"
    )
    print(
        f"    PASS:           {summary['features_pass']}"
    )
    print(
        f"    REVIEW:         {summary['features_review']}"
    )
    print(
        f"    FAIL:           {summary['features_fail']}"
    )
    print(
        f"    Overall status: {summary['overall_status']}"
    )


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(audit, summary):

    print_section(
        "Saving prediction-time provenance outputs"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_out = audit.copy()

    # Normalize booleans explicitly.
    bool_columns = [
        "exists_in_gold",
        "exists_in_registry",
        "policy_defined",
        "uses_future_information",
        "available_at_prediction",
        "review_required",
    ]

    for col in bool_columns:

        if col in audit_out.columns:

            if audit_out[col].isna().any():
                # Nullable boolean is safer for Parquet.
                audit_out[col] = (
                    audit_out[col]
                    .astype("boolean")
                )
            else:
                audit_out[col] = (
                    audit_out[col]
                    .astype(bool)
                )

    # Numeric normalization.
    numeric_columns = [
        "n_rows",
        "n_missing",
        "missing_pct",
        "n_unique",
    ]

    for col in numeric_columns:

        if col in audit_out.columns:
            audit_out[col] = pd.to_numeric(
                audit_out[col],
                errors="coerce",
            )

    parquet_path = (
        OUTPUT_DIR /
        "prediction_time_provenance.parquet"
    )

    csv_path = (
        OUTPUT_DIR /
        "prediction_time_provenance.csv"
    )

    json_path = (
        OUTPUT_DIR /
        "prediction_time_provenance_summary.json"
    )

    audit_out.to_parquet(
        parquet_path,
        index=False,
    )

    audit_out.to_csv(
        csv_path,
        index=False,
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print(
        f"  ✓ Saved: {parquet_path}"
    )

    print(
        f"  ✓ Saved: {csv_path}"
    )

    print(
        f"  ✓ Saved: {json_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "  INDIA CREDIT RISK — "
        "PREDICTION-TIME PROVENANCE AUDIT"
    )

    print("=" * 70)

    warnings.filterwarnings(
        "ignore"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_gold()

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    validate_target(df)

    # --------------------------------------------------------
    # Registry
    # --------------------------------------------------------

    registry, features = load_registry()

    print(
        "\n  Features selected for provenance audit:"
    )

    for feature in features:
        print(
            f"    ✓ {feature}"
        )

    # --------------------------------------------------------
    # Existing provenance artifact
    # --------------------------------------------------------

    existing_provenance = (
        load_existing_provenance()
    )

    if existing_provenance is not None:

        print(
            "\n  ✓ Existing provenance artifact found."
        )

        print(
            f"    {EXISTING_PROVENANCE_PATH}"
        )

    else:

        print(
            "\n  ! Existing detailed provenance artifact "
            "not found."
        )

        print(
            "    Continuing with explicit feature policy."
        )

    # --------------------------------------------------------
    # Audit
    # --------------------------------------------------------

    audit = build_audit(
        df=df,
        registry=registry,
        features=features,
        existing_provenance=existing_provenance,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = build_summary(
        audit
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print_results(
        audit,
        summary,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_outputs(
        audit,
        summary,
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "✓ PREDICTION-TIME PROVENANCE AUDIT COMPLETE"
    )

    print("=" * 70)

    print(
        f"""
Features audited:
  {summary["total_features_audited"]}

PASS:
  {summary["features_pass"]}

REVIEW REQUIRED:
  {summary["features_review"]}

FAIL:
  {summary["features_fail"]}

Overall status:
  {summary["overall_status"]}

Outputs:
  {OUTPUT_DIR}

IMPORTANT:
  This audit is a provenance screening layer.
  PASS does not independently prove that a production
  data feed exposes the feature before scoring.

NEXT:
  If PASS:
      Freeze the Conservative XGBoost feature set and
      proceed to final model packaging.

  If REVIEW_REQUIRED:
      Investigate every flagged feature before freezing.

  If FAIL:
      Remove/reconstruct the affected feature and rerun
      the governed ML pipeline.
"""
    )

    # Return non-zero exit status for FAIL.
    if summary["overall_status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()