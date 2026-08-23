"""
leakage_audit.py — Reproducible ML Leakage Audit
=================================================

India Credit Risk Intelligence

Purpose
-------
Systematically audit candidate ML features before model training.

The audit checks:

1. Target leakage
   - Features derived directly or indirectly from the target/decision.
   - Approved_Flag / risk_grade / risk_grade_numeric / default_risk etc.

2. Known suspicious features
   - Credit_Score
   - Features derived from Credit_Score
   - Other engineered risk indicators requiring provenance review.

3. Identifier leakage
   - borrower IDs and ID-like variables.

4. Feature provenance / availability
   - Whether a feature can plausibly exist at prediction time.

5. Single-feature predictive strength
   - Logistic regression AUC for every numeric candidate.
   - Extremely high AUC is treated as an investigation trigger,
     NOT automatic proof of leakage.

6. Missingness leakage
   - Whether the fact that a feature is missing is strongly associated
     with the target.

7. Final ML eligibility
   - Produces a conservative SAFE / REVIEW / EXCLUDED classification.

Outputs
-------
data/gold/exports/analytics/
    feature_leakage_audit.parquet
    single_feature_auc_audit.parquet
    missingness_leakage_audit.parquet

Usage
-----
    python src/analytics/leakage_audit.py

Important
---------
This audit is a governance / diagnostic layer.

It does NOT claim that statistical association proves causality
or leakage. Provenance and prediction-time availability remain
the final authority.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

GOLD_DIR = Path("data/gold/exports")
ANALYTICS_DIR = GOLD_DIR / "analytics"

FACT_PATH = GOLD_DIR / "fact_credit_risk.parquet"

ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PROJECT TARGET / LEAKAGE DEFINITIONS
# ============================================================

TARGET = "default_risk"

BORROWER_ID = "borrower_id"


# ----------------------------------------------------------------
# DEFINITE TARGET / DECISION LEAKAGE
# ----------------------------------------------------------------
#
# These must NEVER enter the production ML feature matrix.
#
# Approved_Flag is the original decision/risk grade.
# default_risk is the canonical target.
# risk_grade_numeric is a numeric encoding of the decision grade.
# risk_grade is another representation of the decision grade.
#
# risk_band is also treated conservatively because it is an
# engineered risk classification rather than a raw borrower attribute.
# ----------------------------------------------------------------

DEFINITE_LEAKAGE = {
    "default_risk": "TARGET",
    "Approved_Flag": "POST_DECISION_TARGET_PROXY",
    "risk_grade": "POST_DECISION_TARGET_PROXY",
    "risk_grade_numeric": "TARGET_DERIVED_FEATURE",
    "risk_band": "TARGET_DERIVED_OR_RISK_CLASSIFICATION",
}


# ----------------------------------------------------------------
# KNOWN / HIGH PRIORITY REVIEW FEATURES
# ----------------------------------------------------------------
#
# Credit_Score was previously identified in this project as having
# extremely suspicious predictive power (historically ~0.9998 AUC
# when used alone).
#
# We therefore do NOT silently call it safe.
#
# Derived variables using Credit_Score also require review.
# ----------------------------------------------------------------

KNOWN_REVIEW = {
    "cibil_score": (
        "KNOWN_HIGH_PRIORITY_REVIEW: historical single-feature "
        "AUC was approximately 0.9998; investigate provenance."
    ),
    "Credit_Score": (
        "KNOWN_HIGH_PRIORITY_REVIEW: historical single-feature "
        "AUC was approximately 0.9998; investigate provenance."
    ),
    "cibil_band": (
        "DERIVED_FROM_CREDIT_SCORE: review because Credit_Score "
        "itself requires provenance investigation."
    ),
    "score_per_income_lakh": (
        "DERIVED_FROM_CREDIT_SCORE: inherits provenance concern "
        "from Credit_Score."
    ),
}


# ----------------------------------------------------------------
# SAFE / EXPECTED PRE-DECISION FEATURES
# ----------------------------------------------------------------
#
# These are not blindly declared safe.
# They are classified as "LIKELY_PRE_DECISION" and still receive
# statistical checks.
# ----------------------------------------------------------------

LIKELY_PRE_DECISION = {
    "age",
    "gender",
    "education",
    "marital_status",
    "monthly_income_inr",
    "income_tier",
    "months_with_employer",
    "cibil_score",
    "credit_history_months",
    "total_delinquencies",
    "times_30dpd",
    "times_60dpd",
    "delinquency_score",
    "total_missed_payments",
    "time_since_recent_payment",
    "total_enquiries",
    "enquiries_last_6m",
    "enquiries_last_12m",
    "last_product_enquired",
    "first_product_enquired",
    "total_loans_ever",
    "active_loans",
    "closed_loans",
    "active_loan_ratio",
    "loan_type_diversity",
    "gold_loans",
    "home_loans",
    "personal_loans",
    "credit_card_loans",
    "auto_loans",
    "consumer_loans",
    "secured_loans",
    "unsecured_loans",
    "has_gold_loan",
    "has_home_loan",
    "has_personal_loan",
    "has_credit_card",
    "recently_active",
    "num_times_delinquent",
    "num_times_60p_dpd",
    "recent_enquiries_6m",
    "recent_enquiries_12m",
    "total_enquiries",
    "missed_payment_ratio",
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_header(title: str):
    print("\n" + "━" * 70)
    print(title)
    print("━" * 70)


def load_fact() -> pd.DataFrame:
    """Load the Gold fact table."""

    print_header("Loading Gold fact table")

    if not FACT_PATH.exists():
        raise FileNotFoundError(
            f"Gold fact table not found:\n  {FACT_PATH}\n\n"
            "Run first:\n"
            "  python src/modeling/build_gold.py"
        )

    df = pd.read_parquet(FACT_PATH)

    print(f"  ✓ Loaded: {len(df):,} rows × {len(df.columns)} columns")
    return df


def validate_fact(df: pd.DataFrame):
    """Validate basic assumptions before running the audit."""

    print_header("Validating Gold fact table")

    required = {
        BORROWER_ID,
        TARGET,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Gold fact table is missing required columns:\n"
            + "\n".join(f"  - {c}" for c in sorted(missing))
        )

    if df[BORROWER_ID].isna().any():
        raise ValueError("borrower_id contains null values.")

    if df[BORROWER_ID].duplicated().any():
        raise ValueError("borrower_id is not unique.")

    if df[TARGET].isna().any():
        raise ValueError("default_risk contains null values.")

    target_values = set(df[TARGET].dropna().unique())

    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"default_risk must be binary 0/1. "
            f"Found: {sorted(target_values)}"
        )

    print(f"  ✓ Rows: {len(df):,}")
    print("  ✓ borrower_id unique and non-null")
    print("  ✓ default_risk binary and non-null")

    print(
        f"  ✓ Target distribution: "
        f"{int((df[TARGET] == 0).sum()):,} low risk / "
        f"{int((df[TARGET] == 1).sum()):,} high risk"
    )


# ============================================================
# 1. FEATURE PROVENANCE / GOVERNANCE AUDIT
# ============================================================

def classify_feature(feature: str) -> dict:
    """
    Assign a governance classification.

    SAFE
        Plausibly available before the lending decision.

    REVIEW
        Requires provenance investigation.

    EXCLUDED
        Target, target proxy, post-decision variable, or identifier.
    """

    lower = feature.lower()

    # --------------------------------------------
    # Definite target leakage
    # --------------------------------------------

    if feature in DEFINITE_LEAKAGE:
        reason = DEFINITE_LEAKAGE[feature]

        return {
            "feature": feature,
            "classification": "EXCLUDED",
            "severity": "CRITICAL",
            "prediction_time_status": "NOT_AVAILABLE",
            "reason": reason,
            "ml_eligible": False,
        }

    # --------------------------------------------
    # Identifier detection
    # --------------------------------------------

    id_patterns = [
        "id",
        "prospect",
        "borrower_id",
        "customer_id",
        "application_id",
        "account_id",
    ]

    if (
        lower == "borrower_id"
        or lower == "prospectid"
        or any(pattern in lower for pattern in id_patterns)
    ):
        return {
            "feature": feature,
            "classification": "EXCLUDED",
            "severity": "HIGH",
            "prediction_time_status": "IDENTIFIER",
            "reason": "Identifier-like field; excluded from predictive features.",
            "ml_eligible": False,
        }

    # --------------------------------------------
    # Known review features
    # --------------------------------------------

    if feature in KNOWN_REVIEW:
        return {
            "feature": feature,
            "classification": "REVIEW",
            "severity": "HIGH",
            "prediction_time_status": "REQUIRES_PROVENANCE",
            "reason": KNOWN_REVIEW[feature],
            "ml_eligible": False,
        }

    # --------------------------------------------
    # Generic risk/decision-derived features
    # --------------------------------------------

    suspicious_terms = [
        "approved",
        "approval",
        "decision",
        "default",
        "target",
        "risk_score",
        "risk_grade",
    ]

    if any(term in lower for term in suspicious_terms):
        return {
            "feature": feature,
            "classification": "REVIEW",
            "severity": "HIGH",
            "prediction_time_status": "REQUIRES_PROVENANCE",
            "reason": (
                "Feature name suggests it may contain a decision, "
                "target, or risk-derived signal."
            ),
            "ml_eligible": False,
        }

    # --------------------------------------------
    # Known pre-decision features
    # --------------------------------------------

    if feature in LIKELY_PRE_DECISION:
        return {
            "feature": feature,
            "classification": "LIKELY_SAFE",
            "severity": "LOW",
            "prediction_time_status": "PLAUSIBLY_PRE_DECISION",
            "reason": (
                "Borrower/bureau attribute that plausibly exists "
                "before the lending decision."
            ),
            "ml_eligible": True,
        }

    # --------------------------------------------
    # Unknown engineered / numeric feature
    # --------------------------------------------

    return {
        "feature": feature,
        "classification": "REVIEW",
        "severity": "MEDIUM",
        "prediction_time_status": "UNKNOWN",
        "reason": (
            "Feature was not explicitly included in the project's "
            "approved pre-decision feature registry."
        ),
        "ml_eligible": False,
    }


def build_feature_leakage_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature governance table."""

    print_header("Feature Provenance & Governance Audit")

    rows = []

    for feature in df.columns:
        if feature == TARGET:
            # Already represented in definite leakage registry.
            pass

        result = classify_feature(feature)

        # Add basic dataset metadata
        result["dtype"] = str(df[feature].dtype)
        result["missing_count"] = int(df[feature].isna().sum())
        result["missing_pct"] = round(
            df[feature].isna().mean() * 100, 3
        )
        result["n_unique"] = int(df[feature].nunique(dropna=True))

        rows.append(result)

    audit = pd.DataFrame(rows)

    severity_order = {
        "CRITICAL": 0,
        "HIGH": 1,
        "MEDIUM": 2,
        "LOW": 3,
    }

    audit["_severity_order"] = audit["severity"].map(severity_order)

    audit = (
        audit
        .sort_values(
            ["_severity_order", "classification", "feature"]
        )
        .drop(columns="_severity_order")
        .reset_index(drop=True)
    )

    output = ANALYTICS_DIR / "feature_leakage_audit.parquet"
    audit.to_parquet(output, index=False)

    print(f"\n  Feature audit summary:")

    print(
        audit["classification"]
        .value_counts()
        .to_string()
    )

    print("\n  Critical / high-risk features:")

    high = audit[
        audit["severity"].isin(["CRITICAL", "HIGH"])
    ]

    if len(high):
        print(
            high[
                [
                    "feature",
                    "classification",
                    "prediction_time_status",
                    "reason",
                ]
            ].to_string(index=False)
        )
    else:
        print("    None")

    print(f"\n  ✓ Saved: {output}")

    return audit


# ============================================================
# 2. SINGLE FEATURE AUC AUDIT
# ============================================================

def single_feature_auc_audit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test each numeric feature independently.

    A very high AUC is treated as a leakage investigation trigger.

    IMPORTANT:
    High AUC != proof of leakage.
    """

    print_header("Single-Feature Predictive Strength Audit")

    numeric_features = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    exclude = {
        TARGET,
        BORROWER_ID,
    }

    features = [
        f for f in numeric_features
        if f not in exclude
    ]

    results = []

    print(f"  Testing {len(features)} numeric features...")

    for feature in features:

        x = df[[feature]]
        y = df[TARGET]

        valid = x[feature].notna() & y.notna()

        x_valid = x.loc[valid]
        y_valid = y.loc[valid]

        if len(x_valid) < 100:
            continue

        if x_valid[feature].nunique() < 2:
            continue

        try:
            pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median"),
                    ),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=1000,
                            random_state=42,
                        ),
                    ),
                ]
            )

            pipeline.fit(x_valid, y_valid)

            probability = pipeline.predict_proba(
                x_valid
            )[:, 1]

            auc = roc_auc_score(
                y_valid,
                probability,
            )

            # A feature can have an inverse relationship.
            # AUC and 1-AUC tell us strength independent of direction.
            association_auc = max(
                auc,
                1 - auc,
            )

            if association_auc >= 0.99:
                flag = "CRITICAL_REVIEW"
            elif association_auc >= 0.95:
                flag = "HIGH_REVIEW"
            elif association_auc >= 0.85:
                flag = "MODERATE_REVIEW"
            else:
                flag = "NORMAL"

            results.append(
                {
                    "feature": feature,
                    "n_observations": len(x_valid),
                    "raw_auc": round(auc, 6),
                    "association_strength_auc": round(
                        association_auc,
                        6,
                    ),
                    "flag": flag,
                    "missing_pct": round(
                        df[feature].isna().mean() * 100,
                        3,
                    ),
                }
            )

        except Exception as exc:
            results.append(
                {
                    "feature": feature,
                    "n_observations": len(x_valid),
                    "raw_auc": np.nan,
                    "association_strength_auc": np.nan,
                    "flag": "ERROR",
                    "missing_pct": round(
                        df[feature].isna().mean() * 100,
                        3,
                    ),
                }
            )

    auc_df = pd.DataFrame(results)

    if not auc_df.empty:
        auc_df = auc_df.sort_values(
            "association_strength_auc",
            ascending=False,
        ).reset_index(drop=True)

        auc_df["rank"] = np.arange(
            1,
            len(auc_df) + 1,
        )

    output = (
        ANALYTICS_DIR
        / "single_feature_auc_audit.parquet"
    )

    auc_df.to_parquet(
        output,
        index=False,
    )

    print("\n  Highest single-feature AUC signals:")

    if not auc_df.empty:
        print(
            auc_df[
                [
                    "rank",
                    "feature",
                    "association_strength_auc",
                    "flag",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

    print(f"\n  ✓ Saved: {output}")

    return auc_df


# ============================================================
# 3. MISSINGNESS LEAKAGE AUDIT
# ============================================================

def missingness_leakage_audit(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Test whether missingness itself is strongly associated
    with default_risk.

    This catches situations where:
        missing → target information
    """

    print_header("Missingness Leakage Audit")

    results = []

    features = [
        f for f in df.columns
        if f not in {TARGET, BORROWER_ID}
    ]

    for feature in features:

        missing_mask = df[feature].isna()

        missing_n = int(missing_mask.sum())
        present_n = int((~missing_mask).sum())

        if missing_n == 0 or present_n == 0:
            continue

        missing_risk_rate = df.loc[
            missing_mask,
            TARGET,
        ].mean()

        present_risk_rate = df.loc[
            ~missing_mask,
            TARGET,
        ].mean()

        difference = (
            missing_risk_rate
            - present_risk_rate
        )

        absolute_difference = abs(difference)

        if absolute_difference >= 0.20:
            flag = "CRITICAL_REVIEW"
        elif absolute_difference >= 0.10:
            flag = "HIGH_REVIEW"
        elif absolute_difference >= 0.05:
            flag = "MODERATE_REVIEW"
        else:
            flag = "NORMAL"

        results.append(
            {
                "feature": feature,
                "missing_count": missing_n,
                "present_count": present_n,
                "missing_pct": round(
                    missing_mask.mean() * 100,
                    3,
                ),
                "default_rate_when_missing_pct": round(
                    missing_risk_rate * 100,
                    3,
                ),
                "default_rate_when_present_pct": round(
                    present_risk_rate * 100,
                    3,
                ),
                "absolute_rate_difference_pct_points": round(
                    absolute_difference * 100,
                    3,
                ),
                "flag": flag,
            }
        )

    missing_df = pd.DataFrame(results)

    if not missing_df.empty:
        missing_df = (
            missing_df
            .sort_values(
                "absolute_rate_difference_pct_points",
                ascending=False,
            )
            .reset_index(drop=True)
        )

    output = (
        ANALYTICS_DIR
        / "missingness_leakage_audit.parquet"
    )

    missing_df.to_parquet(
        output,
        index=False,
    )

    print("\n  Strongest missingness associations:")

    if not missing_df.empty:
        print(
            missing_df[
                [
                    "feature",
                    "missing_pct",
                    "default_rate_when_missing_pct",
                    "default_rate_when_present_pct",
                    "absolute_rate_difference_pct_points",
                    "flag",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

    print(f"\n  ✓ Saved: {output}")

    return missing_df


# ============================================================
# 4. FINAL ML FEATURE REGISTRY
# ============================================================

def build_ml_feature_registry(
    df: pd.DataFrame,
    feature_audit: pd.DataFrame,
) -> list:
    """
    Produce a conservative list of features that are eligible
    for further ML investigation.

    REVIEW features are deliberately excluded until provenance
    is explicitly resolved.
    """

    print_header("Building Conservative ML Feature Registry")

    eligible = feature_audit[
        feature_audit["ml_eligible"] == True
    ]["feature"].tolist()

    # Defensive checks
    forbidden = set(DEFINITE_LEAKAGE)

    eligible = [
        f for f in eligible
        if f not in forbidden
        and f != BORROWER_ID
    ]

    print(
        f"  Candidate ML features: {len(eligible)}"
    )

    print("\n  Candidate features:")

    for feature in eligible:
        print(f"    ✓ {feature}")

    print(
        "\n  IMPORTANT:"
        "\n  These features passed the structural governance "
        "screen only."
        "\n  Single-feature AUC and missingness audits must still "
        "be reviewed."
    )

    return eligible


# ============================================================
# 5. FINAL REPORT
# ============================================================

def print_final_report(
    feature_audit: pd.DataFrame,
    auc_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    ml_features: list,
):
    """Print concise final governance report."""

    print_header("LEAKAGE AUDIT SUMMARY")

    critical_features = feature_audit[
        feature_audit["severity"] == "CRITICAL"
    ]["feature"].tolist()

    high_features = feature_audit[
        feature_audit["severity"] == "HIGH"
    ]["feature"].tolist()

    auc_critical = []

    if not auc_df.empty:
        auc_critical = auc_df[
            auc_df["flag"] == "CRITICAL_REVIEW"
        ]["feature"].tolist()

    missing_critical = []

    if not missing_df.empty:
        missing_critical = missing_df[
            missing_df["flag"] == "CRITICAL_REVIEW"
        ]["feature"].tolist()

    print(
        f"  Total columns audited:       {len(feature_audit)}"
    )

    print(
        f"  Structurally eligible:       {len(ml_features)}"
    )

    print(
        f"  Critical structural leakage: {len(critical_features)}"
    )

    print(
        f"  High-priority review:        {len(high_features)}"
    )

    print(
        f"  AUC critical-review signals: {len(auc_critical)}"
    )

    print(
        f"  Missingness critical signals:{len(missing_critical)}"
    )

    print("\n  DEFINITE EXCLUSIONS:")

    for feature in critical_features:
        print(f"    🔴 {feature}")

    if auc_critical:
        print(
            "\n  SINGLE-FEATURE AUC INVESTIGATION FLAGS:"
        )

        for feature in auc_critical[:15]:
            print(f"    🟠 {feature}")

    if missing_critical:
        print(
            "\n  MISSINGNESS INVESTIGATION FLAGS:"
        )

        for feature in missing_critical[:15]:
            print(f"    🟠 {feature}")

    print(
        "\n  Interpretation:"
        "\n  • Structural exclusion is based on feature provenance/"
        "prediction-time logic."
        "\n  • High single-feature AUC is an investigation trigger,"
        " not proof of leakage."
        "\n  • Missingness differences indicate association, not"
        "automatic leakage."
        "\n  • Final model eligibility requires both statistical"
        "review and provenance review."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("  INDIA CREDIT RISK — FEATURE LEAKAGE AUDIT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_fact()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_fact(df)

    # --------------------------------------------------------
    # Structural feature audit
    # --------------------------------------------------------

    feature_audit = build_feature_leakage_audit(df)

    # --------------------------------------------------------
    # Single feature predictive strength
    # --------------------------------------------------------

    auc_df = single_feature_auc_audit(df)

    # --------------------------------------------------------
    # Missingness leakage
    # --------------------------------------------------------

    missing_df = missingness_leakage_audit(df)

    # --------------------------------------------------------
    # Conservative ML registry
    # --------------------------------------------------------

    ml_features = build_ml_feature_registry(
        df,
        feature_audit,
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print_final_report(
        feature_audit,
        auc_df,
        missing_df,
        ml_features,
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("✓ LEAKAGE AUDIT COMPLETE")
    print("=" * 70)

    print(
        "\nFiles saved to:"
        "\n  data/gold/exports/analytics/"
        "\n"
        "\n  → feature_leakage_audit.parquet"
        "\n  → single_feature_auc_audit.parquet"
        "\n  → missingness_leakage_audit.parquet"
    )

    print(
        "\nNext:"
        "\n  Review the audit outputs before final ML training."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()