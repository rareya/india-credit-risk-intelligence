"""
feature_semantics_audit.py

INDIA CREDIT RISK — FEATURE SEMANTICS AUDIT

Purpose
-------
Audits the semantic direction of the conservative ML feature set.

This is NOT a leakage test.

It checks whether:

    1. Feature values have plausible distributions.
    2. Feature/target relationships are measurable.
    3. Observed risk direction agrees with expected domain direction.
    4. Suspicious / contradictory features are flagged for review.

Outputs
-------
data/gold/exports/analytics/ml/semantics/

    feature_semantics_audit.parquet
    feature_semantics_audit.csv
    feature_semantics_summary.parquet
    feature_semantics_summary.csv
    feature_semantics_summary.json

Important
---------
Observed association does NOT establish causality.

A direction mismatch is an investigation signal, not proof
of incorrect feature construction or leakage.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path("data/gold/exports/analytics")
ML_DIR = BASE_DIR / "ml"
OUTPUT_DIR = ML_DIR / "semantics"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Actual Gold fact table location in this project.
GOLD_PATH = Path(
    "data/gold/exports/fact_credit_risk.parquet"
)

TARGET = "default_risk"


# ============================================================
# FEATURE REGISTRY CANDIDATES
# ============================================================

REGISTRY_CANDIDATE_PATHS = [
    BASE_DIR / "ml_feature_registry.parquet",
    BASE_DIR / "feature_registry.parquet",
    BASE_DIR / "feature_leakage_audit.parquet",
    ML_DIR / "feature_registry.parquet",
]


# ============================================================
# EXPECTED SEMANTIC DIRECTIONS
# ============================================================

# +1 = higher value is expected to increase risk
# -1 = higher value is expected to decrease risk
#  0 = ambiguous / no simple monotonic expectation
#
# These are domain hypotheses, NOT causal assumptions.

EXPECTED_DIRECTION = {

    # --------------------------------------------------------
    # Delinquency / repayment
    # --------------------------------------------------------

    "delinquency_score": +1,
    "num_times_delinquent": +1,
    "num_times_30p_dpd": +1,
    "num_times_60p_dpd": +1,
    "total_missed_payments": +1,
    "missed_payment_ratio": +1,

    # --------------------------------------------------------
    # Income / financial capacity
    # --------------------------------------------------------

    "monthly_income_inr": -1,
    "income_tier": 0,

    # --------------------------------------------------------
    # Loan structure
    # --------------------------------------------------------

    "total_loans": 0,
    "active_loans": 0,
    "closed_loans": -1,
    "active_loan_ratio": 0,
    "loan_type_diversity": 0,

    "gold_loans": 0,
    "home_loans": 0,
    "personal_loans": 0,
    "credit_card_loans": 0,
    "auto_loans": 0,

    "secured_loans": -1,
    "unsecured_loans": +1,

    # --------------------------------------------------------
    # Demographics
    # --------------------------------------------------------

    "age": 0,
    "gender": 0,
    "education": 0,
    "marital_status": 0,
    "age_band": 0,

    # --------------------------------------------------------
    # Credit history
    # --------------------------------------------------------

    "credit_history_months": -1,

    # --------------------------------------------------------
    # Binary loan indicators
    # --------------------------------------------------------

    "has_gold_loan": 0,
    "has_home_loan": 0,
    "has_personal_loan": 0,
    "has_credit_card": 0,
}


# ============================================================
# PRINT HELPERS
# ============================================================

def print_section(title):
    print("\n" + "━" * 70)
    print(title)
    print("━" * 70)


# ============================================================
# LOAD GOLD DATA
# ============================================================

def load_gold():
    print_section("Loading Gold fact table")

    if not GOLD_PATH.exists():

        # Helpful fallback search in case the project is moved.
        candidates = list(
            Path("data").rglob("fact_credit_risk.parquet")
        )

        if candidates:
            fallback = candidates[0]

            print(
                f"  ! Expected path not found:"
                f" {GOLD_PATH}"
            )

            print(
                f"  ✓ Found Gold fact table at:"
                f" {fallback}"
            )

            path = fallback

        else:
            raise FileNotFoundError(
                f"\nGold fact table not found:\n\n"
                f"{GOLD_PATH}\n\n"
                f"Run:\n"
                f"  python -c "
                f"\"from pathlib import Path; "
                f"[print(p) for p in Path('data').rglob("
                f"'fact_credit_risk.parquet')]\"\n"
            )

    else:
        path = GOLD_PATH

    df = pd.read_parquet(path)

    print(
        f"  ✓ Loaded: {len(df):,} rows × "
        f"{len(df.columns)} columns"
    )

    return df


# ============================================================
# FIND FEATURE REGISTRY
# ============================================================

def load_feature_registry():

    print_section(
        "Loading conservative feature registry"
    )

    registry_path = None

    for path in REGISTRY_CANDIDATE_PATHS:
        if path.exists():
            registry_path = path
            break

    if registry_path is None:

        print(
            "  ! Feature registry not found."
        )

        print(
            "  Using hard-coded conservative "
            "feature list."
        )

        return list(
            EXPECTED_DIRECTION.keys()
        )

    registry = pd.read_parquet(
        registry_path
    )

    print(
        f"  ✓ Registry: {registry_path}"
    )

    print(
        f"  ✓ Registry rows: {len(registry)}"
    )

    # --------------------------------------------------------
    # Conservative eligibility columns
    # --------------------------------------------------------

    eligibility_columns = [
        "ml_eligible_conservative",
        "conservative_eligible",
        "eligible_conservative",
    ]

    selected_column = None

    for col in eligibility_columns:

        if col in registry.columns:
            selected_column = col
            break

    if selected_column is not None:

        mask = (
            registry[selected_column]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(
                [
                    "true",
                    "1",
                    "yes",
                ]
            )
        )

        features = (
            registry.loc[
                mask,
                "feature",
            ]
            .astype(str)
            .tolist()
        )

    elif "classification" in registry.columns:

        features = (
            registry.loc[
                registry["classification"]
                .astype(str)
                .str.upper()
                .eq("LIKELY_SAFE"),
                "feature",
            ]
            .astype(str)
            .tolist()
        )

    elif "feature" in registry.columns:

        features = (
            registry["feature"]
            .astype(str)
            .tolist()
        )

    else:

        features = list(
            EXPECTED_DIRECTION.keys()
        )

    # --------------------------------------------------------
    # Keep only features with semantic definitions
    # --------------------------------------------------------

    features = [
        feature
        for feature in features
        if feature in EXPECTED_DIRECTION
    ]

    if not features:

        features = list(
            EXPECTED_DIRECTION.keys()
        )

    print(
        f"  ✓ Conservative features found:"
        f" {len(features)}"
    )

    return features


# ============================================================
# VALIDATE TARGET
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

    # Convert safely for validation.
    target_values = (
        pd.to_numeric(
            df[TARGET],
            errors="coerce",
        )
    )

    if target_values.isna().any():

        raise ValueError(
            "Target contains non-numeric values."
        )

    unique_values = set(
        target_values.unique()
    )

    if not unique_values.issubset({0, 1}):

        raise ValueError(
            "Target must be binary 0/1. "
            f"Found: {unique_values}"
        )

    print("  ✓ Target validated")

    print(
        f"  Low risk:  "
        f"{(target_values == 0).sum():,}"
    )

    print(
        f"  High risk: "
        f"{(target_values == 1).sum():,}"
    )


# ============================================================
# NUMERIC ASSOCIATION
# ============================================================

def calculate_numeric_association(
    series,
    y,
):

    numeric_series = pd.to_numeric(
        series,
        errors="coerce",
    )

    target = pd.to_numeric(
        y,
        errors="coerce",
    )

    mask = (
        numeric_series.notna()
        & target.notna()
    )

    x = numeric_series.loc[mask]
    target = target.loc[mask]

    if len(x) < 20:
        return np.nan, np.nan

    if x.nunique() <= 1:
        return np.nan, np.nan

    # Pearson correlation.
    try:

        correlation = x.corr(
            target
        )

    except Exception:

        correlation = np.nan

    # Single-feature AUC.
    try:

        auc = roc_auc_score(
            target,
            x,
        )

    except Exception:

        auc = np.nan

    return correlation, auc


# ============================================================
# EXPECTED DIRECTION LABEL
# ============================================================

def expected_direction_label(
    value
):

    if value == 1:

        return (
            "HIGHER_VALUE_INCREASES_RISK"
        )

    if value == -1:

        return (
            "HIGHER_VALUE_DECREASES_RISK"
        )

    return "AMBIGUOUS"


# ============================================================
# OBSERVED DIRECTION
# ============================================================

def observed_direction_from_auc(
    auc
):

    if pd.isna(auc):

        return "UNAVAILABLE"

    if auc >= 0.55:

        return (
            "HIGHER_VALUE_INCREASES_RISK"
        )

    if auc <= 0.45:

        return (
            "HIGHER_VALUE_DECREASES_RISK"
        )

    return "WEAK_OR_NON_MONOTONIC"


# ============================================================
# DIRECTION CHECK
# ============================================================

def compare_direction(
    expected,
    auc,
):

    if expected == 0:

        return (
            "AMBIGUOUS_EXPECTATION"
        )

    if pd.isna(auc):

        return "UNAVAILABLE"

    observed = (
        observed_direction_from_auc(
            auc
        )
    )

    if (
        observed
        == "WEAK_OR_NON_MONOTONIC"
    ):

        return "WEAK_ASSOCIATION"

    expected_label = (
        expected_direction_label(
            expected
        )
    )

    if observed == expected_label:

        return "CONSISTENT"

    return "DIRECTION_MISMATCH"


# ============================================================
# FEATURE AUDIT
# ============================================================

def audit_feature(
    df,
    feature,
):

    series = df[feature]

    expected = (
        EXPECTED_DIRECTION.get(
            feature,
            0,
        )
    )

    row = {

        "feature": feature,

        "dtype": str(
            series.dtype
        ),

        "n_rows": int(
            len(series)
        ),

        "n_missing": int(
            series.isna().sum()
        ),

        "missing_pct": float(
            round(
                series.isna().mean()
                * 100,
                4,
            )
        ),

        "n_unique": int(
            series.nunique(
                dropna=True
            )
        ),

        "expected_direction":
            expected_direction_label(
                expected
            ),

        "observed_direction":
            "UNAVAILABLE",

        "direction_status":
            "UNAVAILABLE",

        "pearson_correlation":
            np.nan,

        "single_feature_auc":
            np.nan,

        "min":
            np.nan,

        "max":
            np.nan,

        "mean":
            np.nan,

        "median":
            np.nan,

        "std":
            np.nan,

        "is_numeric":
            False,

        "is_categorical":
            False,

        "strong_signal":
            False,

        "requires_review":
            False,

        "review_reason":
            "",
    }

    # ========================================================
    # NUMERIC FEATURE
    # ========================================================

    if pd.api.types.is_numeric_dtype(
        series
    ):

        row["is_numeric"] = True

        numeric = pd.to_numeric(
            series,
            errors="coerce",
        )

        # Explicit float conversion prevents
        # numpy bool/object contamination.
        row["min"] = (
            float(numeric.min())
            if numeric.notna().any()
            else np.nan
        )

        row["max"] = (
            float(numeric.max())
            if numeric.notna().any()
            else np.nan
        )

        row["mean"] = (
            float(numeric.mean())
            if numeric.notna().any()
            else np.nan
        )

        row["median"] = (
            float(numeric.median())
            if numeric.notna().any()
            else np.nan
        )

        row["std"] = (
            float(numeric.std())
            if numeric.notna().any()
            else np.nan
        )

        correlation, auc = (
            calculate_numeric_association(
                numeric,
                df[TARGET],
            )
        )

        row[
            "pearson_correlation"
        ] = (
            float(correlation)
            if not pd.isna(correlation)
            else np.nan
        )

        row[
            "single_feature_auc"
        ] = (
            float(auc)
            if not pd.isna(auc)
            else np.nan
        )

        row[
            "observed_direction"
        ] = (
            observed_direction_from_auc(
                auc
            )
        )

        status = (
            compare_direction(
                expected,
                auc,
            )
        )

        row[
            "direction_status"
        ] = status

        # ----------------------------------------------------
        # Direction mismatch
        # ----------------------------------------------------

        if status == "DIRECTION_MISMATCH":

            row[
                "requires_review"
            ] = True

            row[
                "review_reason"
            ] = (
                "Observed single-feature risk "
                "direction conflicts with expected "
                "semantic direction."
            )

        # ----------------------------------------------------
        # Very strong predictive feature
        # ----------------------------------------------------

        elif (
            not pd.isna(auc)
            and auc >= 0.90
        ):

            row[
                "strong_signal"
            ] = True

            row[
                "requires_review"
            ] = True

            row[
                "review_reason"
            ] = (
                "Very strong single-feature "
                "predictive association. "
                "Review provenance and construction."
            )

        elif (
            not pd.isna(auc)
            and auc <= 0.10
        ):

            row[
                "strong_signal"
            ] = True

            row[
                "requires_review"
            ] = True

            row[
                "review_reason"
            ] = (
                "Very strong inverse predictive "
                "association. Review feature semantics "
                "and provenance."
            )

    # ========================================================
    # CATEGORICAL FEATURE
    # ========================================================

    else:

        row[
            "is_categorical"
        ] = True

        row[
            "observed_direction"
        ] = "CATEGORICAL"

        row[
            "direction_status"
        ] = (
            "CATEGORICAL_REVIEW"
            if expected != 0
            else "CATEGORICAL"
        )

        # Missing categorical values are worth
        # documenting, but they are not automatically
        # considered leakage or failure.
        if series.isna().any():

            row[
                "requires_review"
            ] = True

            row[
                "review_reason"
            ] = (
                "Categorical feature contains "
                "missing values. Review whether "
                "missingness has semantic meaning."
            )

    return row


# ============================================================
# BUILD AUDIT
# ============================================================

def build_audit(
    df,
    features,
):

    print_section(
        "Feature semantic audit"
    )

    rows = []

    for feature in features:

        if feature not in df.columns:

            print(
                f"  ⚠ Missing from Gold: "
                f"{feature}"
            )

            rows.append(
                {
                    "feature": feature,
                    "dtype": "MISSING",
                    "n_rows": int(len(df)),
                    "n_missing": int(len(df)),
                    "missing_pct": 100.0,
                    "n_unique": 0,
                    "expected_direction":
                        expected_direction_label(
                            EXPECTED_DIRECTION[
                                feature
                            ]
                        ),
                    "observed_direction":
                        "UNAVAILABLE",
                    "direction_status":
                        "MISSING_FEATURE",
                    "pearson_correlation":
                        np.nan,
                    "single_feature_auc":
                        np.nan,
                    "min":
                        np.nan,
                    "max":
                        np.nan,
                    "mean":
                        np.nan,
                    "median":
                        np.nan,
                    "std":
                        np.nan,
                    "is_numeric":
                        False,
                    "is_categorical":
                        False,
                    "strong_signal":
                        False,
                    "requires_review":
                        True,
                    "review_reason":
                        (
                            "Feature listed in "
                            "conservative registry "
                            "but missing from Gold."
                        ),
                }
            )

            continue

        print(
            f"  Auditing: {feature}"
        )

        rows.append(
            audit_feature(
                df,
                feature,
            )
        )

    audit = pd.DataFrame(
        rows
    )

    return audit


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    audit
):

    print_section(
        "SEMANTIC AUDIT RESULTS"
    )

    # --------------------------------------------------------
    # Direction mismatches
    # --------------------------------------------------------

    print(
        "\n  Direction mismatches:"
    )

    mismatches = audit[
        audit["direction_status"]
        == "DIRECTION_MISMATCH"
    ]

    if mismatches.empty:

        print(
            "    ✓ None detected"
        )

    else:

        print(
            mismatches[
                [
                    "feature",
                    "expected_direction",
                    "observed_direction",
                    "single_feature_auc",
                ]
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Strong signals
    # --------------------------------------------------------

    print(
        "\n  Strong single-feature signals:"
    )

    strong = audit[
        audit["strong_signal"] == True
    ]

    if strong.empty:

        print(
            "    ✓ None detected"
        )

    else:

        print(
            strong[
                [
                    "feature",
                    "single_feature_auc",
                    "direction_status",
                    "requires_review",
                ]
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Review required
    # --------------------------------------------------------

    print(
        "\n  Features requiring review:"
    )

    review = audit[
        audit["requires_review"] == True
    ]

    if review.empty:

        print(
            "    ✓ None"
        )

    else:

        print(
            review[
                [
                    "feature",
                    "direction_status",
                    "review_reason",
                ]
            ].to_string(
                index=False
            )
        )


# ============================================================
# BUILD SUMMARY
# ============================================================

def build_summary(
    audit
):

    mismatches = int(
        (
            audit["direction_status"]
            == "DIRECTION_MISMATCH"
        ).sum()
    )

    strong_signals = int(
        (
            audit["strong_signal"]
            == True
        ).sum()
    )

    review_required = int(
        (
            audit["requires_review"]
            == True
        ).sum()
    )

    missing_features = int(
        (
            audit["direction_status"]
            == "MISSING_FEATURE"
        ).sum()
    )

    categorical_features = int(
        (
            audit["is_categorical"]
            == True
        ).sum()
    )

    numeric_features = int(
        (
            audit["is_numeric"]
            == True
        ).sum()
    )

    consistent_features = int(
        (
            audit["direction_status"]
            == "CONSISTENT"
        ).sum()
    )

    weak_features = int(
        (
            audit["direction_status"]
            == "WEAK_ASSOCIATION"
        ).sum()
    )

    ambiguous_features = int(
        (
            audit["direction_status"]
            == "AMBIGUOUS_EXPECTATION"
        ).sum()
    )

    summary = {

        "features_audited":
            int(len(audit)),

        "numeric_features":
            numeric_features,

        "categorical_features":
            categorical_features,

        "consistent_direction_features":
            consistent_features,

        "weak_association_features":
            weak_features,

        "ambiguous_expectation_features":
            ambiguous_features,

        "direction_mismatches":
            mismatches,

        "strong_single_feature_signals":
            strong_signals,

        "features_requiring_review":
            review_required,

        "missing_features":
            missing_features,

        "audit_status":
            (
                "PASS"
                if (
                    mismatches == 0
                    and strong_signals == 0
                    and missing_features == 0
                )
                else "REVIEW_REQUIRED"
            ),

        "interpretation":
            (
                "Semantic associations are "
                "consistent with the configured "
                "domain expectations."
                if (
                    mismatches == 0
                    and strong_signals == 0
                    and missing_features == 0
                )
                else
                "One or more features require "
                "semantic/provenance review."
            ),

        "important_note":
            (
                "Observed association does not "
                "establish causality. A direction "
                "mismatch or strong association is "
                "an investigation signal, not proof "
                "of leakage."
            ),
    }

    return summary


# ============================================================
# PYARROW-SAFE NORMALIZATION
# ============================================================

def normalize_for_parquet(
    df
):

    out = df.copy()

    # --------------------------------------------------------
    # Convert pandas nullable / NumPy scalar values
    # into ordinary Python-compatible values.
    # --------------------------------------------------------

    for col in out.columns:

        if pd.api.types.is_bool_dtype(
            out[col]
        ):

            out[col] = (
                out[col]
                .astype(bool)
            )

        elif pd.api.types.is_numeric_dtype(
            out[col]
        ):

            # Force numeric columns into
            # consistent float/int representations.
            out[col] = pd.to_numeric(
                out[col],
                errors="coerce",
            )

        else:

            # Object/string columns should contain
            # strings or missing values only.
            out[col] = (
                out[col]
                .map(
                    lambda x:
                        None
                        if pd.isna(x)
                        else str(x)
                )
            )

    return out


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(
    audit,
    summary,
    output_dir=OUTPUT_DIR,
):

    print_section(
        "Saving semantic audit outputs"
    )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Normalize audit
    # --------------------------------------------------------

    audit_out = normalize_for_parquet(
        audit
    )

    audit_parquet = (
        output_dir
        / "feature_semantics_audit.parquet"
    )

    audit_csv = (
        output_dir
        / "feature_semantics_audit.csv"
    )

    audit_out.to_parquet(
        audit_parquet,
        index=False,
    )

    audit_out.to_csv(
        audit_csv,
        index=False,
    )

    print(
        f"  ✓ Saved: {audit_parquet}"
    )

    print(
        f"  ✓ Saved: {audit_csv}"
    )

    # --------------------------------------------------------
    # Summary JSON
    # --------------------------------------------------------

    summary_json = (
        output_dir
        / "feature_semantics_summary.json"
    )

    with open(
        summary_json,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            default=str,
        )

    print(
        f"  ✓ Saved: {summary_json}"
    )

    # --------------------------------------------------------
    # Summary DataFrame
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        [summary]
    )

    summary_df = normalize_for_parquet(
        summary_df
    )

    summary_parquet = (
        output_dir
        / "feature_semantics_summary.parquet"
    )

    summary_csv = (
        output_dir
        / "feature_semantics_summary.csv"
    )

    summary_df.to_parquet(
        summary_parquet,
        index=False,
    )

    summary_df.to_csv(
        summary_csv,
        index=False,
    )

    print(
        f"  ✓ Saved: {summary_parquet}"
    )

    print(
        f"  ✓ Saved: {summary_csv}"
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "  INDIA CREDIT RISK — "
        "FEATURE SEMANTICS AUDIT"
    )

    print("=" * 70)

    warnings.filterwarnings(
        "ignore"
    )

    # --------------------------------------------------------
    # Load Gold
    # --------------------------------------------------------

    df = load_gold()

    # --------------------------------------------------------
    # Validate target
    # --------------------------------------------------------

    validate_target(
        df
    )

    # --------------------------------------------------------
    # Load feature registry
    # --------------------------------------------------------

    features = (
        load_feature_registry()
    )

    print(
        "\n  Features selected "
        "for semantic audit:"
    )

    for feature in features:

        print(
            f"    ✓ {feature}"
        )

    # --------------------------------------------------------
    # Ensure features exist / audit
    # --------------------------------------------------------

    audit = build_audit(
        df,
        features,
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print_results(
        audit
    )

    # --------------------------------------------------------
    # Build summary
    # --------------------------------------------------------

    summary = build_summary(
        audit
    )

    # --------------------------------------------------------
    # Save outputs
    #
    # IMPORTANT:
    # This is the corrected call.
    # save_outputs() now receives all required
    # arguments explicitly.
    # --------------------------------------------------------

    summary = save_outputs(
        audit=audit,
        summary=summary,
        output_dir=OUTPUT_DIR,
    )

    # --------------------------------------------------------
    # Final governance message
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "✓ FEATURE SEMANTICS AUDIT COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"""
Features audited:
  {summary["features_audited"]}

Numeric features:
  {summary["numeric_features"]}

Categorical features:
  {summary["categorical_features"]}

Direction-consistent features:
  {summary["consistent_direction_features"]}

Weak associations:
  {summary["weak_association_features"]}

Direction mismatches:
  {summary["direction_mismatches"]}

Strong single-feature signals:
  {summary["strong_single_feature_signals"]}

Features requiring review:
  {summary["features_requiring_review"]}

Missing features:
  {summary["missing_features"]}

Audit status:
  {summary["audit_status"]}

Outputs:
  {OUTPUT_DIR}

IMPORTANT:
  A direction mismatch is an investigation signal.
  It does NOT by itself prove leakage or causality.

Next:
  Review any flagged features against their
  construction logic, provenance, and
  prediction-time availability.
"""
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()