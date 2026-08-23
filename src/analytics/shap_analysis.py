"""
shap_analysis.py

INDIA CREDIT RISK — SHAP EXPLAINABILITY ANALYSIS
================================================

Explains the Conservative XGBoost model selected by the
feature-governance / leakage-audit pipeline.

IMPORTANT:
    This file does NOT perform probability calibration.

    Calibration is handled separately by:
        calibrate_model.py

This module:

    1. Loads Gold fact table
    2. Loads the conservative ML feature registry
    3. Validates that no structurally excluded features are used
    4. Prepares numeric features
    5. Trains the Conservative XGBoost model
    6. Computes SHAP values
    7. Produces global feature importance
    8. Produces mean absolute SHAP importance
    9. Produces direction of feature effects
   10. Produces individual borrower explanations
   11. Saves outputs for dashboard / research use

Run:

    python src/analytics/shap_analysis.py

Outputs:

    data/gold/exports/analytics/ml/shap/
        shap_feature_importance.parquet
        shap_feature_importance.csv
        shap_summary.parquet
        shap_borrower_explanations.parquet
        shap_metadata.json
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

import shap

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

GOLD_DIR = Path("data/gold/exports")

ANALYTICS_DIR = GOLD_DIR / "analytics"

ML_DIR = ANALYTICS_DIR / "ml"

SHAP_DIR = ML_DIR / "shap"

SHAP_DIR.mkdir(
    parents=True,
    exist_ok=True
)


FACT_PATH = (
    GOLD_DIR /
    "fact_credit_risk.parquet"
)

FEATURE_REGISTRY_PATH = (
    ANALYTICS_DIR /
    "ml_feature_registry.parquet"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

SHAP_SAMPLE_SIZE = 5000

BORROWER_EXPLANATION_SAMPLE = 100


# Conservative XGBoost configuration
#
# IMPORTANT:
# Keep this aligned with run_ml_model.py.
#
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 2.0,
    "reg_alpha": 0.1,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ============================================================
# EXPECTED STRUCTURAL EXCLUSIONS
# ============================================================

STRUCTURAL_EXCLUSIONS = {
    "default_risk",
    "risk_band",
    "risk_grade",
    "risk_grade_numeric",
    "borrower_id",
}


# ============================================================
# LOAD DATA
# ============================================================

def load_fact():

    print("=" * 70)
    print("  INDIA CREDIT RISK — SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 70)

    print("\nLoading Gold fact table...")

    if not FACT_PATH.exists():

        raise FileNotFoundError(
            f"Missing Gold fact table:\n{FACT_PATH}\n\n"
            "Run build_gold.py first."
        )

    df = pd.read_parquet(FACT_PATH)

    print(
        f"  ✓ Loaded: "
        f"{df.shape[0]:,} rows × "
        f"{df.shape[1]} columns"
    )

    return df


# ============================================================
# VALIDATE TARGET
# ============================================================

def validate_target(df):

    print("\n" + "━" * 70)
    print("Validating target")
    print("━" * 70)

    if "default_risk" not in df.columns:

        raise ValueError(
            "default_risk column missing."
        )

    if df["default_risk"].isna().any():

        raise ValueError(
            "default_risk contains missing values."
        )

    unique_target = set(
        df["default_risk"].unique()
    )

    if not unique_target.issubset({0, 1}):

        raise ValueError(
            f"default_risk must be binary. "
            f"Found: {unique_target}"
        )

    print("  ✓ Target validated")

    print(
        f"  Low risk:  "
        f"{(df['default_risk'] == 0).sum():,}"
    )

    print(
        f"  High risk: "
        f"{(df['default_risk'] == 1).sum():,}"
    )


# ============================================================
# LOAD FEATURE REGISTRY
# ============================================================

def load_feature_registry(df):

    print("\n" + "━" * 70)
    print("Loading Conservative ML feature registry")
    print("━" * 70)

    if not FEATURE_REGISTRY_PATH.exists():

        raise FileNotFoundError(
            f"""
Missing feature registry:

{FEATURE_REGISTRY_PATH}

Run leakage_audit.py first.
"""
        )

    registry = pd.read_parquet(
        FEATURE_REGISTRY_PATH
    )

    print(
        f"  Registry rows: "
        f"{len(registry)}"
    )

    print(
        "\n  Registry columns:"
    )

    print(
        "  "
        + ", ".join(registry.columns)
    )

    # --------------------------------------------------------
    # Try to identify feature column
    # --------------------------------------------------------

    possible_feature_columns = [
        "feature",
        "feature_name",
        "column",
        "column_name",
    ]

    feature_column = None

    for col in possible_feature_columns:

        if col in registry.columns:

            feature_column = col
            break

    if feature_column is None:

        raise ValueError(
            "Could not identify feature-name column "
            "in ml_feature_registry.parquet."
        )

    # --------------------------------------------------------
    # Try to identify eligibility classification
    # --------------------------------------------------------

    possible_classification_columns = [
        "classification",
        "eligibility",
        "status",
        "decision",
    ]

    classification_column = None

    for col in possible_classification_columns:

        if col in registry.columns:

            classification_column = col
            break

    # --------------------------------------------------------
    # Conservative feature selection
    # --------------------------------------------------------

    if classification_column is not None:

        eligible_values = {
            "LIKELY_SAFE",
            "SAFE",
            "ELIGIBLE",
            "APPROVED",
            "CANDIDATE",
        }

        mask = (
            registry[classification_column]
            .astype(str)
            .str.upper()
            .isin(eligible_values)
        )

        features = (
            registry.loc[
                mask,
                feature_column
            ]
            .astype(str)
            .tolist()
        )

    else:

        features = (
            registry[feature_column]
            .astype(str)
            .tolist()
        )

    # --------------------------------------------------------
    # Remove structural exclusions defensively
    # --------------------------------------------------------

    features = [
        f for f in features
        if f not in STRUCTURAL_EXCLUSIONS
    ]

    # --------------------------------------------------------
    # Keep only columns actually present
    # --------------------------------------------------------

    missing = [
        f for f in features
        if f not in df.columns
    ]

    if missing:

        raise ValueError(
            "Feature registry contains columns "
            "missing from Gold fact table:\n"
            + "\n".join(missing)
        )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    features = list(
        dict.fromkeys(features)
    )

    if not features:

        raise ValueError(
            "No eligible Conservative ML features found."
        )

    print(
        f"\n  ✓ Conservative feature count: "
        f"{len(features)}"
    )

    print("\n  Conservative features:")

    for feature in features:

        print(
            f"    ✓ {feature}"
        )

    # --------------------------------------------------------
    # Final structural leakage guard
    # --------------------------------------------------------

    leakage_found = (
        set(features)
        & STRUCTURAL_EXCLUSIONS
    )

    if leakage_found:

        raise ValueError(
            "STRUCTURAL LEAKAGE GUARD FAILED.\n"
            "Excluded features found:\n"
            + "\n".join(sorted(leakage_found))
        )

    print(
        "\n  ✓ Structural leakage guard passed"
    )

    return features


# ============================================================
# FEATURE NORMALIZATION
# ============================================================

def normalize_features(X):

    """
    Convert boolean / boolean-like columns to numeric 0/1.

    This is necessary because the Gold fact table may contain:

        bool
        BooleanDtype
        object columns containing True / False

    XGBoost and SHAP require a clean numeric matrix.
    """

    X = X.copy()

    # --------------------------------------------------------
    # Handle actual boolean columns
    # --------------------------------------------------------

    for col in X.columns:

        if pd.api.types.is_bool_dtype(
            X[col]
        ):

            X[col] = (
                X[col]
                .astype(int)
            )

    # --------------------------------------------------------
    # Handle object columns containing True / False
    # --------------------------------------------------------

    for col in X.columns:

        if X[col].dtype == "object":

            non_null = (
                X[col]
                .dropna()
                .astype(str)
                .str.lower()
                .unique()
            )

            unique_values = set(
                non_null
            )

            if unique_values.issubset(
                {"true", "false"}
            ):

                X[col] = (
                    X[col]
                    .astype(str)
                    .str.lower()
                    .map({
                        "true": 1,
                        "false": 0,
                    })
                )

    # --------------------------------------------------------
    # Handle categorical text
    # --------------------------------------------------------

    categorical_columns = [
        col
        for col in X.columns
        if X[col].dtype == "object"
    ]

    if categorical_columns:

        print(
            "\n  Encoding categorical features:"
        )

        for col in categorical_columns:

            print(
                f"    → {col}"
            )

            # Stable category encoding
            #
            # category codes are sufficient here because
            # the purpose is consistency with the model
            # input rather than semantic ordering.
            X[col] = (
                X[col]
                .astype("category")
                .cat.codes
                .replace(-1, np.nan)
            )

    # --------------------------------------------------------
    # Convert numeric-looking columns
    # --------------------------------------------------------

    for col in X.columns:

        if not pd.api.types.is_numeric_dtype(
            X[col]
        ):

            X[col] = pd.to_numeric(
                X[col],
                errors="coerce"
            )

    # --------------------------------------------------------
    # Final numeric validation
    # --------------------------------------------------------

    non_numeric = (
        X.select_dtypes(
            exclude=[np.number]
        )
        .columns
        .tolist()
    )

    if non_numeric:

        raise ValueError(
            "Non-numeric features remain:\n"
            + "\n".join(non_numeric)
        )

    return X


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(
    df,
    features
):

    print("\n" + "━" * 70)
    print("Preparing SHAP feature matrix")
    print("━" * 70)

    X = df[
        features
    ].copy()

    y = df[
        "default_risk"
    ].astype(int)

    print(
        f"  Raw feature matrix: "
        f"{X.shape[0]:,} × "
        f"{X.shape[1]}"
    )

    # Normalize booleans / categorical values
    X = normalize_features(X)

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_before = (
        X.isna()
        .sum()
        .sum()
    )

    if missing_before > 0:

        print(
            f"\n  Missing feature values: "
            f"{missing_before:,}"
        )

        print(
            "  Applying median imputation..."
        )

        for col in X.columns:

            if X[col].isna().any():

                median = X[col].median()

                if pd.isna(median):

                    median = 0

                X[col] = (
                    X[col]
                    .fillna(median)
                )

        print(
            "  ✓ Missing values handled"
        )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if X.isna().any().any():

        raise ValueError(
            "NaN values remain after preprocessing."
        )

    if np.isinf(
        X.to_numpy()
    ).any():

        raise ValueError(
            "Infinite values found in feature matrix."
        )

    print(
        "\n  ✓ Feature matrix numeric"
    )

    print(
        f"  ✓ Features: {X.shape[1]}"
    )

    print(
        f"  ✓ Rows:     {X.shape[0]:,}"
    )

    return X, y


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_data(
    X,
    y
):

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    print("\n" + "━" * 70)
    print("Train / test split")
    print("━" * 70)

    print(
        f"  Train: {len(X_train):,}"
    )

    print(
        f"  Test:  {len(X_test):,}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# ============================================================
# TRAIN CONSERVATIVE XGBOOST
# ============================================================

def train_model(
    X_train,
    y_train
):

    print("\n" + "━" * 70)
    print("Training Conservative XGBoost")
    print("━" * 70)

    model = XGBClassifier(
        **XGB_PARAMS
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "  ✓ Model training complete"
    )

    return model


# ============================================================
# MODEL PERFORMANCE CHECK
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    probability = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    roc_auc = roc_auc_score(
        y_test,
        probability
    )

    pr_auc = average_precision_score(
        y_test,
        probability
    )

    brier = brier_score_loss(
        y_test,
        probability
    )

    print("\n" + "━" * 70)
    print("Model performance check")
    print("━" * 70)

    print(
        f"  ROC-AUC: {roc_auc:.4f}"
    )

    print(
        f"  PR-AUC:  {pr_auc:.4f}"
    )

    print(
        f"  Brier:   {brier:.4f}"
    )

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "brier_score": float(brier),
    }


# ============================================================
# COMPUTE SHAP VALUES
# ============================================================

def compute_shap(
    model,
    X_test
):

    print("\n" + "━" * 70)
    print("Computing SHAP values")
    print("━" * 70)

    sample_size = min(
        SHAP_SAMPLE_SIZE,
        len(X_test)
    )

    # Reproducible sample
    X_sample = (
        X_test
        .sample(
            n=sample_size,
            random_state=RANDOM_STATE
        )
        .copy()
    )

    print(
        f"  SHAP sample: "
        f"{len(X_sample):,} borrowers"
    )

    print(
        "  Building TreeExplainer..."
    )

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = (
        explainer.shap_values(
            X_sample
        )
    )

    # --------------------------------------------------------
    # SHAP versions can return:
    #
    #   ndarray
    #   list
    #   Explanation object
    #
    # Normalize everything.
    # --------------------------------------------------------

    if isinstance(
        shap_values,
        list
    ):

        shap_values = shap_values[-1]

    elif hasattr(
        shap_values,
        "values"
    ):

        shap_values = (
            shap_values.values
        )

    shap_values = np.asarray(
        shap_values
    )

    # Some SHAP versions may return
    # (rows, features, classes)
    if shap_values.ndim == 3:

        shap_values = (
            shap_values[:, :, -1]
        )

    if shap_values.ndim != 2:

        raise ValueError(
            f"Unexpected SHAP shape: "
            f"{shap_values.shape}"
        )

    print(
        f"  ✓ SHAP matrix: "
        f"{shap_values.shape}"
    )

    return (
        X_sample,
        shap_values,
        explainer,
    )


# ============================================================
# GLOBAL SHAP IMPORTANCE
# ============================================================

def build_global_importance(
    X_sample,
    shap_values
):

    print("\n" + "━" * 70)
    print("Global SHAP feature importance")
    print("━" * 70)

    features = list(
        X_sample.columns
    )

    mean_abs_shap = (
        np.abs(shap_values)
        .mean(axis=0)
    )

    mean_shap = (
        shap_values
        .mean(axis=0)
    )

    # --------------------------------------------------------
    # Direction
    #
    # This is a simple global directional indicator.
    # Positive mean SHAP means the feature generally pushes
    # predictions toward the positive/default class.
    # --------------------------------------------------------

    direction = np.where(
        mean_shap > 0,
        "higher_values_generally_increase_risk",
        "higher_values_generally_decrease_risk",
    )

    importance = pd.DataFrame({

        "feature":
            features,

        "mean_abs_shap":
            mean_abs_shap,

        "mean_shap":
            mean_shap,

        "direction":
            direction,

    })

    importance = (
        importance
        .sort_values(
            "mean_abs_shap",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    importance["rank"] = (
        importance.index + 1
    )

    importance = importance[
        [
            "rank",
            "feature",
            "mean_abs_shap",
            "mean_shap",
            "direction",
        ]
    ]

    print(
        "\n  Top 20 SHAP features:"
    )

    print(
        importance
        .head(20)
        .to_string(
            index=False
        )
    )

    return importance


# ============================================================
# SHAP SUMMARY DATASET
# ============================================================

def build_shap_summary(
    X_sample,
    shap_values
):

    """
    Long-format SHAP dataset.

    One row = one borrower-feature combination.

    Useful for:
        dashboard
        distribution analysis
        feature effect analysis
    """

    rows = []

    for j, feature in enumerate(
        X_sample.columns
    ):

        values = (
            X_sample[feature]
            .to_numpy()
        )

        shap_feature = (
            shap_values[:, j]
        )

        for i in range(
            len(X_sample)
        ):

            rows.append({

                "row_index": i,

                "feature": feature,

                "feature_value":
                    float(values[i]),

                "shap_value":
                    float(shap_feature[i]),

                "absolute_shap_value":
                    float(
                        abs(
                            shap_feature[i]
                        )
                    ),

            })

    summary = pd.DataFrame(
        rows
    )

    return summary


# ============================================================
# INDIVIDUAL BORROWER EXPLANATIONS
# ============================================================

def build_borrower_explanations(
    X_sample,
    shap_values
):

    """
    Creates a compact explanation table.

    For each sampled borrower, stores the strongest
    features pushing risk upward or downward.
    """

    n = min(
        BORROWER_EXPLANATION_SAMPLE,
        len(X_sample)
    )

    rows = []

    for i in range(n):

        borrower_features = []

        for j, feature in enumerate(
            X_sample.columns
        ):

            value = (
                float(
                    X_sample.iloc[
                        i,
                        j
                    ]
                )
            )

            shap_value = (
                float(
                    shap_values[
                        i,
                        j
                    ]
                )
            )

            borrower_features.append({

                "feature": feature,

                "feature_value": value,

                "shap_value": shap_value,

                "abs_shap":
                    abs(shap_value),

            })

        borrower_features = sorted(
            borrower_features,
            key=lambda x:
                x["abs_shap"],
            reverse=True
        )

        top_features = (
            borrower_features[:10]
        )

        for rank, item in enumerate(
            top_features,
            start=1
        ):

            rows.append({

                "sample_borrower_index":
                    i,

                "rank":
                    rank,

                "feature":
                    item["feature"],

                "feature_value":
                    item["feature_value"],

                "shap_value":
                    item["shap_value"],

                "effect":
                    (
                        "increases_risk"
                        if item["shap_value"] > 0
                        else "decreases_risk"
                    ),

            })

    return pd.DataFrame(
        rows
    )


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(
    importance,
    summary,
    borrower_explanations,
    metadata
):

    print("\n" + "━" * 70)
    print("Saving SHAP outputs")
    print("━" * 70)

    importance.to_parquet(
        SHAP_DIR /
        "shap_feature_importance.parquet",
        index=False,
    )

    importance.to_csv(
        SHAP_DIR /
        "shap_feature_importance.csv",
        index=False,
    )

    summary.to_parquet(
        SHAP_DIR /
        "shap_summary.parquet",
        index=False,
    )

    borrower_explanations.to_parquet(
        SHAP_DIR /
        "shap_borrower_explanations.parquet",
        index=False,
    )

    with open(
        SHAP_DIR /
        "shap_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(
        "  ✓ shap_feature_importance.parquet"
    )

    print(
        "  ✓ shap_feature_importance.csv"
    )

    print(
        "  ✓ shap_summary.parquet"
    )

    print(
        "  ✓ shap_borrower_explanations.parquet"
    )

    print(
        "  ✓ shap_metadata.json"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_fact()

    validate_target(
        df
    )

    features = load_feature_registry(
        df
    )

    X, y = prepare_features(
        df,
        features
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_data(
        X,
        y
    )

    model = train_model(
        X_train,
        y_train
    )

    performance = evaluate_model(
        model,
        X_test,
        y_test
    )

    (
        X_shap,
        shap_values,
        explainer,
    ) = compute_shap(
        model,
        X_test
    )

    importance = (
        build_global_importance(
            X_shap,
            shap_values
        )
    )

    summary = (
        build_shap_summary(
            X_shap,
            shap_values
        )
    )

    borrower_explanations = (
        build_borrower_explanations(
            X_shap,
            shap_values
        )
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {

        "analysis":
            "SHAP explainability analysis",

        "model":
            "Conservative XGBoost",

        "random_state":
            RANDOM_STATE,

        "total_rows":
            int(len(df)),

        "training_rows":
            int(len(X_train)),

        "testing_rows":
            int(len(X_test)),

        "feature_count":
            int(len(features)),

        "features":
            features,

        "shap_sample_size":
            int(len(X_shap)),

        "performance":
            performance,

        "structural_exclusions":
            sorted(
                STRUCTURAL_EXCLUSIONS
            ),

        "calibration":
            "NOT PERFORMED IN THIS FILE",

        "interpretation_note":
            (
                "SHAP values describe model associations "
                "and contribution to predictions. They do "
                "not establish causal relationships."
            ),

    }

    save_outputs(
        importance,
        summary,
        borrower_explanations,
        metadata
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("✓ SHAP ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        "\nCandidate model:"
        "\n  Conservative XGBoost"
    )

    print(
        f"\nFeatures explained: "
        f"{len(features)}"
    )

    print(
        f"Borrowers in SHAP sample: "
        f"{len(X_shap):,}"
    )

    print(
        "\nTop 10 features by mean |SHAP|:"
    )

    print(
        importance[
            [
                "rank",
                "feature",
                "mean_abs_shap",
                "direction",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    print(
        "\nOutputs:"
    )

    print(
        f"  {SHAP_DIR}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "  SHAP explains model behavior."
    )

    print(
        "  It does NOT establish causality."
    )

    print(
        "\nNext:"
    )

    print(
        "  → Review calibration"
    )

    print(
        "  → Review subgroup robustness"
    )

    print(
        "  → Run final_model_selection.py"
    )

    print(
        "\n" + "=" * 70
    )


if __name__ == "__main__":

    main()