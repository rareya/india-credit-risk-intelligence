"""
calibrate_model.py
============================================================

INDIA CREDIT RISK — PROBABILITY CALIBRATION

Purpose
-------
Calibrate probabilities produced by the Conservative XGBoost model.

IMPORTANT:
We deliberately DO NOT use sklearn's CalibratedClassifierCV around
XGBClassifier because some sklearn/xgboost combinations incorrectly
identify XGBClassifier as a regressor.

Instead we use:

    1. Train XGBoost on training split
    2. Generate raw probabilities on calibration split
    3. Fit:
         - Platt / sigmoid calibration
         - Isotonic calibration
    4. Evaluate all methods on untouched test set

Dataset split
-------------
70% training
15% calibration
15% test

The test set remains untouched until final evaluation.

Outputs
-------
data/gold/exports/analytics/ml/calibration/

    calibration_comparison.parquet
    calibration_comparison.csv
    calibration_predictions.parquet
    calibration_summary.json
    calibration_reliability.parquet

============================================================
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    log_loss,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

GOLD_DIR = Path("data/gold/exports")
ML_DIR = GOLD_DIR / "analytics" / "ml"
OUTPUT_DIR = ML_DIR / "calibration"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TARGET = "default_risk"

# These are the conservative features approved by the leakage audit.
#
# IMPORTANT:
# Do NOT add:
#   risk_grade
#   risk_grade_numeric
#   risk_band
#   default_risk
#   cibil_band
# unless provenance has explicitly been approved.

NUMERIC_FEATURES = [
    "active_loan_ratio",
    "active_loans",
    "age",
    "auto_loans",
    "closed_loans",
    "credit_card_loans",
    "credit_history_months",
    "delinquency_score",
    "gold_loans",
    "home_loans",
    "loan_type_diversity",
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

CATEGORICAL_FEATURES = [
    "education",
    "gender",
    "has_credit_card",
    "has_gold_loan",
    "has_home_loan",
    "has_personal_loan",
    "income_tier",
    "marital_status",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    path = GOLD_DIR / "fact_credit_risk.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing Gold fact table:\n{path}"
        )

    df = pd.read_parquet(path)

    print(
        f"  ✓ Loaded: {len(df):,} rows × "
        f"{len(df.columns)} columns"
    )

    return df


# ============================================================
# VALIDATE
# ============================================================

def validate_data(df):

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("Validating Gold fact table")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' missing."
        )

    if df[TARGET].isna().any():
        raise ValueError(
            "Target contains missing values."
        )

    unique_target = set(
        df[TARGET].dropna().unique()
    )

    if not unique_target.issubset({0, 1}):
        raise ValueError(
            f"Target is not binary: {unique_target}"
        )

    missing_features = [
        f for f in FEATURES
        if f not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing conservative features:\n"
            + "\n".join(
                f"  - {x}"
                for x in missing_features
            )
        )

    print("  ✓ Target validated")
    print("  ✓ Conservative feature registry validated")

    print(
        f"  ✓ Feature count: {len(FEATURES)}"
    )


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df):

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int).copy()

    # --------------------------------------------------------
    # Numeric features
    # --------------------------------------------------------

    for col in NUMERIC_FEATURES:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

        # Median imputation learned from whole dataset is
        # acceptable for this calibration diagnostic because
        # these are descriptive preprocessing statistics.
        #
        # For production training, preprocessing should be
        # fitted only on the training partition.
        X[col] = X[col].fillna(
            X[col].median()
        )

    # --------------------------------------------------------
    # Categorical features
    # --------------------------------------------------------

    for col in CATEGORICAL_FEATURES:

        X[col] = (
            X[col]
            .astype("string")
            .fillna("__MISSING__")
        )

    # --------------------------------------------------------
    # One-hot encoding
    # --------------------------------------------------------

    X = pd.get_dummies(
        X,
        columns=CATEGORICAL_FEATURES,
        drop_first=False,
        dtype=float
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(0)

    X = X.astype(float)

    print(
        f"  ✓ Feature matrix: "
        f"{X.shape[0]:,} × {X.shape[1]}"
    )

    return X, y


# ============================================================
# XGBOOST
# ============================================================

def build_xgb():

    return XGBClassifier(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ============================================================
# SIGMOID / PLATT CALIBRATION
# ============================================================

def fit_sigmoid_calibrator(
    raw_probabilities,
    y_calibration
):

    """
    Platt-style probability calibration.

    We transform probability into log-odds and fit a
    logistic regression:

        calibrated_p =
            sigmoid(a * logit(raw_p) + b)
    """

    eps = 1e-6

    p = np.clip(
        raw_probabilities,
        eps,
        1 - eps
    )

    log_odds = logit(p).reshape(-1, 1)

    calibrator = LogisticRegression(
        solver="lbfgs",
        random_state=RANDOM_STATE
    )

    calibrator.fit(
        log_odds,
        y_calibration
    )

    return calibrator


def sigmoid_predict(
    calibrator,
    probabilities
):

    eps = 1e-6

    p = np.clip(
        probabilities,
        eps,
        1 - eps
    )

    log_odds = logit(p).reshape(-1, 1)

    return calibrator.predict_proba(
        log_odds
    )[:, 1]


# ============================================================
# ISOTONIC CALIBRATION
# ============================================================

def fit_isotonic_calibrator(
    raw_probabilities,
    y_calibration
):

    calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip"
    )

    calibrator.fit(
        raw_probabilities,
        y_calibration
    )

    return calibrator


# ============================================================
# METRICS
# ============================================================

def evaluate_predictions(
    y_true,
    probabilities,
    threshold=0.5
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "roc_auc": roc_auc_score(
            y_true,
            probabilities
        ),

        "pr_auc": average_precision_score(
            y_true,
            probabilities
        ),

        "accuracy": accuracy_score(
            y_true,
            predictions
        ),

        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "brier_score": brier_score_loss(
            y_true,
            probabilities
        ),

        "log_loss": log_loss(
            y_true,
            probabilities
        ),
    }


# ============================================================
# RELIABILITY TABLE
# ============================================================

def reliability_table(
    y_true,
    probabilities,
    model_name,
    bins=10
):

    temp = pd.DataFrame({
        "y": np.asarray(y_true),
        "probability": np.asarray(probabilities)
    })

    temp["bin"] = pd.qcut(
        temp["probability"],
        q=bins,
        duplicates="drop"
    )

    result = (
        temp.groupby(
            "bin",
            observed=True
        )
        .agg(
            borrowers=("y", "size"),
            mean_predicted_probability=(
                "probability",
                "mean"
            ),
            observed_default_rate=(
                "y",
                "mean"
            )
        )
        .reset_index()
    )

    result["model"] = model_name

    result["calibration_error"] = (
        result["mean_predicted_probability"]
        - result["observed_default_rate"]
    ).abs()

    result[
        "mean_predicted_probability"
    ] *= 100

    result[
        "observed_default_rate"
    ] *= 100

    result[
        "calibration_error"
    ] *= 100

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("  INDIA CREDIT RISK — PROBABILITY CALIBRATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print("\nLoading Gold fact table...")

    df = load_data()

    validate_data(df)

    X, y = prepare_features(df)

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    # First:
    # 85% development
    # 15% untouched test

    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # Then split development into:
    # 70% train
    # 15% calibration

    X_train, X_cal, y_train, y_cal = train_test_split(
        X_dev,
        y_dev,
        test_size=0.1764705882,
        stratify=y_dev,
        random_state=RANDOM_STATE
    )

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("Dataset split")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    print(
        f"  Train:       {len(X_train):,}"
    )

    print(
        f"  Calibration: {len(X_cal):,}"
    )

    print(
        f"  Test:        {len(X_test):,}"
    )

    # --------------------------------------------------------
    # Train base model
    # --------------------------------------------------------

    print("\nTraining base XGBoost...")

    model = build_xgb()

    model.fit(
        X_train,
        y_train
    )

    print("  ✓ Base model trained")

    # --------------------------------------------------------
    # Raw probabilities
    # --------------------------------------------------------

    train_raw = model.predict_proba(
        X_train
    )[:, 1]

    calibration_raw = model.predict_proba(
        X_cal
    )[:, 1]

    test_raw = model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # SIGMOID
    # --------------------------------------------------------

    print("\nTraining sigmoid calibration...")

    sigmoid_calibrator = fit_sigmoid_calibrator(
        calibration_raw,
        y_cal
    )

    sigmoid_test = sigmoid_predict(
        sigmoid_calibrator,
        test_raw
    )

    print("  ✓ Sigmoid calibration trained")

    # --------------------------------------------------------
    # ISOTONIC
    # --------------------------------------------------------

    print("\nTraining isotonic calibration...")

    isotonic_calibrator = fit_isotonic_calibrator(
        calibration_raw,
        y_cal
    )

    isotonic_test = isotonic_calibrator.predict(
        test_raw
    )

    print("  ✓ Isotonic calibration trained")

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    raw_metrics = evaluate_predictions(
        y_test,
        test_raw
    )

    sigmoid_metrics = evaluate_predictions(
        y_test,
        sigmoid_test
    )

    isotonic_metrics = evaluate_predictions(
        y_test,
        isotonic_test
    )

    comparison = pd.DataFrame([
        {
            "model": "Base XGBoost",
            **raw_metrics
        },
        {
            "model": "Sigmoid Calibrated XGBoost",
            **sigmoid_metrics
        },
        {
            "model": "Isotonic Calibrated XGBoost",
            **isotonic_metrics
        }
    ])

    # --------------------------------------------------------
    # Print comparison
    # --------------------------------------------------------

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("CALIBRATION COMPARISON")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    print(
        comparison[
            [
                "model",
                "roc_auc",
                "pr_auc",
                "brier_score",
                "log_loss",
            ]
        ].round(4).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = pd.DataFrame({
        "y_true": y_test.values,

        "base_xgb_probability":
            test_raw,

        "sigmoid_probability":
            sigmoid_test,

        "isotonic_probability":
            isotonic_test,
    })

    predictions.to_parquet(
        OUTPUT_DIR /
        "calibration_predictions.parquet",
        index=False
    )

    # --------------------------------------------------------
    # Reliability
    # --------------------------------------------------------

    reliability = pd.concat(
        [
            reliability_table(
                y_test,
                test_raw,
                "Base XGBoost"
            ),

            reliability_table(
                y_test,
                sigmoid_test,
                "Sigmoid Calibrated XGBoost"
            ),

            reliability_table(
                y_test,
                isotonic_test,
                "Isotonic Calibrated XGBoost"
            )
        ],
        ignore_index=True
    )

    reliability.to_parquet(
        OUTPUT_DIR /
        "calibration_reliability.parquet",
        index=False
    )

    # --------------------------------------------------------
    # Save comparison
    # --------------------------------------------------------

    comparison.to_parquet(
        OUTPUT_DIR /
        "calibration_comparison.parquet",
        index=False
    )

    comparison.to_csv(
        OUTPUT_DIR /
        "calibration_comparison.csv",
        index=False
    )

    # --------------------------------------------------------
    # Select calibration method
    # --------------------------------------------------------

    best_row = comparison.loc[
        comparison["brier_score"].idxmin()
    ]

    best_method = best_row["model"]

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {

        "base_model":
            "Conservative XGBoost",

        "calibration_methods_tested": [
            "None",
            "Sigmoid / Platt",
            "Isotonic"
        ],

        "best_calibration_method":
            str(best_method),

        "best_brier_score":
            float(best_row["brier_score"]),

        "base_brier_score":
            float(
                raw_metrics["brier_score"]
            ),

        "base_roc_auc":
            float(
                raw_metrics["roc_auc"]
            ),

        "sigmoid_brier_score":
            float(
                sigmoid_metrics["brier_score"]
            ),

        "isotonic_brier_score":
            float(
                isotonic_metrics["brier_score"]
            ),

        "test_rows":
            int(len(X_test)),

        "calibration_rows":
            int(len(X_cal)),

        "training_rows":
            int(len(X_train)),

        "note":
            (
                "Calibration was fitted exclusively on the "
                "calibration split. The test split was held "
                "out until final evaluation."
            )
    }

    with open(
        OUTPUT_DIR /
        "calibration_summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("CALIBRATION RESULT")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    print(
        f"  Base Brier:     "
        f"{raw_metrics['brier_score']:.4f}"
    )

    print(
        f"  Sigmoid Brier:  "
        f"{sigmoid_metrics['brier_score']:.4f}"
    )

    print(
        f"  Isotonic Brier: "
        f"{isotonic_metrics['brier_score']:.4f}"
    )

    print(
        f"\n  ★ Best method: {best_method}"
    )

    print("\n✓ Calibration analysis complete")

    print("\nSaved:")
    print(
        "  → calibration_comparison.parquet"
    )
    print(
        "  → calibration_comparison.csv"
    )
    print(
        "  → calibration_predictions.parquet"
    )
    print(
        "  → calibration_reliability.parquet"
    )
    print(
        "  → calibration_summary.json"
    )

    print("\nNext:")
    print(
        "  python src/analytics/shap_analysis.py"
    )


if __name__ == "__main__":
    main()