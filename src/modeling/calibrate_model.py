"""
calibrate_model.py

Probability calibration for the SELECTED
Conservative XGBoost model.

The model itself is already selected.

This script determines whether its raw probabilities
should remain unchanged or be calibrated using:

    1. Sigmoid / Platt calibration
    2. Isotonic calibration

The calibration split is used ONLY to fit the calibrators.

The test split remains untouched until final evaluation.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

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

from scipy.special import logit

from src.modeling.final_model_config import (
    MODEL_NAME,
    RANDOM_STATE,
    FEATURES,
    TARGET_COLUMN,
    build_xgb_model,
    load_gold_data,
)

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

##GOLD_DIR = Path("data/gold/exports")

OUTPUT_DIR = (
    "data\model\evaluation\calibration"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CALIBRATION CONFIGURATION
# ============================================================

TEST_SIZE = 0.15

CALIBRATION_FRACTION_OF_DEVELOPMENT = (
    0.1764705882
)


# ============================================================
# SIGMOID CALIBRATION
# ============================================================

def fit_sigmoid_calibrator(
    raw_probabilities,
    y_calibration,
):

    eps = 1e-6

    probabilities = np.clip(
        raw_probabilities,
        eps,
        1 - eps,
    )

    log_odds = logit(
        probabilities
    ).reshape(-1, 1)

    calibrator = LogisticRegression(
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )

    calibrator.fit(
        log_odds,
        y_calibration,
    )

    return calibrator


def sigmoid_predict(
    calibrator,
    probabilities,
):

    eps = 1e-6

    probabilities = np.clip(
        probabilities,
        eps,
        1 - eps,
    )

    log_odds = logit(
        probabilities
    ).reshape(-1, 1)

    return calibrator.predict_proba(
        log_odds
    )[:, 1]


# ============================================================
# ISOTONIC CALIBRATION
# ============================================================

def fit_isotonic_calibrator(
    raw_probabilities,
    y_calibration,
):

    calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )

    calibrator.fit(
        raw_probabilities,
        y_calibration,
    )

    return calibrator


# ============================================================
# METRICS
# ============================================================

def evaluate_predictions(
    y_true,
    probabilities,
    threshold=0.5,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "roc_auc":
            roc_auc_score(
                y_true,
                probabilities,
            ),

        "pr_auc":
            average_precision_score(
                y_true,
                probabilities,
            ),

        "accuracy":
            accuracy_score(
                y_true,
                predictions,
            ),

        "precision":
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            ),

        "f1":
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            ),

        "brier_score":
            brier_score_loss(
                y_true,
                probabilities,
            ),

        "log_loss":
            log_loss(
                y_true,
                probabilities,
            ),
    }


# ============================================================
# RELIABILITY TABLE
# ============================================================

def reliability_table(
    y_true,
    probabilities,
    model_name,
    bins=10,
):

    temp = pd.DataFrame({
        "y_true":
            np.asarray(y_true),

        "probability":
            np.asarray(probabilities),
    })

    temp["bin"] = pd.qcut(
        temp["probability"],
        q=bins,
        duplicates="drop",
    )

    result = (
        temp
        .groupby(
            "bin",
            observed=True,
        )
        .agg(
            borrowers=(
                "y_true",
                "size",
            ),

            mean_predicted_probability=(
                "probability",
                "mean",
            ),

            observed_default_rate=(
                "y_true",
                "mean",
            ),
        )
        .reset_index()
    )

    result["model"] = (
        model_name
    )

    result["calibration_error"] = (
        result[
            "mean_predicted_probability"
        ]
        -
        result[
            "observed_default_rate"
        ]
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
    print(
        "  INDIA CREDIT RISK — "
        "CONSERVATIVE XGBOOST CALIBRATION"
    )
    print("=" * 70)

    print(
        f"\nModel: {MODEL_NAME}"
    )

    print(
        f"Features: {len(FEATURES)}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print(
        "\nLoading Gold fact table..."
    )

    df = load_gold_data()

    X = df[
        FEATURES
    ].copy()

    y = df[
        TARGET_COLUMN
    ].astype(int)

    # --------------------------------------------------------
    # 70 / 15 / 15 split
    # --------------------------------------------------------

    X_dev, X_test, y_dev, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=RANDOM_STATE,
        )
    )

    X_train, X_cal, y_train, y_cal = (
        train_test_split(
            X_dev,
            y_dev,
            test_size=(
                CALIBRATION_FRACTION_OF_DEVELOPMENT
            ),
            stratify=y_dev,
            random_state=RANDOM_STATE,
        )
    )

    print(
        "\n" + "━" * 70
    )

    print(
        "DATASET SPLIT"
    )

    print(
        "━" * 70
    )

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
    # Train EXACT selected model
    # --------------------------------------------------------

    print(
        "\nTraining Conservative XGBoost..."
    )

    model = build_xgb_model(
        X_train
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "  ✓ Model trained"
    )

    # --------------------------------------------------------
    # Raw probabilities
    # --------------------------------------------------------

    calibration_raw = (
        model.predict_proba(
            X_cal
        )[:, 1]
    )

    test_raw = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    # --------------------------------------------------------
    # Sigmoid
    # --------------------------------------------------------

    print(
        "\nTraining sigmoid calibrator..."
    )

    sigmoid_calibrator = (
        fit_sigmoid_calibrator(
            calibration_raw,
            y_cal,
        )
    )

    sigmoid_test = (
        sigmoid_predict(
            sigmoid_calibrator,
            test_raw,
        )
    )

    print(
        "  ✓ Sigmoid calibration complete"
    )

    # --------------------------------------------------------
    # Isotonic
    # --------------------------------------------------------

    print(
        "\nTraining isotonic calibrator..."
    )

    isotonic_calibrator = (
        fit_isotonic_calibrator(
            calibration_raw,
            y_cal,
        )
    )

    isotonic_test = (
        isotonic_calibrator.predict(
            test_raw
        )
    )

    print(
        "  ✓ Isotonic calibration complete"
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    raw_metrics = (
        evaluate_predictions(
            y_test,
            test_raw,
        )
    )

    sigmoid_metrics = (
        evaluate_predictions(
            y_test,
            sigmoid_test,
        )
    )

    isotonic_metrics = (
        evaluate_predictions(
            y_test,
            isotonic_test,
        )
    )

    comparison = pd.DataFrame([
        {
            "model":
                "Base Conservative XGBoost",
            **raw_metrics,
        },
        {
            "model":
                "Sigmoid Calibrated Conservative XGBoost",
            **sigmoid_metrics,
        },
        {
            "model":
                "Isotonic Calibrated Conservative XGBoost",
            **isotonic_metrics,
        },
    ])

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        "\n" + "━" * 70
    )

    print(
        "CALIBRATION COMPARISON"
    )

    print(
        "━" * 70
    )

    print(
        comparison[
            [
                "model",
                "roc_auc",
                "pr_auc",
                "brier_score",
                "log_loss",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = pd.DataFrame({
        "y_true":
            y_test.values,

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
        index=False,
    )

    # --------------------------------------------------------
    # Reliability
    # --------------------------------------------------------

    reliability = pd.concat(
        [
            reliability_table(
                y_test,
                test_raw,
                "Base Conservative XGBoost",
            ),

            reliability_table(
                y_test,
                sigmoid_test,
                "Sigmoid Calibrated Conservative XGBoost",
            ),

            reliability_table(
                y_test,
                isotonic_test,
                "Isotonic Calibrated Conservative XGBoost",
            ),
        ],
        ignore_index=True,
    )

    reliability.to_parquet(
        OUTPUT_DIR /
        "calibration_reliability.parquet",
        index=False,
    )

    reliability.to_csv(
        OUTPUT_DIR /
        "calibration_reliability.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save comparison
    # --------------------------------------------------------

    comparison.to_parquet(
        OUTPUT_DIR /
        "calibration_comparison.parquet",
        index=False,
    )

    comparison.to_csv(
        OUTPUT_DIR /
        "calibration_comparison.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Select calibration method
    # --------------------------------------------------------

    best_row = comparison.loc[
        comparison[
            "brier_score"
        ].idxmin()
    ]

    best_method = str(
        best_row["model"]
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {

        "base_model":
            MODEL_NAME,

        "model_version":
            "conservative_xgb_v1",

        "feature_count":
            len(FEATURES),

        "calibration_methods_tested": [
            "None",
            "Sigmoid / Platt",
            "Isotonic",
        ],

        "best_calibration_method":
            best_method,

        "best_brier_score":
            float(
                best_row[
                    "brier_score"
                ]
            ),

        "base_brier_score":
            float(
                raw_metrics[
                    "brier_score"
                ]
            ),

        "base_roc_auc":
            float(
                raw_metrics[
                    "roc_auc"
                ]
            ),

        "sigmoid_brier_score":
            float(
                sigmoid_metrics[
                    "brier_score"
                ]
            ),

        "isotonic_brier_score":
            float(
                isotonic_metrics[
                    "brier_score"
                ]
            ),

        "training_rows":
            int(len(X_train)),

        "calibration_rows":
            int(len(X_cal)),

        "test_rows":
            int(len(X_test)),

        "note":
            (
                "The Conservative XGBoost model was "
                "selected before this calibration analysis. "
                "Calibration methods were evaluated only "
                "to determine whether probability outputs "
                "should be adjusted."
            ),
    }

    with open(
        OUTPUT_DIR /
        "calibration_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print(
        "\n" + "━" * 70
    )

    print(
        "CALIBRATION RESULT"
    )

    print(
        "━" * 70
    )

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
        f"\n  ★ Best probability method: "
        f"{best_method}"
    )

    print(
        "\n✓ Calibration analysis complete"
    )


if __name__ == "__main__":
    main()