"""
threshold_analysis.py

Threshold and operating-policy analysis for the SELECTED
Conservative XGBoost model.

This script does NOT select a new model.

It evaluates how the already-selected Conservative XGBoost
behaves at different probability thresholds.

Pipeline:

    Gold data
        ↓
    30 Conservative features
        ↓
    Train / test split
        ↓
    Selected Conservative XGBoost
        ↓
    Test-set probabilities
        ↓
    Threshold analysis
        ↓
    Operating-policy candidates

IMPORTANT:

    The 0.50 threshold is NOT assumed to be optimal.

    Threshold selection should consider:
        - precision
        - recall
        - specificity
        - F1
        - false-positive rate
        - false-negative rate
        - predicted default rate
        - expected loss proxy

    This script is an analytical tool.
    It does NOT automatically declare a final business threshold.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from final_model_config import (
    MODEL_NAME,
    MODEL_VERSION,
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

GOLD_DIR = Path("data/gold/exports")

OUTPUT_DIR = (
    GOLD_DIR
    / "analytics"
    / "ml"
    / "threshold_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

TEST_SIZE = 0.20

# Thresholds evaluated for operating-policy analysis.
THRESHOLDS = np.round(
    np.arange(
        0.05,
        0.96,
        0.05
    ),
    2
)

# Additional fine-grained thresholds around potentially
# useful operating regions.
FINE_THRESHOLDS = np.round(
    np.arange(
        0.10,
        0.51,
        0.01
    ),
    2
)

# Cost assumptions are explicitly analytical assumptions.
#
# These are NOT claimed to represent an actual lender's
# economics. They allow us to compare the relative trade-off
# between false positives and false negatives.
#
# False negative:
#   predicted non-default, actual default
#
# False positive:
#   predicted default, actual non-default
#
FALSE_NEGATIVE_COST = 5.0
FALSE_POSITIVE_COST = 1.0


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_dataset(df):
    """
    Validate that the Gold dataset contains the target and
    all governed Conservative model features.
    """

    print("\nValidating dataset...")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            "not found in Gold fact table."
        )

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing model features:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing_features
            )
        )

    target_values = (
        df[TARGET_COLUMN]
        .dropna()
        .unique()
    )

    if not set(target_values).issubset({0, 1}):
        raise ValueError(
            f"Target '{TARGET_COLUMN}' must contain "
            "only binary values {0, 1}."
        )

    print("  ✓ Target validated")
    print(
        f"  ✓ Model features validated: "
        f"{len(FEATURES)}"
    )


# ============================================================
# THRESHOLD METRICS
# ============================================================

def calculate_threshold_metrics(
    y_true,
    probabilities,
    threshold,
):
    """
    Calculate classification and operating metrics
    at one probability threshold.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    total = len(y_true)

    actual_defaults = int(
        np.sum(y_true == 1)
    )

    predicted_defaults = int(
        np.sum(predictions == 1)
    )

    actual_non_defaults = int(
        np.sum(y_true == 0)
    )

    # Core classification metrics
    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    # Specificity
    if (tn + fp) > 0:
        specificity = (
            tn / (tn + fp)
        )
    else:
        specificity = np.nan

    # False-positive rate
    if (fp + tn) > 0:
        false_positive_rate = (
            fp / (fp + tn)
        )
    else:
        false_positive_rate = np.nan

    # False-negative rate
    if (fn + tp) > 0:
        false_negative_rate = (
            fn / (fn + tp)
        )
    else:
        false_negative_rate = np.nan

    # Population-level rates
    predicted_default_rate = (
        predicted_defaults / total
    )

    actual_default_rate = (
        actual_defaults / total
    )

    # Expected-loss proxy.
    #
    # This is deliberately a simple analytical proxy,
    # not a financial estimate.
    expected_loss_proxy = (
        FALSE_NEGATIVE_COST * fn
        + FALSE_POSITIVE_COST * fp
    ) / total

    return {
        "threshold": float(threshold),

        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),

        "false_positive_rate": float(
            false_positive_rate
        ),
        "false_negative_rate": float(
            false_negative_rate
        ),

        "actual_default_rate": float(
            actual_default_rate
        ),
        "predicted_default_rate": float(
            predicted_default_rate
        ),

        "actual_defaults": actual_defaults,
        "predicted_defaults": predicted_defaults,
        "actual_non_defaults": actual_non_defaults,

        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),

        "expected_loss_proxy": float(
            expected_loss_proxy
        ),
    }


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

def evaluate_thresholds(
    y_test,
    probabilities,
    thresholds,
):
    """
    Evaluate a collection of thresholds.
    """

    rows = []

    for threshold in thresholds:
        rows.append(
            calculate_threshold_metrics(
                y_true=y_test,
                probabilities=probabilities,
                threshold=threshold,
            )
        )

    return pd.DataFrame(rows)


# ============================================================
# OPERATING-POLICY CANDIDATES
# ============================================================

def identify_candidates(
    threshold_results,
):
    """
    Identify useful analytical candidates.

    These are NOT automatically approved operating policies.

    Candidate categories:

        1. Maximum F1
        2. Minimum expected-loss proxy
        3. High-recall candidate
        4. High-precision candidate
        5. Balanced candidate

    The final policy must be selected using domain/business
    requirements rather than this script alone.
    """

    candidates = []

    # --------------------------------------------------------
    # Maximum F1
    # --------------------------------------------------------

    max_f1_row = (
        threshold_results
        .loc[
            threshold_results["f1"].idxmax()
        ]
        .copy()
    )

    candidates.append({
        "candidate": "maximum_f1",
        "threshold": float(
            max_f1_row["threshold"]
        ),
        "reason": (
            "Threshold with the highest "
            "F1 score on the test set."
        ),
    })

    # --------------------------------------------------------
    # Minimum expected loss proxy
    # --------------------------------------------------------

    min_loss_row = (
        threshold_results
        .loc[
            threshold_results[
                "expected_loss_proxy"
            ].idxmin()
        ]
        .copy()
    )

    candidates.append({
        "candidate": "minimum_expected_loss_proxy",
        "threshold": float(
            min_loss_row["threshold"]
        ),
        "reason": (
            "Threshold minimizing the analytical "
            "false-negative / false-positive "
            "cost proxy."
        ),
    })

    # --------------------------------------------------------
    # High recall candidate
    # --------------------------------------------------------

    high_recall = (
        threshold_results[
            threshold_results["recall"] >= 0.80
        ]
    )

    if not high_recall.empty:

        # Among thresholds meeting recall >= 0.80,
        # choose the one with highest precision.
        row = (
            high_recall
            .sort_values(
                [
                    "precision",
                    "specificity"
                ],
                ascending=False
            )
            .iloc[0]
        )

        candidates.append({
            "candidate": "high_recall",
            "threshold": float(
                row["threshold"]
            ),
            "reason": (
                "Highest-precision threshold among "
                "thresholds achieving recall >= 0.80."
            ),
        })

    # --------------------------------------------------------
    # High precision candidate
    # --------------------------------------------------------

    high_precision = (
        threshold_results[
            threshold_results["precision"] >= 0.70
        ]
    )

    if not high_precision.empty:

        # Among thresholds meeting precision >= 0.70,
        # choose the one with highest recall.
        row = (
            high_precision
            .sort_values(
                [
                    "recall",
                    "specificity"
                ],
                ascending=False
            )
            .iloc[0]
        )

        candidates.append({
            "candidate": "high_precision",
            "threshold": float(
                row["threshold"]
            ),
            "reason": (
                "Highest-recall threshold among "
                "thresholds achieving precision >= 0.70."
            ),
        })

    # --------------------------------------------------------
    # Balanced candidate
    # --------------------------------------------------------

    balanced = (
        threshold_results
        .assign(
            balance_score=lambda x:
                np.minimum(
                    x["precision"],
                    x["recall"]
                )
        )
    )

    row = (
        balanced
        .loc[
            balanced["balance_score"].idxmax()
        ]
    )

    candidates.append({
        "candidate": "balanced_precision_recall",
        "threshold": float(
            row["threshold"]
        ),
        "reason": (
            "Threshold maximizing the lower of "
            "precision and recall."
        ),
    })

    return pd.DataFrame(candidates)


# ============================================================
# OVERALL MODEL PERFORMANCE
# ============================================================

def calculate_overall_performance(
    y_test,
    probabilities,
):
    """
    Calculate threshold-independent model metrics.
    """

    return {
        "roc_auc": float(
            roc_auc_score(
                y_test,
                probabilities
            )
        ),

        "pr_auc": float(
            average_precision_score(
                y_test,
                probabilities
            )
        ),

        "brier_score": float(
            brier_score_loss(
                y_test,
                probabilities
            )
        ),

        "rows": int(len(y_test)),

        "actual_default_rate": float(
            np.mean(y_test)
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "  INDIA CREDIT RISK — "
        "THRESHOLD ANALYSIS"
    )
    print("=" * 70)

    print(
        f"\nModel: {MODEL_NAME}"
    )

    print(
        f"Version: {MODEL_VERSION}"
    )

    print(
        f"Features: {len(FEATURES)}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This script evaluates operating thresholds "
        "for the already-selected Conservative XGBoost."
    )

    print(
        "It does NOT select a new ML model."
    )

    print(
        "Threshold candidates are analytical and "
        "require business/domain validation."
    )

    # --------------------------------------------------------
    # Load Gold data
    # --------------------------------------------------------

    print(
        "\nLoading Gold fact table..."
    )

    df = load_gold_data()

    print(
        f"  ✓ Loaded: "
        f"{len(df):,} rows"
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_dataset(df)

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    X = df[
        FEATURES
    ].copy()

    y = df[
        TARGET_COLUMN
    ].astype(int)

    # --------------------------------------------------------
    # Train / test split
    # --------------------------------------------------------

    print(
        "\n" + "━" * 70
    )

    print(
        "TRAIN / TEST SPLIT"
    )

    print(
        "━" * 70
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=RANDOM_STATE,
        )
    )

    print(
        f"  Train: "
        f"{len(X_train):,}"
    )

    print(
        f"  Test:  "
        f"{len(X_test):,}"
    )

    # --------------------------------------------------------
    # Train selected model
    # --------------------------------------------------------

    print(
        "\nTraining selected "
        "Conservative XGBoost..."
    )

    model = build_xgb_model(
        X_train
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "  ✓ Selected model trained"
    )

    # --------------------------------------------------------
    # Test probabilities
    # --------------------------------------------------------

    print(
        "\nGenerating test-set probabilities..."
    )

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    print(
        "  ✓ Probabilities generated"
    )

    # --------------------------------------------------------
    # Overall performance
    # --------------------------------------------------------

    overall = (
        calculate_overall_performance(
            y_test,
            probabilities,
        )
    )

    print(
        "\n" + "━" * 70
    )

    print(
        "OVERALL TEST PERFORMANCE"
    )

    print(
        "━" * 70
    )

    print(
        f"  ROC-AUC: "
        f"{overall['roc_auc']:.4f}"
    )

    print(
        f"  PR-AUC:  "
        f"{overall['pr_auc']:.4f}"
    )

    print(
        f"  Brier:   "
        f"{overall['brier_score']:.4f}"
    )

    print(
        f"  Default rate: "
        f"{overall['actual_default_rate']:.4f}"
    )

    # --------------------------------------------------------
    # Coarse threshold analysis
    # --------------------------------------------------------

    print(
        "\n" + "━" * 70
    )

    print(
        "THRESHOLD ANALYSIS"
    )

    print(
        "━" * 70
    )

    threshold_results = (
        evaluate_thresholds(
            y_test=y_test,
            probabilities=probabilities,
            thresholds=THRESHOLDS,
        )
    )

    display_columns = [
        "threshold",
        "precision",
        "recall",
        "specificity",
        "f1",
        "false_positive_rate",
        "false_negative_rate",
        "predicted_default_rate",
        "expected_loss_proxy",
    ]

    print(
        threshold_results[
            display_columns
        ]
        .round(4)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Fine threshold analysis
    # --------------------------------------------------------

    print(
        "\n" + "━" * 70
    )

    print(
        "FINE-GRAINED THRESHOLD ANALYSIS"
    )

    print(
        "━" * 70
    )

    fine_threshold_results = (
        evaluate_thresholds(
            y_test=y_test,
            probabilities=probabilities,
            thresholds=FINE_THRESHOLDS,
        )
    )

    fine_threshold_results.to_parquet(
        OUTPUT_DIR
        / "threshold_metrics_fine.parquet",
        index=False,
    )

    fine_threshold_results.to_csv(
        OUTPUT_DIR
        / "threshold_metrics_fine.csv",
        index=False,
    )

    print(
        "  ✓ Fine-grained threshold metrics saved"
    )

    # --------------------------------------------------------
    # Candidate policies
    # --------------------------------------------------------

    candidates = identify_candidates(
        threshold_results
    )

    print(
        "\n" + "━" * 70
    )

    print(
        "ANALYTICAL THRESHOLD CANDIDATES"
    )

    print(
        "━" * 70
    )

    print(
        candidates.to_string(
            index=False
        )
    )

    print(
        "\nNOTE:"
    )

    print(
        "These are analytical candidates only."
    )

    print(
        "No threshold is automatically approved "
        "for operational use."
    )

    # --------------------------------------------------------
    # Save coarse results
    # --------------------------------------------------------

    threshold_results.to_parquet(
        OUTPUT_DIR
        / "threshold_metrics.parquet",
        index=False,
    )

    threshold_results.to_csv(
        OUTPUT_DIR
        / "threshold_metrics.csv",
        index=False,
    )

    candidates.to_parquet(
        OUTPUT_DIR
        / "threshold_candidates.parquet",
        index=False,
    )

    candidates.to_csv(
        OUTPUT_DIR
        / "threshold_candidates.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Confusion matrices for candidate thresholds
    # --------------------------------------------------------

    confusion_rows = []

    for _, candidate in candidates.iterrows():

        threshold = (
            candidate["threshold"]
        )

        metrics = calculate_threshold_metrics(
            y_true=y_test,
            probabilities=probabilities,
            threshold=threshold,
        )

        confusion_rows.append({
            "candidate":
                candidate["candidate"],

            "threshold":
                threshold,

            "true_negatives":
                metrics["true_negatives"],

            "false_positives":
                metrics["false_positives"],

            "false_negatives":
                metrics["false_negatives"],

            "true_positives":
                metrics["true_positives"],
        })

    candidate_confusion = pd.DataFrame(
        confusion_rows
    )

    candidate_confusion.to_parquet(
        OUTPUT_DIR
        / "candidate_confusion_matrices.parquet",
        index=False,
    )

    candidate_confusion.to_csv(
        OUTPUT_DIR
        / "candidate_confusion_matrices.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {

        "model":
            MODEL_NAME,

        "model_version":
            MODEL_VERSION,

        "features":
            FEATURES,

        "feature_count":
            len(FEATURES),

        "target":
            TARGET_COLUMN,

        "random_state":
            RANDOM_STATE,

        "test_size":
            TEST_SIZE,

        "test_rows":
            len(y_test),

        "thresholds_evaluated":
            [
                float(x)
                for x in THRESHOLDS
            ],

        "fine_thresholds_evaluated":
            [
                float(x)
                for x in FINE_THRESHOLDS
            ],

        "false_negative_cost":
            FALSE_NEGATIVE_COST,

        "false_positive_cost":
            FALSE_POSITIVE_COST,

        "cost_proxy_note":
            (
                "Expected-loss proxy uses analytical "
                "relative costs only. It is not a "
                "financial loss estimate and must not "
                "be interpreted as lender economics."
            ),

        "overall_performance":
            overall,

        "threshold_selection_note":
            (
                "Threshold candidates are presented "
                "for analytical comparison. No final "
                "operating threshold is automatically "
                "approved by this script."
            ),
    }

    with open(
        OUTPUT_DIR
        / "threshold_analysis_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "✓ THRESHOLD ANALYSIS COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nSaved to:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )

    print(
        "\nFiles:"
    )

    print(
        "  → threshold_metrics.parquet"
    )

    print(
        "  → threshold_metrics.csv"
    )

    print(
        "  → threshold_metrics_fine.parquet"
    )

    print(
        "  → threshold_metrics_fine.csv"
    )

    print(
        "  → threshold_candidates.parquet"
    )

    print(
        "  → threshold_candidates.csv"
    )

    print(
        "  → candidate_confusion_matrices.parquet"
    )

    print(
        "  → candidate_confusion_matrices.csv"
    )

    print(
        "  → threshold_analysis_metadata.json"
    )

    print(
        "\nNEXT:"
    )

    print(
        "Review threshold trade-offs before declaring "
        "an operating threshold."
    )

    print(
        "Do NOT treat the highest-F1 threshold as "
        "automatically optimal for credit decisions."
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()