"""
subgroup_robustness.py

SUBGROUP ROBUSTNESS ANALYSIS
for the SELECTED Conservative XGBoost model.

Purpose
-------
Evaluate whether the selected credit-risk model behaves consistently
across important borrower subgroups.

Selected model:
    Conservative XGBoost
    conservative_xgb_v1

The model definition, feature set, target, random state, preprocessing,
and XGBoost configuration are imported from final_model_config.py.

This script does NOT:
    - train a new candidate model
    - perform model selection
    - change the selected feature set
    - use subgroup variables as model inputs unless they are already
      part of FEATURES

Instead, subgroup variables are used only for evaluation.

Primary subgroup dimensions:
    - gender
    - age_band
    - education
    - marital_status
    - income_tier

Additional evaluation-only dimensions:
    - cibil_band, if present in the Gold table

Metrics:
    - sample size
    - default rate
    - mean predicted probability
    - ROC-AUC
    - PR-AUC
    - Brier score
    - precision
    - recall
    - specificity
    - false-positive rate
    - false-negative rate
    - F1

A 0.50 classification threshold is used ONLY for reporting
classification metrics. It is NOT treated as the final operating
threshold.

Outputs:
    data/gold/exports/analytics/ml/subgroup_robustness/

        subgroup_metrics.parquet
        subgroup_metrics.csv
        subgroup_robustness_summary.json
        subgroup_robustness_metadata.json
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
   "data\model\evaluation\subgroup_robustness"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

TEST_SIZE = 0.20

# IMPORTANT:
# This threshold is only for reporting classification metrics.
# It is NOT the final operational threshold.
CLASSIFICATION_THRESHOLD = 0.50


# These are evaluation dimensions.
#
# They are NOT automatically added to the model.
#
# Only dimensions that actually exist in the Gold table will
# be evaluated.
SUBGROUP_DIMENSIONS = [
    "gender",
    "age_band",
    "education",
    "marital_status",
    "income_tier",
    "cibil_band",
]


# Minimum subgroup size.
#
# Small groups can produce unstable estimates, particularly
# ROC-AUC and PR-AUC.
MIN_SUBGROUP_SIZE = 100


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_data(df):
    """
    Validate the Gold dataset before modeling.
    """

    print("\nValidating dataset...")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            f"not found in Gold fact table."
        )

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "The following governed model features "
            "are missing from the Gold fact table:\n"
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
            f"only 0/1 values."
        )

    print("  ✓ Target validated")
    print(
        f"  ✓ Model features validated: "
        f"{len(FEATURES)}"
    )


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def create_test_split(df):
    """
    Create the same deterministic 80/20 stratified split
    used by the selected-model analyses.
    """

    X = df[FEATURES].copy()
    y = df[TARGET_COLUMN].astype(int)

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# ============================================================
# TRAIN SELECTED MODEL
# ============================================================

def train_selected_model(
    X_train,
    y_train,
):
    """
    Train the exact selected Conservative XGBoost model
    from final_model_config.py.
    """

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

    return model


# ============================================================
# OVERALL TEST PERFORMANCE
# ============================================================

def calculate_overall_performance(
    model,
    X_test,
    y_test,
):
    """
    Calculate overall test-set performance.

    This provides a reference point against which subgroup
    metrics can be compared.
    """

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    predictions = (
        probabilities
        >= CLASSIFICATION_THRESHOLD
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else np.nan
    )

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else np.nan
    )

    return {
        "group_type": "overall",
        "group": "ALL",
        "sample_size": int(len(y_test)),
        "default_count": int(y_test.sum()),
        "default_rate": float(y_test.mean()),
        "mean_predicted_probability": float(
            probabilities.mean()
        ),
        "roc_auc": float(
            roc_auc_score(
                y_test,
                probabilities,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_test,
                probabilities,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y_test,
                probabilities,
            )
        ),
        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "specificity": float(
            specificity
        ),
        "false_positive_rate": float(
            false_positive_rate
        ),
        "false_negative_rate": float(
            false_negative_rate
        ),
        "f1": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


# ============================================================
# SUBGROUP METRICS
# ============================================================

def calculate_subgroup_metrics(
    y_true,
    probabilities,
    subgroup_values,
    dimension,
):
    """
    Calculate performance metrics for each subgroup
    within a single evaluation dimension.
    """

    rows = []

    subgroup_series = (
        subgroup_values
        .copy()
        .reset_index(drop=True)
    )

    y_true = (
        pd.Series(y_true)
        .reset_index(drop=True)
    )

    probabilities = (
        pd.Series(probabilities)
        .reset_index(drop=True)
    )

    evaluation_df = pd.DataFrame({
        "y_true": y_true,
        "probability": probabilities,
        "subgroup": subgroup_series,
    })

    # Missing subgroup values are represented explicitly.
    evaluation_df["subgroup"] = (
        evaluation_df["subgroup"]
        .astype("object")
        .where(
            evaluation_df["subgroup"].notna(),
            "MISSING",
        )
    )

    for subgroup_name, group_df in (
        evaluation_df
        .groupby(
            "subgroup",
            dropna=False,
        )
    ):

        n = len(group_df)

        # Small groups can create extremely unstable
        # performance estimates.
        if n < MIN_SUBGROUP_SIZE:
            continue

        y_group = (
            group_df["y_true"]
            .astype(int)
            .to_numpy()
        )

        p_group = (
            group_df["probability"]
            .astype(float)
            .to_numpy()
        )

        predictions = (
            p_group
            >= CLASSIFICATION_THRESHOLD
        ).astype(int)

        positive_count = int(
            y_group.sum()
        )

        negative_count = int(
            len(y_group) - positive_count
        )

        # ROC-AUC and PR-AUC require appropriate
        # target variation.
        if (
            len(np.unique(y_group))
            == 2
        ):
            roc_auc = float(
                roc_auc_score(
                    y_group,
                    p_group,
                )
            )

            pr_auc = float(
                average_precision_score(
                    y_group,
                    p_group,
                )
            )
        else:
            roc_auc = np.nan
            pr_auc = np.nan

        tn, fp, fn, tp = confusion_matrix(
            y_group,
            predictions,
            labels=[0, 1],
        ).ravel()

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else np.nan
        )

        false_positive_rate = (
            fp / (fp + tn)
            if (fp + tn) > 0
            else np.nan
        )

        false_negative_rate = (
            fn / (fn + tp)
            if (fn + tp) > 0
            else np.nan
        )

        rows.append({
            "group_type": dimension,
            "group": str(
                subgroup_name
            ),
            "sample_size": int(n),
            "default_count": positive_count,
            "non_default_count": negative_count,
            "default_rate": float(
                y_group.mean()
            ),
            "mean_predicted_probability": float(
                p_group.mean()
            ),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "brier_score": float(
                brier_score_loss(
                    y_group,
                    p_group,
                )
            ),
            "precision": float(
                precision_score(
                    y_group,
                    predictions,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_group,
                    predictions,
                    zero_division=0,
                )
            ),
            "specificity": float(
                specificity
            ),
            "false_positive_rate": float(
                false_positive_rate
            ),
            "false_negative_rate": float(
                false_negative_rate
            ),
            "f1": float(
                f1_score(
                    y_group,
                    predictions,
                    zero_division=0,
                )
            ),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        })

    return rows


# ============================================================
# ROBUSTNESS COMPARISON
# ============================================================

def build_robustness_summary(
    subgroup_metrics,
):
    """
    Summarize the range of subgroup performance for each
    subgroup dimension.

    This does not declare a subgroup 'fair' or 'unfair'.
    It identifies dimensions where material performance
    variation exists and therefore deserves investigation.
    """

    metrics_df = pd.DataFrame(
        subgroup_metrics
    )

    summary_rows = []

    dimensions = (
        metrics_df[
            metrics_df["group_type"]
            != "overall"
        ]["group_type"]
        .drop_duplicates()
        .tolist()
    )

    monitored_metrics = [
        "roc_auc",
        "pr_auc",
        "brier_score",
        "precision",
        "recall",
        "specificity",
        "false_positive_rate",
        "false_negative_rate",
        "f1",
    ]

    for dimension in dimensions:

        dimension_df = metrics_df[
            metrics_df["group_type"]
            == dimension
        ].copy()

        row = {
            "group_type": dimension,
            "groups_evaluated": int(
                len(dimension_df)
            ),
        }

        for metric in monitored_metrics:

            values = (
                pd.to_numeric(
                    dimension_df[metric],
                    errors="coerce",
                )
                .dropna()
            )

            if len(values) == 0:
                row[
                    f"{metric}_min"
                ] = np.nan

                row[
                    f"{metric}_max"
                ] = np.nan

                row[
                    f"{metric}_range"
                ] = np.nan

            else:
                row[
                    f"{metric}_min"
                ] = float(
                    values.min()
                )

                row[
                    f"{metric}_max"
                ] = float(
                    values.max()
                )

                row[
                    f"{metric}_range"
                ] = float(
                    values.max()
                    - values.min()
                )

        summary_rows.append(row)

    return pd.DataFrame(
        summary_rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "  INDIA CREDIT RISK — "
        "SUBGROUP ROBUSTNESS ANALYSIS"
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
        f"Classification threshold: "
        f"{CLASSIFICATION_THRESHOLD:.2f}"
    )

    print(
        "\nNOTE:"
    )

    print(
        "The 0.50 threshold is used only for "
        "classification diagnostics."
    )

    print(
        "It is NOT the final operating threshold."
    )

    # --------------------------------------------------------
    # LOAD GOLD DATA
    # --------------------------------------------------------

    print(
        "\nLoading Gold fact table..."
    )

    df = load_gold_data()

    print(
        f"  ✓ Loaded: "
        f"{len(df):,} rows"
    )

    validate_data(df)

    # --------------------------------------------------------
    # CREATE SPLIT
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = create_test_split(df)

    print(
        "\n" + "━" * 70
    )

    print(
        "TRAIN / TEST SPLIT"
    )

    print(
        "━" * 70
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
    # TRAIN SELECTED MODEL
    # --------------------------------------------------------

    model = train_selected_model(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # TEST PREDICTIONS
    # --------------------------------------------------------

    print(
        "\nGenerating test-set predictions..."
    )

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    print(
        "  ✓ Predictions generated"
    )

    # --------------------------------------------------------
    # OVERALL PERFORMANCE
    # --------------------------------------------------------

    overall_metrics = (
        calculate_overall_performance(
            model=model,
            X_test=X_test,
            y_test=y_test,
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
        f"{overall_metrics['roc_auc']:.4f}"
    )

    print(
        f"  PR-AUC:  "
        f"{overall_metrics['pr_auc']:.4f}"
    )

    print(
        f"  Brier:   "
        f"{overall_metrics['brier_score']:.4f}"
    )

    print(
        f"  Recall:  "
        f"{overall_metrics['recall']:.4f}"
    )

    print(
        f"  F1:      "
        f"{overall_metrics['f1']:.4f}"
    )

    # --------------------------------------------------------
    # ALIGN TEST ROWS WITH ORIGINAL DATA
    # --------------------------------------------------------

    #
    # X_test retains the original dataframe indices after
    # train_test_split.
    #
    test_indices = X_test.index

    subgroup_df = df.loc[
        test_indices
    ].copy()

    # --------------------------------------------------------
    # SUBGROUP ANALYSIS
    # --------------------------------------------------------

    print(
        "\n" + "━" * 70
    )

    print(
        "SUBGROUP ANALYSIS"
    )

    print(
        "━" * 70
    )

    subgroup_metrics = [
        overall_metrics
    ]

    dimensions_evaluated = []

    for dimension in SUBGROUP_DIMENSIONS:

        if dimension not in subgroup_df.columns:
            print(
                f"\n  ⚠ Skipping "
                f"{dimension}: "
                f"column not present."
            )
            continue

        print(
            f"\nEvaluating: "
            f"{dimension}"
        )

        rows = calculate_subgroup_metrics(
            y_true=y_test,
            probabilities=probabilities,
            subgroup_values=subgroup_df[
                dimension
            ],
            dimension=dimension,
        )

        if not rows:
            print(
                "  ⚠ No sufficiently large "
                "subgroups found."
            )
            continue

        dimensions_evaluated.append(
            dimension
        )

        subgroup_metrics.extend(
            rows
        )

        for row in rows:

            print(
                f"  {row['group']:<20}"
                f" n={row['sample_size']:>6,}"
                f"  default={row['default_rate']:.3f}"
                f"  ROC-AUC={row['roc_auc']:.4f}"
                if not np.isnan(
                    row["roc_auc"]
                )
                else
                f"  {row['group']:<20}"
                f" n={row['sample_size']:>6,}"
                f"  default={row['default_rate']:.3f}"
                f"  ROC-AUC=N/A"
            )

    # --------------------------------------------------------
    # DATAFRAME
    # --------------------------------------------------------

    metrics_df = pd.DataFrame(
        subgroup_metrics
    )

    # Ensure stable ordering.
    metrics_df = metrics_df[
        [
            "group_type",
            "group",
            "sample_size",
            "default_count",
            "non_default_count",
            "default_rate",
            "mean_predicted_probability",
            "roc_auc",
            "pr_auc",
            "brier_score",
            "precision",
            "recall",
            "specificity",
            "false_positive_rate",
            "false_negative_rate",
            "f1",
            "true_negatives",
            "false_positives",
            "false_negatives",
            "true_positives",
        ]
    ]

    # --------------------------------------------------------
    # ROBUSTNESS SUMMARY
    # --------------------------------------------------------

    robustness_summary = (
        build_robustness_summary(
            subgroup_metrics
        )
    )

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    metrics_df.to_parquet(
        OUTPUT_DIR
        / "subgroup_metrics.parquet",
        index=False,
    )

    metrics_df.to_csv(
        OUTPUT_DIR
        / "subgroup_metrics.csv",
        index=False,
    )

    robustness_summary.to_parquet(
        OUTPUT_DIR
        / "subgroup_robustness_summary.parquet",
        index=False,
    )

    robustness_summary.to_csv(
        OUTPUT_DIR
        / "subgroup_robustness_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------

    print(
        "\n" + "━" * 70
    )

    print(
        "SUBGROUP ROBUSTNESS SUMMARY"
    )

    print(
        "━" * 70
    )

    if len(robustness_summary) > 0:

        display_columns = [
            "group_type",
            "groups_evaluated",
            "roc_auc_range",
            "pr_auc_range",
            "brier_score_range",
            "recall_range",
            "specificity_range",
            "f1_range",
        ]

        print(
            robustness_summary[
                display_columns
            ]
            .round(4)
            .to_string(
                index=False
            )
        )

    else:

        print(
            "No subgroup summary available."
        )

    # --------------------------------------------------------
    # IDENTIFY LARGEST PERFORMANCE RANGES
    # --------------------------------------------------------

    investigation_flags = []

    for _, row in (
        robustness_summary.iterrows()
    ):

        dimension = row[
            "group_type"
        ]

        for metric in [
            "roc_auc",
            "pr_auc",
            "brier_score",
            "recall",
            "specificity",
            "f1",
        ]:

            range_column = (
                f"{metric}_range"
            )

            value = row.get(
                range_column
            )

            if pd.notna(value):
                investigation_flags.append({
                    "group_type":
                        dimension,
                    "metric":
                        metric,
                    "range":
                        float(value),
                })

    flags_df = pd.DataFrame(
        investigation_flags
    )

    if len(flags_df) > 0:

        flags_df = (
            flags_df
            .sort_values(
                "range",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

    flags_df.to_parquet(
        OUTPUT_DIR
        / "subgroup_variation_flags.parquet",
        index=False,
    )

    flags_df.to_csv(
        OUTPUT_DIR
        / "subgroup_variation_flags.csv",
        index=False,
    )

    # --------------------------------------------------------
    # JSON SUMMARY
    # --------------------------------------------------------

    json_summary = {
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "target": TARGET_COLUMN,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "classification_threshold":
            CLASSIFICATION_THRESHOLD,
        "minimum_subgroup_size":
            MIN_SUBGROUP_SIZE,
        "dimensions_evaluated":
            dimensions_evaluated,
        "overall_performance":
            overall_metrics,
        "note": (
            "Subgroup variables are used for "
            "evaluation only. They are not added "
            "to the selected model unless they are "
            "already part of the governed feature "
            "set. Classification metrics use a "
            "0.50 threshold for diagnostic purposes "
            "only; this is not the final operating "
            "threshold."
        ),
    }

    with open(
        OUTPUT_DIR
        / "subgroup_robustness_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            json_summary,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    metadata = {
        "model":
            MODEL_NAME,
        "model_version":
            MODEL_VERSION,
        "feature_count":
            len(FEATURES),
        "features":
            FEATURES,
        "target":
            TARGET_COLUMN,
        "random_state":
            RANDOM_STATE,
        "test_size":
            TEST_SIZE,
        "classification_threshold":
            CLASSIFICATION_THRESHOLD,
        "minimum_subgroup_size":
            MIN_SUBGROUP_SIZE,
        "subgroup_dimensions_requested":
            SUBGROUP_DIMENSIONS,
        "subgroup_dimensions_evaluated":
            dimensions_evaluated,
        "purpose": (
            "Evaluate robustness of the selected "
            "Conservative XGBoost across borrower "
            "subgroups."
        ),
    }

    with open(
        OUTPUT_DIR
        / "subgroup_robustness_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "✓ SUBGROUP ROBUSTNESS ANALYSIS COMPLETE"
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
        "  → subgroup_metrics.parquet"
    )

    print(
        "  → subgroup_metrics.csv"
    )

    print(
        "  → subgroup_robustness_summary.parquet"
    )

    print(
        "  → subgroup_robustness_summary.csv"
    )

    print(
        "  → subgroup_variation_flags.parquet"
    )

    print(
        "  → subgroup_variation_flags.csv"
    )

    print(
        "  → subgroup_robustness_summary.json"
    )

    print(
        "  → subgroup_robustness_metadata.json"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Do not interpret subgroup variation as "
        "a fairness violation automatically."
    )

    print(
        "Use the results to identify groups requiring "
        "further investigation before operational deployment."
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()