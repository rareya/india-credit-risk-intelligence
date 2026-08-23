"""
shap_analysis.py

SHAP explainability analysis for the SELECTED
Conservative XGBoost model.

IMPORTANT:

    This script explains the exact same model definition
    used by run_ml_model.py, cross_validate.py and
    calibrate_model.py.

Pipeline:

    Gold data
        ↓
    30 Conservative features
        ↓
    shared preprocessing
        ↓
    Conservative XGBoost
        ↓
    SHAP
        ↓
    global feature importance
        ↓
    original-feature aggregation
        ↓
    borrower-level explanations

Outputs:

    shap_transformed_feature_importance.parquet
    shap_feature_importance.parquet
    shap_feature_importance.csv
    shap_borrower_explanations.parquet
    shap_borrower_explanations.csv
    shap_summary.parquet
    shap_metadata.json
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
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
    / "shap"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

TEST_SIZE = 0.20

SHAP_SAMPLE_SIZE = 5000

BORROWER_EXPLANATION_SAMPLE = 100


# ============================================================
# SHAP VALUE EXTRACTION
# ============================================================

def calculate_shap_values(
    model,
    X,
):
    """
    Calculate SHAP values on the transformed feature matrix.

    The model is a sklearn Pipeline:

        preprocessor
            ↓
        XGBClassifier

    The fitted preprocessor and underlying XGBoost model
    are extracted explicitly.
    """

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    xgb_model = model.named_steps[
        "model"
    ]

    X_transformed = (
        preprocessor.transform(X)
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    explainer = shap.TreeExplainer(
        xgb_model
    )

    shap_values = (
        explainer.shap_values(
            X_transformed
        )
    )

    shap_values = np.asarray(
        shap_values
    )

    # --------------------------------------------------------
    # Defensive handling for SHAP output shape.
    #
    # Binary XGBoost should normally produce:
    #
    #     (rows, transformed_features)
    #
    # But some SHAP versions can return an extra dimension.
    # --------------------------------------------------------

    if shap_values.ndim == 3:

        if shap_values.shape[-1] == 2:
            shap_values = shap_values[:, :, 1]

        elif shap_values.shape[0] == 2:
            shap_values = shap_values[1, :, :]

        else:
            raise ValueError(
                "Unexpected 3-dimensional SHAP output "
                f"shape: {shap_values.shape}"
            )

    if shap_values.ndim != 2:
        raise ValueError(
            "Unexpected SHAP output shape: "
            f"{shap_values.shape}"
        )

    if shap_values.shape[0] != len(X):
        raise ValueError(
            "SHAP row count does not match input rows. "
            f"SHAP rows={shap_values.shape[0]}, "
            f"X rows={len(X)}"
        )

    if shap_values.shape[1] != len(feature_names):
        raise ValueError(
            "SHAP feature count does not match "
            "transformed feature names. "
            f"SHAP features={shap_values.shape[1]}, "
            f"feature names={len(feature_names)}"
        )

    return (
        shap_values,
        X_transformed,
        feature_names,
    )


# ============================================================
# MAP TRANSFORMED FEATURES BACK TO ORIGINAL FEATURES
# ============================================================

def original_feature_name(
    transformed_name,
):
    """
    Convert sklearn ColumnTransformer names such as:

        numeric__age
        categorical__gender_Male

    back to the original feature:

        age
        gender
    """

    name = str(
        transformed_name
    )

    if "__" in name:
        name = name.split(
            "__",
            1
        )[1]

    # Match longest known feature first.
    for feature in sorted(
        FEATURES,
        key=len,
        reverse=True,
    ):

        if (
            name == feature
            or name.startswith(
                feature + "_"
            )
        ):
            return feature

    return name


# ============================================================
# GLOBAL FEATURE IMPORTANCE
# ============================================================

def build_feature_importance(
    shap_values,
    feature_names,
):
    """
    Aggregate one-hot encoded SHAP values back to
    the original 30 governed features.
    """

    absolute_values = np.abs(
        shap_values
    )

    rows = []

    for index, name in enumerate(
        feature_names
    ):

        rows.append({

            "transformed_feature":
                str(name),

            "original_feature":
                original_feature_name(
                    name
                ),

            "mean_abs_shap":
                float(
                    absolute_values[
                        :,
                        index
                    ].mean()
                ),

            "mean_shap":
                float(
                    shap_values[
                        :,
                        index
                    ].mean()
                ),
        })

    transformed_df = pd.DataFrame(
        rows
    )

    # Aggregate one-hot columns back
    # to original governed features.
    feature_importance = (
        transformed_df
        .groupby(
            "original_feature",
            as_index=False,
        )
        .agg(
            mean_abs_shap=(
                "mean_abs_shap",
                "sum",
            ),
            mean_shap=(
                "mean_shap",
                "sum",
            ),
        )
    )

    feature_importance[
        "importance_rank"
    ] = (
        feature_importance[
            "mean_abs_shap"
        ]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    feature_importance = (
        feature_importance
        .sort_values(
            "mean_abs_shap",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    return (
        transformed_df,
        feature_importance,
    )


# ============================================================
# BORROWER EXPLANATIONS
# ============================================================

def build_borrower_explanations(
    df,
    shap_values,
    feature_names,
    sample_indices,
):
    """
    Produce borrower-level top SHAP drivers.

    SHAP values are first aggregated from one-hot encoded
    columns back to original feature names.
    """

    aggregated = {}

    for index, name in enumerate(
        feature_names
    ):

        original = (
            original_feature_name(
                name
            )
        )

        if original not in aggregated:

            aggregated[
                original
            ] = np.zeros(
                len(shap_values)
            )

        aggregated[
            original
        ] += shap_values[
            :,
            index
        ]

    aggregated_df = pd.DataFrame(
        aggregated
    )

    rows = []

    for local_position, source_index in enumerate(
        sample_indices
    ):

        borrower_row = df.iloc[
            source_index
        ]

        shap_row = (
            aggregated_df
            .iloc[
                local_position
            ]
        )

        sorted_features = (
            shap_row
            .abs()
            .sort_values(
                ascending=False
            )
        )

        top_features = (
            sorted_features
            .head(10)
            .index
            .tolist()
        )

        for rank, feature in enumerate(
            top_features,
            start=1,
        ):

            shap_value = float(
                shap_row[
                    feature
                ]
            )

            if "borrower_id" in borrower_row.index:
                borrower_id = borrower_row[
                    "borrower_id"
                ]
            else:
                borrower_id = None

            rows.append({

                "row_index":
                    int(source_index),

                "borrower_id":
                    borrower_id,

                "feature":
                    feature,

                "rank":
                    int(rank),

                "shap_value":
                    shap_value,

                "absolute_shap":
                    float(
                        abs(
                            shap_value
                        )
                    ),

                "direction":
                    (
                        "increases_risk"
                        if shap_value > 0
                        else
                        "decreases_risk"
                    ),

                "feature_value":
                    str(
                        borrower_row[
                            feature
                        ]
                    ),
            })

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "  INDIA CREDIT RISK — "
        "SHAP EXPLAINABILITY"
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

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print(
        "\nLoading Gold fact table..."
    )

    df = load_gold_data()

    # --------------------------------------------------------
    # Validate selected model configuration
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Selected model features missing "
            "from Gold fact table: "
            f"{missing_features}"
        )

    if TARGET_COLUMN not in df.columns:

        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            "is missing from Gold fact table."
        )

    X = df[
        FEATURES
    ].copy()

    y = df[
        TARGET_COLUMN
    ].astype(int)

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

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
        "\n" + "━" * 70
    )

    print(
        "TRAIN / TEST SPLIT"
    )

    print(
        "━" * 70
    )

    print(
        f"  Train: {len(X_train):,}"
    )

    print(
        f"  Test:  {len(X_test):,}"
    )

    # --------------------------------------------------------
    # Train exact selected model
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
    # Evaluate
    # --------------------------------------------------------

    test_probability = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    roc_auc = roc_auc_score(
        y_test,
        test_probability
    )

    pr_auc = average_precision_score(
        y_test,
        test_probability
    )

    brier = brier_score_loss(
        y_test,
        test_probability
    )

    print(
        "\nModel performance:"
    )

    print(
        f"  ROC-AUC: {roc_auc:.4f}"
    )

    print(
        f"  PR-AUC:  {pr_auc:.4f}"
    )

    print(
        f"  Brier:   {brier:.4f}"
    )

    # --------------------------------------------------------
    # SHAP sample
    # --------------------------------------------------------

    sample_size = min(
        SHAP_SAMPLE_SIZE,
        len(X_test)
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    sample_positions = (
        rng.choice(
            len(X_test),
            size=sample_size,
            replace=False,
        )
    )

    X_shap = X_test.iloc[
        sample_positions
    ]

    print(
        "\nCalculating SHAP values..."
    )

    (
        shap_values,
        X_transformed,
        feature_names,
    ) = calculate_shap_values(
        model,
        X_shap,
    )

    print(
        f"  ✓ SHAP rows: "
        f"{len(X_shap):,}"
    )

    print(
        f"  ✓ Transformed features: "
        f"{len(feature_names)}"
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    (
        transformed_importance,
        feature_importance,
    ) = build_feature_importance(
        shap_values,
        feature_names,
    )

    # --------------------------------------------------------
    # Save transformed SHAP metadata
    # --------------------------------------------------------

    transformed_importance.to_parquet(
        OUTPUT_DIR
        / "shap_transformed_feature_importance.parquet",
        index=False,
    )

    # --------------------------------------------------------
    # Save original feature importance
    # --------------------------------------------------------

    feature_importance.to_parquet(
        OUTPUT_DIR
        / "shap_feature_importance.parquet",
        index=False,
    )

    feature_importance.to_csv(
        OUTPUT_DIR
        / "shap_feature_importance.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Print ranking
    # --------------------------------------------------------

    print(
        "\n" + "━" * 70
    )

    print(
        "GLOBAL SHAP FEATURE IMPORTANCE"
    )

    print(
        "━" * 70
    )

    print(
        feature_importance[
            [
                "importance_rank",
                "original_feature",
                "mean_abs_shap",
                "mean_shap",
            ]
        ]
        .head(30)
        .round(6)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Borrower explanations
    # --------------------------------------------------------

    borrower_sample_size = min(
        BORROWER_EXPLANATION_SAMPLE,
        len(X_test)
    )

    borrower_positions = (
        rng.choice(
            len(X_test),
            size=borrower_sample_size,
            replace=False,
        )
    )

    X_borrowers = X_test.iloc[
        borrower_positions
    ]

    (
        borrower_shap_values,
        _,
        borrower_feature_names,
    ) = calculate_shap_values(
        model,
        X_borrowers,
    )

    # Convert test-local positions back
    # to original dataframe indices.
    borrower_source_indices = (
        X_test.index[
            borrower_positions
        ]
        .tolist()
    )

    borrower_explanations = (
        build_borrower_explanations(
            df=df,
            shap_values=borrower_shap_values,
            feature_names=borrower_feature_names,
            sample_indices=borrower_source_indices,
        )
    )

    borrower_explanations.to_parquet(
        OUTPUT_DIR
        / "shap_borrower_explanations.parquet",
        index=False,
    )

    borrower_explanations.to_csv(
        OUTPUT_DIR
        / "shap_borrower_explanations.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    #
    # IMPORTANT:
    #
    # Do NOT mix strings and numbers in the same
    # parquet column.
    #
    # The previous implementation created:
    #
    #     value = MODEL_NAME       -> string
    #     value = len(FEATURES)    -> integer
    #     value = roc_auc           -> float
    #
    # PyArrow correctly rejected that mixed object
    # column.
    #
    # We therefore explicitly store all summary values
    # as strings.
    # --------------------------------------------------------

    summary_rows = [

        {
            "metric": "model",
            "value": str(MODEL_NAME),
        },

        {
            "metric": "model_version",
            "value": str(MODEL_VERSION),
        },

        {
            "metric": "feature_count",
            "value": str(len(FEATURES)),
        },

        {
            "metric": "shap_sample_size",
            "value": str(sample_size),
        },

        {
            "metric": "borrower_explanation_sample",
            "value": str(
                borrower_sample_size
            ),
        },

        {
            "metric": "test_roc_auc",
            "value": f"{roc_auc:.6f}",
        },

        {
            "metric": "test_pr_auc",
            "value": f"{pr_auc:.6f}",
        },

        {
            "metric": "test_brier_score",
            "value": f"{brier:.6f}",
        },
    ]

    summary = pd.DataFrame(
        summary_rows,
        columns=[
            "metric",
            "value",
        ],
    )

    summary[
        "metric"
    ] = summary[
        "metric"
    ].astype(str)

    summary[
        "value"
    ] = summary[
        "value"
    ].astype(str)

    summary.to_parquet(
        OUTPUT_DIR
        / "shap_summary.parquet",
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

        "feature_count":
            len(FEATURES),

        "features":
            FEATURES,

        "random_state":
            RANDOM_STATE,

        "test_size":
            TEST_SIZE,

        "shap_sample_size":
            sample_size,

        "borrower_explanation_sample":
            borrower_sample_size,

        "transformed_feature_count":
            len(feature_names),

        "roc_auc":
            float(roc_auc),

        "pr_auc":
            float(pr_auc),

        "brier_score":
            float(brier),

        "note":
            (
                "SHAP values were calculated on the "
                "same preprocessed feature representation "
                "used by the selected Conservative XGBoost. "
                "One-hot encoded SHAP values were aggregated "
                "back to the original governed features."
            ),
    }

    with open(
        OUTPUT_DIR
        / "shap_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "✓ SHAP ANALYSIS COMPLETE"
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
        "  → shap_transformed_feature_importance.parquet"
    )

    print(
        "  → shap_feature_importance.parquet"
    )

    print(
        "  → shap_feature_importance.csv"
    )

    print(
        "  → shap_borrower_explanations.parquet"
    )

    print(
        "  → shap_borrower_explanations.csv"
    )

    print(
        "  → shap_summary.parquet"
    )

    print(
        "  → shap_metadata.json"
    )


if __name__ == "__main__":
    main()