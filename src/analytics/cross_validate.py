"""
cross_validate.py

5-fold stratified cross-validation for the SELECTED
Conservative XGBoost model.

IMPORTANT:
    This is NOT model selection.

    Conservative XGBoost has already been selected.

    This script evaluates:
        - stability
        - variance
        - generalization consistency
        - probability quality

    across five independent stratified folds.
"""

from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.model_selection import StratifiedKFold

from final_model_config import (
    MODEL_NAME,
    RANDOM_STATE,
    FEATURES,
    TARGET_COLUMN,
    build_xgb_model,
    load_gold_data,
)


# ============================================================
# PATHS
# ============================================================

GOLD_DIR = Path("data/gold/exports")

OUTPUT_DIR = (
    GOLD_DIR /
    "analytics" /
    "ml" /
    "cross_validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

N_SPLITS = 5


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "  INDIA CREDIT RISK — "
        "5-FOLD CROSS-VALIDATION"
    )
    print("=" * 70)

    print(
        f"\nModel: {MODEL_NAME}"
    )

    print(
        f"Features: {len(FEATURES)}"
    )

    print(
        f"Target: {TARGET_COLUMN}"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_gold_data()

    X = df[FEATURES].copy()

    y = df[
        TARGET_COLUMN
    ].astype(int)

    print(
        f"Rows: {len(df):,}"
    )

    # --------------------------------------------------------
    # CV definition
    # --------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = []

    # --------------------------------------------------------
    # Fold loop
    # --------------------------------------------------------

    for fold, (
        train_idx,
        test_idx
    ) in enumerate(
        cv.split(X, y),
        start=1,
    ):

        print(
            f"\n{'━' * 70}"
        )

        print(
            f"Fold {fold}/{N_SPLITS}"
        )

        print(
            f"{'━' * 70}"
        )

        X_train = X.iloc[
            train_idx
        ]

        X_test = X.iloc[
            test_idx
        ]

        y_train = y.iloc[
            train_idx
        ]

        y_test = y.iloc[
            test_idx
        ]

        print(
            f"  Training rows: {len(X_train):,}"
        )

        print(
            f"  Testing rows:  {len(X_test):,}"
        )

        # ----------------------------------------------------
        # Build exact selected model
        # ----------------------------------------------------

        model = build_xgb_model(
            X_train
        )

        print(
            "\n  Training..."
        )

        model.fit(
            X_train,
            y_train
        )

        print(
            "  ✓ Training complete"
        )

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        probabilities = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )

        pr_auc = average_precision_score(
            y_test,
            probabilities
        )

        brier = brier_score_loss(
            y_test,
            probabilities
        )

        results.append({
            "model": MODEL_NAME,
            "fold": fold,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "brier_score": brier,
            "train_rows": len(train_idx),
            "test_rows": len(test_idx),
        })

        print(
            f"\n  ROC-AUC: {roc_auc:.4f}"
        )

        print(
            f"  PR-AUC:  {pr_auc:.4f}"
        )

        print(
            f"  Brier:   {brier:.4f}"
        )

    # --------------------------------------------------------
    # Results dataframe
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = pd.DataFrame([
        {
            "model": MODEL_NAME,
            "metric": "roc_auc",
            "mean":
                results_df["roc_auc"].mean(),
            "std":
                results_df["roc_auc"].std(
                    ddof=1
                ),
            "min":
                results_df["roc_auc"].min(),
            "max":
                results_df["roc_auc"].max(),
        },
        {
            "model": MODEL_NAME,
            "metric": "pr_auc",
            "mean":
                results_df["pr_auc"].mean(),
            "std":
                results_df["pr_auc"].std(
                    ddof=1
                ),
            "min":
                results_df["pr_auc"].min(),
            "max":
                results_df["pr_auc"].max(),
        },
        {
            "model": MODEL_NAME,
            "metric": "brier_score",
            "mean":
                results_df["brier_score"].mean(),
            "std":
                results_df["brier_score"].std(
                    ddof=1
                ),
            "min":
                results_df["brier_score"].min(),
            "max":
                results_df["brier_score"].max(),
        },
    ])

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results_df.to_parquet(
        OUTPUT_DIR /
        "cv_fold_results.parquet",
        index=False,
    )

    results_df.to_csv(
        OUTPUT_DIR /
        "cv_fold_results.csv",
        index=False,
    )

    summary.to_parquet(
        OUTPUT_DIR /
        "cv_summary.parquet",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR /
        "cv_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Print final summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "CROSS-VALIDATION SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        summary.round(4).to_string(
            index=False
        )
    )

    print(
        "\n✓ Conservative XGBoost "
        "cross-validation complete"
    )

    print(
        "\nSaved to:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()