"""
cross_validate.py

5-fold stratified cross-validation for Conservative XGBoost.

Uses ONLY the conservative feature registry.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from feature_registry import (
    TARGET,
    CONSERVATIVE_FEATURES,
    validate_registry,
)

GOLD_DIR = Path("data/gold/exports")
OUTPUT_DIR = GOLD_DIR / "analytics" / "ml" / "cross_validation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


RANDOM_STATE = 42
N_SPLITS = 5


def load_data():

    df = pd.read_parquet(
        GOLD_DIR / "fact_credit_risk.parquet"
    )

    validate_registry(df.columns)

    return df


def build_pipeline(X):

    categorical = [
        c for c in X.columns
        if X[c].dtype == "object"
    ]

    numeric = [
        c for c in X.columns
        if c not in categorical
    ]

    numeric_pipe = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        )
    ])

    categorical_pipe = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def main():

    print("=" * 70)
    print("  INDIA CREDIT RISK — 5-FOLD CROSS-VALIDATION")
    print("=" * 70)

    df = load_data()

    X = df[CONSERVATIVE_FEATURES]
    y = df[TARGET].astype(int)

    print(f"\nFeatures: {len(CONSERVATIVE_FEATURES)}")
    print(f"Rows: {len(df):,}")

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y),
        start=1,
    ):

        print(f"\n━━━ Fold {fold}/{N_SPLITS} ━━━")

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        pipeline = build_pipeline(X_train)

        pipeline.fit(X_train, y_train)

        probabilities = pipeline.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(
            y_test,
            probabilities,
        )

        pr_auc = average_precision_score(
            y_test,
            probabilities,
        )

        brier = brier_score_loss(
            y_test,
            probabilities,
        )

        results.append({
            "fold": fold,
            "roc_auc": auc,
            "pr_auc": pr_auc,
            "brier_score": brier,
            "train_rows": len(train_idx),
            "test_rows": len(test_idx),
        })

        print(f"  ROC-AUC: {auc:.4f}")
        print(f"  PR-AUC:  {pr_auc:.4f}")
        print(f"  Brier:   {brier:.4f}")

    results_df = pd.DataFrame(results)

    summary = pd.DataFrame([{
        "metric": "roc_auc",
        "mean": results_df.roc_auc.mean(),
        "std": results_df.roc_auc.std(ddof=1),
        "min": results_df.roc_auc.min(),
        "max": results_df.roc_auc.max(),
    }, {
        "metric": "pr_auc",
        "mean": results_df.pr_auc.mean(),
        "std": results_df.pr_auc.std(ddof=1),
        "min": results_df.pr_auc.min(),
        "max": results_df.pr_auc.max(),
    }, {
        "metric": "brier_score",
        "mean": results_df.brier_score.mean(),
        "std": results_df.brier_score.std(ddof=1),
        "min": results_df.brier_score.min(),
        "max": results_df.brier_score.max(),
    }])

    results_df.to_parquet(
        OUTPUT_DIR / "cv_fold_results.parquet",
        index=False,
    )

    summary.to_parquet(
        OUTPUT_DIR / "cv_summary.parquet",
        index=False,
    )

    results_df.to_csv(
        OUTPUT_DIR / "cv_fold_results.csv",
        index=False,
    )

    print("\n" + "=" * 70)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 70)

    print(summary.to_string(index=False))

    print("\n✓ Cross-validation complete")


if __name__ == "__main__":
    main()