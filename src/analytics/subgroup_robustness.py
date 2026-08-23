"""
subgroup_robustness.py

Evaluate Conservative XGBoost performance across borrower subgroups.

Subgroups:
    - gender
    - age_band
    - income_tier
    - cibil_band
    - has_gold_loan
    - education
"""

from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from feature_registry import (
    TARGET,
    CONSERVATIVE_FEATURES,
    validate_registry,
)

GOLD_DIR = Path("data/gold/exports")
OUTPUT_DIR = GOLD_DIR / "analytics" / "ml" / "subgroup"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def build_model(X):

    categorical = [
        c for c in X.columns
        if X[c].dtype == "object"
    ]

    numeric = [
        c for c in X.columns
        if c not in categorical
    ]

    preprocessor = ColumnTransformer([
        (
            "numeric",
            SimpleImputer(strategy="median"),
            numeric,
        ),
        (
            "categorical",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]),
            categorical,
        ),
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


def evaluate_subgroup(
    group_name,
    group_value,
    group,
    threshold=0.5,
):

    if len(group) < 100:
        return None

    if group[TARGET].nunique() < 2:
        return None

    y = group[TARGET]
    probability = group["predicted_probability"]

    prediction = (
        probability >= threshold
    ).astype(int)

    return {
        "subgroup": group_name,
        "group": str(group_value),
        "n": len(group),
        "default_rate_pct": y.mean() * 100,
        "roc_auc": roc_auc_score(
            y,
            probability,
        ),
        "pr_auc": average_precision_score(
            y,
            probability,
        ),
        "precision": precision_score(
            y,
            prediction,
            zero_division=0,
        ),
        "recall": recall_score(
            y,
            prediction,
            zero_division=0,
        ),
    }


def main():

    print("=" * 70)
    print("  INDIA CREDIT RISK — SUBGROUP ROBUSTNESS")
    print("=" * 70)

    df = pd.read_parquet(
        GOLD_DIR / "fact_credit_risk.parquet"
    )

    validate_registry(df.columns)

    X = df[CONSERVATIVE_FEATURES]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    model = build_model(X_train)

    print("\nTraining Conservative XGBoost...")
    model.fit(X_train, y_train)

    test_df = df.loc[
        X_test.index
    ].copy()

    test_df["predicted_probability"] = (
        model.predict_proba(X_test)[:, 1]
    )

    # --------------------------------------------------------
    # Age bands
    # --------------------------------------------------------

    test_df["age_band"] = pd.cut(
        test_df["age"],
        bins=[0, 24, 34, 44, 54, 200],
        labels=[
            "under_25",
            "25_to_34",
            "35_to_44",
            "45_to_54",
            "55_plus",
        ],
    )

    subgroup_columns = [
        "gender",
        "age_band",
        "income_tier",
        "cibil_band",
        "has_gold_loan",
        "education",
    ]

    rows = []

    for column in subgroup_columns:

        if column not in test_df.columns:
            continue

        for value, group in test_df.groupby(
            column,
            dropna=False,
        ):

            result = evaluate_subgroup(
                column,
                value,
                group,
            )

            if result:
                rows.append(result)

    result_df = pd.DataFrame(rows)

    result_df.to_parquet(
        OUTPUT_DIR / "subgroup_performance.parquet",
        index=False,
    )

    result_df.to_csv(
        OUTPUT_DIR / "subgroup_performance.csv",
        index=False,
    )

    print("\nSubgroup performance:")

    print(
        result_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # --------------------------------------------------------
    # Robustness summary
    # --------------------------------------------------------

    summary = pd.DataFrame([{
        "metric": "minimum_subgroup_auc",
        "value": result_df.roc_auc.min(),
    }, {
        "metric": "maximum_subgroup_auc",
        "value": result_df.roc_auc.max(),
    }, {
        "metric": "auc_range",
        "value": (
            result_df.roc_auc.max()
            - result_df.roc_auc.min()
        ),
    }, {
        "metric": "minimum_subgroup_recall",
        "value": result_df.recall.min(),
    }, {
        "metric": "maximum_subgroup_recall",
        "value": result_df.recall.max(),
    }])

    summary.to_parquet(
        OUTPUT_DIR / "subgroup_robustness_summary.parquet",
        index=False,
    )

    print("\n✓ Subgroup robustness analysis complete")


if __name__ == "__main__":
    main()