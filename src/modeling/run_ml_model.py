"""
run_ml_model.py — Leakage-Aware Credit Risk Modeling

Models:

    Model 0
        Logistic Regression baseline

    Model 1
        Conservative XGBoost
        Uses features approved for conservative ML use.

    Model 2
        Expanded XGBoost
        Uses features approved for expanded ML use.
        Expanded-only features require additional
        provenance/availability review before operational use.

The feature sets are obtained from feature_governance.py.

IMPORTANT:
    No target-derived features are allowed.

Evaluation:
    ROC-AUC
    PR-AUC
    Accuracy
    Precision
    Recall
    F1
    Brier score
    Confusion matrix
    Calibration
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

GOLD_DIR = Path("data/gold/exports")
ANALYTICS_DIR = GOLD_DIR / "analytics"

FACT_FILE = GOLD_DIR / "fact_credit_risk.parquet"
REGISTRY_FILE = (
    ANALYTICS_DIR /
    "feature_governance.parquet"
)

OUTPUT_DIR = "data/model/evaluation"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


TARGET = "default_risk"

RANDOM_STATE = 42


# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------

def load_data():

    print("=" * 70)
    print("  INDIA CREDIT RISK — ML MODELING")
    print("=" * 70)

    print("\nLoading Gold fact table...")

    df = pd.read_parquet(FACT_FILE)

    print(
        f"  ✓ Loaded: "
        f"{df.shape[0]:,} rows × {df.shape[1]} columns"
    )

    return df


# ---------------------------------------------------------------------
# LOAD GOVERNED FEATURES
# ---------------------------------------------------------------------

def load_feature_sets():

    registry = pd.read_parquet(
        REGISTRY_FILE
    )

    conservative = registry[
        registry["ml_eligible_conservative"]
    ]["feature"].tolist()

    expanded = registry[
        registry["ml_eligible_expanded"]
    ]["feature"].tolist()

    # ---------------------------------------------------------------
    # Governance sanity checks
    # ---------------------------------------------------------------

    conservative_set = set(conservative)
    expanded_set = set(expanded)

    # Every conservative feature must also be allowed in expanded.
    if not conservative_set.issubset(expanded_set):
        invalid = sorted(
            conservative_set - expanded_set
        )

        raise ValueError(
            "GOVERNANCE ERROR: "
            "Conservative features missing from expanded set: "
            f"{invalid}"
        )

    # Expanded-only features are explicitly identified.
    expanded_only = sorted(
        expanded_set - conservative_set
    )

    print("\nFeature governance:")
    print(
        f"  Conservative features: "
        f"{len(conservative)}"
    )

    print(
        f"  Expanded features:     "
        f"{len(expanded)}"
    )

    print(
        f"  Expanded-only features:"
        f" {len(expanded_only)}"
    )

    if expanded_only:
        print("\n  Expanded-only features:")

        for feature in expanded_only:
            print(f"    - {feature}")

    return conservative, expanded, registry

# ---------------------------------------------------------------------
# PREPROCESSOR
# ---------------------------------------------------------------------

def build_preprocessor(X):

    numeric_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        ),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    return preprocessor


# ---------------------------------------------------------------------
# MODEL BUILDERS
# ---------------------------------------------------------------------

def build_logistic_model(X):

    preprocessor = build_preprocessor(X)

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def build_xgb_model(X):

    preprocessor = build_preprocessor(X)

    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


# ---------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------

def evaluate_model(
    model,
    X_test,
    y_test,
    model_name
):

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    metrics = {
        "model": model_name,

        "roc_auc":
            roc_auc_score(
                y_test,
                probabilities
            ),

        "pr_auc":
            average_precision_score(
                y_test,
                probabilities
            ),

        "accuracy":
            accuracy_score(
                y_test,
                predictions
            ),

        "precision":
            precision_score(
                y_test,
                predictions,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_test,
                predictions,
                zero_division=0
            ),

        "f1":
            f1_score(
                y_test,
                predictions,
                zero_division=0
            ),

        "brier_score":
            brier_score_loss(
                y_test,
                probabilities
            ),

        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }

    return metrics


# ---------------------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------------------

def train_model(
    df,
    features,
    model_name,
    model
):

    print("\n" + "━" * 70)
    print(model_name)
    print("━" * 70)

    X = df[features].copy()
    y = df[TARGET].astype(int)

    print(
        f"  Features: {len(features)}"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(
        f"  Training rows: {len(X_train):,}"
    )

    print(
        f"  Testing rows:  {len(X_test):,}"
    )

    print("\n  Training...")

    model.fit(
        X_train,
        y_train
    )

    print("  ✓ Training complete")

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
        model_name
    )

    print("\n  Performance:")

    print(
        f"    ROC-AUC:    {metrics['roc_auc']:.4f}"
    )

    print(
        f"    PR-AUC:     {metrics['pr_auc']:.4f}"
    )

    print(
        f"    Accuracy:   {metrics['accuracy']:.4f}"
    )

    print(
        f"    Precision:  {metrics['precision']:.4f}"
    )

    print(
        f"    Recall:     {metrics['recall']:.4f}"
    )

    print(
        f"    F1:         {metrics['f1']:.4f}"
    )

    print(
        f"    Brier:      {metrics['brier_score']:.4f}"
    )

    print("\n  Confusion matrix:")

    print(
        f"    TN={metrics['true_negatives']:,}"
        f"  FP={metrics['false_positives']:,}"
    )

    print(
        f"    FN={metrics['false_negatives']:,}"
        f"  TP={metrics['true_positives']:,}"
    )

    return model, metrics


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    df = load_data()

    print("\nValidating target...")

    assert TARGET in df.columns

    assert set(
        df[TARGET].dropna().unique()
    ).issubset({0, 1})

    print("  ✓ Target validated")

    conservative, expanded, registry = (
    load_feature_sets()
    )

    # ---------------------------------------------------------------
    # Safety check
    # ---------------------------------------------------------------

    forbidden = {
        "borrower_id",
        "default_risk",
        "risk_band",
        "risk_grade",
        "risk_grade_numeric",
    }

    for feature in (
        conservative + expanded
    ):
        if feature in forbidden:
            raise ValueError(
                f"LEAKAGE BLOCKED: "
                f"{feature} entered ML feature set."
            )

    print("\n✓ Leakage guard passed")

        # ---------------------------------------------------------------
    # Review-feature governance check
    # ---------------------------------------------------------------

    review_features = registry[
        registry["feature"].isin(expanded)
        & (
            registry["reason"]
            .fillna("")
            .str.contains(
                "review|confirm|provenance|available",
                case=False,
                regex=True
            )
        )
    ]["feature"].tolist()

    if review_features:

        print("\n⚠ Expanded model contains review features:")

        for feature in review_features:
            print(f"    - {feature}")

        print(
            "\n  These features are included for analytical "
            "comparison only and should not be treated as "
            "operationally approved until provenance is confirmed."
        )

    # ---------------------------------------------------------------
    # Model 0 — Logistic baseline
    # ---------------------------------------------------------------

    baseline_features = [
        feature
        for feature in conservative
        if feature in [
            "age",
            "monthly_income_inr",
            "cibil_score",
            "credit_history_months",
            "num_times_delinquent",
            "total_loans",
            "active_loan_ratio",
            "recent_enquiries_6m",
            "recent_enquiries_12m",
        ]
    ]

    baseline_model = build_logistic_model(
        df[baseline_features]
    )

    baseline_model, baseline_metrics = train_model(
        df,
        baseline_features,
        "Model 0 — Logistic Baseline",
        baseline_model
    )

    # ---------------------------------------------------------------
    # Model 1 — Conservative XGBoost
    # ---------------------------------------------------------------

    conservative_model = build_xgb_model(
        df[conservative]
    )

    conservative_model, conservative_metrics = train_model(
        df,
        conservative,
        "Model 1 — Conservative XGBoost",
        conservative_model
    )

    # ---------------------------------------------------------------
    # Model 2 — Expanded XGBoost
    # ---------------------------------------------------------------

    expanded_model = build_xgb_model(
        df[expanded]
    )

    expanded_model, expanded_metrics = train_model(
        df,
        expanded,
        "Model 2 — Expanded XGBoost",
        expanded_model
    )

    # ---------------------------------------------------------------
    # Comparison
    # ---------------------------------------------------------------

    comparison = pd.DataFrame([
        baseline_metrics,
        conservative_metrics,
        expanded_metrics,
    ])

    comparison.to_parquet(
        OUTPUT_DIR /
        "model_comparison.parquet",
        index=False
    )

    comparison.to_csv(
        OUTPUT_DIR /
        "model_comparison.csv",
        index=False
    )

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    display_cols = [
        "model",
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "brier_score",
    ]

    print(
        comparison[display_cols]
        .round(4)
        .to_string(index=False)
    )

    # ---------------------------------------------------------------
    # Difference between models
    # ---------------------------------------------------------------

    conservative_auc = (
        conservative_metrics["roc_auc"]
    )

    expanded_auc = (
        expanded_metrics["roc_auc"]
    )

    auc_difference = (
        expanded_auc -
        conservative_auc
    )

    print("\n" + "━" * 70)
    print("Governance comparison")
    print("━" * 70)

    print(
        f"\n  Conservative ROC-AUC: "
        f"{conservative_auc:.4f}"
    )

    print(
        f"  Expanded ROC-AUC:     "
        f"{expanded_auc:.4f}"
    )

    print(
        f"  Difference:           "
        f"{auc_difference:+.4f}"
    )

    if abs(auc_difference) < 0.01:

        interpretation = (
            "Expanded features provide little "
            "additional predictive benefit."
        )

    elif auc_difference > 0:

        interpretation = (
            "Expanded features improve predictive "
            "performance. Provenance of review "
            "features should therefore be investigated "
            "before using them operationally."
        )

    else:

        interpretation = (
            "Expanded features did not improve "
            "predictive performance."
        )

    print(
        f"\n  Interpretation:\n  {interpretation}"
    )

    # ---------------------------------------------------------------
    # Save experiment metadata
    # ---------------------------------------------------------------

    metadata = {
        "rows": int(len(df)),
        "target": TARGET,
        "random_state": RANDOM_STATE,
        "conservative_feature_count":
            len(conservative),
        "expanded_feature_count":
            len(expanded),
        "baseline_feature_count":
            len(baseline_features),
        "conservative_features":
            conservative,
        "expanded_features":
            expanded,
        "baseline_features":
            baseline_features,
        "conservative_roc_auc":
            float(conservative_auc),
        "expanded_roc_auc":
            float(expanded_auc),
        "auc_difference":
            float(auc_difference),
        "interpretation":
            interpretation,
    }

    with open(
        OUTPUT_DIR /
        "model_experiment_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    print("\n" + "=" * 70)
    print("✓ ML EXPERIMENT COMPLETE")
    print("=" * 70)

    print("\nSaved to:")
    print(
        f"  {OUTPUT_DIR}"
    )

    print("\nFiles:")
    print(
        "  → model_comparison.parquet"
    )

    print(
        "  → model_comparison.csv"
    )

    print(
        "  → model_experiment_metadata.json"
    )

    print("\nIMPORTANT:")
    print(
        "Do NOT select the final model yet."
    )

    print(
        "Next: calibration + cross-validation + "
        "SHAP + subgroup robustness analysis."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()