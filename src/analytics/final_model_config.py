"""
final_model_config.py

Single source of truth for the SELECTED production model.

Selected model:
    Conservative XGBoost
    conservative_xgb_v1

This module defines:
    - frozen feature governance
    - target definition
    - preprocessing
    - exact XGBoost hyperparameters
    - model construction
    - Gold dataset loading
    - validation helpers

IMPORTANT
---------
This file does NOT:
    - compare models
    - run cross-validation
    - perform calibration
    - run SHAP
    - select thresholds
    - train/save the final artifact

Those are separate analytical/deployment steps.

All downstream production/analytical scripts should import
this module so that they use the exact same model definition.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xgboost as xgb

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GOLD_PATH = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "gold_fact_table.parquet"
)


# ============================================================
# MODEL IDENTITY
# ============================================================

MODEL_NAME = "Conservative XGBoost"

MODEL_VERSION = "conservative_xgb_v1"

FEATURE_SET_NAME = "conservative_v1"

FEATURE_SET_STATUS = "FROZEN"


# ============================================================
# TARGET
# ============================================================

TARGET_COLUMN = "default_risk"

TARGET_DEFINITION = {
    "0": "P1/P2 low-risk classification proxy",
    "1": "P3/P4 high-risk classification proxy",
}


# ============================================================
# FROZEN CONSERVATIVE FEATURE SET
# ============================================================

FEATURES = [
    "active_loan_ratio",
    "active_loans",
    "age",
    "auto_loans",
    "closed_loans",
    "credit_card_loans",
    "credit_history_months",
    "delinquency_score",
    "education",
    "gender",
    "gold_loans",
    "has_credit_card",
    "has_gold_loan",
    "has_home_loan",
    "has_personal_loan",
    "home_loans",
    "income_tier",
    "loan_type_diversity",
    "marital_status",
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


# ============================================================
# FEATURE TYPES
# ============================================================

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
    "has_credit_card",
    "has_gold_loan",
    "has_home_loan",
    "has_personal_loan",
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
    "income_tier",
    "marital_status",
]


# ============================================================
# GOVERNANCE
# ============================================================

REVIEW_FEATURES_INCLUDED = [
    "recent_enquiries_6m",
    "recent_enquiries_12m",
    "total_enquiries",
]

# These are explicitly NOT part of the selected feature set.
EXCLUDED_FEATURES = [
    "cibil_score",
    "cibil_band",
]


# ============================================================
# HYPERPARAMETERS
# ============================================================

RANDOM_STATE = 42

N_ESTIMATORS = 300
MAX_DEPTH = 4
LEARNING_RATE = 0.05
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8

OBJECTIVE = "binary:logistic"
EVAL_METRIC = "logloss"


# ============================================================
# PREPROCESSING
# ============================================================

def build_preprocessor() -> ColumnTransformer:
    """
    Build the exact preprocessing pipeline used by
    conservative_xgb_v1.
    """

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


# ============================================================
# MODEL BUILDER
# ============================================================

def build_xgb_model() -> Pipeline:
    """
    Construct the exact selected Conservative XGBoost.

    IMPORTANT:
    Do not change hyperparameters here without creating
    a new model version.
    """

    preprocessor = build_preprocessor()

    classifier = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        objective=OBJECTIVE,
        eval_metric=EVAL_METRIC,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                classifier,
            ),
        ]
    )


# ============================================================
# DATA LOADING
# ============================================================

def load_gold_data() -> pd.DataFrame:
    """
    Load the Gold fact table used by the ML pipeline.
    """

    if not GOLD_PATH.exists():
        raise FileNotFoundError(
            f"Gold fact table not found:\n{GOLD_PATH}"
        )

    df = pd.read_parquet(GOLD_PATH)

    return df


# ============================================================
# GOVERNANCE VALIDATION
# ============================================================

def validate_model_configuration(
    df: pd.DataFrame,
) -> None:
    """
    Validate that the dataset and frozen configuration agree.
    """

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            "is missing from Gold data."
        )

    if len(FEATURES) != 30:
        raise ValueError(
            f"Expected exactly 30 frozen features; "
            f"found {len(FEATURES)}."
        )

    if set(FEATURES) != (
        set(NUMERIC_FEATURES)
        | set(CATEGORICAL_FEATURES)
    ):
        raise ValueError(
            "Feature type definitions do not match "
            "the frozen feature list."
        )

    missing = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            "Frozen model features missing from Gold data:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing
            )
        )

    forbidden = [
        feature
        for feature in FEATURES
        if feature in EXCLUDED_FEATURES
    ]

    if forbidden:
        raise ValueError(
            "Forbidden features entered the selected model:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in forbidden
            )
        )


# ============================================================
# CONFIGURATION METADATA
# ============================================================

def get_model_metadata() -> dict:
    """
    Return the authoritative configuration metadata.
    """

    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_type": (
            "sklearn Pipeline "
            "(ColumnTransformer + XGBClassifier)"
        ),
        "feature_set_name": FEATURE_SET_NAME,
        "feature_set_status": FEATURE_SET_STATUS,
        "target": TARGET_COLUMN,
        "target_definition": TARGET_DEFINITION,
        "n_features": len(FEATURES),
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "review_features_included": (
            REVIEW_FEATURES_INCLUDED
        ),
        "excluded_features": EXCLUDED_FEATURES,
        "hyperparameters": {
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "learning_rate": LEARNING_RATE,
            "subsample": SUBSAMPLE,
            "colsample_bytree": COLSAMPLE_BYTREE,
            "objective": OBJECTIVE,
            "eval_metric": EVAL_METRIC,
            "random_state": RANDOM_STATE,
        },
        "preprocessing": {
            "numeric": (
                "SimpleImputer(strategy=median)"
            ),
            "categorical": (
                "SimpleImputer(strategy=most_frequent)"
                " -> OneHotEncoder(handle_unknown=ignore)"
            ),
        },
    }