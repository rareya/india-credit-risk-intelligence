"""
final_model_config.py

Canonical configuration for the SELECTED production candidate model.

Selected model:
    Conservative XGBoost

This file is the single source of truth for:
    - feature registry
    - XGBoost parameters
    - preprocessing
    - random state

Downstream validation scripts such as:
    - cross_validate.py
    - calibrate_model.py
    - shap_analysis.py
    - subgroup_robustness.py

must use this configuration so that they evaluate the SAME model.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from feature_registry import (
    TARGET,
    CONSERVATIVE_FEATURES,
    EXCLUDED_FEATURES,
    validate_registry,
)


# ============================================================
# PATHS
# ============================================================

GOLD_DIR = Path("data/gold/exports")

FACT_FILE = (
    GOLD_DIR /
    "fact_credit_risk.parquet"
)


# ============================================================
# MODEL IDENTITY
# ============================================================

MODEL_NAME = "Conservative XGBoost"

MODEL_VERSION = "conservative_xgb_v1"

RANDOM_STATE = 42

TARGET_COLUMN = TARGET

FEATURES = list(CONSERVATIVE_FEATURES)


# ============================================================
# XGBOOST PARAMETERS
# ============================================================

XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 2.0,
    "reg_alpha": 0.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ============================================================
# VALIDATION
# ============================================================

def validate_final_model_registry(df):
    """
    Validate that the Gold table contains exactly the
    governed Conservative feature set.
    """

    validate_registry(df.columns)

    forbidden_present = (
        set(FEATURES)
        & set(EXCLUDED_FEATURES)
    )

    if forbidden_present:
        raise ValueError(
            "FINAL MODEL LEAKAGE GUARD FAILED.\n"
            "Forbidden features detected:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in sorted(forbidden_present)
            )
        )

    missing = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            "Final Conservative model features missing:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing
            )
        )

    return True


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor(X):
    """
    Exactly matches the preprocessing used by run_ml_model.py.
    """

    numeric_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        ),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        ),
    ])

    return ColumnTransformer(
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


# ============================================================
# MODEL
# ============================================================

def build_xgb_model(X):
    """
    Build the exact selected Conservative XGBoost pipeline.
    """

    preprocessor = build_preprocessor(X)

    model = XGBClassifier(
        **XGB_PARAMS
    )

    return Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        ),
    ])


# ============================================================
# DATA LOADING
# ============================================================

def load_gold_data():
    """
    Load and validate the Gold fact table.
    """

    if not FACT_FILE.exists():
        raise FileNotFoundError(
            f"Missing Gold fact table:\n{FACT_FILE}"
        )

    df = pd.read_parquet(
        FACT_FILE
    )

    validate_final_model_registry(df)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target '{TARGET_COLUMN}' not found."
        )

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            "Target contains missing values."
        )

    unique_target = set(
        df[TARGET_COLUMN]
        .dropna()
        .unique()
    )

    if not unique_target.issubset({0, 1}):
        raise ValueError(
            f"Target must be binary. "
            f"Found: {unique_target}"
        )

    return df


# ============================================================
# METADATA
# ============================================================

def final_model_metadata():
    """
    Return metadata describing the frozen model.
    """

    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "target": TARGET_COLUMN,
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "random_state": RANDOM_STATE,
        "xgb_params": XGB_PARAMS,
        "feature_governance": "conservative",
    }