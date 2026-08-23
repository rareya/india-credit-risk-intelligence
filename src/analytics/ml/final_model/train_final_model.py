from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[4]

GOLD_DIR = ROOT / "data" / "gold" / "exports"
FACT_FILE = GOLD_DIR / "fact_credit_risk.parquet"

FINAL_DIR = ROOT / "src" / "analytics" / "ml" / "final_model"

MODEL_FILE = FINAL_DIR / "conservative_xgb_v1.joblib"
METADATA_FILE = FINAL_DIR / "model_metadata.json"

TARGET = "default_risk"
RANDOM_STATE = 42


# ============================================================
# FROZEN CONSERVATIVE V1 FEATURE SET
# ============================================================

FEATURES = [
    "delinquency_score",
    "num_times_delinquent",
    "total_missed_payments",
    "monthly_income_inr",
    "income_tier",
    "age",
    "gender",
    "marital_status",
    "education",
    "total_loans",
    "active_loans",
    "closed_loans",
    "gold_loans",
    "home_loans",
    "personal_loans",
    "credit_card_loans",
    "auto_loans",
    "secured_loans",
    "unsecured_loans",
    "loan_type_diversity",
    "active_loan_ratio",
    "missed_payment_ratio",
    "has_home_loan",
    "has_gold_loan",
    "has_personal_loan",
    "has_credit_card",
    "credit_history_months",
    "recent_enquiries_6m",
    "recent_enquiries_12m",
    "total_enquiries",
    "num_times_60p_dpd",
]


# ============================================================
# EXACT PREPROCESSING
# ============================================================

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
            OneHotEncoder(handle_unknown="ignore")
        ),
    ])

    return ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        ),
    ])


# ============================================================
# EXACT CONSERVATIVE XGBOOST
# ============================================================

def build_model(X):

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


# ============================================================
# MAIN
# ============================================================

def main():

    FINAL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("INDIA CREDIT RISK — FINAL MODEL BUILD")
    print("=" * 70)

    print("\nLoading Gold fact table...")

    if not FACT_FILE.exists():
        raise FileNotFoundError(
            f"Gold fact table not found:\n{FACT_FILE}"
        )

    df = pd.read_parquet(FACT_FILE)

    print(f"✓ Loaded: {len(df):,} rows")

    # --------------------------------------------------------
    # VALIDATE FEATURES
    # --------------------------------------------------------

    print("\nValidating frozen feature set...")

    missing = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing conservative features: {missing}"
        )

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found."
        )

    print(f"✓ Features validated: {len(FEATURES)}")
    print("✓ Target validated")

    # --------------------------------------------------------
    # LEAKAGE CHECK
    # --------------------------------------------------------

    forbidden = {
        "borrower_id",
        "default_risk",
        "risk_band",
        "risk_grade",
        "risk_grade_numeric",
    }

    leakage = sorted(
        set(FEATURES) & forbidden
    )

    if leakage:
        raise ValueError(
            f"LEAKAGE BLOCKED: {leakage}"
        )

    print("✓ Leakage check passed")

    # --------------------------------------------------------
    # BUILD X / y
    # --------------------------------------------------------

    X = df[FEATURES].copy()

    y = df[TARGET].astype(int)

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(
            "Target must contain only 0 and 1."
        )

    print(f"✓ Training rows: {len(X):,}")

    # --------------------------------------------------------
    # BUILD EXACT SELECTED MODEL
    # --------------------------------------------------------

    print("\nBuilding Conservative XGBoost...")

    final_model = build_model(X)

    # --------------------------------------------------------
    # FINAL FIT
    # --------------------------------------------------------

    print("Training final model...")

    final_model.fit(
        X,
        y
    )

    print("✓ Final model trained")

    # --------------------------------------------------------
    # SAVE JOBLIB
    # --------------------------------------------------------

    print("\nSaving final artifact...")

    joblib.dump(
        final_model,
        MODEL_FILE,
        compress=3
    )

    print(
        f"✓ Saved:\n  {MODEL_FILE}"
    )

    # --------------------------------------------------------
    # RELOAD TEST
    # --------------------------------------------------------

    print("\nTesting saved artifact...")

    loaded_model = joblib.load(
        MODEL_FILE
    )

    test_predictions = loaded_model.predict_proba(
        X.head(5)
    )[:, 1]

    if not np.isfinite(
        test_predictions
    ).all():

        raise RuntimeError(
            "Artifact returned invalid probabilities."
        )

    print("✓ Artifact reload successful")
    print("✓ Prediction smoke test passed")

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    metadata = {

        "model_name":
            "Conservative XGBoost",

        "model_version":
            "conservative_xgb_v1",

        "status":
            "FINAL",

        "target":
            TARGET,

        "feature_set":
            "conservative_v1",

        "n_features":
            len(FEATURES),

        "features":
            FEATURES,

        "training_rows":
            int(len(df)),

        "random_state":
            RANDOM_STATE,

        "calibration":
            "Base Conservative XGBoost",

        "xgboost_parameters": {

            "n_estimators":
                400,

            "max_depth":
                4,

            "learning_rate":
                0.05,

            "subsample":
                0.8,

            "colsample_bytree":
                0.8,

            "min_child_weight":
                5,

            "reg_lambda":
                2.0,

            "objective":
                "binary:logistic",

            "eval_metric":
                "logloss",

            "n_jobs":
                -1,

            "random_state":
                RANDOM_STATE,
        },

        "preprocessing": {

            "numeric":
                [
                    "median imputation",
                    "StandardScaler"
                ],

            "categorical":
                [
                    "most_frequent imputation",
                    "OneHotEncoder(handle_unknown='ignore')"
                ]
        },

        "validation": {

            "cross_validation": {

                "folds":
                    5,

                "roc_auc_mean":
                    0.8962,

                "roc_auc_std":
                    0.0042,

                "pr_auc_mean":
                    0.7728,

                "pr_auc_std":
                    0.0053,

                "brier_mean":
                    0.1077,

                "brier_std":
                    0.0016
            },

            "test_reference": {

                "train_rows":
                    41068,

                "test_rows":
                    10268,

                "roc_auc":
                    0.8976,

                "pr_auc":
                    0.7781,

                "brier":
                    0.1070
            }
        },

        "governance": {

            "feature_leakage_audit":
                "PASS",

            "feature_semantics_audit":
                "PASS",

            "prediction_time_provenance":
                "PASS",

            "model_selection":
                "Conservative XGBoost",

            "expanded_xgboost":
                "REJECTED_FOR_LEAKAGE"
        },

        "threshold": {

            "status":
                "NOT_APPROVED",

            "analytical_candidates": {

                "maximum_f1":
                    0.40,

                "minimum_expected_loss_proxy":
                    0.15,

                "high_recall":
                    0.25,

                "high_precision":
                    0.45,

                "balanced_precision_recall":
                    0.40
            }
        },

        "artifact": {

            "filename":
                "conservative_xgb_v1.joblib",

            "type":
                "sklearn Pipeline",

            "contains":
                "preprocessing + fitted Conservative XGBoost",

            "training_script":
                "train_final_model.py"
        }
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    print(
        f"\n✓ Metadata saved:\n  {METADATA_FILE}"
    )

    print("\n" + "=" * 70)
    print("FINAL MODEL BUILD COMPLETE")
    print("=" * 70)

    print("\nGenerated:")

    print(
        f"  ✓ {MODEL_FILE}"
    )

    print(
        f"  ✓ {METADATA_FILE}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "  Operating threshold remains NOT APPROVED."
    )

    print(
        "  Conservative XGBoost remains the selected model."
    )


if __name__ == "__main__":
    main()