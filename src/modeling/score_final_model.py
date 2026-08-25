"""
score_final_model.py

Score the FINAL frozen Conservative XGBoost artifact.

This script does NOT:
    - select a model
    - train a model
    - tune a model
    - evaluate generalization performance

The final model has already been selected and built as:

    conservative_xgb_v1

This script:

    Gold fact table
        ↓
    validate frozen model configuration
        ↓
    load final .joblib artifact
        ↓
    validate exact 30-feature schema
        ↓
    generate borrower-level risk probabilities
        ↓
    assign risk bands / decisions
        ↓
    save canonical prediction dataset

IMPORTANT
---------
The predictions generated here are IN-SAMPLE predictions because the
final model was refit on 100% of the Gold dataset.

They MUST NOT be used as unbiased generalization metrics.

These predictions are intended for downstream:
    - SQL analytics
    - Streamlit dashboard
    - Power BI
    - portfolio monitoring
    - risk segmentation
    - policy analysis

The honest generalization metrics remain those produced from the
held-out evaluation / model-selection pipeline.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from final_model_config import (
    MODEL_NAME,
    MODEL_VERSION,
    FEATURE_SET_NAME,
    FEATURE_SET_STATUS,
    FEATURES,
    TARGET_COLUMN,
    validate_model_configuration,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GOLD_FACT_TABLE = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "exports"
    / "fact_credit_risk.parquet"
)

FINAL_MODEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "exports"
    / "analytics"
    / "ml"
    / "final_model"
)

MODEL_PATH = (
    FINAL_MODEL_DIR
    / "conservative_xgb_v1.joblib"
)

METADATA_PATH = (
    FINAL_MODEL_DIR
    / "conservative_xgb_v1_metadata.json"
)

PREDICTION_DIR = (
    PROJECT_ROOT
    / "data"
    / "model"
    / "predictions"
)

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PREDICTIONS_PATH = (
    PREDICTION_DIR
    / "predictions.parquet"
)

PREDICTION_METADATA_PATH = (
    PREDICTION_DIR
    / "prediction_metadata.json"
)


# ============================================================
# EXPECTED DATASET
# ============================================================

EXPECTED_ROWS = 51_336
EXPECTED_FEATURE_COUNT = 30


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path: Path) -> str:
    """Calculate SHA-256 checksum of a file."""

    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# LOAD GOLD DATA
# ============================================================

def load_gold_data() -> pd.DataFrame:
    """Load and validate the canonical Gold fact table."""

    print("\nLoading Gold fact table...")

    if not GOLD_FACT_TABLE.is_file():
        raise FileNotFoundError(
            "Gold fact table not found:\n"
            f"  {GOLD_FACT_TABLE}"
        )

    df = pd.read_parquet(GOLD_FACT_TABLE)

    print(
        f"  ✓ Loaded: "
        f"{len(df):,} rows × {len(df.columns)} columns"
    )

    if df.empty:
        raise ValueError(
            "Gold fact table is empty."
        )

    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            "Unexpected Gold row count.\n"
            f"Expected: {EXPECTED_ROWS:,}\n"
            f"Found:    {len(df):,}\n\n"
            "Refusing to score an unexpected dataset."
        )

    return df


# ============================================================
# LOAD FINAL MODEL
# ============================================================

def load_final_model():
    """Load the authoritative frozen final model."""

    print("\nLoading final model artifact...")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Final model artifact not found:\n"
            f"  {MODEL_PATH}\n\n"
            "Run train_final_model.py first."
        )

    model = joblib.load(MODEL_PATH)

    print(
        "  ✓ Loaded:"
    )
    print(
        f"    {MODEL_PATH.relative_to(PROJECT_ROOT)}"
    )

    return model


# ============================================================
# VALIDATE MODEL METADATA
# ============================================================

def validate_model_metadata() -> dict:
    """Validate metadata belonging to the final model."""

    print("\nValidating final model metadata...")

    if not METADATA_PATH.is_file():
        raise FileNotFoundError(
            "Final model metadata not found:\n"
            f"  {METADATA_PATH}"
        )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as handle:
        metadata = json.load(handle)

    if metadata.get("model_name") != MODEL_NAME:
        raise ValueError(
            "Model name mismatch.\n"
            f"Expected: {MODEL_NAME}\n"
            f"Found:    {metadata.get('model_name')}"
        )

    if metadata.get("model_version") != MODEL_VERSION:
        raise ValueError(
            "Model version mismatch.\n"
            f"Expected: {MODEL_VERSION}\n"
            f"Found:    {metadata.get('model_version')}"
        )

    if metadata.get("feature_set_name") != FEATURE_SET_NAME:
        raise ValueError(
            "Feature-set mismatch.\n"
            f"Expected: {FEATURE_SET_NAME}\n"
            f"Found:    {metadata.get('feature_set_name')}"
        )

    print("  ✓ Model metadata validated")

    return metadata


# ============================================================
# VALIDATE FEATURE SCHEMA
# ============================================================

def validate_feature_schema(df: pd.DataFrame) -> None:
    """
    Validate that the Gold table contains exactly the frozen
    feature set required by the final model.
    """

    print("\nValidating frozen feature schema...")

    if len(FEATURES) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            "Unexpected frozen feature count.\n"
            f"Expected: {EXPECTED_FEATURE_COUNT}\n"
            f"Found:    {len(FEATURES)}"
        )

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing frozen model features:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing_features
            )
        )

    print(
        f"  ✓ Frozen feature set validated: "
        f"{len(FEATURES)} features"
    )


# ============================================================
# VALIDATE BORROWER IDENTIFIER
# ============================================================

def validate_borrower_id(df: pd.DataFrame) -> None:
    """Validate borrower identifier required for predictions."""

    print("\nValidating borrower identifiers...")

    if "borrower_id" not in df.columns:
        raise ValueError(
            "Gold fact table does not contain "
            "'borrower_id'."
        )

    if df["borrower_id"].isna().any():
        raise ValueError(
            "borrower_id contains missing values."
        )

    if df["borrower_id"].duplicated().any():
        duplicate_count = int(
            df["borrower_id"].duplicated().sum()
        )

        raise ValueError(
            "Duplicate borrower_id values detected.\n"
            f"Duplicates: {duplicate_count}"
        )

    print(
        f"  ✓ {df['borrower_id'].nunique():,} "
        "unique borrowers"
    )


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

def generate_predictions(
    model,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate borrower-level probability predictions.

    The model is already trained. No fitting occurs here.
    """

    print("\nGenerating final model predictions...")

    X = df[FEATURES].copy()

    probabilities = model.predict_proba(X)[:, 1]

    if len(probabilities) != len(df):
        raise ValueError(
            "Prediction count does not match "
            "Gold row count."
        )

    if pd.Series(probabilities).isna().any():
        raise ValueError(
            "Model generated missing probabilities."
        )

    predictions = pd.DataFrame(
        {
            "borrower_id": df["borrower_id"].values,
            "risk_probability": probabilities,
        }
    )

    return predictions


# ============================================================
# ASSIGN RISK BAND
# ============================================================

def assign_risk_band(
    probability: pd.Series,
) -> pd.Series:
    """
    Assign business-facing risk bands.

    IMPORTANT:
    These bands are presentation / segmentation labels.
    They are not model evaluation metrics.
    """

    return pd.cut(
        probability,
        bins=[
            -float("inf"),
            0.20,
            0.50,
            float("inf"),
        ],
        labels=[
            "Low",
            "Medium",
            "High",
        ],
        right=False,
    ).astype(str)


# ============================================================
# ASSIGN DECISION
# ============================================================

def assign_risk_decision(
    probability: pd.Series,
) -> pd.Series:
    """
    Assign a simple downstream decision using the approved
    operational threshold.

    IMPORTANT:
    The threshold must match the frozen model's approved
    decision policy.
    """

    threshold = 0.50

    return pd.Series(
        [
            "Review"
            if value >= threshold
            else "Approve"
            for value in probability
        ],
        index=probability.index,
    )


# ============================================================
# BUILD PREDICTION DATASET
# ============================================================

def build_prediction_dataset(
    df: pd.DataFrame,
    model,
    model_metadata: dict,
) -> pd.DataFrame:

    predictions = generate_predictions(
        model,
        df,
    )

    predictions["risk_band"] = assign_risk_band(
        predictions["risk_probability"]
    )

    predictions["risk_decision"] = assign_risk_decision(
        predictions["risk_probability"]
    )

    predictions["model_name"] = MODEL_NAME
    predictions["model_version"] = MODEL_VERSION
    predictions["feature_set_name"] = FEATURE_SET_NAME
    predictions["feature_set_status"] = FEATURE_SET_STATUS

    predictions["prediction_timestamp_utc"] = (
        datetime.now(timezone.utc).isoformat()
    )

    return predictions


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("  INDIA CREDIT RISK — FINAL MODEL SCORING")
    print("=" * 70)

    print(
        f"\nModel:        {MODEL_NAME}"
    )

    print(
        f"Version:      {MODEL_VERSION}"
    )

    print(
        f"Feature set:  {FEATURE_SET_NAME}"
    )

    print(
        f"Features:     {len(FEATURES)}"
    )

    # --------------------------------------------------------
    # Load Gold
    # --------------------------------------------------------

    df = load_gold_data()

    # --------------------------------------------------------
    # Validate model configuration
    # --------------------------------------------------------

    print(
        "\nValidating frozen model configuration..."
    )

    validate_model_configuration(df)

    print(
        "  ✓ Frozen configuration validated"
    )

    # --------------------------------------------------------
    # Validate schema
    # --------------------------------------------------------

    validate_feature_schema(df)

    validate_borrower_id(df)

    # --------------------------------------------------------
    # Load final model
    # --------------------------------------------------------

    model = load_final_model()

    model_metadata = validate_model_metadata()

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    predictions = build_prediction_dataset(
        df,
        model,
        model_metadata,
    )

    # --------------------------------------------------------
    # Validate output
    # --------------------------------------------------------

    if len(predictions) != EXPECTED_ROWS:
        raise ValueError(
            "Prediction dataset has unexpected row count.\n"
            f"Expected: {EXPECTED_ROWS:,}\n"
            f"Found:    {len(predictions):,}"
        )

    expected_columns = {
        "borrower_id",
        "risk_probability",
        "risk_band",
        "risk_decision",
        "model_name",
        "model_version",
        "feature_set_name",
        "feature_set_status",
        "prediction_timestamp_utc",
    }

    missing_columns = (
        expected_columns
        - set(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Prediction output is missing columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in sorted(missing_columns)
            )
        )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    print("\nSaving canonical prediction dataset...")

    predictions.to_parquet(
        PREDICTIONS_PATH,
        index=False,
    )

    print(
        "  ✓ Saved:"
    )

    print(
        f"    {PREDICTIONS_PATH.relative_to(PROJECT_ROOT)}"
    )

    # --------------------------------------------------------
    # Prediction metadata
    # --------------------------------------------------------

    prediction_metadata = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "feature_set_name": FEATURE_SET_NAME,
        "feature_count": len(FEATURES),
        "gold_data_path": str(
            GOLD_FACT_TABLE.relative_to(PROJECT_ROOT)
        ),
        "gold_rows_scored": len(df),
        "prediction_rows": len(predictions),
        "prediction_file": str(
            PREDICTIONS_PATH.relative_to(PROJECT_ROOT)
        ),
        "prediction_file_sha256": sha256_file(
            PREDICTIONS_PATH
        ),
        "scoring_scope": (
            "100% of Gold dataset"
        ),
        "prediction_type": (
            "in-sample predictions from final "
            "model refit on full Gold dataset"
        ),
        "generalization_warning": (
            "These predictions must not be used "
            "as unbiased generalization metrics. "
            "Use held-out evaluation artifacts "
            "for model performance claims."
        ),
        "trained_model_artifact": str(
            MODEL_PATH.relative_to(PROJECT_ROOT)
        ),
        "trained_model_sha256": sha256_file(
            MODEL_PATH
        ),
        "scored_at_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
    }

    with open(
        PREDICTION_METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            prediction_metadata,
            handle,
            indent=2,
        )

    print(
        "  ✓ Prediction metadata saved:"
    )

    print(
        f"    {PREDICTION_METADATA_PATH.relative_to(PROJECT_ROOT)}"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("✓ FINAL MODEL SCORING COMPLETE")
    print("=" * 70)

    print("\nPrediction summary:")

    print(
        f"  Borrowers scored: "
        f"{len(predictions):,}"
    )

    print(
        f"  Mean predicted risk: "
        f"{predictions['risk_probability'].mean():.4f}"
    )

    print(
        f"  Median predicted risk: "
        f"{predictions['risk_probability'].median():.4f}"
    )

    print("\nRisk bands:")

    print(
        predictions["risk_band"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nDecision distribution:")

    print(
        predictions["risk_decision"]
        .value_counts()
        .to_string()
    )

    print("\nArtifacts:")

    print(
        f"  → {PREDICTIONS_PATH}"
    )

    print(
        f"  → {PREDICTION_METADATA_PATH}"
    )

    print("\nIMPORTANT:")

    print(
        "  These are final-model scoring outputs."
    )

    print(
        "  They are NOT unbiased model evaluation metrics."
    )

    print(
        "  Use held-out evaluation artifacts for performance claims."
    )


if __name__ == "__main__":
    main()