"""
train_final_model.py

Build the FINAL Conservative XGBoost artifact.

This script does NOT select the model.

The model has already been selected and frozen as:

    conservative_xgb_v1

This script:

    Gold data
        ↓
    validate frozen configuration
        ↓
    build exact selected model
        ↓
    fit on 100% of Gold data
        ↓
    save .joblib
        ↓
    save authoritative metadata
        ↓
    save manifest

IMPORTANT
---------
The 80/20 test-set metrics reported elsewhere are the honest
generalization metrics.

This final artifact is refit on all 51,336 rows for downstream
pipeline use.

Do NOT evaluate this final artifact on the same full dataset
and report those results as generalization performance.
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
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    TARGET_DEFINITION,
    REVIEW_FEATURES_INCLUDED,
    EXCLUDED_FEATURES,
    RANDOM_STATE,
    build_xgb_model,
    validate_model_configuration,
    get_model_metadata,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FINAL_MODEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "exports"
    / "analytics"
    / "ml"
    / "final_model"
)

FINAL_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


MODEL_PATH = (
    FINAL_MODEL_DIR
    / "conservative_xgb_v1.joblib"
)

METADATA_PATH = (
    FINAL_MODEL_DIR
    / "conservative_xgb_v1_metadata.json"
)

MANIFEST_PATH = (
    FINAL_MODEL_DIR
    / "conservative_xgb_v1_manifest.json"
)


# ============================================================
# DATA PATH
# ============================================================

# VERIFIED repository Gold fact table used by the ML experiments.
GOLD_FACT_TABLE = (
    PROJECT_ROOT / "data" / "gold" / "exports" / "fact_credit_risk.parquet"
)

def load_gold_data_for_final_model() -> pd.DataFrame:
    """Load the verified Gold fact table used by the ML experiments."""
    if not GOLD_FACT_TABLE.is_file():
        raise FileNotFoundError(
            "Verified Gold fact table not found:\n"
            f"  {GOLD_FACT_TABLE}\n\n"
            "Expected repository path:\n"
            "  data/gold/exports/fact_credit_risk.parquet"
        )

    print("  ✓ Gold fact table found:")
    print(f"    {GOLD_FACT_TABLE.relative_to(PROJECT_ROOT)}")

    df = pd.read_parquet(GOLD_FACT_TABLE)
    if df.empty:
        raise ValueError(f"Gold fact table is empty: {GOLD_FACT_TABLE}")

    expected_rows = 51336
    expected_columns = 40
    if len(df) != expected_rows or len(df.columns) != expected_columns:
        raise ValueError(
            "Unexpected Gold fact table shape.\n"
            f"  Expected: {expected_rows:,} rows × {expected_columns} columns\n"
            f"  Found:    {len(df):,} rows × {len(df.columns)} columns\n\n"
            "Refusing to train the final model on an unexpected dataset."
        )
    return df


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path: Path) -> str:
    """
    Calculate SHA-256 checksum of a file.
    """

    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def validate_target(df):
    """
    Validate target before final fitting.
    """

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"Target '{TARGET_COLUMN}' contains missing values."
        )

    unique_values = set(
        df[TARGET_COLUMN]
        .astype(int)
        .unique()
    )

    if not unique_values.issubset({0, 1}):
        raise ValueError(
            f"Target must be binary {{0,1}}. "
            f"Found: {sorted(unique_values)}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print(
        "  INDIA CREDIT RISK — FINAL MODEL BUILD"
    )
    print("=" * 70)

    print(
        f"\nModel: {MODEL_NAME}"
    )

    print(
        f"Version: {MODEL_VERSION}"
    )

    print(
        f"Feature set: {FEATURE_SET_NAME}"
    )

    print(
        f"Feature status: {FEATURE_SET_STATUS}"
    )

    # --------------------------------------------------------
    # Load Gold
    # --------------------------------------------------------

    print(
        "\nLoading Gold fact table..."
    )

    df = load_gold_data_for_final_model()

    print(
        f"  ✓ Loaded: "
        f"{len(df):,} rows × {len(df.columns)} columns"
    )

    if len(df) != 51336:
        raise ValueError(
            "Unexpected Gold dataset row count. "
            f"Expected 51,336 rows for the selected model build, "
            f"but found {len(df):,}. "
            "Refusing to train the final artifact on an unexpected dataset."
        )

    # --------------------------------------------------------
    # Validate configuration
    # --------------------------------------------------------

    print(
        "\nValidating frozen model configuration..."
    )

    validate_model_configuration(df)

    validate_target(df)

    print(
        "  ✓ Target validated"
    )

    print(
        f"  ✓ Frozen features validated: "
        f"{len(FEATURES)}"
    )

    print(
        "  ✓ Feature governance validated"
    )

    # --------------------------------------------------------
    # Build final matrices
    # --------------------------------------------------------

    X = df[FEATURES].copy()

    y = df[TARGET_COLUMN].astype(int)

    print(
        "\nFinal training dataset:"
    )

    print(
        f"  Rows:       {len(X):,}"
    )

    print(
        f"  Features:   {len(FEATURES)}"
    )

    print(
        f"  Positives:  {int(y.sum()):,}"
    )

    print(
        f"  Positive rate: {y.mean():.6f}"
    )

    # --------------------------------------------------------
    # Build exact selected model
    # --------------------------------------------------------

    print(
        "\nBuilding selected Conservative XGBoost..."
    )

    model = build_xgb_model()

    print(
        "  ✓ Exact frozen model definition loaded"
    )

    # --------------------------------------------------------
    # Final training
    # --------------------------------------------------------

    print(
        "\nTraining final artifact on 100% of Gold data..."
    )

    model.fit(
        X,
        y,
    )

    print(
        "  ✓ Final model trained"
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    print(
        "\nSaving final model..."
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print(
        f"  ✓ {MODEL_PATH}"
    )

    # --------------------------------------------------------
    # Build metadata
    # --------------------------------------------------------

    metadata = get_model_metadata()

    metadata.update(
        {
            "gold_data_path": str(
                GOLD_FACT_TABLE.relative_to(PROJECT_ROOT)
            ),
            "training_rows_full_dataset": int(
                len(df)
            ),
            "train_positive_rate": float(
                y.mean()
            ),
            "training_scope": (
                "100% of Gold dataset"
            ),
            "deployment_note": (
                "This artifact is refit on the full "
                "Gold dataset after model selection "
                "and held-out evaluation. "
                "Its full-dataset predictions must "
                "not be used as an unbiased estimate "
                "of generalization performance."
            ),
            "held_out_evaluation_reference": {
                "split": (
                    "80/20 stratified, random_state=42"
                ),
                "note": (
                    "See model selection / evaluation "
                    "artifacts for honest held-out "
                    "performance."
                ),
            },
            "cross_validation_reference": {
                "mean_roc_auc": 0.8962,
                "std_roc_auc": 0.0042,
                "source": (
                    "cross_validation/cv_summary.parquet"
                ),
            },
            "review_features_included": (
                REVIEW_FEATURES_INCLUDED
            ),
            "excluded_features": (
                EXCLUDED_FEATURES
            ),
            "random_state": RANDOM_STATE,
            "trained_at_utc": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "artifact_path": str(
                MODEL_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
        }
    )

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            metadata,
            handle,
            indent=2,
        )

    print(
        f"  ✓ {METADATA_PATH}"
    )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    model_hash = sha256_file(
        MODEL_PATH
    )

    manifest = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "artifact": MODEL_PATH.name,
        "artifact_sha256": model_hash,
        "metadata_file": METADATA_PATH.name,
        "feature_set_name": FEATURE_SET_NAME,
        "feature_set_status": FEATURE_SET_STATUS,
        "target": TARGET_COLUMN,
        "n_features": len(FEATURES),
        "training_rows": len(df),
        "gold_data_path": metadata["gold_data_path"],
        "trained_at_utc": metadata[
            "trained_at_utc"
        ],
        "status": "FINAL",
    }

    with open(
        MANIFEST_PATH,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            manifest,
            handle,
            indent=2,
        )

    print(
        f"  ✓ {MANIFEST_PATH}"
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "✓ FINAL MODEL BUILD COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nSelected model:"
    )

    print(
        f"  {MODEL_NAME}"
    )

    print(
        f"  Version: {MODEL_VERSION}"
    )

    print(
        f"  Features: {len(FEATURES)}"
    )

    print(
        f"  Training rows: {len(df):,}"
    )

    print(
        "\nGold dataset used:"
    )
    print(
        f"  → {metadata['gold_data_path']}"
    )

    print(
        "\nArtifacts:"
    )

    print(
        f"  → {MODEL_PATH}"
    )

    print(
        f"  → {METADATA_PATH}"
    )

    print(
        f"  → {MANIFEST_PATH}"
    )

    print(
        "\nSHA-256:"
    )

    print(
        f"  {model_hash}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "  This script does NOT select or tune the model."
    )

    print(
        "  It only builds the already-selected "
        "Conservative XGBoost artifact."
    )


if __name__ == "__main__":
    main()