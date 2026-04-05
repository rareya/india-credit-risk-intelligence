"""
export_powerbi_dataset.py
────────────────────────────────────────────────────────────────────
Creates a Power BI-ready scored borrower dataset from the project's
silver analytical layer + trained ML model.

Output:
    data/powerbi/credit_risk_powerbi_input.csv

What it does:
1. Loads silver dataset from data/silver/silver_master.parquet
2. Loads trained model from data/processed/credit_risk_model_v2.pkl
   (fallback: credit_risk_model.pkl)
3. Aligns features with model expectations
4. Scores predicted probability (PD)
5. Adds business-friendly columns for Power BI:
   - risk_band
   - approved_flag / rejected_flag
   - ead_proxy
   - lgd_assumption
   - expected_loss
   - watchlist_flag
   - priority_score
   - age_band
   - income_band
   - enquiry_band
   - delinquency_band
6. Exports a flat CSV for Power BI

Run:
    python src/data/export_powerbi_dataset.py
"""

from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import joblib

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]

SILVER_PATH = ROOT / "data" / "silver" / "silver_master.parquet"
PROCESSED_DIR = ROOT / "data" / "processed"
POWERBI_DIR = ROOT / "data" / "powerbi"

MODEL_CANDIDATES = [
    PROCESSED_DIR / "credit_risk_model_v2.pkl",
    PROCESSED_DIR / "credit_risk_model.pkl",
]

OUTPUT_PATH = POWERBI_DIR / "credit_risk_powerbi_input.csv"

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def load_silver() -> pd.DataFrame:
    if not SILVER_PATH.exists():
        raise FileNotFoundError(f"Silver dataset not found: {SILVER_PATH}")
    df = pd.read_parquet(SILVER_PATH)
    print(f"✓ Loaded silver dataset: {SILVER_PATH}")
    print(f"  Shape: {df.shape}")
    return df


def load_model():
    for model_path in MODEL_CANDIDATES:
        if model_path.exists():
            model = joblib.load(model_path)
            print(f"✓ Loaded model: {model_path}")
            return model, model_path
    raise FileNotFoundError(
        "No model found. Expected one of:\n" +
        "\n".join(str(p) for p in MODEL_CANDIDATES)
    )


def safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def ensure_columns(df: pd.DataFrame, required_cols: list[str]) -> pd.DataFrame:
    """
    Ensures all model-required columns exist.
    Missing columns are created with 0.0 so scoring doesn't crash.
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"⚠ Missing columns added as 0.0: {missing}")
        for col in missing:
            df[col] = 0.0
    return df


def get_model_feature_names(model) -> list[str]:
    """
    Tries to extract feature names from sklearn-style models / pipelines.
    """
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    # Pipeline support: final estimator or preprocessor may expose feature_names_in_
    if hasattr(model, "named_steps"):
        # Try last step first
        for _, step in reversed(model.named_steps.items()):
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    raise ValueError(
        "Could not infer model feature names automatically. "
        "Model must expose feature_names_in_."
    )


def score_predictions(df: pd.DataFrame, model) -> tuple[pd.DataFrame, list[str]]:
    feature_cols = get_model_feature_names(model)

    # Ensure all required columns exist
    df = ensure_columns(df, feature_cols)

    # Make numeric where possible
    df = safe_numeric(df, feature_cols)

    # Fill nulls conservatively
    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    # Predict probability
    if hasattr(model, "predict_proba"):
        pred_proba = model.predict_proba(X)[:, 1]
    else:
        # fallback for models without predict_proba
        pred_raw = model.predict(X)
        pred_proba = np.clip(pred_raw.astype(float), 0, 1)

    df["predicted_pd"] = pred_proba
    return df, feature_cols


# ──────────────────────────────────────────────────────────────────────────────
# BUSINESS DERIVED COLUMNS
# ──────────────────────────────────────────────────────────────────────────────
def add_business_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Safe numerics for commonly used columns
    numeric_cols = [
        "AGE",
        "NETMONTHLYINCOME",
        "enq_L6m",
        "enq_L12m",
        "tot_enq",
        "num_times_delinquent",
        "num_times_60p_dpd",
        "delinquency_score",
        "missed_payment_ratio",
        "Total_TL",
        "active_loan_ratio",
        "loan_type_diversity",
        "Age_Oldest_TL",
        "Gold_TL",
        "Home_TL",
        "default_risk",
    ]
    df = safe_numeric(df, numeric_cols)

    # Fill key missing values
    if "NETMONTHLYINCOME" not in df.columns:
        df["NETMONTHLYINCOME"] = 0.0
    if "AGE" not in df.columns:
        df["AGE"] = 0.0

    df["NETMONTHLYINCOME"] = df["NETMONTHLYINCOME"].fillna(0)
    df["AGE"] = df["AGE"].fillna(0)

    # Risk band
    df["risk_band"] = pd.cut(
        df["predicted_pd"],
        bins=[-np.inf, 0.25, 0.50, np.inf],
        labels=["Low Risk", "Medium Risk", "High Risk"]
    )

    # Approval logic
    df["approved_flag"] = (df["predicted_pd"] < 0.50).astype(int)
    df["rejected_flag"] = (df["predicted_pd"] >= 0.50).astype(int)

    # Exposure / loss assumptions
    df["ead_proxy"] = df["NETMONTHLYINCOME"].clip(lower=0) * 12
    df["lgd_assumption"] = 0.45
    df["expected_loss"] = df["predicted_pd"] * df["lgd_assumption"] * df["ead_proxy"]

    # Watchlist logic
    df["high_risk_flag"] = (df["predicted_pd"] >= 0.50).astype(int)
    df["watchlist_flag"] = (df["predicted_pd"] >= 0.65).astype(int)

    # Priority score (good for ranking risky cases)
    df["priority_score"] = df["predicted_pd"] * df["expected_loss"]

    # Optional actual label copy if available
    if "default_risk" in df.columns:
        df["default_actual_label"] = df["default_risk"].fillna(0).astype(int)
    else:
        df["default_actual_label"] = np.nan

    # Segmentation bands
    df["age_band"] = pd.cut(
        df["AGE"],
        bins=[0, 25, 35, 45, 55, np.inf],
        labels=["<=25", "26-35", "36-45", "46-55", "55+"]
    )

    df["income_band"] = pd.cut(
        df["NETMONTHLYINCOME"],
        bins=[-np.inf, 25000, 50000, 100000, 200000, np.inf],
        labels=["<=25k", "25k-50k", "50k-1L", "1L-2L", "2L+"]
    )

    enquiry_source = None
    for c in ["tot_enq", "enq_L12m", "enq_L6m"]:
        if c in df.columns:
            enquiry_source = c
            break

    if enquiry_source is None:
        df["enquiry_value"] = 0
    else:
        df["enquiry_value"] = pd.to_numeric(df[enquiry_source], errors="coerce").fillna(0)

    df["enquiry_band"] = pd.cut(
        df["enquiry_value"],
        bins=[-np.inf, 1, 3, 6, np.inf],
        labels=["0-1", "2-3", "4-6", "7+"]
    )

    delinquency_source = None
    for c in ["num_times_delinquent", "num_times_60p_dpd", "delinquency_score"]:
        if c in df.columns:
            delinquency_source = c
            break

    if delinquency_source is None:
        df["delinquency_value"] = 0
    else:
        df["delinquency_value"] = pd.to_numeric(df[delinquency_source], errors="coerce").fillna(0)

    df["delinquency_band"] = pd.cut(
        df["delinquency_value"],
        bins=[-np.inf, 0, 2, 5, np.inf],
        labels=["0", "1-2", "3-5", "6+"]
    )

    # Rank for top-risk tables in Power BI
    df["risk_rank"] = df["predicted_pd"].rank(method="dense", ascending=False).astype(int)
    df["loss_rank"] = df["expected_loss"].rank(method="dense", ascending=False).astype(int)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# COLUMN ORDERING
# ──────────────────────────────────────────────────────────────────────────────
def reorder_columns(df: pd.DataFrame, model_features: list[str]) -> pd.DataFrame:
    priority_cols = []

    # If you have an ID column in your dataset, these will be kept first if present
    possible_id_cols = [
        "customer_id", "Customer_ID", "CUSTOMER_ID",
        "app_id", "application_id", "loan_id", "ID"
    ]
    for c in possible_id_cols:
        if c in df.columns:
            priority_cols.append(c)

    business_cols = [
        "predicted_pd",
        "risk_band",
        "approved_flag",
        "rejected_flag",
        "ead_proxy",
        "lgd_assumption",
        "expected_loss",
        "high_risk_flag",
        "watchlist_flag",
        "priority_score",
        "default_actual_label",
        "age_band",
        "income_band",
        "enquiry_band",
        "delinquency_band",
        "risk_rank",
        "loss_rank",
    ]

    ordered = []
    for c in priority_cols + business_cols + model_features:
        if c in df.columns and c not in ordered:
            ordered.append(c)

    # Append the rest
    remaining = [c for c in df.columns if c not in ordered]
    final_cols = ordered + remaining

    return df[final_cols]


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("\n=== Exporting Power BI dataset ===\n")

    POWERBI_DIR.mkdir(parents=True, exist_ok=True)

    df = load_silver()
    model, model_path = load_model()

    df, model_features = score_predictions(df, model)
    print(f"✓ Scored predictions using {len(model_features)} model features")

    df = add_business_columns(df)
    print("✓ Added business-friendly Power BI columns")

    df = reorder_columns(df, model_features)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✓ Exported Power BI dataset: {OUTPUT_PATH}")
    print(f"  Final shape: {df.shape}")

    # Small summary
    print("\n=== Quick Summary ===")
    print(f"Total rows: {len(df):,}")
    print(f"Approval rate: {df['approved_flag'].mean():.2%}")
    print(f"Avg PD: {df['predicted_pd'].mean():.4f}")
    print(f"High-risk share: {df['high_risk_flag'].mean():.2%}")
    print(f"Total Expected Loss: {df['expected_loss'].sum():,.2f}")

    print("\nDone. Use this CSV in Power BI:")
    print(f"  {OUTPUT_PATH}")


if __name__ == "__main__":
    main()