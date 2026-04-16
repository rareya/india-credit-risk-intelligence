"""
export_powerbi_dataset.py
────────────────────────────────────────────────────────────────────
Creates a Power BI-ready scored borrower dataset.

FIX APPLIED:
    pd.cut() returns Categorical dtype. NaN categories become blank
    strings in CSV. All pd.cut() calls now include:
        .astype(str).replace("nan", "Unknown")
    This eliminates blank category values in Power BI.

Output: data/powerbi/credit_risk_powerbi_input.csv

Run: python src/data/export_powerbi_dataset.py
────────────────────────────────────────────────────────────────────
"""

from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import joblib

warnings.filterwarnings("ignore")

ROOT          = Path(__file__).resolve().parents[2]
SILVER_PATH   = ROOT / "data" / "silver" / "silver_master.parquet"
PROCESSED_DIR = ROOT / "data" / "processed"
POWERBI_DIR   = ROOT / "data" / "powerbi"

MODEL_CANDIDATES = [
    PROCESSED_DIR / "credit_risk_model_v2.pkl",
    PROCESSED_DIR / "credit_risk_model.pkl",
]

OUTPUT_PATH = POWERBI_DIR / "credit_risk_powerbi_input.csv"


def safe_cut(series: pd.Series, bins, labels, fill_value="Unknown") -> pd.Series:
    """
    pd.cut wrapper that prevents blank/NaN categories in output.
    All out-of-range or NaN values become fill_value instead of blank.
    """
    result = pd.cut(series, bins=bins, labels=labels, right=True)
    return result.astype(str).replace("nan", fill_value).fillna(fill_value)


def load_silver() -> pd.DataFrame:
    if not SILVER_PATH.exists():
        raise FileNotFoundError(f"Silver dataset not found: {SILVER_PATH}")
    df = pd.read_parquet(SILVER_PATH)
    print(f"✓ Loaded silver dataset: {df.shape}")
    return df


def load_model():
    for model_path in MODEL_CANDIDATES:
        if model_path.exists():
            model = joblib.load(model_path)
            print(f"✓ Loaded model: {model_path.name}")
            return model, model_path
    raise FileNotFoundError(
        "No model found. Run: python src/analytics/run_ml_model.py")


def safe_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def ensure_columns(df: pd.DataFrame, required_cols: list) -> pd.DataFrame:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"⚠ Missing columns added as 0.0: {missing}")
        for col in missing:
            df[col] = 0.0
    return df


def get_model_feature_names(model) -> list:
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "named_steps"):
        for _, step in reversed(model.named_steps.items()):
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    raise ValueError("Could not infer model feature names. Model must expose feature_names_in_.")


def score_predictions(df: pd.DataFrame, model) -> tuple:
    feature_cols = get_model_feature_names(model)
    df = ensure_columns(df, feature_cols)
    df = safe_numeric(df, feature_cols)

    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    pred_proba = (model.predict_proba(X)[:, 1]
                  if hasattr(model, "predict_proba")
                  else np.clip(model.predict(X).astype(float), 0, 1))

    df["predicted_pd"] = pred_proba
    return df, feature_cols


def add_business_columns(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "AGE", "NETMONTHLYINCOME", "enq_L6m", "enq_L12m", "tot_enq",
        "num_times_delinquent", "num_times_60p_dpd", "delinquency_score",
        "missed_payment_ratio", "Total_TL", "active_loan_ratio",
        "loan_type_diversity", "Age_Oldest_TL", "Gold_TL", "Home_TL", "default_risk",
    ]
    df = safe_numeric(df, numeric_cols)
    df["NETMONTHLYINCOME"] = df["NETMONTHLYINCOME"].fillna(0) if "NETMONTHLYINCOME" in df.columns else 0.0
    df["AGE"]              = df["AGE"].fillna(0) if "AGE" in df.columns else 0.0

    # ── Risk band — FIX: use safe_cut to prevent blank categories ────────────
    df["risk_band"] = safe_cut(
        df["predicted_pd"],
        bins   = [-np.inf, 0.25, 0.50, np.inf],
        labels = ["Low Risk", "Medium Risk", "High Risk"],
    )

    # Approval flags
    df["approved_flag"] = (df["predicted_pd"] < 0.50).astype(int)
    df["rejected_flag"] = (df["predicted_pd"] >= 0.50).astype(int)

    # Expected Loss
    df["ead_proxy"]        = df["NETMONTHLYINCOME"].clip(lower=0) * 12
    df["lgd_assumption"]   = 0.45
    df["expected_loss"]    = df["predicted_pd"] * df["lgd_assumption"] * df["ead_proxy"]

    # Watchlist
    df["high_risk_flag"]   = (df["predicted_pd"] >= 0.50).astype(int)
    df["watchlist_flag"]   = (df["predicted_pd"] >= 0.65).astype(int)
    df["priority_score"]   = df["predicted_pd"] * df["expected_loss"]

    # Actual label
    df["default_actual_label"] = (
        df["default_risk"].fillna(0).astype(int)
        if "default_risk" in df.columns else np.nan
    )

    # ── Segmentation bands — FIX: all use safe_cut ───────────────────────────
    df["age_band"] = safe_cut(
        df["AGE"],
        bins   = [0, 25, 35, 45, 55, np.inf],
        labels = ["<=25", "26-35", "36-45", "46-55", "55+"],
    )

    df["income_band"] = safe_cut(
        df["NETMONTHLYINCOME"],
        bins   = [-np.inf, 25000, 50000, 100000, 200000, np.inf],
        labels = ["<=25k", "25k-50k", "50k-1L", "1L-2L", "2L+"],
    )

    # Enquiry band
    enq_source = next((c for c in ["tot_enq", "enq_L12m", "enq_L6m"] if c in df.columns), None)
    df["enquiry_value"] = (
        pd.to_numeric(df[enq_source], errors="coerce").fillna(0)
        if enq_source else 0
    )
    df["enquiry_band"] = safe_cut(
        df["enquiry_value"],
        bins   = [-np.inf, 1, 3, 6, np.inf],
        labels = ["0-1", "2-3", "4-6", "7+"],
    )

    # Delinquency band
    delinq_source = next((c for c in ["num_times_delinquent", "num_times_60p_dpd", "delinquency_score"] if c in df.columns), None)
    df["delinquency_value"] = (
        pd.to_numeric(df[delinq_source], errors="coerce").fillna(0)
        if delinq_source else 0
    )
    df["delinquency_band"] = safe_cut(
        df["delinquency_value"],
        bins   = [-np.inf, 0, 2, 5, np.inf],
        labels = ["0", "1-2", "3-5", "6+"],
    )

    # Credit history band — FIX: was missing, added for Power BI
    if "Age_Oldest_TL" in df.columns:
        df["credit_history_band"] = safe_cut(
            df["Age_Oldest_TL"],
            bins   = [-np.inf, 12, 24, 48, 96, np.inf],
            labels = ["<1yr", "1-2yr", "2-4yr", "4-8yr", "8yr+"],
        )

    # Priority rank
    df["risk_rank"] = df["predicted_pd"].rank(method="dense", ascending=False).astype(int)
    df["loss_rank"] = df["expected_loss"].rank(method="dense", ascending=False).astype(int)

    # ── Validation: check for blank categories ────────────────────────────────
    band_cols = ["risk_band", "age_band", "income_band", "enquiry_band", "delinquency_band"]
    print("\n  Category validation (no blanks expected):")
    for col in band_cols:
        if col in df.columns:
            blanks = df[col].isin(["nan", "", "None", "Unknown"]).sum()
            status = "✓" if blanks == 0 else f"⚠ {blanks} blanks"
            print(f"    {col}: {status}  |  values: {sorted(df[col].unique().tolist())}")

    return df


def reorder_columns(df: pd.DataFrame, model_features: list) -> pd.DataFrame:
    id_cols = [c for c in ["customer_id", "Customer_ID", "PROSPECTID", "borrower_id"] if c in df.columns]
    business_cols = [
        "predicted_pd", "risk_band", "approved_flag", "rejected_flag",
        "ead_proxy", "lgd_assumption", "expected_loss",
        "high_risk_flag", "watchlist_flag", "priority_score",
        "default_actual_label", "age_band", "income_band",
        "enquiry_band", "delinquency_band", "credit_history_band",
        "risk_rank", "loss_rank",
    ]
    ordered = []
    for c in id_cols + business_cols + model_features:
        if c in df.columns and c not in ordered:
            ordered.append(c)
    remaining = [c for c in df.columns if c not in ordered]
    return df[ordered + remaining]


def main():
    print("\n" + "="*55)
    print("  INDIA CREDIT RISK — Power BI Export")
    print("="*55 + "\n")

    POWERBI_DIR.mkdir(parents=True, exist_ok=True)

    df = load_silver()
    model, _ = load_model()

    df, model_features = score_predictions(df, model)
    print(f"✓ Scored {len(df):,} borrowers using {len(model_features)} features")

    df = add_business_columns(df)
    df = reorder_columns(df, model_features)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✓ Exported: {OUTPUT_PATH}")
    print(f"  Shape: {df.shape}")

    print("\n  Summary:")
    print(f"  Total rows:       {len(df):,}")
    print(f"  Approval rate:    {df['approved_flag'].mean():.1%}")
    print(f"  Avg PD:           {df['predicted_pd'].mean():.4f}")
    print(f"  High-risk share:  {df['high_risk_flag'].mean():.1%}")
    print(f"  Watchlist count:  {df['watchlist_flag'].sum():,}")
    print(f"  Total Exp. Loss:  ₹{df['expected_loss'].sum():,.0f}")

    print("\n  risk_band distribution:")
    print(df["risk_band"].value_counts().to_string())

    print(f"\n  Power BI file: {OUTPUT_PATH}")
    print("  Import in Power BI: Get Data → Text/CSV → select this file")


if __name__ == "__main__":
    main()