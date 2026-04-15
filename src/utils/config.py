"""
config.py — Central configuration for India Credit Risk Intelligence
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
DATA_DIR    = ROOT / "data"
SILVER_DIR  = DATA_DIR / "silver"
ML_DIR      = DATA_DIR / "gold" / "exports" / "ml"
SQL_DIR     = DATA_DIR / "gold" / "exports" / "sql"
MODEL_DIR   = DATA_DIR / "processed"
DB_PATH     = DATA_DIR / "credit_risk.db"
ASSETS_DIR  = ROOT / "assets"
DOCS_DIR    = ROOT / "docs"
SQL_QUERIES = ROOT / "sql"

# ── Model config ───────────────────────────────────────────────────────────
MODEL_FILES      = ["credit_risk_model_v2.pkl", "credit_risk_model.pkl"]
TARGET           = "default_risk"
DECISION_THRESHOLD = 0.50
SCALE_POS_WEIGHT   = 1.43

# ── Business config ────────────────────────────────────────────────────────
# Expected Loss assumptions (industry standard simplification)
LGD_ASSUMPTION  = 0.45   # Loss Given Default — 45% industry average for unsecured India loans
EAD_PROXY_COL   = "NETMONTHLYINCOME"  # Exposure At Default proxy: 12x monthly income
EAD_MULTIPLIER  = 12

# Risk bands (probability of default thresholds)
RISK_BANDS = {
    "Low Risk":    (0.00, 0.25),
    "Medium Risk": (0.25, 0.50),
    "High Risk":   (0.50, 1.00),
}

# ── Feature groups ─────────────────────────────────────────────────────────
FEATURES = [
    "num_times_delinquent", "num_times_60p_dpd", "delinquency_score",
    "missed_payment_ratio", "Total_TL", "active_loan_ratio",
    "loan_type_diversity", "Age_Oldest_TL", "AGE", "NETMONTHLYINCOME",
    "enq_L6m", "enq_L12m", "tot_enq", "Gold_TL", "Home_TL",
]

FEATURE_GROUPS = {
    "Delinquency Behaviour": ["num_times_delinquent","num_times_60p_dpd","delinquency_score","missed_payment_ratio"],
    "Loan Portfolio":        ["Total_TL","active_loan_ratio","loan_type_diversity","Age_Oldest_TL"],
    "Credit Seeking":        ["enq_L6m","enq_L12m","tot_enq"],
    "Demographics":          ["AGE","NETMONTHLYINCOME"],
    "India-Specific":        ["Gold_TL","Home_TL"],
}

ENGINEERED_FEATURES = [
    "enq_per_credit_year", "delinquency_rate",
    "enq_acceleration", "severe_delinquency_ratio",
]