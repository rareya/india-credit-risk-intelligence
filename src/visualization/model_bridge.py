"""
model_bridge.py — single source of truth for the dashboard's model,
predictions, metrics, and SQL access.

Replaces the dashboard's previous direct references to:
  - data/processed/credit_risk_model_v2.pkl   (old 19-feature model)
  - data/gold/exports/ml/model_metrics.json   (stale: AUC 0.898)
  - data/credit_risk.db (SQLite, old schema, id mismatch vs gold)

...with:
  - data/gold/exports/analytics/ml/final/conservative_xgb_v1.joblib
  - data/gold/exports/analytics/ml/final/model_metadata.json
  - data/gold/exports/analytics/ml/final/model_predictions.parquet
  - data/gold/credit_risk.duckdb (fact_credit_risk + model_predictions,
    plus borrowers/risk_segments views for backward-compatible SQL)

Row alignment note: silver_master.parquet (PROSPECTID 1..N, no gaps)
and fact_credit_risk.parquet (borrower_id 1..N, no gaps) were verified
row-aligned by position -- both strictly monotonic 1..51336 in file
order. Safe to concat positionally. Do not assume this holds after
any future re-run of the gold pipeline without re-checking.
"""

from pathlib import Path
import json

import duckdb
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

FINAL_DIR = ROOT / "data" / "gold" / "exports" / "analytics" / "ml" / "final"
MODEL_PATH = FINAL_DIR / "conservative_xgb_v1.joblib"
METADATA_PATH = FINAL_DIR / "model_metadata.json"
PREDICTIONS_PATH = FINAL_DIR / "model_predictions.parquet"

GOLD_FACT_PATH = ROOT / "data" / "gold" / "exports" / "fact_credit_risk.parquet"
DUCKDB_PATH = ROOT / "data" / "gold" / "credit_risk.duckdb"

SQL_QUERIES_DIR = ROOT / "sql" / "queries"
SQL_MODEL_QUERIES_DIR = ROOT / "sql" / "model_queries"
SHAP_IMPORTANCE_PATH = ROOT / "data" / "gold" / "exports" / "analytics" / "ml" / "shap" / "shap_feature_importance.parquet"

OPERATING_THRESHOLD = 0.40  # analytical candidate from threshold_analysis.py -- not business-signed-off


def model_available() -> bool:
    return MODEL_PATH.exists()


def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def load_model_metadata() -> dict:
    """Real, current metrics -- replaces the dashboard's hardcoded
    0.8994/69.1%/71.0% fallback, which matched none of the three
    metrics files that actually exist in this repo."""
    if not METADATA_PATH.exists():
        return {}
    with open(METADATA_PATH) as f:
        return json.load(f)


def load_gold_fact() -> pd.DataFrame:
    if not GOLD_FACT_PATH.exists():
        return None
    return pd.read_parquet(GOLD_FACT_PATH)


def load_predictions() -> pd.DataFrame:
    """Pre-scored predictions -- no live re-scoring needed, no risk of
    the dashboard's feature-engineering code drifting from the actual
    training pipeline. Run score_portfolio.py to regenerate."""
    if not PREDICTIONS_PATH.exists():
        return None
    return pd.read_parquet(PREDICTIONS_PATH)


def load_shap_importance() -> pd.DataFrame:
    """Real SHAP output for conservative_xgb_v1 (from shap_analysis.py),
    remapped to the column names the dashboard's chart code expects."""
    if not SHAP_IMPORTANCE_PATH.exists():
        return None
    df = pd.read_parquet(SHAP_IMPORTANCE_PATH)
    return df.rename(columns={
        "original_feature": "feature",
        "mean_abs_shap": "shap_importance",
    })


SQL_DASHBOARD_DIR = ROOT / "sql" / "dashboard"


def _read_sql(path: Path, **params) -> str:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return text.format(**params) if params else text


def get_portfolio_kpis(con) -> dict:
    """Runs sql/dashboard/portfolio_kpis.sql -- replaces what was
    previously ad-hoc pandas/numpy math on the loaded silver
    dataframe + a raw prediction array."""
    sql = _read_sql(SQL_DASHBOARD_DIR / "portfolio_kpis.sql", threshold=OPERATING_THRESHOLD)
    df = run_sql(con, sql)
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def get_confusion_matrix(con):
    """Runs sql/dashboard/confusion_matrix.sql -- same aggregation
    as sql/model_queries/01_model_vs_actual_confusion.sql, reused
    rather than reimplemented so dashboard and query library can't
    show different numbers for the same thing."""
    sql = _read_sql(SQL_DASHBOARD_DIR / "confusion_matrix.sql")
    df = run_sql(con, sql)
    if df.empty:
        return None
    counts = dict(zip(df["outcome_bucket"], df["borrower_count"]))
    tn = int(counts.get("true_negative", 0))
    fp = int(counts.get("false_positive", 0))
    fn = int(counts.get("false_negative", 0))
    tp = int(counts.get("true_positive", 0))
    return [[tn, fp], [fn, tp]]


def get_roc_curve(con) -> pd.DataFrame:
    """Runs sql/dashboard/roc_curve_sweep.sql -- 101-point ROC curve
    computed live via SQL threshold sweep, no separately-persisted
    (and driftable) roc_curve.parquet."""
    sql = _read_sql(SQL_DASHBOARD_DIR / "roc_curve_sweep.sql")
    return run_sql(con, sql)


def get_threshold_sensitivity(con) -> pd.DataFrame:
    """Runs sql/dashboard/threshold_sensitivity.sql."""
    sql = _read_sql(SQL_DASHBOARD_DIR / "threshold_sensitivity.sql", op_threshold=OPERATING_THRESHOLD)
    return run_sql(con, sql)


def duckdb_available() -> bool:
    return DUCKDB_PATH.exists()


def get_connection():
    """One DuckDB connection with everything registered:
      - fact_credit_risk   (gold layer, 30 conservative features + demographics)
      - model_predictions  (conservative_xgb_v1 scores, all 51,336 borrowers)
      - borrowers          (compatibility view over fact_credit_risk, for the
                             original 10 sql/queries/*.sql which expect this name)
      - risk_segments      (compatibility view, rule-based segment matching
                             the same logic create_database.py used)
    Run score_portfolio.py first if this file doesn't exist yet.
    """
    if not DUCKDB_PATH.exists():
        return None
    con = duckdb.connect(str(DUCKDB_PATH), read_only=False)

    # Compatibility views so the ORIGINAL 10 queries (written against a
    # `borrowers` table with old-style column names) still run unmodified.
    con.execute(f"""
        CREATE OR REPLACE VIEW borrowers AS
        SELECT
            borrower_id,
            default_risk,
            age                    AS AGE,
            monthly_income_inr     AS NETMONTHLYINCOME,
            credit_history_months  AS Age_Oldest_TL,
            recent_enquiries_6m    AS enq_L6m,
            recent_enquiries_12m   AS enq_L12m,
            total_enquiries        AS tot_enq,
            total_loans            AS Total_TL,
            gold_loans             AS Gold_TL,
            home_loans             AS Home_TL,
            num_times_delinquent,
            num_times_60p_dpd,
            missed_payment_ratio,
            active_loan_ratio,
            loan_type_diversity
        FROM fact_credit_risk
    """)

    con.execute("""
        CREATE OR REPLACE VIEW risk_segments AS
        SELECT
            borrower_id,
            default_risk,
            age AS AGE,
            monthly_income_inr AS NETMONTHLYINCOME,
            CASE
                WHEN recent_enquiries_6m >= 4 AND credit_history_months < 24 THEN 'extreme_risk'
                WHEN recent_enquiries_6m >= 3 OR num_times_60p_dpd >= 1       THEN 'high_risk'
                ELSE 'standard_risk'
            END AS risk_segment
        FROM fact_credit_risk
    """)

    return con


def run_sql(con, query: str) -> pd.DataFrame:
    if con is None:
        return pd.DataFrame()
    try:
        return con.execute(query).fetchdf()
    except Exception:
        return pd.DataFrame()


def load_all_query_files() -> dict:
    """Reads every .sql file from both sql/queries/ (original 10,
    descriptive/historical) and sql/model_queries/ (new 10, model-layer)
    so the dashboard's SQL Query Library panel can show all 20 without
    the queries being hand-copied into Python dicts (the original
    dashboard hardcoded each query's SQL text inline -- this reads the
    actual files instead, so the panel can never drift from the repo's
    real .sql files again)."""
    catalogue = {}
    for label, directory, layer in [
        ("Historical / Descriptive", SQL_QUERIES_DIR, "historical"),
        ("Model Predictions", SQL_MODEL_QUERIES_DIR, "model"),
    ]:
        if not directory.exists():
            continue
        for f in sorted(directory.glob("*.sql")):
            with open(f, encoding="utf-8") as fh:
                text = fh.read()
            # pull the header comment block for business_q / output lines
            lines = text.splitlines()
            business_q, output_desc = "", ""
            for line in lines:
                if "Business Question:" in line:
                    business_q = line.split("Business Question:")[-1].strip(" -")
                elif line.strip().startswith("--") and "Output:" in line:
                    output_desc = line.split("Output:")[-1].strip(" -")
            catalogue[f.stem] = {
                "layer": layer,
                "layer_label": label,
                "file": f.name,
                "business_q": business_q,
                "output": output_desc,
                "sql": text,
            }
    return catalogue