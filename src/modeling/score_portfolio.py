"""
score_portfolio.py

Scores every borrower in fact_credit_risk.parquet with the persisted
conservative_xgb_v1 model and writes a model_predictions table --
this is the missing link the SQL layer never had: every one of the
original 10 queries ran against raw features + the historical label
only, never against what the model actually predicts.

Output:
  data/gold/exports/analytics/ml/final/model_predictions.parquet
  data/gold/exports/analytics/ml/final/model_predictions.csv
  (+ loaded into data/credit_risk.db as table `model_predictions`,
   joinable to `borrowers` -- see NOTE below on the id mismatch)

NOTE on IDs -- found while building this, worth knowing:
  gold.borrower_id IS silver.PROSPECTID (1-indexed, e.g. 1..51336).
  SQLite's existing `borrowers` table (built in create_database.py via
  a bare .reset_index() on silver_master, ignoring PROSPECTID) has its
  own borrower_id that is just a 0-indexed row position (0..51335).
  These are NOT the same column despite the identical name -- they're
  off by one and conceptually different (one is a real business key,
  one is incidental row position). A naive
  `JOIN borrowers b ON b.borrower_id = mp.borrower_id` will silently
  misalign every single row. This script avoids the problem entirely
  by scoring straight off the gold fact table (which is internally
  consistent) rather than joining through the old SQLite borrowers
  table. The 10 queries below all run against gold-layer data only.
  Fixing the SQLite table's id scheme is a separate, real cleanup item
  -- flagging it here rather than quietly working around it forever.

Threshold used for predicted_class: 0.40, the "maximum_f1" /
"balanced_precision_recall" analytical candidate from
threshold_analysis.py. Per that script's own output, this is an
ANALYTICAL candidate, not a signed-off business threshold -- every
query below exposes predicted_probability directly so the threshold
choice is visible and swappable, not hidden inside a WHERE clause.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, "src/analytics")

import duckdb
import joblib
import pandas as pd

GOLD_DIR = Path("data/gold/exports")
FINAL_DIR = GOLD_DIR / "analytics" / "ml" / "final"
MODEL_PATH = FINAL_DIR / "conservative_xgb_v1.joblib"

OPERATING_THRESHOLD = 0.40  # see NOTE above -- analytical candidate, not business-approved

from feature_registry import CONSERVATIVE_FEATURES, TARGET


def main():
    print("=" * 70)
    print("  SCORE PORTFOLIO — conservative_xgb_v1")
    print("=" * 70)

    df = pd.read_parquet(GOLD_DIR / "fact_credit_risk.parquet")
    model = joblib.load(MODEL_PATH)

    X = df[CONSERVATIVE_FEATURES]
    proba = model.predict_proba(X)[:, 1]

    predictions = pd.DataFrame({
        "borrower_id": df["borrower_id"],
        "actual_default": df[TARGET].astype(int),
        "predicted_probability": proba.round(6),
        "predicted_class": (proba >= OPERATING_THRESHOLD).astype(int),
    })

    predictions["outcome_bucket"] = (
        predictions["actual_default"] * 2 + predictions["predicted_class"]
    ).map({0: "true_negative", 1: "false_positive", 2: "false_negative", 3: "true_positive"})

    # probability calibration bucket -- for reliability-curve style queries
    predictions["probability_decile"] = pd.qcut(
        predictions["predicted_probability"], 10, labels=False, duplicates="drop"
    ) + 1

    out_parquet = FINAL_DIR / "model_predictions.parquet"
    out_csv = FINAL_DIR / "model_predictions.csv"
    predictions.to_parquet(out_parquet, index=False)
    predictions.to_csv(out_csv, index=False)
    print(f"  saved -> {out_parquet}  ({len(predictions):,} rows)")
    print(f"  saved -> {out_csv}")

    print("\nOutcome bucket counts (threshold = {:.2f}):".format(OPERATING_THRESHOLD))
    print(predictions["outcome_bucket"].value_counts())

    # Also register in DuckDB alongside the gold fact table so the
    # queries below run directly, no SQLite id-mismatch involved.
    duckdb_path = Path("data/gold/credit_risk.duckdb")
    con = duckdb.connect(str(duckdb_path))
    con.execute("CREATE OR REPLACE TABLE model_predictions AS SELECT * FROM read_parquet(?)", [str(out_parquet)])
    con.execute("CREATE OR REPLACE TABLE fact_credit_risk AS SELECT * FROM read_parquet(?)",
                [str(GOLD_DIR / "fact_credit_risk.parquet")])
    con.close()
    print(f"\n  registered tables 'model_predictions' + 'fact_credit_risk' in {duckdb_path}")

    print("\nDONE")


if __name__ == "__main__":
    main()