"""
create_database.py  —  Bronze/Silver → SQLite
────────────────────────────────────────────────────────────────────
Converts the Silver master parquet into a queryable SQLite database.

Run:
    python src/data/create_database.py

Creates:
    data/credit_risk.db   ← SQLite database with 3 tables
────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import sqlite3
from pathlib import Path

SILVER_DIR = Path("data/silver")
DB_PATH    = Path("data/credit_risk.db")

def create_database():
    print("=" * 55)
    print("  INDIA CREDIT RISK — SQLite Database Builder")
    print("=" * 55)

    # ── Load silver master ────────────────────────────────────────
    print("\n[1/3] Loading silver master...")
    df = pd.read_parquet(SILVER_DIR / "silver_master.parquet")

    # Fix boolean columns for SQLite
    bool_cols = df.select_dtypes(include=["bool"]).columns
    for col in bool_cols:
        df[col] = df[col].astype(int)

    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # ── Build derived tables ──────────────────────────────────────
    print("\n[2/3] Building tables...")

    # Table 1: borrowers — full feature set
    borrowers = df.copy()
    borrowers.index.name = "borrower_id"
    borrowers = borrowers.reset_index()

    # Table 2: delinquency_summary — aggregated delinquency stats
    delinquency_cols = [c for c in df.columns if any(
        x in c.lower() for x in ["delinq", "dpd", "missed"]
    )]
    delinquency_summary = df[delinquency_cols + ["default_risk"]].copy()
    delinquency_summary = delinquency_summary.reset_index(drop=True)
    delinquency_summary.index.name = "borrower_id"
    delinquency_summary = delinquency_summary.reset_index()

    # Table 3: risk_segments — derived segment labels
    def segment(row):
        if row.get("enq_L6m", 0) >= 4 and row.get("Age_Oldest_TL", 999) < 24:
            return "extreme_risk"
        elif row.get("enq_L6m", 0) >= 3 or row.get("num_times_60p_dpd", 0) >= 1:
            return "high_risk"
        else:
            return "standard_risk"

    risk_df = pd.DataFrame({
        "borrower_id":   range(len(df)),
        "risk_segment":  df.apply(segment, axis=1),
        "default_risk":  df["default_risk"].values,
        "AGE":           df["AGE"].values if "AGE" in df.columns else None,
        "NETMONTHLYINCOME": df["NETMONTHLYINCOME"].values if "NETMONTHLYINCOME" in df.columns else None,
    })

    # ── Write to SQLite ───────────────────────────────────────────
    print("\n[3/3] Writing to SQLite...")
    conn = sqlite3.connect(DB_PATH)

    borrowers.to_sql("borrowers", conn,
                     if_exists="replace", index=False)
    print(f"  ✓ Table 'borrowers'           ({len(borrowers):,} rows)")

    delinquency_summary.to_sql("delinquency_summary", conn,
                                if_exists="replace", index=False)
    print(f"  ✓ Table 'delinquency_summary' ({len(delinquency_summary):,} rows)")

    risk_df.to_sql("risk_segments", conn,
                   if_exists="replace", index=False)
    print(f"  ✓ Table 'risk_segments'       ({len(risk_df):,} rows)")

    # Verify
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
    print(f"\n  Database: {DB_PATH}")
    print(f"  Tables:   {tables['name'].tolist()}")
    conn.close()

    print(f"""
{'='*55}
✓ DATABASE CREATED: {DB_PATH}

  Tables:
  → borrowers           Full borrower feature set
  → delinquency_summary Delinquency signals only
  → risk_segments       Derived segment labels

  Next: python src/analytics/run_queries.py
{'='*55}
    """)

if __name__ == "__main__":
    create_database()