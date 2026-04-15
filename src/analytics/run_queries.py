"""
run_queries.py  —  SQL Query Executor
────────────────────────────────────────────────────────────────────
Runs all 10 business SQL queries against the SQLite database
and prints formatted results.

    python src/analytics/run_queries.py

Runs: All 10 queries in sql/queries/
Output: Printed results + saved to data/gold/exports/sql/ as CSV
────────────────────────────────────────────────────────────────────
"""

import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH   = Path("data/credit_risk.db")
SQL_DIR   = Path("sql/queries")
OUT_DIR   = Path("data/gold/exports/sql")
OUT_DIR.mkdir(parents=True, exist_ok=True)

QUERY_DESCRIPTIONS = {
    "01_portfolio_health":         "Portfolio KPIs — how risky is our overall loan book?",
    "02_default_by_segment":       "Default rate by risk segment and demographics",
    "03_enquiry_default_correlation": "Does credit-seeking behaviour predict default?",
    "04_delinquency_funnel":       "Repayment journey: active → 30DPD → 60DPD → default",
    "05_credit_history_vs_default": "Does credit history length protect against default?",
    "06_high_risk_identification": "Which borrower profiles trigger automatic review?",
    "07_income_default_analysis":  "Is income a reliable risk predictor vs behaviour?",
    "08_gold_loan_analysis":       "Are gold loan borrowers higher risk? (India-specific)",
    "09_early_intervention_candidates": "Which borrowers need collections outreach NOW?",
    "10_policy_impact_simulation": "Estimated NPA reduction from 3 credit policies",
}


def run_query(conn: sqlite3.Connection, sql_path: Path, query_name: str) -> pd.DataFrame:
    """Execute a single SQL file and return results as DataFrame."""
    with open(sql_path, "r") as f:
        sql = f.read()

    # Strip comments for execution
    lines = [l for l in sql.split("\n") if not l.strip().startswith("--")]
    clean_sql = "\n".join(lines).strip()

    if not clean_sql:
        return pd.DataFrame()

    try:
        df = pd.read_sql_query(clean_sql, conn)
        return df
    except Exception as e:
        print(f"  ✗ Query failed: {e}")
        return pd.DataFrame()


def main():
    print("=" * 65)
    print("  INDIA CREDIT RISK — SQL QUERY LIBRARY")
    print("=" * 65)

    if not DB_PATH.exists():
        print(f"\n  ✗ Database not found: {DB_PATH}")
        print(f"  Run: python src/data/create_database.py")
        return

    conn = sqlite3.connect(DB_PATH)

    # Check available tables
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table'", conn
    )["name"].tolist()
    print(f"\n  Connected: {DB_PATH}")
    print(f"  Tables:    {tables}")

    query_files = sorted(SQL_DIR.glob("*.sql")) if SQL_DIR.exists() else []
    if not query_files:
        # Fallback: try sql/ directly
        query_files = sorted(Path("sql").glob("*.sql"))

    if not query_files:
        print(f"\n  ✗ No SQL files found in {SQL_DIR}")
        return

    print(f"\n  Running {len(query_files)} queries...\n")

    results = {}
    for sql_path in query_files:
        query_name = sql_path.stem
        description = QUERY_DESCRIPTIONS.get(query_name, query_name)

        print(f"{'─'*65}")
        print(f"  {query_name}")
        print(f"  {description}")
        print(f"{'─'*65}")

        df = run_query(conn, sql_path, query_name)

        if df.empty:
            print("  (no results)\n")
            continue

        print(df.to_string(index=False))
        print(f"\n  → {len(df)} rows returned")

        # Save to CSV
        out_path = OUT_DIR / f"{query_name}.csv"
        df.to_csv(out_path, index=False)
        print(f"  → Saved: {out_path}\n")

        results[query_name] = df

    conn.close()

    print("=" * 65)
    print(f"✓ QUERIES COMPLETE — {len(results)}/{len(query_files)} successful")
    print(f"  Results saved to: {OUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()