"""
build_gold.py  —  Gold Layer
────────────────────────────────────────────────────────────────────
Builds a DuckDB star schema from Silver data.
Exports all tables to Parquet for dashboard consumption.

    python src/modeling/build_gold.py

Star Schema:
    dim_borrower     — who borrowed (demographics)
    dim_credit       — credit bureau profile
    dim_loan_type    — what kind of loans they have
    dim_risk         — risk classification
    fact_credit_risk — one row per borrower, all measures
────────────────────────────────────────────────────────────────────
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

SILVER_DIR = Path("data/silver")
GOLD_DIR   = Path("data/gold")
EXPORTS    = GOLD_DIR / "exports"
DB_PATH    = GOLD_DIR / "credit_risk.duckdb"

GOLD_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS.mkdir(parents=True, exist_ok=True)


def build_gold():
    print("=" * 60)
    print("  INDIA CREDIT RISK — GOLD LAYER (DuckDB)")
    print("=" * 60)

    # Load Silver
    print("\n[1/6] Loading Silver master...")
    df = pd.read_parquet(SILVER_DIR / "silver_master.parquet")
    print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")

    # Connect to DuckDB
    con = duckdb.connect(str(DB_PATH))
    con.register("silver_master", df)

    # ── Dimension 1: Borrower Demographics ────────────────────────────
    print("\n[2/6] Building dim_borrower...")
    con.execute("""
        CREATE OR REPLACE TABLE dim_borrower AS
        SELECT
            PROSPECTID                          AS borrower_id,
            AGE                                 AS age,
            GENDER                              AS gender,
            MARITALSTATUS                       AS marital_status,
            EDUCATION                           AS education,
            NETMONTHLYINCOME                    AS monthly_income_inr,
            income_tier,
            Time_With_Curr_Empr                 AS months_with_employer,
            CASE
                WHEN AGE < 25 THEN 'under_25'
                WHEN AGE < 35 THEN '25_to_34'
                WHEN AGE < 45 THEN '35_to_44'
                WHEN AGE < 55 THEN '45_to_54'
                ELSE '55_plus'
            END                                 AS age_band
        FROM silver_master
    """)
    count = con.execute("SELECT COUNT(*) FROM dim_borrower").fetchone()[0]
    print(f"  ✓ dim_borrower: {count:,} rows")

    # ── Dimension 2: Credit Profile ────────────────────────────────────
    print("\n[3/6] Building dim_credit...")
    con.execute("""
        CREATE OR REPLACE TABLE dim_credit AS
        SELECT
            PROSPECTID                          AS borrower_id,
            Credit_Score                        AS cibil_score,
            cibil_band,
            credit_history_months,
            num_times_delinquent                AS total_delinquencies,
            num_times_30p_dpd                   AS times_30dpd,
            num_times_60p_dpd                   AS times_60dpd,
            delinquency_score,
            Tot_Missed_Pmnt                     AS total_missed_payments,
            time_since_recent_payment,
            tot_enq                             AS total_enquiries,
            enq_L6m                             AS enquiries_last_6m,
            enq_L12m                            AS enquiries_last_12m,
            last_prod_enq2                      AS last_product_enquired,
            first_prod_enq2                     AS first_product_enquired
        FROM silver_master
    """)
    count = con.execute("SELECT COUNT(*) FROM dim_credit").fetchone()[0]
    print(f"  ✓ dim_credit: {count:,} rows")

    # ── Dimension 3: Loan Portfolio ────────────────────────────────────
    print("\n[4/6] Building dim_loan_portfolio...")
    con.execute("""
        CREATE OR REPLACE TABLE dim_loan_portfolio AS
        SELECT
            PROSPECTID                          AS borrower_id,
            Total_TL                            AS total_loans_ever,
            Tot_Active_TL                       AS active_loans,
            Tot_Closed_TL                       AS closed_loans,
            active_loan_ratio,
            loan_type_diversity,
            Gold_TL                             AS gold_loans,
            Home_TL                             AS home_loans,
            PL_TL                               AS personal_loans,
            CC_TL                               AS credit_card_loans,
            Auto_TL                             AS auto_loans,
            Consumer_TL                         AS consumer_loans,
            Secured_TL                          AS secured_loans,
            Unsecured_TL                        AS unsecured_loans,
            GL_Flag                             AS has_gold_loan,
            HL_Flag                             AS has_home_loan,
            PL_Flag                             AS has_personal_loan,
            CC_Flag                             AS has_credit_card,
            recently_active
        FROM silver_master
    """)
    count = con.execute("SELECT COUNT(*) FROM dim_loan_portfolio").fetchone()[0]
    print(f"  ✓ dim_loan_portfolio: {count:,} rows")

    # ── Fact Table: Credit Risk ────────────────────────────────────────
    print("\n[5/6] Building fact_credit_risk...")
    con.execute("""
        CREATE OR REPLACE TABLE fact_credit_risk AS
        SELECT
            s.PROSPECTID                        AS borrower_id,
            s.default_risk,
            s.Approved_Flag                     AS risk_grade,
            s.risk_grade                        AS risk_score,
            s.Credit_Score                      AS cibil_score,
            s.NETMONTHLYINCOME                  AS monthly_income_inr,
            s.AGE                               AS age,
            s.delinquency_score,
            s.missed_payment_ratio,
            s.active_loan_ratio,
            s.score_per_income_lakh,
            s.loan_type_diversity,
            s.credit_history_months,
            s.income_tier,
            s.cibil_band,
            CASE
                WHEN s.AGE < 25 THEN 'under_25'
                WHEN s.AGE < 35 THEN '25_to_34'
                WHEN s.AGE < 45 THEN '35_to_44'
                WHEN s.AGE < 55 THEN '45_to_54'
                ELSE '55_plus'
            END AS age_band,
            s.GENDER                            AS gender,
            s.EDUCATION                         AS education,
            s.MARITALSTATUS                     AS marital_status,
            s.Gold_TL > 0                       AS has_gold_loan,
            s.Home_TL > 0                       AS has_home_loan,
            s.PL_TL > 0                         AS has_personal_loan,
            s.Total_TL                          AS total_loans,
            s.num_times_delinquent,
            s.num_times_60p_dpd,
            s.enq_L6m                           AS recent_enquiries_6m
        FROM silver_master s
    """)
    count = con.execute("SELECT COUNT(*) FROM fact_credit_risk").fetchone()[0]
    print(f"  ✓ fact_credit_risk: {count:,} rows")

    # ── Analytical Views ───────────────────────────────────────────────
    print("\n[6/6] Building analytical views...")

    # View 1: Risk by CIBIL band
    con.execute("""
        CREATE OR REPLACE VIEW v_risk_by_cibil_band AS
        SELECT
            cibil_band,
            COUNT(*)                            AS total_borrowers,
            SUM(default_risk)                   AS high_risk_count,
            ROUND(AVG(default_risk) * 100, 2)   AS default_rate_pct,
            ROUND(AVG(cibil_score), 0)          AS avg_cibil_score,
            ROUND(AVG(monthly_income_inr), 0)   AS avg_income_inr
        FROM fact_credit_risk
        GROUP BY cibil_band
        ORDER BY avg_cibil_score
    """)

    # View 2: Risk by income tier
    con.execute("""
        CREATE OR REPLACE VIEW v_risk_by_income AS
        SELECT
            income_tier,
            COUNT(*)                            AS total_borrowers,
            ROUND(AVG(default_risk) * 100, 2)   AS default_rate_pct,
            ROUND(AVG(cibil_score), 1)          AS avg_cibil_score,
            ROUND(AVG(monthly_income_inr), 0)   AS avg_income_inr,
            ROUND(AVG(total_loans), 1)          AS avg_total_loans
        FROM fact_credit_risk
        GROUP BY income_tier
        ORDER BY avg_income_inr
    """)

    # View 3: Risk by age band
    con.execute("""
        CREATE OR REPLACE VIEW v_risk_by_age AS
        SELECT
            age_band,
            COUNT(*)                            AS total_borrowers,
            ROUND(AVG(default_risk) * 100, 2)   AS default_rate_pct,
            ROUND(AVG(cibil_score), 1)          AS avg_cibil_score,
            ROUND(AVG(monthly_income_inr), 0)   AS avg_income_inr
        FROM fact_credit_risk
        GROUP BY age_band
        ORDER BY age_band
    """)

    # View 4: Gold loan analysis (uniquely Indian)
    con.execute("""
        CREATE OR REPLACE VIEW v_gold_loan_analysis AS
        SELECT
            has_gold_loan,
            COUNT(*)                            AS total_borrowers,
            ROUND(AVG(default_risk) * 100, 2)   AS default_rate_pct,
            ROUND(AVG(cibil_score), 1)          AS avg_cibil_score,
            ROUND(AVG(monthly_income_inr), 0)   AS avg_income_inr,
            ROUND(AVG(total_loans), 1)          AS avg_total_loans
        FROM fact_credit_risk
        GROUP BY has_gold_loan
    """)

    # View 5: Education vs default rate
    con.execute("""
        CREATE OR REPLACE VIEW v_risk_by_education AS
        SELECT
            education,
            COUNT(*)                            AS total_borrowers,
            ROUND(AVG(default_risk) * 100, 2)   AS default_rate_pct,
            ROUND(AVG(cibil_score), 1)          AS avg_cibil_score,
            ROUND(AVG(monthly_income_inr), 0)   AS avg_income_inr
        FROM fact_credit_risk
        GROUP BY education
        ORDER BY default_rate_pct DESC
    """)

    print("  ✓ 5 analytical views created")

    # ── Export to Parquet ──────────────────────────────────────────────
    print("\n  Exporting to Parquet...")

    tables = [
        "dim_borrower",
        "dim_credit",
        "dim_loan_portfolio",
        "fact_credit_risk",
    ]

    views = [
        "v_risk_by_cibil_band",
        "v_risk_by_income",
        "v_risk_by_age",
        "v_gold_loan_analysis",
        "v_risk_by_education",
    ]

    for name in tables + views:
        df_out = con.execute(f"SELECT * FROM {name}").df()
        path   = EXPORTS / f"{name}.parquet"
        df_out.to_parquet(path, index=False)
        print(f"  ✓ {name}.parquet ({len(df_out):,} rows)")

    # ── Print key findings from views ──────────────────────────────────
    print("\n━━━ Key Findings from Gold Layer ━━━")

    print("\n  Default rate by CIBIL band:")
    print(con.execute("SELECT * FROM v_risk_by_cibil_band").df().to_string(index=False))

    print("\n  Default rate by income tier:")
    print(con.execute("SELECT * FROM v_risk_by_income").df().to_string(index=False))

    print("\n  Gold loan vs default rate:")
    print(con.execute("SELECT * FROM v_gold_loan_analysis").df().to_string(index=False))

    con.close()

    print(f"""
{'='*60}
✓ GOLD LAYER COMPLETE

  Database: data/gold/credit_risk.duckdb
  Exports:  data/gold/exports/

  Tables:   {len(tables)} dimension + fact tables
  Views:    {len(views)} analytical views

  Next: python src/analytics/run_analytics.py
{'='*60}
    """)


if __name__ == "__main__":
    build_gold()