"""
India Credit Risk Intelligence — Gold Layer

Purpose
-------
Build a DuckDB analytical layer from the validated Silver borrower master.

Silver architecture
-------------------
silver_master.parquet
    51,336 borrower-level records
    Bank + CIBIL joined using PROSPECTID

silver_loan_applications.parquet
    Separate 87,020-row application dataset
    No PROSPECTID
    NOT joined to borrower master

Gold architecture
-----------------
DuckDB database:
    data/gold/credit_risk.duckdb

Tables:
    dim_borrower
    dim_credit
    dim_loan_portfolio
    dim_risk
    fact_credit_risk

Views:
    v_risk_by_cibil_band
    v_risk_by_income
    v_risk_by_age
    v_gold_loan_analysis
    v_risk_by_education
    v_risk_by_gender
    v_risk_by_delinquency
    v_risk_summary

Parquet exports are generated for dashboard consumption.

Run
---
python src/modeling/build_gold.py
"""

from pathlib import Path

import duckdb
import pandas as pd


# ============================================================
# PATHS
# ============================================================

SILVER_DIR = Path("data/silver")
GOLD_DIR = Path("data/gold")
EXPORTS = GOLD_DIR / "exports"
DB_PATH = GOLD_DIR / "credit_risk.duckdb"

GOLD_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS.mkdir(parents=True, exist_ok=True)


# ============================================================
# REQUIRED SILVER COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "PROSPECTID",
    "Approved_Flag",
    "default_risk",
    "risk_grade",
    "risk_band",
    "Credit_Score",
    "NETMONTHLYINCOME",
    "AGE",
    "GENDER",
    "EDUCATION",
    "MARITALSTATUS",
    "income_tier",
    "cibil_band",
    "credit_history_months",
    "delinquency_score",
    "num_times_delinquent",
    "num_times_30p_dpd",
    "num_times_60p_dpd",
    "Tot_Missed_Pmnt",
    "time_since_recent_payment",
    "tot_enq",
    "enq_L6m",
    "enq_L12m",
    "last_prod_enq2",
    "first_prod_enq2",
    "Total_TL",
    "Tot_Active_TL",
    "Tot_Closed_TL",
    "active_loan_ratio",
    "loan_type_diversity",
    "Gold_TL",
    "Home_TL",
    "PL_TL",
    "CC_TL",
    "Auto_TL",
    "Consumer_TL",
    "Secured_TL",
    "Unsecured_TL",
    "GL_Flag",
    "HL_Flag",
    "PL_Flag",
    "CC_Flag",
    "recently_active",
    "missed_payment_ratio",
    
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def validate_silver(df: pd.DataFrame) -> None:
    """
    Validate the Silver master before creating Gold.

    Gold should never silently build from an invalid Silver layer.
    """

    print("\n[2/7] Validating Silver master...")

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Silver master is missing required columns:\n"
            + "\n".join(f"  - {col}" for col in missing_columns)
        )

    print(
        f"  ✓ Required columns present: "
        f"{len(REQUIRED_COLUMNS)}/{len(REQUIRED_COLUMNS)}"
    )

    # --------------------------------------------------------
    # Borrower key
    # --------------------------------------------------------

    if df["PROSPECTID"].isna().any():
        raise ValueError(
            "PROSPECTID contains null values."
        )

    duplicate_ids = df["PROSPECTID"].duplicated().sum()

    if duplicate_ids:
        raise ValueError(
            f"Silver contains {duplicate_ids:,} "
            "duplicate PROSPECTID values."
        )

    print("  ✓ PROSPECTID unique and non-null")

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    if df["default_risk"].isna().any():
        raise ValueError(
            "default_risk contains null values."
        )

    unique_targets = sorted(
        df["default_risk"].dropna().unique().tolist()
    )

    if unique_targets != [0, 1]:
        raise ValueError(
            f"default_risk must contain only 0/1. "
            f"Found: {unique_targets}"
        )

    print("  ✓ default_risk is binary and non-null")

    # --------------------------------------------------------
    # Risk grade
    # --------------------------------------------------------

    approved_values = sorted(
        df["Approved_Flag"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    expected_values = ["P1", "P2", "P3", "P4"]

    unexpected = [
        value for value in approved_values
        if value not in expected_values
    ]

    if unexpected:
        raise ValueError(
            "Unexpected Approved_Flag values: "
            f"{unexpected}"
        )

    print("  ✓ Approved_Flag contains P1/P2/P3/P4")

    # --------------------------------------------------------
    # Credit score
    # --------------------------------------------------------

    if df["Credit_Score"].isna().any():
        raise ValueError(
            "Credit_Score contains null values."
        )

    invalid_scores = (
        (df["Credit_Score"] < 300)
        | (df["Credit_Score"] > 900)
    ).sum()

    if invalid_scores:
        raise ValueError(
            f"{invalid_scores:,} Credit_Score values "
            "fall outside 300–900."
        )

    print("  ✓ Credit_Score within 300–900")

    # --------------------------------------------------------
    # Income
    # --------------------------------------------------------

    negative_income = (
        df["NETMONTHLYINCOME"] < 0
    ).sum()

    if negative_income:
        raise ValueError(
            f"{negative_income:,} negative income values."
        )

    print("  ✓ NETMONTHLYINCOME has no negative values")

    # --------------------------------------------------------
    # Risk consistency
    # --------------------------------------------------------

    expected_target = (
        df["Approved_Flag"]
        .isin(["P3", "P4"])
        .astype(int)
    )

    mismatches = (
        expected_target != df["default_risk"]
    ).sum()

    if mismatches:
        raise ValueError(
            f"{mismatches:,} rows have inconsistent "
            "Approved_Flag/default_risk mapping."
        )

    print(
        "  ✓ Approved_Flag → default_risk mapping consistent"
    )

    print("  ✓ Silver validation passed")


def create_table_exports(
    con: duckdb.DuckDBPyConnection,
    names: list[str],
) -> None:
    """
    Export DuckDB tables/views to Parquet.
    """

    for name in names:

        df_out = con.execute(
            f"SELECT * FROM {name}"
        ).df()

        path = EXPORTS / f"{name}.parquet"

        df_out.to_parquet(
            path,
            index=False
        )

        print(
            f"  ✓ {name}.parquet "
            f"({len(df_out):,} rows)"
        )


def validate_gold(
    con: duckdb.DuckDBPyConnection,
    expected_rows: int,
) -> None:
    """
    Validate Gold star-schema tables.
    """

    print("\n[6/7] Validating Gold layer...")

    tables = [
        "dim_borrower",
        "dim_credit",
        "dim_loan_portfolio",
        "dim_risk",
        "fact_credit_risk",
    ]

    for table in tables:

        count = con.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]

        if count != expected_rows:
            raise ValueError(
                f"{table} contains {count:,} rows; "
                f"expected {expected_rows:,}."
            )

        print(
            f"  ✓ {table}: {count:,} rows"
        )

    # --------------------------------------------------------
    # Fact uniqueness
    # --------------------------------------------------------

    duplicate_fact_ids = con.execute("""
        SELECT COUNT(*)
        FROM (
            SELECT borrower_id
            FROM fact_credit_risk
            GROUP BY borrower_id
            HAVING COUNT(*) > 1
        )
    """).fetchone()[0]

    if duplicate_fact_ids:
        raise ValueError(
            "fact_credit_risk contains duplicate borrower IDs."
        )

    print(
        "  ✓ fact_credit_risk has one row per borrower"
    )

    # --------------------------------------------------------
    # Target validation
    # --------------------------------------------------------

    invalid_targets = con.execute("""
        SELECT COUNT(*)
        FROM fact_credit_risk
        WHERE default_risk NOT IN (0, 1)
           OR default_risk IS NULL
    """).fetchone()[0]

    if invalid_targets:
        raise ValueError(
            f"Invalid default_risk values: {invalid_targets:,}"
        )

    print(
        "  ✓ fact_credit_risk target is valid"
    )

    # --------------------------------------------------------
    # Risk consistency
    # --------------------------------------------------------

    inconsistent_risk = con.execute("""
        SELECT COUNT(*)
        FROM fact_credit_risk
        WHERE
            (
                risk_grade IN ('P3', 'P4')
                AND default_risk != 1
            )
            OR
            (
                risk_grade IN ('P1', 'P2')
                AND default_risk != 0
            )
    """).fetchone()[0]

    if inconsistent_risk:
        raise ValueError(
            "Risk grade/default risk inconsistency detected."
        )

    print(
        "  ✓ Risk grade/default risk consistency passed"
    )

    print("  ✓ GOLD VALIDATION PASSED")


# ============================================================
# MAIN GOLD BUILD
# ============================================================

def build_gold():

    print("=" * 60)
    print("  INDIA CREDIT RISK — GOLD LAYER (DuckDB)")
    print("=" * 60)

    # ========================================================
    # 1. LOAD SILVER
    # ========================================================

    print("\n[1/7] Loading Silver master...")

    silver_path = (
        SILVER_DIR / "silver_master.parquet"
    )

    if not silver_path.exists():
        raise FileNotFoundError(
            f"Silver master not found: {silver_path}"
        )

    df = pd.read_parquet(silver_path)

    print(
        f"  Loaded: {df.shape[0]:,} rows × "
        f"{df.shape[1]} columns"
    )

    # ========================================================
    # 2. VALIDATE SILVER
    # ========================================================

    validate_silver(df)

    # ========================================================
    # 3. CONNECT DUCKDB
    # ========================================================

    print("\n[3/7] Connecting to DuckDB...")

    con = duckdb.connect(
        str(DB_PATH)
    )

    con.register(
        "silver_master",
        df
    )

    print(
        f"  ✓ Database: {DB_PATH}"
    )

    # ========================================================
    # 4. DIMENSION: BORROWER
    # ========================================================

    print("\n[4/7] Building analytical dimensions...")

    con.execute("""
        CREATE OR REPLACE TABLE dim_borrower AS
        SELECT
            PROSPECTID AS borrower_id,
            AGE AS age,
            GENDER AS gender,
            MARITALSTATUS AS marital_status,
            EDUCATION AS education,
            NETMONTHLYINCOME AS monthly_income_inr,
            income_tier,
            Time_With_Curr_Empr AS months_with_employer,

            CASE
                WHEN AGE < 25 THEN 'under_25'
                WHEN AGE < 35 THEN '25_to_34'
                WHEN AGE < 45 THEN '35_to_44'
                WHEN AGE < 55 THEN '45_to_54'
                ELSE '55_plus'
            END AS age_band

        FROM silver_master
    """)

    print(
        f"  ✓ dim_borrower: "
        f"{con.execute('SELECT COUNT(*) FROM dim_borrower').fetchone()[0]:,}"
    )

    # ========================================================
    # DIMENSION: CREDIT
    # ========================================================

    con.execute("""
        CREATE OR REPLACE TABLE dim_credit AS
        SELECT
            PROSPECTID AS borrower_id,

            Credit_Score AS cibil_score,
            cibil_band,
            credit_history_months,

            num_times_delinquent
                AS total_delinquencies,

            num_times_30p_dpd
                AS times_30dpd,

            num_times_60p_dpd
                AS times_60dpd,

            delinquency_score,

            Tot_Missed_Pmnt
                AS total_missed_payments,

            time_since_recent_payment,

            tot_enq
                AS total_enquiries,

            enq_L6m
                AS enquiries_last_6m,

            enq_L12m
                AS enquiries_last_12m,

            last_prod_enq2
                AS last_product_enquired,

            first_prod_enq2
                AS first_product_enquired

        FROM silver_master
    """)

    print(
        f"  ✓ dim_credit: "
        f"{con.execute('SELECT COUNT(*) FROM dim_credit').fetchone()[0]:,}"
    )

    # ========================================================
    # DIMENSION: LOAN PORTFOLIO
    # ========================================================

    con.execute("""
        CREATE OR REPLACE TABLE dim_loan_portfolio AS
        SELECT
            PROSPECTID AS borrower_id,

            Total_TL AS total_loans_ever,
            Tot_Active_TL AS active_loans,
            Tot_Closed_TL AS closed_loans,

            active_loan_ratio,
            loan_type_diversity,

            Gold_TL AS gold_loans,
            Home_TL AS home_loans,
            PL_TL AS personal_loans,
            CC_TL AS credit_card_loans,
            Auto_TL AS auto_loans,
            Consumer_TL AS consumer_loans,

            Secured_TL AS secured_loans,
            Unsecured_TL AS unsecured_loans,

            GL_Flag AS has_gold_loan,
            HL_Flag AS has_home_loan,
            PL_Flag AS has_personal_loan,
            CC_Flag AS has_credit_card,

            recently_active

        FROM silver_master
    """)

    print(
        f"  ✓ dim_loan_portfolio: "
        f"{con.execute('SELECT COUNT(*) FROM dim_loan_portfolio').fetchone()[0]:,}"
    )

    # ========================================================
    # DIMENSION: RISK
    # ========================================================

    con.execute("""
        CREATE OR REPLACE TABLE dim_risk AS
        SELECT
            PROSPECTID AS borrower_id,

            Approved_Flag AS risk_grade,

            CASE
                WHEN Approved_Flag IN ('P1', 'P2')
                    THEN 0
                WHEN Approved_Flag IN ('P3', 'P4')
                    THEN 1
            END AS default_risk,

            risk_grade AS risk_grade_numeric,

            risk_band

        FROM silver_master
    """)

    print(
        f"  ✓ dim_risk: "
        f"{con.execute('SELECT COUNT(*) FROM dim_risk').fetchone()[0]:,}"
    )

    # ========================================================
    # FACT: CREDIT RISK
    # ========================================================

    print("\n[5/7] Building fact_credit_risk...")

    con.execute("""
        CREATE OR REPLACE TABLE fact_credit_risk AS
        SELECT
            s.PROSPECTID AS borrower_id,

            --------------------------------------------------
            -- TARGET / OUTCOME
            --------------------------------------------------

            s.default_risk,
            s.Approved_Flag AS risk_grade,
            s.risk_grade AS risk_grade_numeric,
            s.risk_band,

            --------------------------------------------------
            -- CORE CREDIT MEASURES
            --------------------------------------------------

            s.Credit_Score AS cibil_score,
            s.delinquency_score,
            s.num_times_delinquent,
            s.num_times_30p_dpd,
            s.num_times_60p_dpd,
            s.Tot_Missed_Pmnt AS total_missed_payments,
            s.missed_payment_ratio,

            --------------------------------------------------
            -- FINANCIAL MEASURES
            --------------------------------------------------

            s.NETMONTHLYINCOME AS monthly_income_inr,
            s.income_tier,
            

            --------------------------------------------------
            -- LOAN PORTFOLIO
            --------------------------------------------------

            s.Total_TL AS total_loans,
            s.Tot_Active_TL AS active_loans,
            s.Tot_Closed_TL AS closed_loans,
            s.active_loan_ratio,
            s.loan_type_diversity,

            s.Gold_TL AS gold_loans,
            s.Home_TL AS home_loans,
            s.PL_TL AS personal_loans,
            s.CC_TL AS credit_card_loans,
            s.Auto_TL AS auto_loans,

            s.Secured_TL AS secured_loans,
            s.Unsecured_TL AS unsecured_loans,

            --------------------------------------------------
            -- ENQUIRIES
            --------------------------------------------------

            s.tot_enq AS total_enquiries,
            s.enq_L6m AS recent_enquiries_6m,
            s.enq_L12m AS recent_enquiries_12m,

            --------------------------------------------------
            -- DEMOGRAPHICS
            --------------------------------------------------

            s.AGE AS age,
            s.GENDER AS gender,
            s.EDUCATION AS education,
            s.MARITALSTATUS AS marital_status,

            CASE
                WHEN s.AGE < 25 THEN 'under_25'
                WHEN s.AGE < 35 THEN '25_to_34'
                WHEN s.AGE < 45 THEN '35_to_44'
                WHEN s.AGE < 55 THEN '45_to_54'
                ELSE '55_plus'
            END AS age_band,

            --------------------------------------------------
            -- CREDIT BANDS
            --------------------------------------------------

            s.cibil_band,
            s.credit_history_months,

            --------------------------------------------------
            -- INDIAN-SPECIFIC PORTFOLIO SIGNAL
            --------------------------------------------------

            s.Gold_TL > 0 AS has_gold_loan,
            s.Home_TL > 0 AS has_home_loan,
            s.PL_TL > 0 AS has_personal_loan,
            s.CC_TL > 0 AS has_credit_card

        FROM silver_master s
    """)

    fact_count = con.execute(
        "SELECT COUNT(*) FROM fact_credit_risk"
    ).fetchone()[0]

    print(
        f"  ✓ fact_credit_risk: "
        f"{fact_count:,} rows"
    )

    # ========================================================
    # 6. ANALYTICAL VIEWS
    # ========================================================

    print("\n[6/7] Building analytical views...")

    # --------------------------------------------------------
    # Risk by CIBIL band
    # --------------------------------------------------------

    con.execute("""
        CREATE OR REPLACE VIEW v_risk_by_cibil_band AS
        SELECT
            cibil_band,
            COUNT(*) AS total_borrowers,
            SUM(default_risk) AS high_risk_count,
            ROUND(
                AVG(default_risk) * 100,
                2
            ) AS default_rate_pct,
            ROUND(
                AVG(cibil_score),
                1
            ) AS avg_cibil_score,
            ROUND(
                AVG(monthly_income_inr),
                0
            ) AS avg_income_inr
        FROM fact_credit_risk
        GROUP BY cibil_band
        ORDER BY avg_cibil_score
    """)

    # --------------------------------------------------------
    # Risk by income
    # --------------------------------------------------------

    con.execute("""
        CREATE OR REPLACE VIEW v_risk_by_income AS
        SELECT
            income_tier,
            COUNT(*) AS total_borrowers,

            SUM(default_risk)
                AS high_risk_count,

            ROUND(
                AVG(default_risk) * 100,
                2
            ) AS default_rate_pct,

            ROUND(
                AVG(cibil_score),
                1
            ) AS avg_cibil_score,

            ROUND(
                AVG(monthly_income_inr),
                0
            ) AS avg_income_inr,

            ROUND(
                AVG(total_loans),
                1
            ) AS avg_total_loans

        FROM fact_credit_risk

        GROUP BY income_tier

        ORDER BY avg_income_inr
    """)

    # --------------------------------------------------------
    # Risk by age
    # --------------------------------------------------------

    con.execute("""
        CREATE OR REPLACE VIEW v_risk_by_age AS
        SELECT
            age_band,
            COUNT(*) AS total_borrowers,

            SUM(default_risk)
                AS high_risk_count,

            ROUND(
                AVG(default_risk) * 100,
                2
            ) AS default_rate_pct,

            ROUND(
                AVG(cibil_score),
                1
            ) AS avg_cibil_score,

            ROUND(
                AVG(monthly_income_inr),
                0
            ) AS avg_income_inr

        FROM fact_credit_risk

        GROUP BY age_band

        ORDER BY
            CASE age_band
                WHEN 'under_25' THEN 1
                WHEN '25_to_34' THEN 2
                WHEN '35_to_44' THEN 3
                WHEN '45_to_54' THEN 4
                WHEN '55_plus' THEN 5
            END
    """)

    # --------------------------------------------------------
    # Gold loan analysis
    # --------------------------------------------------------

    con.execute("""
        CREATE OR REPLACE VIEW v_gold_loan_analysis AS
        SELECT
            has_gold_loan,
            COUNT(*) AS total_borrowers,

            SUM(default_risk)
                AS high_risk_count,

            ROUND(
                AVG(default_risk) * 100,
                2
            ) AS default_rate_pct,

            ROUND(
                AVG(cibil_score),
                1
            ) AS avg_cibil_score,

            ROUND(
                AVG(monthly_income_inr),
                0
            ) AS avg_income_inr,

            ROUND(
                AVG(total_loans),
                1
            ) AS avg_total_loans

        FROM fact_credit_risk

        GROUP BY has_gold_loan
    """)

    # --------------------------------------------------------
    # Education
    # --------------------------------------------------------

    con.execute("""
        CREATE OR REPLACE VIEW v_risk_by_education AS
        SELECT
            education,
            COUNT(*) AS total_borrowers,

            SUM(default_risk)
                AS high_risk_count,

            ROUND(
                AVG(default_risk) * 100,
                2
            ) AS default_rate_pct,

            ROUND(
                AVG(cibil_score),
                1
            ) AS avg_cibil_score,

            ROUND(
                AVG(monthly_income_inr),
                0
            ) AS avg_income_inr

        FROM fact_credit_risk

        GROUP BY education

        ORDER BY default_rate_pct DESC
    """)

    # --------------------------------------------------------
    # Gender
    # --------------------------------------------------------

    con.execute("""
        CREATE OR REPLACE VIEW v_risk_by_gender AS
        SELECT
            gender,
            COUNT(*) AS total_borrowers,

            SUM(default_risk)
                AS high_risk_count,

            ROUND(
                AVG(default_risk) * 100,
                2
            ) AS default_rate_pct,

            ROUND(
                AVG(cibil_score),
                1
            ) AS avg_cibil_score,

            ROUND(
                AVG(monthly_income_inr),
                0
            ) AS avg_income_inr

        FROM fact_credit_risk

        GROUP BY gender

        ORDER BY default_rate_pct DESC
    """)

    # --------------------------------------------------------
    # Delinquency analysis
    # --------------------------------------------------------

    con.execute("""
        CREATE OR REPLACE VIEW v_risk_by_delinquency AS
        SELECT
            CASE
                WHEN num_times_delinquent = 0
                    THEN '0'
                WHEN num_times_delinquent BETWEEN 1 AND 2
                    THEN '1_to_2'
                WHEN num_times_delinquent BETWEEN 3 AND 5
                    THEN '3_to_5'
                ELSE '6_plus'
            END AS delinquency_band,

            COUNT(*) AS total_borrowers,

            SUM(default_risk)
                AS high_risk_count,

            ROUND(
                AVG(default_risk) * 100,
                2
            ) AS default_rate_pct,

            ROUND(
                AVG(cibil_score),
                1
            ) AS avg_cibil_score

        FROM fact_credit_risk

        GROUP BY delinquency_band

        ORDER BY
            CASE delinquency_band
                WHEN '0' THEN 1
                WHEN '1_to_2' THEN 2
                WHEN '3_to_5' THEN 3
                WHEN '6_plus' THEN 4
            END
    """)

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    con.execute("""
        CREATE OR REPLACE VIEW v_risk_summary AS
        SELECT
            COUNT(*) AS total_borrowers,

            SUM(default_risk)
                AS high_risk_borrowers,

            COUNT(*) - SUM(default_risk)
                AS low_risk_borrowers,

            ROUND(
                AVG(default_risk) * 100,
                2
            ) AS overall_default_rate_pct,

            ROUND(
                AVG(cibil_score),
                1
            ) AS avg_cibil_score,

            ROUND(
                AVG(monthly_income_inr),
                0
            ) AS avg_monthly_income_inr,

            ROUND(
                AVG(total_loans),
                2
            ) AS avg_total_loans

        FROM fact_credit_risk
    """)

    print(
        "  ✓ 8 analytical views created"
    )

    # ========================================================
    # 7. VALIDATE + EXPORT
    # ========================================================

    validate_gold(
        con,
        expected_rows=len(df)
    )

    print("\n[7/7] Exporting Gold tables and views...")

    tables = [
        "dim_borrower",
        "dim_credit",
        "dim_loan_portfolio",
        "dim_risk",
        "fact_credit_risk",
    ]

    views = [
        "v_risk_summary",
        "v_risk_by_cibil_band",
        "v_risk_by_income",
        "v_risk_by_age",
        "v_gold_loan_analysis",
        "v_risk_by_education",
        "v_risk_by_gender",
        "v_risk_by_delinquency",
    ]

    create_table_exports(
        con,
        tables + views
    )

    # ========================================================
    # KEY FINDINGS
    # ========================================================

    print("\n" + "=" * 60)
    print("  KEY FINDINGS FROM GOLD LAYER")
    print("=" * 60)

    print("\nOverall portfolio:")
    print(
        con.execute(
            "SELECT * FROM v_risk_summary"
        ).df().to_string(index=False)
    )

    print("\nRisk by CIBIL band:")
    print(
        con.execute(
            "SELECT * FROM v_risk_by_cibil_band"
        ).df().to_string(index=False)
    )

    print("\nRisk by income tier:")
    print(
        con.execute(
            "SELECT * FROM v_risk_by_income"
        ).df().to_string(index=False)
    )

    print("\nGold loan analysis:")
    print(
        con.execute(
            "SELECT * FROM v_gold_loan_analysis"
        ).df().to_string(index=False)
    )

    print("\nRisk by delinquency:")
    print(
        con.execute(
            "SELECT * FROM v_risk_by_delinquency"
        ).df().to_string(index=False)
    )

    con.close()

    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print(
        f"""
{'=' * 60}
✓ GOLD LAYER COMPLETE

Database:
  {DB_PATH}

Exports:
  {EXPORTS}

Tables:
  ✓ dim_borrower
  ✓ dim_credit
  ✓ dim_loan_portfolio
  ✓ dim_risk
  ✓ fact_credit_risk

Analytical views:
  ✓ v_risk_summary
  ✓ v_risk_by_cibil_band
  ✓ v_risk_by_income
  ✓ v_risk_by_age
  ✓ v_gold_loan_analysis
  ✓ v_risk_by_education
  ✓ v_risk_by_gender
  ✓ v_risk_by_delinquency

Borrowers:
  {len(df):,}

Next:
  → Validate Gold
  → Exploratory Data Analysis
  → Statistical analysis
  → Leakage audit
  → ML train/test split
{'=' * 60}
"""
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    build_gold()