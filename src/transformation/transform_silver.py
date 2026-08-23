"""
transform_silver.py
===================

Silver-layer transformation for the India Credit Risk Intelligence project.

Responsibilities
----------------
1. Load Bronze datasets.
2. Normalize the target definition.
3. Clean -99999 sentinel values correctly.
4. Preserve useful missingness information.
5. Clean categorical variables.
6. Engineer business-interpretable features.
7. Validate the Silver dataset.
8. Write Silver Parquet artifacts.

IMPORTANT TARGET DEFINITION
---------------------------
Approved_Flag is an ordinal risk classification:

    P1 = lowest risk
    P2 = low risk
    P3 = higher risk
    P4 = highest risk

For compatibility with the existing project:

    default_risk = 0 -> P1/P2
    default_risk = 1 -> P3/P4

IMPORTANT:
`default_risk` is a HIGH-RISK CLASSIFICATION PROXY.
It is NOT an observed loan-default outcome.

Therefore this project should not claim:
    "probability that the borrower will actually default"

without an observed default/event label.

Run from repository root:

    python src/transformation/transform_silver.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pathlib import Path


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BRONZE_DIR = PROJECT_ROOT / "data" / "bronze" / "parquet"
SILVER_DIR = PROJECT_ROOT / "data" / "silver"

SILVER_DIR.mkdir(parents=True, exist_ok=True)


SENTINEL = -99999


class SilverTransformer:
    """Build and validate the Silver analytical dataset."""

    # -----------------------------------------------------------------
    # 1. LOAD BRONZE
    # -----------------------------------------------------------------

    def load_bronze(self) -> tuple:
    
    
        print("\n[1/6] Loading Bronze data...")

        bank_path = BRONZE_DIR / "bronze_internal_bank.parquet"
        cibil_path = BRONZE_DIR / "bronze_cibil_external.parquet"
        loans_path = BRONZE_DIR / "bronze_loan_applications.parquet"

        # ---------------------------------------------------------
        # Load datasets
        # ---------------------------------------------------------

        df_bank = pd.read_parquet(bank_path)
        df_cibil = pd.read_parquet(cibil_path)
        df_loans = pd.read_parquet(loans_path)

        # ---------------------------------------------------------
        # Remove ingestion metadata
        # ---------------------------------------------------------

        meta_cols = ["_source", "_ingested_at"]

        for df in [df_bank, df_cibil, df_loans]:

            existing_meta_cols = [
                col for col in meta_cols
                if col in df.columns
            ]

            if existing_meta_cols:
                df.drop(
                    columns=existing_meta_cols,
                    inplace=True
                )

        # ---------------------------------------------------------
        # Validate MAIN borrower datasets
        # ---------------------------------------------------------

        if "PROSPECTID" not in df_bank.columns:
            raise ValueError(
                "Internal Bank dataset must contain PROSPECTID."
            )

        if "PROSPECTID" not in df_cibil.columns:
            raise ValueError(
                "CIBIL dataset must contain PROSPECTID."
            )

        # ---------------------------------------------------------
        # Validate uniqueness of borrower keys
        # ---------------------------------------------------------

        bank_duplicates = df_bank["PROSPECTID"].duplicated().sum()
        cibil_duplicates = df_cibil["PROSPECTID"].duplicated().sum()

        if bank_duplicates > 0:
            raise ValueError(
                f"Internal Bank contains {bank_duplicates:,} "
                f"duplicate PROSPECTID values."
            )

        if cibil_duplicates > 0:
            raise ValueError(
                f"CIBIL contains {cibil_duplicates:,} "
                f"duplicate PROSPECTID values."
            )

        # ---------------------------------------------------------
        # Loan Applications are intentionally NOT required to have
        # PROSPECTID.
        # ---------------------------------------------------------

        print(f"  Bank : {df_bank.shape}")
        print(f"  CIBIL: {df_cibil.shape}")
        print(f"  Loans: {df_loans.shape}")

        print(
            "  ✓ Bank + CIBIL validated for PROSPECTID join"
        )

        print(
            "  ✓ Loan Applications retained as a separate "
            "dataset — no PROSPECTID join required"
        )

        return df_bank, df_cibil, df_loans

    # -----------------------------------------------------------------
    # 2. TARGET
    # -----------------------------------------------------------------

    def encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n[2/6] Encoding risk target...")

        if "Approved_Flag" not in df.columns:
            raise ValueError(
                "Approved_Flag is missing from the CIBIL dataset."
            )

        df["Approved_Flag"] = (
            df["Approved_Flag"]
            .astype("string")
            .str.strip()
            .str.upper()
        )

        valid_flags = {"P1", "P2", "P3", "P4"}

        unexpected = set(df["Approved_Flag"].dropna().unique()) - valid_flags

        if unexpected:
            raise ValueError(
                f"Unexpected Approved_Flag values: {unexpected}"
            )

        # Ordinal risk representation.
        risk_map = {
            "P1": 1,
            "P2": 2,
            "P3": 3,
            "P4": 4,
        }

        df["risk_grade"] = df["Approved_Flag"].map(risk_map)

        # Binary high-risk classification.
        df["default_risk"] = (
            df["Approved_Flag"].isin(["P3", "P4"])
        ).astype("int8")

        print("\n  Approved_Flag distribution:")
        print(df["Approved_Flag"].value_counts().sort_index())

        risky = int(df["default_risk"].sum())
        safe = int((df["default_risk"] == 0).sum())

        print("\n  Canonical default_risk:")
        print(
            f"  High risk (P3/P4): {risky:,} "
            f"({risky / len(df) * 100:.1f}%)"
        )
        print(
            f"  Low risk  (P1/P2): {safe:,} "
            f"({safe / len(df) * 100:.1f}%)"
        )

        # -------------------------------------------------------------
        # Sanity check
        # -------------------------------------------------------------

        if "Credit_Score" in df.columns:

            risky_score = df.loc[
                df["default_risk"] == 1,
                "Credit_Score",
            ].mean()

            safe_score = df.loc[
                df["default_risk"] == 0,
                "Credit_Score",
            ].mean()

            print("\n  Sanity check — CIBIL:")

            print(
                f"  High-risk average: {risky_score:.1f}"
            )

            print(
                f"  Low-risk average:  {safe_score:.1f}"
            )

            if risky_score < safe_score:
                print(
                    "  ✓ Risk direction is consistent "
                    "with the expected mapping"
                )
            else:
                raise ValueError(
                    "Target mapping sanity check failed: "
                    "high-risk borrowers have a higher average "
                    "Credit_Score than low-risk borrowers."
                )

        return df

    # -----------------------------------------------------------------
    # 3. SENTINELS
    # -----------------------------------------------------------------

    def clean_sentinels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert -99999 into NaN.

        Why?
        ----
        -99999 is a data-system sentinel, not a real financial value.

        We deliberately DO NOT convert it to:
            0
        or:
            9999

        because those values would create artificial observations.

        We additionally create missingness indicators for every numeric
        column containing the sentinel.

        Example:

            CC_utilization = -99999

        becomes:

            CC_utilization = NaN
            has_cc_utilization = 0

        This lets downstream models distinguish:

            "no applicable value"

        from:

            "real zero".
        """

        print("\n[3/6] Cleaning sentinel values...")

        numeric_columns = df.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        sentinel_columns = [
            col
            for col in numeric_columns
            if (df[col] == SENTINEL).any()
        ]

        print(
            f"  Columns containing -99999: "
            f"{sentinel_columns}"
        )

        for col in sentinel_columns:

            flag_name = (
                "has_"
                + col.lower()
                .replace(" ", "_")
                .replace("-", "_")
            )

            # 1 = valid/applicable value exists
            # 0 = source contained sentinel / not applicable
            df[flag_name] = (
                df[col] != SENTINEL
            ).astype("int8")

            df[col] = df[col].replace(
                SENTINEL,
                np.nan,
            )

        remaining = []

        for col in df.select_dtypes(
            include=[np.number]
        ).columns:

            if (df[col] == SENTINEL).any():
                remaining.append(col)

        print(
            f"  Remaining sentinel columns: "
            f"{len(remaining)}"
        )

        if remaining:
            print(
                f"  ✗ Sentinel cleanup incomplete: "
                f"{remaining}"
            )
            raise ValueError(
                "Sentinel cleanup failed."
            )

        print("  ✓ All -99999 values converted to NaN")

        return df

    # -----------------------------------------------------------------
    # 4. CATEGORICAL CLEANING
    # -----------------------------------------------------------------

    def clean_categorical(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        print("\n[4/6] Cleaning categorical columns...")

        categorical_columns = [
            "MARITALSTATUS",
            "EDUCATION",
            "GENDER",
            "last_prod_enq2",
            "first_prod_enq2",
        ]

        for col in categorical_columns:

            if col not in df.columns:
                continue

            df[col] = (
                df[col]
                .astype("string")
                .str.strip()
            )

        if "MARITALSTATUS" in df.columns:
            df["MARITALSTATUS"] = (
                df["MARITALSTATUS"]
                .str.upper()
            )

        if "GENDER" in df.columns:
            df["GENDER"] = (
                df["GENDER"]
                .str.upper()
            )

        if "EDUCATION" in df.columns:

            df["EDUCATION"] = (
                df["EDUCATION"]
                .str.upper()
                .replace(
                    {
                        "POST-GRADUATE": "POST_GRADUATE",
                        "POST GRADUATE": "POST_GRADUATE",
                    }
                )
            )

            valid_education = {
                "12TH",
                "SSC",
                "GRADUATE",
                "POST_GRADUATE",
                "PROFESSIONAL",
                "OTHERS",
            }

            df["EDUCATION"] = df[
                "EDUCATION"
            ].where(
                df["EDUCATION"].isin(valid_education),
                "OTHERS",
            )

        print("  ✓ Categorical normalization complete")

        return df

    # -----------------------------------------------------------------
    # 5. FEATURE ENGINEERING
    # -----------------------------------------------------------------

    def engineer_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        print("\n[5/6] Engineering business features...")

        # -------------------------------------------------------------
        # Helper for safe numeric columns
        # -------------------------------------------------------------

        def numeric_series(
            name: str,
            default: float = 0.0,
        ) -> pd.Series:

            if name in df.columns:
                return pd.to_numeric(
                    df[name],
                    errors="coerce",
                )

            return pd.Series(
                default,
                index=df.index,
                dtype="float64",
            )

        total_loans = numeric_series("Total_TL")
        active_loans = numeric_series("Tot_Active_TL")
        missed_payments = numeric_series("Tot_Missed_Pmnt")

        # -------------------------------------------------------------
        # Portfolio utilization behaviour
        # -------------------------------------------------------------

        df["active_loan_ratio"] = np.where(
            total_loans > 0,
            active_loans / total_loans,
            0.0,
        )

        df["missed_payment_ratio"] = np.where(
            total_loans > 0,
            missed_payments / total_loans,
            0.0,
        )

        # Avoid impossible ratios from malformed source data.
        df["active_loan_ratio"] = (
            df["active_loan_ratio"]
            .clip(lower=0, upper=1)
        )

        df["missed_payment_ratio"] = (
            df["missed_payment_ratio"]
            .clip(lower=0, upper=1)
        )

        # -------------------------------------------------------------
        # Credit history
        # -------------------------------------------------------------

        if "Age_Oldest_TL" in df.columns:
            df["credit_history_months"] = df[
                "Age_Oldest_TL"
            ]

        # -------------------------------------------------------------
        # Delinquency severity
        # -------------------------------------------------------------

        delinquent = numeric_series(
            "num_times_delinquent"
        )

        dpd30 = numeric_series(
            "num_times_30p_dpd"
        )

        dpd60 = numeric_series(
            "num_times_60p_dpd"
        )

        df["delinquency_score"] = (
            delinquent.fillna(0) * 2
            + dpd30.fillna(0) * 3
            + dpd60.fillna(0) * 5
        )

        # -------------------------------------------------------------
        # Recent credit activity
        # -------------------------------------------------------------

        if "Total_TL_opened_L6M" in df.columns:
            df["recently_active"] = (
                numeric_series(
                    "Total_TL_opened_L6M"
                )
                .fillna(0)
                > 0
            ).astype("int8")
        else:
            df["recently_active"] = 0

        # -------------------------------------------------------------
        # Loan type diversity
        # -------------------------------------------------------------

        loan_type_columns = [
            "Auto_TL",
            "CC_TL",
            "Consumer_TL",
            "Gold_TL",
            "Home_TL",
            "PL_TL",
        ]

        available = [
            col
            for col in loan_type_columns
            if col in df.columns
        ]

        if available:
            df["loan_type_diversity"] = (
                df[available]
                .apply(
                    pd.to_numeric,
                    errors="coerce",
                )
                .fillna(0)
                .gt(0)
                .sum(axis=1)
                .astype("int8")
            )

        # -------------------------------------------------------------
        # Income tier
        # -------------------------------------------------------------

        if "NETMONTHLYINCOME" in df.columns:

            income = pd.to_numeric(
                df["NETMONTHLYINCOME"],
                errors="coerce",
            )

            df["income_tier"] = pd.cut(
                income,
                bins=[
                    -np.inf,
                    15000,
                    30000,
                    60000,
                    100000,
                    np.inf,
                ],
                labels=[
                    "very_low",
                    "low",
                    "middle",
                    "upper_middle",
                    "high",
                ],
            ).astype("string")

        # -------------------------------------------------------------
        # CIBIL bands
        # -------------------------------------------------------------

        if "Credit_Score" in df.columns:

            score = pd.to_numeric(
                df["Credit_Score"],
                errors="coerce",
            )

            df["cibil_band"] = pd.cut(
                score,
                bins=[
                    -np.inf,
                    549,
                    649,
                    699,
                    749,
                    np.inf,
                ],
                labels=[
                    "poor",
                    "fair",
                    "good",
                    "very_good",
                    "excellent",
                ],
            ).astype("string")

        # -------------------------------------------------------------
        # Risk band
        #
        # IMPORTANT:
        # This is descriptive only.
        # It MUST NOT enter the ML feature set.
        # -------------------------------------------------------------

        df["risk_band"] = df[
            "Approved_Flag"
        ].map(
            {
                "P1": "Low Risk",
                "P2": "Low Risk",
                "P3": "Medium Risk",
                "P4": "High Risk",
            }
        )

        if df["risk_band"].isna().any():
            raise ValueError(
                "risk_band contains unknown values."
            )

        print(
            f"  ✓ Feature engineering complete: "
            f"{df.shape[1]} columns"
        )

        print("\n  Risk-band distribution:")
        print(
            df["risk_band"]
            .value_counts()
            .to_string()
        )

        return df

    # -----------------------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------------------

    def validate(
        self,
        df: pd.DataFrame,
    ) -> bool:

        print("\n[6/6] Validating Silver data...")

        checks: list[bool] = []

        # -------------------------------------------------------------
        # Required columns
        # -------------------------------------------------------------

        required_columns = [
            "PROSPECTID",
            "Approved_Flag",
            "risk_grade",
            "default_risk",
            "Credit_Score",
            "NETMONTHLYINCOME",
            "AGE",
        ]

        for col in required_columns:

            exists = col in df.columns

            print(
                f"  {'✓' if exists else '✗'} "
                f"{col} exists"
            )

            checks.append(exists)

            if exists:

                nulls = int(
                    df[col].isna().sum()
                )

                # We require critical columns to be complete.
                required_no_nulls = {
                    "PROSPECTID",
                    "Approved_Flag",
                    "risk_grade",
                    "default_risk",
                    "Credit_Score",
                    "AGE",
                }

                if col in required_no_nulls:

                    passed = nulls == 0

                    print(
                        f"    "
                        f"{'✓' if passed else '✗'} "
                        f"nulls: {nulls}"
                    )

                    checks.append(passed)

        # -------------------------------------------------------------
        # Target
        # -------------------------------------------------------------

        if "default_risk" in df.columns:

            values = set(
                df["default_risk"]
                .dropna()
                .unique()
            )

            binary = values.issubset({0, 1})

            print(
                f"  "
                f"{'✓' if binary else '✗'} "
                f"default_risk is binary"
            )

            checks.append(binary)

        # -------------------------------------------------------------
        # Sentinel audit
        # -------------------------------------------------------------

        remaining_sentinel_columns = []

        for col in df.select_dtypes(
            include=[np.number]
        ).columns:

            if (df[col] == SENTINEL).any():
                remaining_sentinel_columns.append(col)

        passed = len(
            remaining_sentinel_columns
        ) == 0

        print(
            f"  "
            f"{'✓' if passed else '✗'} "
            f"Remaining sentinel columns: "
            f"{len(remaining_sentinel_columns)}"
        )

        if not passed:
            print(
                f"    {remaining_sentinel_columns}"
            )

        checks.append(passed)

        # -------------------------------------------------------------
        # Duplicates
        # -------------------------------------------------------------

        duplicate_count = int(
            df["PROSPECTID"]
            .duplicated()
            .sum()
        )

        passed = duplicate_count == 0

        print(
            f"  "
            f"{'✓' if passed else '✗'} "
            f"Duplicate PROSPECTID: "
            f"{duplicate_count}"
        )

        checks.append(passed)

        # -------------------------------------------------------------
        # Credit score sanity
        # -------------------------------------------------------------

        if "Credit_Score" in df.columns:

            valid_scores = (
                df["Credit_Score"]
                .between(300, 900)
                .all()
            )

            print(
                f"  "
                f"{'✓' if valid_scores else '✗'} "
                f"Credit scores within 300–900"
            )

            checks.append(valid_scores)

        # -------------------------------------------------------------
        # Income
        # -------------------------------------------------------------

        if "NETMONTHLYINCOME" in df.columns:

            negative_income = int(
                (df["NETMONTHLYINCOME"] < 0)
                .sum()
            )

            passed = negative_income == 0

            print(
                f"  "
                f"{'✓' if passed else '✗'} "
                f"Negative income: "
                f"{negative_income}"
            )

            checks.append(passed)

        # -------------------------------------------------------------
        # Target direction
        # -------------------------------------------------------------

        if {
            "default_risk",
            "Credit_Score",
        }.issubset(df.columns):

            risky_score = df.loc[
                df["default_risk"] == 1,
                "Credit_Score",
            ].mean()

            safe_score = df.loc[
                df["default_risk"] == 0,
                "Credit_Score",
            ].mean()

            passed = risky_score < safe_score

            print(
                f"  "
                f"{'✓' if passed else '✗'} "
                f"Risk direction: "
                f"high-risk CIBIL "
                f"({risky_score:.1f}) < "
                f"low-risk CIBIL "
                f"({safe_score:.1f})"
            )

            checks.append(passed)

        # -------------------------------------------------------------
        # Legacy target check
        # -------------------------------------------------------------

        if "risk_target" in df.columns:

            print(
                "  ✗ Legacy column risk_target "
                "must not exist"
            )

            checks.append(False)

        else:

            print(
                "  ✓ No legacy risk_target column"
            )

            checks.append(True)

        # -------------------------------------------------------------
        # Result
        # -------------------------------------------------------------

        all_passed = all(checks)

        print(
            "\n  "
            + (
                "✓ SILVER VALIDATION PASSED"
                if all_passed
                else "✗ SILVER VALIDATION FAILED"
            )
        )

        return all_passed

    # -----------------------------------------------------------------
    # RUN
    # -----------------------------------------------------------------

    def run(self):

        print("=" * 60)
        print("  INDIA CREDIT RISK — SILVER TRANSFORMATION")
        print("=" * 60)

        # =========================================================
        # 1. LOAD BRONZE
        # =========================================================

        df_bank, df_cibil, df_loans = self.load_bronze()

        # =========================================================
        # 2. TRANSFORM CIBIL
        # =========================================================

        df_cibil = self.encode_target(df_cibil)

        df_cibil = self.clean_sentinels(df_cibil)

        df_cibil = self.clean_categorical(df_cibil)

        # =========================================================
        # 3. TRANSFORM INTERNAL BANK
        # =========================================================

        df_bank = self.clean_sentinels(df_bank)

        # =========================================================
        # 4. JOIN BORROWER DATASETS
        # =========================================================

        print(
            "\n  Joining Internal Bank + CIBIL "
            "datasets on PROSPECTID..."
        )

        bank_ids = set(df_bank["PROSPECTID"])
        cibil_ids = set(df_cibil["PROSPECTID"])

        common_ids = bank_ids & cibil_ids

        print(f"  Bank borrower IDs : {len(bank_ids):,}")
        print(f"  CIBIL borrower IDs: {len(cibil_ids):,}")
        print(f"  Matching IDs      : {len(common_ids):,}")

        if len(common_ids) == 0:
            raise ValueError(
                "No matching PROSPECTID values between "
                "Internal Bank and CIBIL datasets."
            )

        df_master = df_cibil.merge(
            df_bank,
            on="PROSPECTID",
            how="inner",
            suffixes=("_cibil", "_bank")
        )

        print(
            f"  ✓ Joined borrower master: "
            f"{df_master.shape[0]:,} rows × "
            f"{df_master.shape[1]} columns"
        )

        # =========================================================
        # 5. FEATURE ENGINEERING
        # =========================================================

        df_master = self.engineer_features(df_master)

        # =========================================================
        # 6. VALIDATE MASTER
        # =========================================================

        validation_passed = self.validate(df_master)

        if not validation_passed:
            raise ValueError(
                "Silver master validation failed. "
                "Fix data quality issues before continuing."
            )

        # =========================================================
        # 7. SAVE SILVER DATASETS
        # =========================================================

        master_path = SILVER_DIR / "silver_master.parquet"

        loans_path = SILVER_DIR / "silver_loan_applications.parquet"

        df_master.to_parquet(
            master_path,
            index=False
        )

        df_loans.to_parquet(
            loans_path,
            index=False
        )

        # =========================================================
        # 8. FINAL SUMMARY
        # =========================================================

        risky = int(
            df_master["default_risk"].sum()
        )

        safe = int(
            (df_master["default_risk"] == 0).sum()
        )

        print(
            f"""
    {'=' * 60}
    ✓ SILVER TRANSFORMATION COMPLETE

    Main borrower dataset:
    → silver_master.parquet
    → {len(df_master):,} rows × {df_master.shape[1]} columns

    Separate loan application dataset:
    → silver_loan_applications.parquet
    → {len(df_loans):,} rows × {df_loans.shape[1]} columns

    Target encoding:
    → default_risk = 1 → P3/P4 = HIGH RISK
    → default_risk = 0 → P1/P2 = LOW RISK

    High risk: {risky:,} ({risky / len(df_master) * 100:.1f}%)
    Low risk : {safe:,} ({safe / len(df_master) * 100:.1f}%)

    Data architecture:
    ✓ Internal Bank + CIBIL → borrower master
    ✓ Loan Applications → separate dataset
    ✓ No artificial PROSPECTID created
    ✓ No invalid cross-dataset join

    Next:
    python src/modeling/build_gold.py
    {'=' * 60}
    """
        )

        return df_master
    

if __name__ == "__main__":
    transformer = SilverTransformer()
    transformer.run()