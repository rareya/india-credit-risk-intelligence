"""
transform_silver.py  —  Silver Layer
────────────────────────────────────────────────────────────────────
Cleans Bronze data, engineers features, encodes target variable.
Produces the master dataset ready for ML and analytics.

    python src/transformation/transform_silver.py

What happens here:
  1. Load Bronze Parquet files
  2. Clean sentinel values (-99999 → meaningful substitutes)
  3. Encode target: Approved_Flag → binary default risk
  4. Join internal bank + CIBIL datasets on PROSPECTID
  5. Engineer derived features
  6. Validate data quality
  7. Save Silver Parquet
────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

BRONZE_DIR = Path("data/bronze/parquet")
SILVER_DIR = Path("data/silver")
SILVER_DIR.mkdir(parents=True, exist_ok=True)

# Sentinel value used throughout the dataset to mean "not applicable"
SENTINEL = -99999


class SilverTransformer:
    """
    Transforms Bronze data into clean, analytics-ready Silver data.
    Every transformation is documented with WHY, not just WHAT.
    """

    def load_bronze(self) -> tuple:
        """Load Bronze Parquet files."""
        print("\n[1/6] Loading Bronze data...")

        df_bank  = pd.read_parquet(BRONZE_DIR / "bronze_internal_bank.parquet")
        df_cibil = pd.read_parquet(BRONZE_DIR / "bronze_cibil_external.parquet")
        df_loans = pd.read_parquet(BRONZE_DIR / "bronze_loan_applications.parquet")

        # Drop metadata columns added during ingestion
        meta_cols = ["_source", "_ingested_at"]
        for df in [df_bank, df_cibil, df_loans]:
            for col in meta_cols:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)

        print(f"  Bank:  {df_bank.shape}")
        print(f"  CIBIL: {df_cibil.shape}")
        print(f"  Loans: {df_loans.shape}")
        return df_bank, df_cibil, df_loans

    def encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode Approved_Flag → binary default risk.

        Approved_Flag values:
            P1 = High risk borrower → default_risk = 1
            P2 = Medium risk        → default_risk = 1
            P3 = Low risk           → default_risk = 0
            P4 = Very low risk      → default_risk = 0

        WHY P1+P2 as positive class:
            In credit risk we want to catch risky borrowers.
            P1 and P2 together represent borrowers the bank
            should scrutinise carefully. P3 and P4 are acceptable.
            This gives us a meaningful binary classification task.

        We also keep the original 4-class label for multiclass analysis.
        """
        print("\n[2/6] Encoding target variable...")

        # Map to ordinal risk score (1=highest risk, 4=lowest)
        risk_map = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}
        df["risk_grade"]   = df["Approved_Flag"].map(risk_map)

        # Binary target: P1 or P2 = high risk = 1
        df["default_risk"] = (df["Approved_Flag"].isin(["P3", "P4"])).astype(int)
        print(f"  Target distribution:")
        print(f"  {df['Approved_Flag'].value_counts().to_string()}")
        print(f"\n  Binary target:")
        print(f"  High risk (P1+P2): {df['default_risk'].sum():,} ({df['default_risk'].mean()*100:.1f}%)")
        print(f"  Safe (P3+P4):      {(df['default_risk']==0).sum():,} ({(1-df['default_risk'].mean())*100:.1f}%)")

        return df

    def clean_sentinels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle -99999 sentinel values.

        WHY SENTINELS EXIST:
            -99999 means "this feature doesn't apply to this borrower"
            e.g. CC_utilization = -99999 means "borrower has no credit card"
            We CANNOT just replace with 0 or mean — that loses the signal.

        STRATEGY:
            For utilization columns: replace with -1 (flag: no product)
            For time-based columns: replace with 999 (flag: very long ago / never)
            For count columns: replace with 0 (no events)
            Always create a binary FLAG column: had_product_CC, had_product_PL etc.
        """
        print("\n[3/6] Cleaning sentinel values (-99999)...")

        sentinel_cols = [col for col in df.columns
                         if (df[col] == SENTINEL).any()]
        print(f"  Columns with sentinels: {sentinel_cols}")

        # Utilization columns — -99999 means no product
        for col in ["CC_utilization", "PL_utilization"]:
            if col in df.columns:
                flag_col = f"has_{col.replace('_utilization','').lower()}"
                df[flag_col] = (df[col] != SENTINEL).astype(int)
                df[col]      = df[col].replace(SENTINEL, 0)

        # Time-since columns — -99999 means never happened
        time_cols = [c for c in df.columns
                     if "time_since" in c.lower() or "age" in c.lower()]
        for col in time_cols:
            if col in df.columns and (df[col] == SENTINEL).any():
                # Replace with 9999 = "a very long time ago / never"
                df[col] = df[col].replace(SENTINEL, 9999)

        # Delinquency level columns
        deliq_level_cols = [c for c in df.columns
                            if "max_deliq" in c.lower() or "level_of_deliq" in c.lower()]
        for col in deliq_level_cols:
            if col in df.columns and (df[col] == SENTINEL).any():
                # -99999 = never been delinquent → replace with 0
                df[col] = df[col].replace(SENTINEL, 0)

        remaining = sum((df[col] == SENTINEL).any()
                        for col in df.columns
                        if df[col].dtype in [np.int64, np.float64])
        print(f"  Remaining sentinel columns after cleaning: {remaining}")

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features that carry business meaning.

        WHY FEATURE ENGINEERING MATTERS:
            Raw columns like "Total_TL" and "Tot_Active_TL" are less
            informative than "what % of loans are still active?"
            Derived ratios and flags often predict better than raw counts.
        """
        print("\n[4/6] Engineering features...")

        # ── Loan portfolio health ──────────────────────────────────────
        # Active ratio: what % of ever-opened loans are still active?
        # Low ratio = paid off most loans (good) OR very new borrower
        df["active_loan_ratio"] = np.where(
            df["Total_TL"] > 0,
            df["Tot_Active_TL"] / df["Total_TL"],
            0
        )

        # Stress indicator: missed payments as % of total loans
        df["missed_payment_ratio"] = np.where(
            df["Total_TL"] > 0,
            df["Tot_Missed_Pmnt"] / df["Total_TL"],
            0
        )

        # ── Credit history age ─────────────────────────────────────────
        # Long credit history = more data for assessment
        df["credit_history_months"] = df["Age_Oldest_TL"]

        # ── Income-based features ──────────────────────────────────────
        # CIBIL score per rupee of income — combines creditworthiness + capacity
        df["score_per_income_lakh"] = np.where(
            df["NETMONTHLYINCOME"] > 0,
            df["Credit_Score"] / (df["NETMONTHLYINCOME"] / 100000),
            0
        )

        # Income tier — Indian-context brackets
        df["income_tier"] = pd.cut(
            df["NETMONTHLYINCOME"],
            bins    = [0, 15000, 30000, 60000, 100000, float("inf")],
            labels  = ["very_low", "low", "middle", "upper_middle", "high"],
            right   = True
        ).astype(str)

        # ── Delinquency severity score ─────────────────────────────────
        # Combines frequency and recency of delinquencies
        df["delinquency_score"] = (
            df["num_times_delinquent"] * 2 +
            df["num_times_30p_dpd"]    * 3 +
            df["num_times_60p_dpd"]    * 5
        )

        # ── Recent activity flags ──────────────────────────────────────
        # Any new loans in last 6 months = increased credit seeking
        df["recently_active"] = (df["Total_TL_opened_L6M"] > 0).astype(int)

        # ── Loan type diversity ────────────────────────────────────────
        # Count distinct loan types used
        loan_type_cols = ["Auto_TL", "CC_TL", "Consumer_TL",
                          "Gold_TL", "Home_TL", "PL_TL"]
        available = [c for c in loan_type_cols if c in df.columns]
        if available:
            df["loan_type_diversity"] = (df[available] > 0).sum(axis=1)

        # ── CIBIL score bands ──────────────────────────────────────────
        # Standard Indian credit bureau score interpretation
        df["cibil_band"] = pd.cut(
            df["Credit_Score"],
            bins   = [0, 549, 649, 699, 749, 900],
            labels = ["poor", "fair", "good", "very_good", "excellent"],
            right  = True
        ).astype(str)

        print(f"  ✓ {df.shape[1]} total features after engineering")
        return df

    def clean_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardise categorical columns.
        Inconsistent casing and whitespace breaks groupby and encoding.
        """
        print("\n[5/6] Cleaning categorical columns...")

        cat_cols = {
            "MARITALSTATUS": lambda x: x.strip().title(),
            "EDUCATION":     lambda x: x.strip().upper(),
            "GENDER":        lambda x: x.strip().upper(),
            "last_prod_enq2":  lambda x: x.strip(),
            "first_prod_enq2": lambda x: x.strip(),
        }

        for col, clean_fn in cat_cols.items():
            if col in df.columns:
                before = df[col].nunique()
                df[col] = df[col].astype(str).apply(clean_fn)
                after   = df[col].nunique()
                if before != after:
                    print(f"  {col}: {before} → {after} unique values (consolidated)")

        # Standardise education values
        if "EDUCATION" in df.columns:
            edu_map = {
                "12TH":     "12TH",
                "SSC":      "SSC",
                "GRADUATE": "GRADUATE",
                "POST-GRADUATE": "POST_GRADUATE",
                "PROFESSIONAL": "PROFESSIONAL",
                "OTHERS":   "OTHERS",
            }
            df["EDUCATION"] = df["EDUCATION"].map(edu_map).fillna("OTHERS")

        return df

    def validate(self, df: pd.DataFrame) -> bool:
        """
        Data quality checks before saving.
        Returns True if all checks pass.
        """
        print("\n[6/6] Validating Silver data...")

        checks = []

        # Check 1: No nulls in key columns
        key_cols = ["PROSPECTID", "Credit_Score", "default_risk",
                    "NETMONTHLYINCOME", "AGE"]
        for col in key_cols:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                status     = "✓" if null_count == 0 else "✗"
                checks.append(null_count == 0)
                print(f"  {status} {col}: {null_count} nulls")

        # Check 2: CIBIL score in valid range
        if "Credit_Score" in df.columns:
            valid_score = df["Credit_Score"].between(300, 900).all()
            checks.append(valid_score)
            print(f"  {'✓' if valid_score else '✗'} Credit scores in 300-900 range")

        # Check 3: No duplicate PROSPECTID
        dupes = df["PROSPECTID"].duplicated().sum()
        checks.append(dupes == 0)
        print(f"  {'✓' if dupes == 0 else '✗'} Duplicate PROSPECTID: {dupes}")

        # Check 4: Target variable is binary
        if "default_risk" in df.columns:
            is_binary = set(df["default_risk"].unique()).issubset({0, 1})
            checks.append(is_binary)
            print(f"  {'✓' if is_binary else '✗'} Target is binary (0/1)")

        # Check 5: Positive income values
        if "NETMONTHLYINCOME" in df.columns:
            neg_income = (df["NETMONTHLYINCOME"] < 0).sum()
            checks.append(neg_income == 0)
            print(f"  {'✓' if neg_income == 0 else '✗'} Negative income: {neg_income}")

        all_passed = all(checks)
        print(f"\n  {'✓ All checks passed' if all_passed else '✗ Some checks failed'}")
        return all_passed

    def run(self):
        """Run the full Silver transformation pipeline."""
        print("=" * 60)
        print("  INDIA CREDIT RISK — SILVER TRANSFORMATION")
        print("=" * 60)

        # Load
        df_bank, df_cibil, df_loans = self.load_bronze()

        # Transform CIBIL (primary dataset)
        df_cibil = self.encode_target(df_cibil)
        df_cibil = self.clean_sentinels(df_cibil)
        df_cibil = self.clean_categorical(df_cibil)

        # Transform bank dataset
        df_bank = self.clean_sentinels(df_bank)

        # Join bank + CIBIL on PROSPECTID
        print("\n  Joining Internal Bank + CIBIL datasets...")
        df_master = df_cibil.merge(
            df_bank,
            on     = "PROSPECTID",
            how    = "inner",
            suffixes = ("_cibil", "_bank")
        )
        print(f"  Joined shape: {df_master.shape}")

        # Engineer features
        df_master = self.engineer_features(df_master)

        # Validate
        self.validate(df_master)

        # Save
        master_path = SILVER_DIR / "silver_master.parquet"
        loans_path  = SILVER_DIR / "silver_loan_applications.parquet"

        df_master.to_parquet(master_path, index=False)
        df_loans.to_parquet(loans_path, index=False)

        print(f"""
{'='*60}
✓ SILVER TRANSFORMATION COMPLETE

  Files saved to data/silver/:
  → silver_master.parquet          ({len(df_master):,} rows × {df_master.shape[1]} cols)
  → silver_loan_applications.parquet ({len(df_loans):,} rows)

  Target: default_risk
  High risk borrowers: {df_master['default_risk'].sum():,} ({df_master['default_risk'].mean()*100:.1f}%)
  Safe borrowers:      {(df_master['default_risk']==0).sum():,} ({(1-df_master['default_risk'].mean())*100:.1f}%)

  Next: python src/modeling/build_gold.py
{'='*60}
        """)

        return df_master


if __name__ == "__main__":
    transformer = SilverTransformer()
    transformer.run()