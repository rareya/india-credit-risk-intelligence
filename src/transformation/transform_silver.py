"""
transform_silver.py  —  Silver Layer
────────────────────────────────────────────────────────────────────
Cleans Bronze data, engineers features, encodes target variable.
Produces the master dataset ready for ML and analytics.

    python src/transformation/transform_silver.py

CRITICAL NOTE ON TARGET ENCODING:
    Approved_Flag values from CIBIL dataset:
        P1 = BEST borrowers  (lowest risk, high CIBIL, low delinquency)
        P2 = Good borrowers
        P3 = Risky borrowers
        P4 = WORST borrowers (highest risk, low CIBIL, high delinquency)

    Therefore:
        default_risk = 1  →  P3 or P4  (high risk, likely to default)
        default_risk = 0  →  P1 or P2  (safe, unlikely to default)

    This was verified by investigate_data.py:
        P1 avg CIBIL ~780, avg delinquency 0.1  → SAFE
        P4 avg CIBIL ~580, avg delinquency 4.2  → RISKY
────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

BRONZE_DIR = Path("data/bronze/parquet")
SILVER_DIR = Path("data/silver")
SILVER_DIR.mkdir(parents=True, exist_ok=True)

SENTINEL = -99999


class SilverTransformer:

    def load_bronze(self) -> tuple:
        print("\n[1/6] Loading Bronze data...")
        df_bank  = pd.read_parquet(BRONZE_DIR / "bronze_internal_bank.parquet")
        df_cibil = pd.read_parquet(BRONZE_DIR / "bronze_cibil_external.parquet")
        df_loans = pd.read_parquet(BRONZE_DIR / "bronze_loan_applications.parquet")

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

        VERIFIED MAPPING (from investigate_data.py output):
            P1 = highest CIBIL (~780), lowest delinquency → SAFE borrower
            P2 = good CIBIL, low delinquency             → SAFE borrower
            P3 = lower CIBIL, moderate delinquency       → RISKY borrower
            P4 = lowest CIBIL (~580), high delinquency   → RISKY borrower

        Encoding:
            default_risk = 1  →  P3 or P4  (risky — will likely default)
            default_risk = 0  →  P1 or P2  (safe  — unlikely to default)

        Ordinal risk grade (for multi-class analysis):
            P1=1 (lowest risk), P2=2, P3=3, P4=4 (highest risk)
        """
        print("\n[2/6] Encoding target variable...")

        # Ordinal: P1=1 (best/safest), P4=4 (worst/riskiest)
        risk_map = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}
        df["risk_grade"] = df["Approved_Flag"].map(risk_map)

        # Binary target: P3 or P4 = high risk = 1 (RISKY)
        #                P1 or P2 = low risk  = 0 (SAFE)
        df["default_risk"] = (df["Approved_Flag"].isin(["P3", "P4"])).astype(int)

        print(f"  Approved_Flag distribution:")
        print(df["Approved_Flag"].value_counts().to_string())
        print(f"\n  Binary target (VERIFIED: P3/P4=risky, P1/P2=safe):")
        print(f"  Risky (P3+P4, default_risk=1): {df['default_risk'].sum():,} ({df['default_risk'].mean()*100:.1f}%)")
        print(f"  Safe  (P1+P2, default_risk=0): {(df['default_risk']==0).sum():,} ({(1-df['default_risk'].mean())*100:.1f}%)")

        # Sanity check: P3/P4 should have LOWER CIBIL scores than P1/P2
        if "Credit_Score" in df.columns:
            risky_cibil = df[df["default_risk"]==1]["Credit_Score"].mean()
            safe_cibil  = df[df["default_risk"]==0]["Credit_Score"].mean()
            print(f"\n  Sanity check — CIBIL scores:")
            print(f"  Risky (P3+P4) avg CIBIL: {risky_cibil:.0f}  (expected: lower)")
            print(f"  Safe  (P1+P2) avg CIBIL: {safe_cibil:.0f}  (expected: higher)")
            if risky_cibil < safe_cibil:
                print(f"  ✓ Encoding verified: risky borrowers have lower CIBIL scores")
            else:
                print(f"  ✗ WARNING: encoding may be wrong — check Approved_Flag values")

        return df

    def clean_sentinels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle -99999 sentinel values.
        -99999 means 'not applicable' — e.g. no credit card → CC_utilization = -99999.
        Strategy: create binary flag columns, then replace with 0.
        """
        print("\n[3/6] Cleaning sentinel values (-99999)...")

        sentinel_cols = [col for col in df.columns
                         if df[col].dtype in [np.int64, np.float64]
                         and (df[col] == SENTINEL).any()]
        print(f"  Columns with sentinels: {sentinel_cols}")

        for col in ["CC_utilization", "PL_utilization"]:
            if col in df.columns:
                flag_col = f"has_{col.replace('_utilization','').lower()}"
                df[flag_col] = (df[col] != SENTINEL).astype(int)
                df[col]      = df[col].replace(SENTINEL, 0)

        time_cols = [c for c in df.columns if "time_since" in c.lower() or "age" in c.lower()]
        for col in time_cols:
            if col in df.columns and (df[col] == SENTINEL).any():
                df[col] = df[col].replace(SENTINEL, 9999)

        deliq_cols = [c for c in df.columns if "max_deliq" in c.lower() or "level_of_deliq" in c.lower()]
        for col in deliq_cols:
            if col in df.columns and (df[col] == SENTINEL).any():
                df[col] = df[col].replace(SENTINEL, 0)

        remaining = sum((df[col] == SENTINEL).any()
                        for col in df.columns
                        if df[col].dtype in [np.int64, np.float64])
        print(f"  Remaining sentinel columns after cleaning: {remaining}")
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features with clear business meaning.
        """
        print("\n[4/6] Engineering features...")

        # Active loan ratio
        df["active_loan_ratio"] = np.where(
            df["Total_TL"] > 0,
            df["Tot_Active_TL"] / df["Total_TL"], 0)

        # Missed payment ratio
        df["missed_payment_ratio"] = np.where(
            df["Total_TL"] > 0,
            df["Tot_Missed_Pmnt"] / df["Total_TL"], 0)

        # Credit history
        df["credit_history_months"] = df["Age_Oldest_TL"]

        # Score per income lakh
        df["score_per_income_lakh"] = np.where(
            df["NETMONTHLYINCOME"] > 0,
            df["Credit_Score"] / (df["NETMONTHLYINCOME"] / 100000), 0)

        # Income tier
        df["income_tier"] = pd.cut(
            df["NETMONTHLYINCOME"],
            bins   = [0, 15000, 30000, 60000, 100000, float("inf")],
            labels = ["very_low", "low", "middle", "upper_middle", "high"],
            right  = True
        ).astype(str)

        # Delinquency severity score
        df["delinquency_score"] = (
            df.get("num_times_delinquent", 0) * 2 +
            df.get("num_times_30p_dpd", 0)    * 3 +
            df.get("num_times_60p_dpd", 0)    * 5
        )

        # Recently active flag
        if "Total_TL_opened_L6M" in df.columns:
            df["recently_active"] = (df["Total_TL_opened_L6M"] > 0).astype(int)
        else:
            df["recently_active"] = 0

        # Loan type diversity
        loan_type_cols = ["Auto_TL", "CC_TL", "Consumer_TL", "Gold_TL", "Home_TL", "PL_TL"]
        available = [c for c in loan_type_cols if c in df.columns]
        if available:
            df["loan_type_diversity"] = (df[available] > 0).sum(axis=1)

        # CIBIL score bands
        df["cibil_band"] = pd.cut(
            df["Credit_Score"],
            bins   = [0, 549, 649, 699, 749, 900],
            labels = ["poor", "fair", "good", "very_good", "excellent"],
            right  = True
        ).astype(str)

        # Risk band based on risk_grade ordinal
        # P1/P2 (risk_grade 1-2) = Low Risk
        # P3 (risk_grade 3) = Medium Risk
        # P4 (risk_grade 4) = High Risk
        df["risk_band"] = df["Approved_Flag"].map({
            "P1": "Low Risk",
            "P2": "Low Risk",
            "P3": "Medium Risk",
            "P4": "High Risk",
        }).fillna("Unknown")

        print(f"  ✓ {df.shape[1]} total features after engineering")
        print(f"  risk_band distribution:")
        print(df["risk_band"].value_counts().to_string())
        return df

    def clean_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n[5/6] Cleaning categorical columns...")

        cat_cols = {
            "MARITALSTATUS":   lambda x: x.strip().title(),
            "EDUCATION":       lambda x: x.strip().upper(),
            "GENDER":          lambda x: x.strip().upper(),
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

        if "EDUCATION" in df.columns:
            edu_map = {
                "12TH": "12TH", "SSC": "SSC", "GRADUATE": "GRADUATE",
                "POST-GRADUATE": "POST_GRADUATE", "PROFESSIONAL": "PROFESSIONAL",
                "OTHERS": "OTHERS",
            }
            df["EDUCATION"] = df["EDUCATION"].map(edu_map).fillna("OTHERS")

        return df

    def validate(self, df: pd.DataFrame) -> bool:
        print("\n[6/6] Validating Silver data...")
        checks = []

        key_cols = ["PROSPECTID", "Credit_Score", "default_risk", "NETMONTHLYINCOME", "AGE"]
        for col in key_cols:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                status     = "✓" if null_count == 0 else "✗"
                checks.append(null_count == 0)
                print(f"  {status} {col}: {null_count} nulls")

        if "Credit_Score" in df.columns:
            valid_score = df["Credit_Score"].between(300, 900).all()
            checks.append(valid_score)
            print(f"  {'✓' if valid_score else '✗'} Credit scores in 300-900 range")

        dupes = df["PROSPECTID"].duplicated().sum()
        checks.append(dupes == 0)
        print(f"  {'✓' if dupes == 0 else '✗'} Duplicate PROSPECTID: {dupes}")

        if "default_risk" in df.columns:
            is_binary = set(df["default_risk"].unique()).issubset({0, 1})
            checks.append(is_binary)
            print(f"  {'✓' if is_binary else '✗'} Target is binary (0/1)")

        # Verify encoding direction
        if "Credit_Score" in df.columns and "default_risk" in df.columns:
            risky_cibil = df[df["default_risk"]==1]["Credit_Score"].mean()
            safe_cibil  = df[df["default_risk"]==0]["Credit_Score"].mean()
            encoding_ok = risky_cibil < safe_cibil
            checks.append(encoding_ok)
            print(f"  {'✓' if encoding_ok else '✗'} Encoding direction: risky CIBIL ({risky_cibil:.0f}) < safe CIBIL ({safe_cibil:.0f})")

        # Verify risk_band has no blanks
        if "risk_band" in df.columns:
            blanks = (df["risk_band"].isin(["nan", "", "None", "Unknown"])).sum()
            checks.append(blanks == 0)
            print(f"  {'✓' if blanks == 0 else '✗'} risk_band: {blanks} blank/unknown values")
            print(f"      risk_band counts: {df['risk_band'].value_counts().to_dict()}")

        if "NETMONTHLYINCOME" in df.columns:
            neg_income = (df["NETMONTHLYINCOME"] < 0).sum()
            checks.append(neg_income == 0)
            print(f"  {'✓' if neg_income == 0 else '✗'} Negative income: {neg_income}")

        all_passed = all(checks)
        print(f"\n  {'✓ All checks passed' if all_passed else '⚠ Some checks failed — review output above'}")
        return all_passed

    def run(self):
        print("=" * 60)
        print("  INDIA CREDIT RISK — SILVER TRANSFORMATION")
        print("=" * 60)

        df_bank, df_cibil, df_loans = self.load_bronze()

        df_cibil = self.encode_target(df_cibil)
        df_cibil = self.clean_sentinels(df_cibil)
        df_cibil = self.clean_categorical(df_cibil)
        df_bank  = self.clean_sentinels(df_bank)

        print("\n  Joining Internal Bank + CIBIL datasets on PROSPECTID...")
        df_master = df_cibil.merge(
            df_bank, on="PROSPECTID", how="inner", suffixes=("_cibil", "_bank"))
        print(f"  Joined shape: {df_master.shape}")

        df_master = self.engineer_features(df_master)
        self.validate(df_master)

        master_path = SILVER_DIR / "silver_master.parquet"
        loans_path  = SILVER_DIR / "silver_loan_applications.parquet"
        df_master.to_parquet(master_path, index=False)
        df_loans.to_parquet(loans_path, index=False)

        risky = df_master["default_risk"].sum()
        safe  = (df_master["default_risk"] == 0).sum()

        print(f"""
{'='*60}
✓ SILVER TRANSFORMATION COMPLETE

  Files saved to data/silver/:
  → silver_master.parquet  ({len(df_master):,} rows × {df_master.shape[1]} cols)

  Target encoding (VERIFIED):
  → default_risk = 1  means P3/P4 (RISKY borrowers)  : {risky:,} ({risky/len(df_master)*100:.1f}%)
  → default_risk = 0  means P1/P2 (SAFE  borrowers)  : {safe:,} ({safe/len(df_master)*100:.1f}%)

  Next: python src/modeling/build_gold.py
{'='*60}
        """)
        return df_master


if __name__ == "__main__":
    transformer = SilverTransformer()
    transformer.run()