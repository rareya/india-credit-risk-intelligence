"""
ingest_kaggle.py  —  Bronze Layer
────────────────────────────────────────────────────────────────────
Reads all raw Kaggle files, adds metadata, saves as Parquet.
This is the Bronze layer — raw data preserved as-is.
No cleaning here. Only type validation and metadata tagging.

    python src/ingestion/ingest_kaggle.py
────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

RAW_DIR    = Path("data/bronze/kaggle_cibil")
OUTPUT_DIR = Path("data/bronze/parquet")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def add_metadata(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Add ingestion metadata to every raw file."""
    df["_source"]      = source
    df["_ingested_at"] = datetime.utcnow().isoformat()
    return df


def ingest_internal_bank() -> pd.DataFrame:
    """
    Internal Bank Dataset — trade line features per borrower.
    case_study1.xlsx and Internal_Bank_Dataset.xlsx are identical.
    We read one and document both.

    Key columns:
        PROSPECTID      — join key to CIBIL data
        Total_TL        — total trade lines (loans) ever opened
        Tot_Active_TL   — currently active loans
        Tot_Missed_Pmnt — total missed payments EVER
        Gold_TL         — gold loans (very Indian — proxy for rural/semi-urban)
        Home_TL         — home loans
        PL_TL           — personal loans
        Age_Oldest_TL   — months since oldest loan (credit history age)
        Age_Newest_TL   — months since newest loan
    """
    print("\n[1/4] Ingesting Internal Bank Dataset...")
    path = RAW_DIR / "case_study1.xlsx"
    df   = pd.read_excel(path)

    print(f"  Raw shape: {df.shape}")
    print(f"  PROSPECTID range: {df['PROSPECTID'].min()} — {df['PROSPECTID'].max()}")
    print(f"  Duplicate PROSPECTID: {df['PROSPECTID'].duplicated().sum()}")

    df = add_metadata(df, "kaggle_internal_bank")
    df.to_parquet(OUTPUT_DIR / "bronze_internal_bank.parquet", index=False)
    print(f"  ✓ Saved → bronze_internal_bank.parquet")
    return df


def ingest_cibil() -> pd.DataFrame:
    """
    External CIBIL Dataset — credit bureau features + TARGET VARIABLE.
    case_study2.xlsx and External_Cibil_Dataset.xlsx are identical.

    Key columns:
        PROSPECTID      — join key
        Credit_Score    — CIBIL score 300-900 (Indian credit bureau)
        Approved_Flag   — TARGET: P1=high risk, P2=med, P3=low, P4=safest
        AGE             — borrower age
        GENDER          — M/F
        MARITALSTATUS   — Married/Single/etc
        EDUCATION       — SSC/12TH/GRADUATE/etc
        NETMONTHLYINCOME— monthly income in INR
        num_times_delinquent — total delinquencies
        num_times_30p_dpd    — times 30+ days past due
        num_times_60p_dpd    — times 60+ days past due
        Tot_Missed_Pmnt — missed payments
        CC_utilization  — credit card utilization (-99999 = no CC)
        PL_utilization  — personal loan utilization

    NOTE on -99999 values:
        These are sentinel values meaning "not applicable"
        e.g. CC_utilization = -99999 means borrower has no credit card
        We handle these in the Silver layer, NOT here
    """
    print("\n[2/4] Ingesting External CIBIL Dataset...")
    path = RAW_DIR / "case_study2.xlsx"
    df   = pd.read_excel(path)

    print(f"  Raw shape: {df.shape}")
    print(f"  Target distribution (Approved_Flag):")
    print(df["Approved_Flag"].value_counts().to_string())
    print(f"  Credit Score range: {df['Credit_Score'].min()} — {df['Credit_Score'].max()}")
    print(f"  Income range: ₹{df['NETMONTHLYINCOME'].min():,} — ₹{df['NETMONTHLYINCOME'].max():,}")

    df = add_metadata(df, "kaggle_cibil_external")
    df.to_parquet(OUTPUT_DIR / "bronze_cibil_external.parquet", index=False)
    print(f"  ✓ Saved → bronze_cibil_external.parquet")
    return df


def ingest_loan_applications() -> pd.DataFrame:
    """
    Loan Application Dataset (train_modified.csv).
    Fintech loan disbursement data — different from CIBIL case studies.

    Key columns:
        Disbursed       — TARGET: 1=loan disbursed, 0=rejected
        Loan_Amount_Applied  — loan amount requested in INR
        Loan_Tenure_Applied  — tenure in years
        Monthly_Income       — monthly income in INR
        Existing_EMI         — existing EMI obligations
        age                  — borrower age (float — slight obfuscation)
    """
    print("\n[3/4] Ingesting Loan Application Dataset...")
    path = RAW_DIR / "train_modified.csv"
    df   = pd.read_csv(path)

    print(f"  Raw shape: {df.shape}")
    print(f"  Disbursement rate: {df['Disbursed'].mean()*100:.1f}%")
    print(f"  Loan amount range: ₹{df['Loan_Amount_Applied'].min():,.0f} — ₹{df['Loan_Amount_Applied'].max():,.0f}")

    df = add_metadata(df, "kaggle_loan_applications")
    df.to_parquet(OUTPUT_DIR / "bronze_loan_applications.parquet", index=False)
    print(f"  ✓ Saved → bronze_loan_applications.parquet")
    return df


def ingest_data_dictionary() -> pd.DataFrame:
    """
    Features and Target Description — data dictionary.
    Not used in modeling but essential for documentation.
    """
    print("\n[4/4] Ingesting Data Dictionary...")
    path = RAW_DIR / "Features_Target_Description.xlsx"
    df   = pd.read_excel(path)

    print(f"  {len(df)} feature descriptions")

    df = add_metadata(df, "kaggle_data_dictionary")
    df.to_parquet(OUTPUT_DIR / "bronze_data_dictionary.parquet", index=False)
    print(f"  ✓ Saved → bronze_data_dictionary.parquet")
    return df


def validate_join_integrity(df_bank: pd.DataFrame, df_cibil: pd.DataFrame):
    """
    Verify that the two main datasets can be joined cleanly on PROSPECTID.
    This is critical — if IDs don't match, our joined dataset is meaningless.
    """
    print("\n━━━ Join Integrity Check ━━━")

    bank_ids  = set(df_bank["PROSPECTID"])
    cibil_ids = set(df_cibil["PROSPECTID"])

    common         = bank_ids & cibil_ids
    only_in_bank   = bank_ids - cibil_ids
    only_in_cibil  = cibil_ids - bank_ids

    print(f"  Bank dataset IDs:    {len(bank_ids):,}")
    print(f"  CIBIL dataset IDs:   {len(cibil_ids):,}")
    print(f"  Matching IDs:        {len(common):,}")
    print(f"  Only in bank:        {len(only_in_bank):,}")
    print(f"  Only in CIBIL:       {len(only_in_cibil):,}")

    if len(common) == len(bank_ids) == len(cibil_ids):
        print("  ✓ Perfect join — all IDs match")
    elif len(common) > 0:
        print(f"  ⚠ Partial join — {len(common):,} records will match")
    else:
        print("  ✗ No matching IDs — check data sources")


def main():
    print("=" * 60)
    print("  INDIA CREDIT RISK — BRONZE INGESTION")
    print("  Reading raw files → Parquet")
    print("=" * 60)

    df_bank  = ingest_internal_bank()
    df_cibil = ingest_cibil()
    df_loans = ingest_loan_applications()
    df_dict  = ingest_data_dictionary()

    validate_join_integrity(df_bank, df_cibil)

    print(f"""
{'='*60}
✓ BRONZE INGESTION COMPLETE

  Files saved to data/bronze/parquet/:
  → bronze_internal_bank.parquet    ({len(df_bank):,} rows)
  → bronze_cibil_external.parquet   ({len(df_cibil):,} rows)
  → bronze_loan_applications.parquet({len(df_loans):,} rows)
  → bronze_data_dictionary.parquet  ({len(df_dict):,} rows)

  Next: python src/transformation/transform_silver.py
{'='*60}
    """)


if __name__ == "__main__":
    main()