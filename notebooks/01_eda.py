"""
explore_data.py
────────────────────────────────────────────────────────────────────
Run this ONCE to understand what's in each downloaded file.
Prints shape, columns, dtypes, sample rows, and null counts.

    python explore_data.py

We read this output before writing any ingestion code.
────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import os
from pathlib import Path

DATA_DIR = Path("data/bronze/kaggle_cibil")

FILES = {
    "case_study1.xlsx":              {"type": "excel"},
    "case_study2.xlsx":              {"type": "excel"},
    "External_Cibil_Dataset.xlsx":   {"type": "excel"},
    "Internal_Bank_Dataset.xlsx":    {"type": "excel"},
    "Features_Target_Description.xlsx": {"type": "excel"},
    "train_modified.csv":            {"type": "csv"},
    "test_modified.csv":             {"type": "csv"},
    "Unseen_Dataset.xlsx":           {"type": "excel"},
}


def explore_file(filename: str, filetype: str):
    path = DATA_DIR / filename
    print(f"\n{'='*65}")
    print(f"FILE: {filename}")
    print(f"{'='*65}")

    try:
        if filetype == "csv":
            df = pd.read_csv(path, nrows=5000)
        else:
            # Try reading first sheet
            df = pd.read_excel(path, nrows=5000)

        print(f"Shape:   {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"\nColumns ({len(df.columns)}):")
        for col in df.columns:
            null_pct = df[col].isnull().mean() * 100
            dtype    = str(df[col].dtype)
            sample   = str(df[col].dropna().iloc[0]) if df[col].dropna().shape[0] > 0 else "N/A"
            sample   = sample[:40] + "..." if len(sample) > 40 else sample
            print(f"  {col:<45} {dtype:<12} nulls:{null_pct:>5.1f}%  sample: {sample}")

        print(f"\nFirst 3 rows:")
        print(df.head(3).to_string())

    except Exception as e:
        print(f"  ✗ Could not read: {e}")
        # Try reading all sheets for Excel
        if filetype == "excel":
            try:
                xl = pd.ExcelFile(path)
                print(f"  Sheets: {xl.sheet_names}")
                for sheet in xl.sheet_names[:3]:
                    df = pd.read_excel(path, sheet_name=sheet, nrows=3)
                    print(f"\n  Sheet '{sheet}': {df.shape}")
                    print(df.head(2).to_string())
            except Exception as e2:
                print(f"  ✗ Excel read also failed: {e2}")


def main():
    print("INDIA CREDIT RISK — DATA EXPLORATION")
    print("Reading all files in data/bronze/kaggle_cibil/")

    for filename, meta in FILES.items():
        explore_file(filename, meta["type"])

    print(f"\n{'='*65}")
    print("EXPLORATION COMPLETE")
    print("Share this output — we use it to design the ingestion pipeline")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()