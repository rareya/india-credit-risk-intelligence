"""
investigate_target.py
Quick investigation to understand what Approved_Flag actually means.
Run: python investigate_target.py
"""

import pandas as pd
import numpy as np

df = pd.read_parquet("data/silver/silver_master.parquet")

print("=" * 60)
print("INVESTIGATING Approved_Flag")
print("=" * 60)

print("\n1. Value counts:")
print(df["Approved_Flag"].value_counts())

print("\n2. CIBIL score by Approved_Flag:")
print(df.groupby("Approved_Flag")["Credit_Score"].describe().round(1))

print("\n3. Income by Approved_Flag:")
print(df.groupby("Approved_Flag")["NETMONTHLYINCOME"].describe().round(0))

print("\n4. Delinquency by Approved_Flag:")
print(df.groupby("Approved_Flag")[
    ["num_times_delinquent", "num_times_60p_dpd", "Tot_Missed_Pmnt"]
].mean().round(2))

print("\n5. Age by Approved_Flag:")
print(df.groupby("Approved_Flag")["AGE"].describe().round(1))

print("\n6. Are P1 borrowers BETTER or WORSE credit profiles?")
p1 = df[df["Approved_Flag"] == "P1"]
p4 = df[df["Approved_Flag"] == "P4"]
print(f"\n  P1 avg CIBIL:       {p1['Credit_Score'].mean():.0f}")
print(f"  P4 avg CIBIL:       {p4['Credit_Score'].mean():.0f}")
print(f"  P1 avg income:      ₹{p1['NETMONTHLYINCOME'].mean():,.0f}")
print(f"  P4 avg income:      ₹{p4['NETMONTHLYINCOME'].mean():,.0f}")
print(f"  P1 avg delinquency: {p1['num_times_delinquent'].mean():.1f}")
print(f"  P4 avg delinquency: {p4['num_times_delinquent'].mean():.1f}")
print(f"  P1 missed payments: {p1['Tot_Missed_Pmnt'].mean():.1f}")
print(f"  P4 missed payments: {p4['Tot_Missed_Pmnt'].mean():.1f}")
print(f"  P1 total loans:     {p1['Total_TL'].mean():.1f}")
print(f"  P4 total loans:     {p4['Total_TL'].mean():.1f}")

print("\n7. What does the data dictionary say about Approved_Flag?")
try:
    dd = pd.read_parquet("data/bronze/parquet/bronze_data_dictionary.parquet")
    flag_rows = dd[dd.apply(
        lambda r: r.astype(str).str.contains("Approved|Flag|P1|P2|P3|P4",
                                              case=False).any(), axis=1
    )]
    print(flag_rows.to_string())
except Exception as e:
    print(f"  Could not load data dictionary: {e}")