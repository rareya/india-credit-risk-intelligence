"""
portfolio_metrics.py — Core Portfolio Business Metrics
────────────────────────────────────────────────────────────────────
Computes approval rate, default rate, expected loss,
risk band distribution, and segment contribution.

All functions accept a DataFrame (silver master or enriched)
and return clean metrics ready for dashboard display.
────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
from typing import Optional

# ── Risk band thresholds ──────────────────────────────────────────────────────
RISK_BANDS = {
    "Low Risk":    (0.00, 0.25),
    "Medium Risk": (0.25, 0.50),
    "High Risk":   (0.50, 1.00),
}

LGD = 0.45      # Loss Given Default — 45% for unsecured Indian loans
EAD_MULT = 12   # EAD proxy = 12 × monthly income


# ── Core metrics ──────────────────────────────────────────────────────────────
def compute_portfolio_summary(df: pd.DataFrame,
                               pred_proba: Optional[np.ndarray] = None) -> dict:
    """
    Compute top-level portfolio KPIs.

    Returns a dict of metrics ready for st.metric() display.
    """
    n = len(df)
    n_defaults = int(df["default_risk"].sum()) if "default_risk" in df.columns else 0
    default_rate = n_defaults / n if n > 0 else 0

    metrics = {
        "total_borrowers":    n,
        "total_defaults":     n_defaults,
        "default_rate_pct":   round(default_rate * 100, 2),
        "safe_borrowers":     n - n_defaults,
    }

    # Predicted PD metrics (when model scores available)
    if pred_proba is not None:
        metrics["avg_predicted_pd_pct"]  = round(pred_proba.mean() * 100, 2)
        metrics["high_risk_count"]       = int((pred_proba >= 0.50).sum())
        metrics["high_risk_pct"]         = round((pred_proba >= 0.50).mean() * 100, 2)
        metrics["medium_risk_count"]     = int(((pred_proba >= 0.25) & (pred_proba < 0.50)).sum())
        metrics["low_risk_count"]        = int((pred_proba < 0.25).sum())

    # Income-based Expected Loss proxy
    if "NETMONTHLYINCOME" in df.columns and pred_proba is not None:
        ead = df["NETMONTHLYINCOME"].fillna(df["NETMONTHLYINCOME"].median()) * EAD_MULT
        el  = pred_proba * LGD * ead
        metrics["total_ead_cr"]            = round(ead.sum() / 1e7, 2)   # crores
        metrics["expected_loss_cr"]        = round(el.sum() / 1e7, 2)    # crores
        metrics["expected_loss_rate_pct"]  = round((el.sum() / ead.sum()) * 100, 2)

    return metrics


def risk_band_distribution(pred_proba: np.ndarray) -> pd.DataFrame:
    """
    Classify borrowers into Low / Medium / High Risk bands.
    Returns counts and percentages.
    """
    n = len(pred_proba)
    rows = []
    for band, (lo, hi) in RISK_BANDS.items():
        mask  = (pred_proba >= lo) & (pred_proba < hi)
        count = int(mask.sum())
        rows.append({
            "Risk Band":    band,
            "Count":        count,
            "% Portfolio":  round(count / n * 100, 1),
            "Avg PD":       round(pred_proba[mask].mean() * 100, 1) if count > 0 else 0,
        })
    return pd.DataFrame(rows)


def segment_default_rates(df: pd.DataFrame, col: str,
                           bins=None, labels=None) -> pd.DataFrame:
    """
    Compute default rate, count, and risk contribution by segment.

    Args:
        df:     DataFrame with default_risk column
        col:    Column to segment by
        bins:   Optional bin edges for numeric columns
        labels: Optional bin labels
    """
    work = df.copy()
    if bins is not None:
        work[f"_{col}_band"] = pd.cut(work[col], bins=bins, labels=labels)
        group_col = f"_{col}_band"
    else:
        group_col = col

    total_defaults = work["default_risk"].sum()
    agg = (work.groupby(group_col, observed=True)
               .agg(
                   Count=("default_risk", "count"),
                   Defaults=("default_risk", "sum"),
               )
               .reset_index()
               .rename(columns={group_col: "Segment"}))

    agg["Default Rate %"]       = (agg["Defaults"] / agg["Count"] * 100).round(1)
    agg["Risk Contribution %"]  = (agg["Defaults"] / total_defaults * 100).round(1)
    agg["% of Portfolio"]       = (agg["Count"] / len(work) * 100).round(1)
    agg["Segment"]              = agg["Segment"].astype(str)

    return agg.sort_values("Default Rate %", ascending=False).reset_index(drop=True)


def expected_loss_by_segment(df: pd.DataFrame, pred_proba: np.ndarray,
                              segment_col: str, bins=None, labels=None) -> pd.DataFrame:
    """
    Compute Expected Loss (EL = PD × LGD × EAD) per segment.
    EAD proxy = 12 × monthly income.
    """
    work = df.copy()
    work["_pd"]  = pred_proba
    work["_ead"] = (work["NETMONTHLYINCOME"].fillna(work["NETMONTHLYINCOME"].median())
                    * EAD_MULT)
    work["_el"]  = work["_pd"] * LGD * work["_ead"]

    if bins is not None:
        work[f"_{segment_col}_band"] = pd.cut(work[segment_col], bins=bins, labels=labels)
        group_col = f"_{segment_col}_band"
    else:
        group_col = segment_col

    agg = (work.groupby(group_col, observed=True)
               .agg(
                   Count=("_pd", "count"),
                   Avg_PD=("_pd", "mean"),
                   Total_EAD=("_ead", "sum"),
                   Total_EL=("_el", "sum"),
               )
               .reset_index()
               .rename(columns={group_col: "Segment"}))

    agg["Avg PD %"]          = (agg["Avg_PD"] * 100).round(1)
    agg["EAD (₹ Cr)"]        = (agg["Total_EAD"] / 1e7).round(2)
    agg["Expected Loss (₹ Cr)"] = (agg["Total_EL"] / 1e7).round(2)
    agg["EL Rate %"]         = (agg["Total_EL"] / agg["Total_EAD"] * 100).round(2)
    agg["Segment"]           = agg["Segment"].astype(str)

    return agg[["Segment","Count","Avg PD %","EAD (₹ Cr)","Expected Loss (₹ Cr)","EL Rate %"]].sort_values("EL Rate %", ascending=False)


def top_risky_segments(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Cross-cut: find the N highest-risk segments across key dimensions.
    Returns a leaderboard table.
    """
    rows = []

    dims = [
        ("Age Group", "AGE", [18,25,35,45,55,75], ["18-25","26-35","36-45","46-55","55+"]),
        ("Income Band", "NETMONTHLYINCOME", [0,10000,20000,35000,60000,500000], ["<10k","10-20k","20-35k","35-60k","60k+"]),
    ]

    for dim_name, col, bins, labels in dims:
        if col not in df.columns: continue
        seg = segment_default_rates(df, col, bins=bins, labels=labels)
        for _, row in seg.iterrows():
            rows.append({
                "Dimension":         dim_name,
                "Segment":           row["Segment"],
                "Count":             row["Count"],
                "Default Rate %":    row["Default Rate %"],
                "Risk Contribution %": row["Risk Contribution %"],
            })

    result = (pd.DataFrame(rows)
                .sort_values("Default Rate %", ascending=False)
                .head(n)
                .reset_index(drop=True))
    result.index += 1
    return result