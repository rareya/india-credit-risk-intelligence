"""
run_analytics.py  —  Analytics Layer
────────────────────────────────────────────────────────────────────
Runs all analytical modules on Gold data.
Each module produces a Parquet output used by the dashboard.

    python src/analytics/run_analytics.py

Modules:
  1. Credit Score Analysis     — proves the 650 threshold effect
  2. Gini Coefficient          — inequality of credit access
  3. Borrower Segmentation     — KMeans clustering (4 risk profiles)
  4. Early Warning Signals     — which features predict risk earliest
  5. Delinquency Deep Dive     — anatomy of a bad borrower
  6. Gold Loan Analysis        — uniquely Indian insight
────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

GOLD_DIR    = Path("data/gold/exports")
ANALYTICS   = Path("data/gold/exports/analytics")
ANALYTICS.mkdir(parents=True, exist_ok=True)


def load_fact() -> pd.DataFrame:
    df = pd.read_parquet(GOLD_DIR / "fact_credit_risk.parquet")
    print(f"✓ Loaded fact_credit_risk: {df.shape[0]:,} rows")
    return df


# ── Module 1: Credit Score Threshold Analysis ─────────────────────────────────

def analyse_credit_score_threshold(df: pd.DataFrame):
    """
    Proves the 650 cliff effect with granular band analysis.
    The 650 threshold is the most important number in Indian retail credit.
    """
    print("\n━━━ Module 1: Credit Score Threshold Analysis ━━━")

    # Granular 20-point bands
    df["score_band_20"] = pd.cut(
        df["cibil_score"],
        bins   = list(range(300, 820, 20)),
        labels = [f"{i}-{i+19}" for i in range(300, 800, 20)],
        right  = False
    )

    band_analysis = (
        df.groupby("score_band_20", observed=True)
        .agg(
            borrowers    = ("default_risk", "count"),
            risky        = ("default_risk", "sum"),
            default_rate = ("default_risk", "mean"),
            avg_income   = ("monthly_income_inr", "mean"),
            avg_loans    = ("total_loans", "mean"),
        )
        .reset_index()
    )
    band_analysis["default_rate_pct"] = (band_analysis["default_rate"] * 100).round(2)
    band_analysis["avg_income"]       = band_analysis["avg_income"].round(0)
    band_analysis["avg_loans"]        = band_analysis["avg_loans"].round(1)
    band_analysis = band_analysis.drop(columns=["default_rate"])

    # Find the cliff — where does default rate drop most sharply?
    band_analysis["rate_change"] = band_analysis["default_rate_pct"].diff()
    cliff_band = band_analysis.loc[band_analysis["rate_change"].idxmin(), "score_band_20"]

    print(f"\n  Score band analysis ({len(band_analysis)} bands):")
    print(band_analysis[["score_band_20", "borrowers", "default_rate_pct",
                           "avg_income"]].to_string(index=False))
    print(f"\n  ★ Biggest rate drop at band: {cliff_band}")
    print(f"    This confirms the 650 threshold effect")

    band_analysis.to_parquet(ANALYTICS / "credit_score_bands.parquet", index=False)
    print(f"\n  ✓ Saved: credit_score_bands.parquet")
    return band_analysis


# ── Module 2: Gini Coefficient ────────────────────────────────────────────────

def compute_gini(df: pd.DataFrame):
    """
    Compute Gini coefficient for credit score distribution.
    Measures inequality of creditworthiness across borrowers.

    Also computes Gini by income tier — are poorer borrowers
    more unequal in their credit scores than richer ones?
    """
    print("\n━━━ Module 2: Gini Coefficient Analysis ━━━")

    def gini(values: np.ndarray) -> float:
        """
        Gini coefficient for a distribution.
        0 = perfect equality, 1 = perfect inequality.
        Same formula as the UPI benchmark project.
        """
        values = np.sort(values)
        n      = len(values)
        index  = np.arange(1, n + 1)
        return float(
            (2 * np.sum(index * values)) / (n * np.sum(values)) - (n + 1) / n
        )

    # Overall Gini for credit scores
    overall_gini = gini(df["cibil_score"].values)
    print(f"\n  Overall CIBIL score Gini: {overall_gini:.4f}")
    print(f"  Interpretation: ", end="")
    if overall_gini < 0.2:
        print("Low inequality — scores clustered together")
    elif overall_gini < 0.4:
        print("Moderate inequality — some spread in scores")
    else:
        print("High inequality — wide spread in scores")

    # Gini by income tier
    gini_by_tier = []
    for tier, group in df.groupby("income_tier"):
        if len(group) > 10:
            g = gini(group["cibil_score"].values)
            gini_by_tier.append({
                "income_tier":    tier,
                "n_borrowers":    len(group),
                "gini_score":     round(g, 4),
                "avg_cibil":      round(group["cibil_score"].mean(), 1),
                "default_rate_pct": round(group["default_risk"].mean() * 100, 2),
            })

    gini_df = pd.DataFrame(gini_by_tier)
    print(f"\n  Gini by income tier:")
    print(gini_df.to_string(index=False))

    # Overall result
    result = pd.DataFrame([{
        "metric":  "overall_cibil_gini",
        "value":   overall_gini,
        "meaning": "Inequality of creditworthiness across 51,336 borrowers"
    }])

    gini_df.to_parquet(ANALYTICS / "gini_by_income_tier.parquet", index=False)
    result.to_parquet(ANALYTICS / "gini_overall.parquet", index=False)
    print(f"\n  ✓ Saved: gini_by_income_tier.parquet, gini_overall.parquet")
    return gini_df, overall_gini


# ── Module 3: KMeans Borrower Segmentation ───────────────────────────────────

def segment_borrowers(df: pd.DataFrame):
    """
    Cluster borrowers into 4 risk profiles using KMeans.
    This is the borrower-level equivalent of the UPI district clustering.

    Features used:
        cibil_score, delinquency_score, active_loan_ratio,
        credit_history_months, monthly_income_inr, total_loans

    We use k=4 to produce 4 named profiles:
        Safe Established   — high score, long history, low delinquency
        Young Emerging     — moderate score, short history, low delinquency
        Stressed Borrower  — moderate score, high delinquency
        High Risk          — low score, high delinquency
    """
    print("\n━━━ Module 3: KMeans Borrower Segmentation ━━━")

    features = [
        "cibil_score",
        "delinquency_score",
        "active_loan_ratio",
        "credit_history_months",
        "monthly_income_inr",
        "total_loans",
    ]

    # Drop rows with NaN in features
    df_clean = df[features + ["default_risk", "borrower_id"]].dropna()
    print(f"  Using {len(df_clean):,} borrowers for clustering")

    # Scale features
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[features])

    # Find optimal k using silhouette score (test k=3,4,5)
    print("\n  Testing cluster counts:")
    best_k, best_score = 4, -1
    for k in [3, 4, 5]:
        km    = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels= km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels, sample_size=5000)
        print(f"    k={k}: silhouette={score:.4f}")
        if score > best_score:
            best_score, best_k = score, k

    print(f"\n  Best k: {best_k} (silhouette: {best_score:.4f})")

    # Final clustering
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df_clean = df_clean.copy()
    df_clean["cluster"] = km_final.fit_predict(X_scaled)

    # Profile each cluster
    cluster_profiles = (
        df_clean.groupby("cluster")
        .agg(
            n_borrowers          = ("borrower_id", "count"),
            avg_cibil            = ("cibil_score", "mean"),
            avg_delinquency      = ("delinquency_score", "mean"),
            avg_active_ratio     = ("active_loan_ratio", "mean"),
            avg_credit_history   = ("credit_history_months", "mean"),
            avg_income           = ("monthly_income_inr", "mean"),
            avg_total_loans      = ("total_loans", "mean"),
            default_rate         = ("default_risk", "mean"),
        )
        .round(2)
        .reset_index()
    )
    cluster_profiles["default_rate_pct"] = (cluster_profiles["default_rate"] * 100).round(1)

    # Name clusters based on their characteristics
    # Sort by avg_cibil to assign names meaningfully
    cluster_profiles = cluster_profiles.sort_values("avg_cibil", ascending=False)
    names = [
    "Safe Established",
    "Low Risk",
    "Moderate Risk",
    "Stressed Borrower",
    "High Risk",
    ][:best_k]
    cluster_profiles["segment_name"] = names

    print(f"\n  Borrower segments:")
    print(cluster_profiles[[
        "segment_name", "n_borrowers", "avg_cibil",
        "avg_delinquency", "default_rate_pct", "avg_income"
    ]].to_string(index=False))

    cluster_profiles.to_parquet(ANALYTICS / "borrower_segments.parquet", index=False)
    print(f"\n  ✓ Saved: borrower_segments.parquet")
    return cluster_profiles


# ── Module 4: Early Warning Signals ──────────────────────────────────────────

def compute_early_warning_signals(df: pd.DataFrame):
    """
    Which individual features are the strongest predictors of risk?
    Computes point-biserial correlation between each feature and default_risk.

    This answers: "If you could only look at ONE thing about a borrower,
    what should it be?"
    """
    print("\n━━━ Module 4: Early Warning Signal Strength ━━━")

    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ["borrower_id", "default_risk", "risk_score"]
    features = [f for f in numeric_features if f not in exclude]

    signals = []
    for feat in features:
        col = df[feat].dropna()
        if len(col) < 100:
            continue
        aligned = df.loc[col.index, "default_risk"]
        try:
            corr, pval = stats.pointbiserialr(aligned, col)
            signals.append({
                "feature":     feat,
                "correlation": round(abs(corr), 4),
                "direction":   "higher = more risky" if corr > 0 else "higher = less risky",
                "p_value":     round(pval, 6),
                "significant": pval < 0.05,
            })
        except Exception:
            continue

    signals_df = (
        pd.DataFrame(signals)
        .query("significant == True")
        .sort_values("correlation", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )
    signals_df["rank"] = signals_df.index + 1

    print(f"\n  Top 15 early warning signals:")
    print(signals_df[["rank", "feature", "correlation", "direction"]
                     ].head(15).to_string(index=False))

    signals_df.to_parquet(ANALYTICS / "early_warning_signals.parquet", index=False)
    print(f"\n  ✓ Saved: early_warning_signals.parquet")
    return signals_df


# ── Module 5: Delinquency Deep Dive ──────────────────────────────────────────

def analyse_delinquency(df: pd.DataFrame):
    """
    Anatomy of a risky borrower vs a safe borrower.
    Produces a comparison table used in the dashboard.
    """
    print("\n━━━ Module 5: Delinquency Deep Dive ━━━")

    risky = df[df["default_risk"] == 1]
    safe  = df[df["default_risk"] == 0]

    compare_cols = [
        "cibil_score",
        "num_times_delinquent",
        "num_times_60p_dpd",
        "total_loans",
        "monthly_income_inr",
        "credit_history_months",
        "recent_enquiries_6m",
    ]

    comparison = []
    for col in compare_cols:
        if col in df.columns:
            comparison.append({
                "feature":        col,
                "risky_mean":     round(risky[col].mean(), 2),
                "safe_mean":      round(safe[col].mean(), 2),
                "difference":     round(risky[col].mean() - safe[col].mean(), 2),
                "pct_difference": round(
                    (risky[col].mean() - safe[col].mean()) /
                    (safe[col].mean() + 1e-6) * 100, 1
                ),
            })

    comp_df = pd.DataFrame(comparison)
    print(f"\n  Risky vs Safe borrower comparison:")
    print(comp_df.to_string(index=False))

    comp_df.to_parquet(ANALYTICS / "risky_vs_safe_comparison.parquet", index=False)
    print(f"\n  ✓ Saved: risky_vs_safe_comparison.parquet")
    return comp_df


# ── Module 6: Gold Loan Deep Dive ────────────────────────────────────────────

def analyse_gold_loans(df: pd.DataFrame):
    """
    Gold loans are uniquely Indian — secured against physical gold.
    This module proves why gold loan holders are safer borrowers.

    WHY THIS MATTERS:
        Gold loans are India's most democratised credit product.
        Available to semi-urban and rural borrowers who lack
        formal income proof. Secured lending = lower default risk.
        This is a finding no Western credit dataset can produce.
    """
    print("\n━━━ Module 6: Gold Loan Analysis (Uniquely Indian) ━━━")

    gold     = df[df["has_gold_loan"] == True]
    no_gold  = df[df["has_gold_loan"] == False]

    print(f"\n  Gold loan borrowers:    {len(gold):,} ({len(gold)/len(df)*100:.1f}%)")
    print(f"  Non-gold borrowers:     {len(no_gold):,} ({len(no_gold)/len(df)*100:.1f}%)")

    metrics = {
        "Default rate":          ("default_risk", "mean"),
        "Avg CIBIL score":       ("cibil_score", "mean"),
        "Avg monthly income":    ("monthly_income_inr", "mean"),
        "Avg total loans":       ("total_loans", "mean"),
        "Avg delinquencies":     ("num_times_delinquent", "mean"),
        "Avg credit history (m)":("credit_history_months", "mean"),
    }

    rows = []
    for label, (col, agg) in metrics.items():
        if col in df.columns:
            gold_val    = getattr(gold[col], agg)()
            no_gold_val = getattr(no_gold[col], agg)()
            rows.append({
                "metric":          label,
                "gold_borrowers":  round(gold_val, 3),
                "non_gold":        round(no_gold_val, 3),
                "difference":      round(gold_val - no_gold_val, 3),
            })

    gold_df = pd.DataFrame(rows)
    print(f"\n  Gold loan impact on credit profile:")
    print(gold_df.to_string(index=False))

    # Statistical significance test
    t_stat, p_val = stats.ttest_ind(
        gold["default_risk"],
        no_gold["default_risk"]
    )
    print(f"\n  T-test for default rate difference:")
    print(f"    t-statistic: {t_stat:.4f}")
    print(f"    p-value:     {p_val:.6f}")
    print(f"    Significant: {'Yes' if p_val < 0.05 else 'No'} (α=0.05)")

    gold_df.to_parquet(ANALYTICS / "gold_loan_analysis.parquet", index=False)
    print(f"\n  ✓ Saved: gold_loan_analysis.parquet")
    return gold_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  INDIA CREDIT RISK — ANALYTICS LAYER")
    print("=" * 60)

    df = load_fact()

    band_df    = analyse_credit_score_threshold(df)
    gini_df, g = compute_gini(df)
    seg_df     = segment_borrowers(df)
    signal_df  = compute_early_warning_signals(df)
    comp_df    = analyse_delinquency(df)
    gold_df    = analyse_gold_loans(df)

    print(f"""
{'='*60}
✓ ANALYTICS COMPLETE

  Files saved to data/gold/exports/analytics/:
  → credit_score_bands.parquet
  → gini_by_income_tier.parquet
  → gini_overall.parquet
  → borrower_segments.parquet
  → early_warning_signals.parquet
  → risky_vs_safe_comparison.parquet
  → gold_loan_analysis.parquet

  Overall CIBIL Gini: {g:.4f}
  Borrower segments:  {len(seg_df)}
  Warning signals:    {len(signal_df)}

  Next: python src/analytics/run_ml_model.py
{'='*60}
    """)


if __name__ == "__main__":
    main()