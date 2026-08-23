"""
run_analytics.py — Analytics Layer
===================================

Runs the analytical layer on the Gold dataset.

Purpose
-------
This layer converts the validated Gold dataset into statistically
defensible business insights for:

1. Credit score threshold / breakpoint analysis
2. CIBIL distribution inequality analysis
3. Borrower segmentation using KMeans
4. Univariate risk association / early-warning signals
5. Risky vs safe borrower profiling
6. Gold-loan portfolio analysis
7. Data quality / coverage diagnostics

IMPORTANT ANALYTICAL PRINCIPLES
--------------------------------
- default_risk is the canonical target:
      1 = P3/P4 = high risk
      0 = P1/P2 = low risk

- These analyses are descriptive / inferential.
- Correlation with default_risk is NOT presented as causal.
- KMeans clusters are unsupervised segments, not ML predictions.
- Gold-loan analysis does NOT claim causality.
- The 650 CIBIL level is investigated rather than assumed to be
  universally predictive.

Run:
    python src/analytics/run_analytics.py

Outputs:
    data/gold/exports/analytics/
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")


# =============================================================================
# PATHS
# =============================================================================

GOLD_DIR = Path("data/gold/exports")
ANALYTICS = GOLD_DIR / "analytics"

ANALYTICS.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONSTANTS
# =============================================================================

RANDOM_STATE = 42

TARGET = "default_risk"

REQUIRED_COLUMNS = [
    "borrower_id",
    "default_risk",
    "risk_grade",
    "cibil_score",
    "monthly_income_inr",
    "total_loans",
    "delinquency_score",
    "active_loan_ratio",
    "credit_history_months",
    "income_tier",
    "cibil_band",
]


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def safe_round(value, digits=2):
    """Return rounded numeric value while safely handling NaN."""
    if pd.isna(value):
        return np.nan
    return round(float(value), digits)


def proportion_ci(successes, n, confidence=0.95):
    """
    Wilson confidence interval for a proportion.

    More appropriate than a simple normal approximation, especially
    when the number of observations is relatively small.
    """
    if n == 0:
        return np.nan, np.nan

    z = stats.norm.ppf(1 - (1 - confidence) / 2)

    p = successes / n

    denominator = 1 + (z ** 2) / n

    centre = (
        p
        + (z ** 2) / (2 * n)
    ) / denominator

    margin = (
        z
        * np.sqrt(
            (p * (1 - p) / n)
            + (z ** 2) / (4 * n ** 2)
        )
        / denominator
    )

    return centre - margin, centre + margin


def cohens_d(group_a, group_b):
    """
    Cohen's d for two independent groups.
    """
    a = pd.Series(group_a).dropna().astype(float)
    b = pd.Series(group_b).dropna().astype(float)

    if len(a) < 2 or len(b) < 2:
        return np.nan

    pooled_std = np.sqrt(
        (
            (len(a) - 1) * a.var(ddof=1)
            + (len(b) - 1) * b.var(ddof=1)
        )
        / (len(a) + len(b) - 2)
    )

    if pooled_std == 0:
        return np.nan

    return (a.mean() - b.mean()) / pooled_std


def validate_fact(df: pd.DataFrame):
    """
    Validate the Gold fact table before analytics.
    """

    print("\n━━━ Validating Gold fact table ━━━")

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "fact_credit_risk is missing required columns:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

    if df["borrower_id"].duplicated().any():
        raise ValueError(
            "fact_credit_risk contains duplicate borrower_id values."
        )

    if df[TARGET].isna().any():
        raise ValueError(
            "default_risk contains null values."
        )

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError(
            "default_risk must contain only 0 and 1."
        )

    if ((df["cibil_score"] < 300) |
        (df["cibil_score"] > 900)).any():
        raise ValueError(
            "cibil_score contains values outside 300–900."
        )

    print(f"  ✓ Rows: {len(df):,}")
    print("  ✓ Required columns present")
    print("  ✓ borrower_id unique")
    print("  ✓ default_risk binary and non-null")
    print("  ✓ CIBIL scores within 300–900")


# =============================================================================
# LOAD GOLD FACT
# =============================================================================

def load_fact() -> pd.DataFrame:

    path = GOLD_DIR / "fact_credit_risk.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"Gold fact table not found:\n{path}\n\n"
            "Run build_gold.py first."
        )

    df = pd.read_parquet(path)

    print(f"✓ Loaded fact_credit_risk: "
          f"{df.shape[0]:,} rows × {df.shape[1]} columns")

    validate_fact(df)

    return df


# =============================================================================
# MODULE 1
# CREDIT SCORE THRESHOLD / BREAKPOINT ANALYSIS
# =============================================================================

def analyse_credit_score_threshold(df: pd.DataFrame):

    """
    Investigate the relationship between CIBIL score and default risk.

    IMPORTANT:
    This does NOT assume that 650 is universally the correct cutoff.

    We produce:

    1. Fine-grained score bands
    2. Explicit <650 vs >=650 comparison
    3. 650-adjacent analysis
    4. Statistical significance
    5. Confidence intervals
    6. Risk-rate changes between neighbouring bands
    """

    print("\n━━━ Module 1: Credit Score Threshold Analysis ━━━")

    work = df[
        [
            "cibil_score",
            "default_risk",
            "monthly_income_inr",
            "total_loans",
        ]
    ].dropna()

    # -------------------------------------------------------------------------
    # 20-point score bands
    # -------------------------------------------------------------------------

    bins = list(range(300, 921, 20))

    labels = [
        f"{start}-{start + 19}"
        for start in range(300, 901, 20)
    ]

    work["score_band_20"] = pd.cut(
        work["cibil_score"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )

    band_analysis = (
        work.groupby("score_band_20", observed=True)
        .agg(
            borrowers=("default_risk", "count"),
            risky=("default_risk", "sum"),
            default_rate=("default_risk", "mean"),
            avg_income=("monthly_income_inr", "mean"),
            avg_loans=("total_loans", "mean"),
            avg_cibil=("cibil_score", "mean"),
        )
        .reset_index()
    )

    band_analysis["default_rate_pct"] = (
        band_analysis["default_rate"] * 100
    ).round(2)

    band_analysis["avg_income"] = (
        band_analysis["avg_income"].round(0)
    )

    band_analysis["avg_loans"] = (
        band_analysis["avg_loans"].round(1)
    )

    band_analysis["avg_cibil"] = (
        band_analysis["avg_cibil"].round(1)
    )

    band_analysis["ci_lower_pct"] = np.nan
    band_analysis["ci_upper_pct"] = np.nan

    for idx, row in band_analysis.iterrows():

        low, high = proportion_ci(
            int(row["risky"]),
            int(row["borrowers"]),
        )

        band_analysis.loc[idx, "ci_lower_pct"] = round(
            low * 100, 2
        )

        band_analysis.loc[idx, "ci_upper_pct"] = round(
            high * 100, 2
        )

    band_analysis["rate_change_pct_points"] = (
        band_analysis["default_rate_pct"].diff().round(2)
    )

    # -------------------------------------------------------------------------
    # Explicit 650 analysis
    # -------------------------------------------------------------------------

    work["above_650"] = (
        work["cibil_score"] >= 650
    )

    below = work.loc[
        work["above_650"] == False,
        "default_risk",
    ]

    above = work.loc[
        work["above_650"] == True,
        "default_risk",
    ]

    below_rate = below.mean()
    above_rate = above.mean()

    rate_difference = (
        below_rate - above_rate
    )

    odds_ratio = np.nan

    # 2x2 contingency table
    contingency = pd.crosstab(
        work["above_650"],
        work["default_risk"],
    )

    if contingency.shape == (2, 2):

        try:
            odds_ratio, fisher_p = stats.fisher_exact(
                contingency
            )
        except Exception:
            fisher_p = np.nan

    else:
        fisher_p = np.nan

    # Chi-square
    try:
        chi2, chi_p, _, _ = stats.chi2_contingency(
            contingency
        )
    except Exception:
        chi2 = np.nan
        chi_p = np.nan

    threshold_result = pd.DataFrame([{

        "threshold": 650,

        "borrowers_below_650": len(below),
        "default_rate_below_650_pct": round(
            below_rate * 100, 2
        ),

        "borrowers_at_or_above_650": len(above),
        "default_rate_at_or_above_650_pct": round(
            above_rate * 100, 2
        ),

        "absolute_rate_difference_pct_points": round(
            rate_difference * 100, 2
        ),

        "chi_square": safe_round(chi2, 4),
        "chi_square_p_value": safe_round(chi_p, 8),

        "fisher_odds_ratio": safe_round(
            odds_ratio,
            4,
        ),

        "fisher_p_value": safe_round(
            fisher_p,
            8,
        ),

        "statistically_significant": (
            chi_p < 0.05
            if not pd.isna(chi_p)
            else False
        ),

        "interpretation":
            "Borrowers below 650 exhibit a higher observed "
            "default rate than borrowers at or above 650."
            if below_rate > above_rate
            else
            "The observed default rate is not higher below 650."

    }])

    # -------------------------------------------------------------------------
    # Local 640–660 analysis
    # -------------------------------------------------------------------------

    local = work[
        work["cibil_score"].between(
            630,
            670,
            inclusive="both",
        )
    ].copy()

    local["score_bucket"] = pd.cut(
        local["cibil_score"],
        bins=[630, 640, 650, 660, 670],
        labels=[
            "630-639",
            "640-649",
            "650-659",
            "660-669",
        ],
        right=False,
    )

    local_analysis = (
        local.groupby("score_bucket", observed=True)
        .agg(
            borrowers=("default_risk", "count"),
            risky=("default_risk", "sum"),
            default_rate=("default_risk", "mean"),
        )
        .reset_index()
    )

    local_analysis["default_rate_pct"] = (
        local_analysis["default_rate"] * 100
    ).round(2)

    local_analysis = local_analysis.drop(
        columns=["default_rate"]
    )

    print("\n  CIBIL threshold analysis:")
    print(
        threshold_result.to_string(index=False)
    )

    print("\n  20-point score bands:")
    print(
        band_analysis[
            [
                "score_band_20",
                "borrowers",
                "default_rate_pct",
                "ci_lower_pct",
                "ci_upper_pct",
            ]
        ].to_string(index=False)
    )

    print("\n  Local analysis around 650:")
    print(
        local_analysis.to_string(index=False)
    )

    band_analysis.to_parquet(
        ANALYTICS / "credit_score_bands.parquet",
        index=False,
    )

    threshold_result.to_parquet(
        ANALYTICS / "credit_score_650_analysis.parquet",
        index=False,
    )

    local_analysis.to_parquet(
        ANALYTICS / "credit_score_650_local.parquet",
        index=False,
    )

    print(
        "\n  ✓ Saved:"
        "\n    credit_score_bands.parquet"
        "\n    credit_score_650_analysis.parquet"
        "\n    credit_score_650_local.parquet"
    )

    return (
        band_analysis,
        threshold_result,
        local_analysis,
    )


# =============================================================================
# MODULE 2
# CIBIL DISTRIBUTION GINI
# =============================================================================

def compute_gini(df: pd.DataFrame):

    """
    Compute the Gini coefficient of the CIBIL score distribution.

    IMPORTANT:
    This measures inequality / dispersion of the score distribution.

    It does NOT measure:
        - inequality of credit access
        - probability of default
        - lending fairness

    Therefore we label the metric correctly.
    """

    print("\n━━━ Module 2: CIBIL Distribution Inequality ━━━")

    def gini(values):

        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]

        if len(values) == 0:
            return np.nan

        if np.any(values < 0):
            raise ValueError(
                "Gini implementation requires non-negative values."
            )

        total = values.sum()

        if total == 0:
            return 0.0

        values = np.sort(values)

        n = len(values)

        index = np.arange(1, n + 1)

        return float(
            (
                2 * np.sum(index * values)
            )
            /
            (
                n * total
            )
            -
            (n + 1) / n
        )

    overall_gini = gini(
        df["cibil_score"].values
    )

    print(
        f"\n  Overall CIBIL distribution Gini: "
        f"{overall_gini:.4f}"
    )

    gini_by_tier = []

    for tier, group in df.groupby(
        "income_tier",
        dropna=False,
    ):

        scores = group["cibil_score"].dropna()

        if len(scores) < 10:
            continue

        g = gini(scores.values)

        gini_by_tier.append({

            "income_tier": tier,

            "n_borrowers": len(group),

            "gini_cibil": round(g, 4),

            "avg_cibil": round(
                scores.mean(),
                1,
            ),

            "median_cibil": round(
                scores.median(),
                1,
            ),

            "cibil_std": round(
                scores.std(),
                1,
            ),

            "default_rate_pct": round(
                group["default_risk"].mean() * 100,
                2,
            ),

        })

    gini_df = pd.DataFrame(
        gini_by_tier
    )

    overall_result = pd.DataFrame([{

        "metric":
            "overall_cibil_distribution_gini",

        "value":
            overall_gini,

        "interpretation":
            "Dispersion of CIBIL scores across borrowers; "
            "not a direct measure of credit-access inequality."

    }])

    print("\n  Gini by income tier:")
    print(
        gini_df.to_string(index=False)
    )

    gini_df.to_parquet(
        ANALYTICS / "gini_by_income_tier.parquet",
        index=False,
    )

    overall_result.to_parquet(
        ANALYTICS / "gini_overall.parquet",
        index=False,
    )

    print(
        "\n  ✓ Saved:"
        "\n    gini_by_income_tier.parquet"
        "\n    gini_overall.parquet"
    )

    return gini_df, overall_gini


# =============================================================================
# MODULE 3
# KMEANS BORROWER SEGMENTATION
# =============================================================================

def segment_borrowers(df: pd.DataFrame):

    """
    Unsupervised borrower segmentation.

    KMeans is NOT trained using default_risk.

    default_risk is used only AFTER clustering to profile
    the observed risk rate of each segment.

    Features:
        cibil_score
        delinquency_score
        active_loan_ratio
        credit_history_months
        monthly_income_inr
        total_loans
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

    missing = [
        f for f in features
        if f not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing clustering features: {missing}"
        )

    clean = df[
        features + [
            "default_risk",
            "borrower_id",
        ]
    ].dropna().copy()

    print(
        f"  Using {len(clean):,} borrowers"
    )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        clean[features]
    )

    # -------------------------------------------------------------------------
    # Evaluate k
    # -------------------------------------------------------------------------

    print("\n  Testing cluster counts:")

    cluster_scores = []

    for k in [3, 4, 5, 6]:

        km = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=20,
        )

        labels = km.fit_predict(
            X_scaled
        )

        score = silhouette_score(
            X_scaled,
            labels,
            sample_size=min(
                5000,
                len(clean),
            ),
            random_state=RANDOM_STATE,
        )

        cluster_scores.append({

            "k": k,

            "silhouette_score":
                round(score, 4),

        })

        print(
            f"    k={k}: "
            f"silhouette={score:.4f}"
        )

    scores_df = pd.DataFrame(
        cluster_scores
    )

    best_row = scores_df.loc[
        scores_df["silhouette_score"].idxmax()
    ]

    best_k = int(best_row["k"])

    best_score = float(
        best_row["silhouette_score"]
    )

    print(
        f"\n  Selected k={best_k}"
        f" (highest silhouette={best_score:.4f})"
    )

    # -------------------------------------------------------------------------
    # Final clustering
    # -------------------------------------------------------------------------

    final_model = KMeans(
        n_clusters=best_k,
        random_state=RANDOM_STATE,
        n_init=20,
    )

    clean["cluster"] = (
        final_model.fit_predict(
            X_scaled
        )
    )

    # -------------------------------------------------------------------------
    # Cluster profile
    # -------------------------------------------------------------------------

    profiles = (
        clean.groupby("cluster")
        .agg(

            n_borrowers=(
                "borrower_id",
                "count",
            ),

            avg_cibil=(
                "cibil_score",
                "mean",
            ),

            avg_delinquency=(
                "delinquency_score",
                "mean",
            ),

            avg_active_ratio=(
                "active_loan_ratio",
                "mean",
            ),

            avg_credit_history=(
                "credit_history_months",
                "mean",
            ),

            avg_income=(
                "monthly_income_inr",
                "mean",
            ),

            avg_total_loans=(
                "total_loans",
                "mean",
            ),

            default_rate=(
                "default_risk",
                "mean",
            ),

        )
        .reset_index()
    )

    profiles["default_rate_pct"] = (
        profiles["default_rate"] * 100
    ).round(2)

    # -------------------------------------------------------------------------
    # Data-driven labels
    #
    # Do NOT pretend KMeans knows "Safe" or "High Risk".
    # We create descriptive labels based on observed profiles.
    # -------------------------------------------------------------------------

    profiles = profiles.sort_values(
        "default_rate",
        ascending=True,
    ).reset_index(drop=True)

    label_count = len(profiles)

    if label_count == 3:

        labels = [
            "Lower Risk Segment",
            "Moderate Risk Segment",
            "Higher Risk Segment",
        ]

    elif label_count == 4:

        labels = [
            "Lower Risk Segment",
            "Emerging Segment",
            "Stressed Segment",
            "Higher Risk Segment",
        ]

    elif label_count == 5:

        labels = [
            "Lowest Risk Segment",
            "Lower Risk Segment",
            "Moderate Risk Segment",
            "Stressed Segment",
            "Highest Risk Segment",
        ]

    else:

        labels = [
            f"Segment {i + 1}"
            for i in range(label_count)
        ]

    profiles["segment_name"] = labels

    # -------------------------------------------------------------------------
    # Restore cluster IDs
    # -------------------------------------------------------------------------

    clean = clean.merge(
        profiles[
            [
                "cluster",
                "segment_name",
            ]
        ],
        on="cluster",
        how="left",
    )

    profiles = profiles[
        [
            "cluster",
            "segment_name",
            "n_borrowers",
            "avg_cibil",
            "avg_delinquency",
            "avg_active_ratio",
            "avg_credit_history",
            "avg_income",
            "avg_total_loans",
            "default_rate_pct",
        ]
    ]

    profiles[
        [
            "avg_cibil",
            "avg_delinquency",
            "avg_active_ratio",
            "avg_credit_history",
            "avg_income",
            "avg_total_loans",
        ]
    ] = profiles[
        [
            "avg_cibil",
            "avg_delinquency",
            "avg_active_ratio",
            "avg_credit_history",
            "avg_income",
            "avg_total_loans",
        ]
    ].round(2)

    print("\n  Borrower segments:")
    print(
        profiles[
            [
                "segment_name",
                "n_borrowers",
                "avg_cibil",
                "avg_delinquency",
                "default_rate_pct",
                "avg_income",
            ]
        ].to_string(index=False)
    )

    scores_df.to_parquet(
        ANALYTICS / "cluster_selection.parquet",
        index=False,
    )

    profiles.to_parquet(
        ANALYTICS / "borrower_segments.parquet",
        index=False,
    )

    clean[
        [
            "borrower_id",
            "cluster",
            "segment_name",
        ]
    ].to_parquet(
        ANALYTICS / "borrower_segment_assignments.parquet",
        index=False,
    )

    print(
        "\n  ✓ Saved:"
        "\n    cluster_selection.parquet"
        "\n    borrower_segments.parquet"
        "\n    borrower_segment_assignments.parquet"
    )

    return profiles


# =============================================================================
# MODULE 4
# UNIVARIATE EARLY WARNING SIGNALS
# =============================================================================

def compute_early_warning_signals(df: pd.DataFrame):

    """
    Measure univariate association between numeric variables
    and default_risk.

    IMPORTANT:
    These are associations, not predictive model feature importance.

    The purpose is exploratory:
        "Which variables are most strongly associated with observed risk?"
    """

    print(
        "\n━━━ Module 4: Early Warning Signal Strength ━━━"
    )

    numeric_features = (
        df.select_dtypes(
            include=[np.number]
        )
        .columns
        .tolist()
    )

    exclude = {
        "borrower_id",
        "default_risk",
        "risk_score",
    }

    features = [
        f for f in numeric_features
        if f not in exclude
    ]

    signals = []

    for feature in features:

        temp = df[
            [
                feature,
                "default_risk",
            ]
        ].dropna()

        if len(temp) < 100:
            continue

        if temp[feature].nunique() <= 1:
            continue

        try:

            corr, p_value = (
                stats.pointbiserialr(
                    temp["default_risk"],
                    temp[feature],
                )
            )

            signals.append({

                "feature": feature,

                "correlation_signed":
                    round(float(corr), 4),

                "association_strength":
                    round(abs(float(corr)), 4),

                "direction":
                    (
                        "higher = more risky"
                        if corr > 0
                        else "higher = less risky"
                    ),

                "p_value":
                    round(float(p_value), 10),

                "sample_size":
                    len(temp),

                "significant":
                    bool(p_value < 0.05),

            })

        except Exception:
            continue

    signals_df = pd.DataFrame(
        signals
    )

    if signals_df.empty:
        raise ValueError(
            "No usable numeric risk signals were found."
        )

    # Multiple testing correction: Benjamini-Hochberg
    signals_df = signals_df.sort_values(
        "p_value"
    ).reset_index(drop=True)

    m = len(signals_df)

    signals_df["bh_adjusted_p"] = (
        signals_df["p_value"]
        * m
        / (signals_df.index + 1)
    ).clip(upper=1)

    # Enforce monotonicity
    signals_df["bh_adjusted_p"] = (
        signals_df["bh_adjusted_p"]
        .iloc[::-1]
        .cummin()
        .iloc[::-1]
        .values
    )

    signals_df["significant_after_bh"] = (
        signals_df["bh_adjusted_p"] < 0.05
    )

    signals_df = signals_df.sort_values(
        "association_strength",
        ascending=False,
    ).reset_index(drop=True)

    signals_df["rank"] = (
        signals_df.index + 1
    )

    print("\n  Top 15 associated features:")

    print(
        signals_df[
            [
                "rank",
                "feature",
                "association_strength",
                "direction",
                "bh_adjusted_p",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    signals_df.to_parquet(
        ANALYTICS / "early_warning_signals.parquet",
        index=False,
    )

    print(
        "\n  ✓ Saved: "
        "early_warning_signals.parquet"
    )

    return signals_df


# =============================================================================
# MODULE 5
# DELINQUENCY DEEP DIVE
# =============================================================================

def analyse_delinquency(df: pd.DataFrame):

    """
    Compare observed characteristics of high-risk and low-risk borrowers.

    Produces:
        - means
        - absolute differences
        - percentage differences
        - effect size
    """

    print(
        "\n━━━ Module 5: Delinquency Deep Dive ━━━"
    )

    risky = df[
        df["default_risk"] == 1
    ]

    safe = df[
        df["default_risk"] == 0
    ]

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

        if col not in df.columns:
            continue

        risky_values = risky[col].dropna()
        safe_values = safe[col].dropna()

        if (
            len(risky_values) == 0
            or len(safe_values) == 0
        ):
            continue

        risky_mean = risky_values.mean()
        safe_mean = safe_values.mean()

        difference = (
            risky_mean - safe_mean
        )

        pct_difference = (
            difference
            /
            (abs(safe_mean) + 1e-9)
            * 100
        )

        comparison.append({

            "feature": col,

            "risky_n":
                len(risky_values),

            "safe_n":
                len(safe_values),

            "risky_mean":
                round(risky_mean, 2),

            "safe_mean":
                round(safe_mean, 2),

            "difference":
                round(difference, 2),

            "pct_difference":
                round(pct_difference, 1),

            "cohens_d":
                round(
                    cohens_d(
                        risky_values,
                        safe_values,
                    ),
                    4,
                ),

        })

    comparison_df = pd.DataFrame(
        comparison
    )

    print(
        "\n  High-risk vs low-risk comparison:"
    )

    print(
        comparison_df.to_string(
            index=False
        )
    )

    comparison_df.to_parquet(
        ANALYTICS / "risky_vs_safe_comparison.parquet",
        index=False,
    )

    print(
        "\n  ✓ Saved: "
        "risky_vs_safe_comparison.parquet"
    )

    return comparison_df


# =============================================================================
# MODULE 6
# GOLD LOAN ANALYSIS
# =============================================================================

def analyse_gold_loans(df: pd.DataFrame):

    """
    Compare gold-loan and non-gold-loan borrowers.

    IMPORTANT:
    This is an observational comparison.

    We do NOT conclude:
        "gold loans cause lower default."

    We only report:
        "gold-loan borrowers have a different observed risk profile."
    """

    print(
        "\n━━━ Module 6: Gold Loan Analysis ━━━"
    )

    if "has_gold_loan" not in df.columns:
        raise ValueError(
            "has_gold_loan is missing from Gold fact table."
        )

    gold = df[
        df["has_gold_loan"] == True
    ]

    no_gold = df[
        df["has_gold_loan"] == False
    ]

    print(
        f"\n  Gold-loan borrowers: "
        f"{len(gold):,} "
        f"({len(gold) / len(df) * 100:.1f}%)"
    )

    print(
        f"  Non-gold borrowers: "
        f"{len(no_gold):,} "
        f"({len(no_gold) / len(df) * 100:.1f}%)"
    )

    metrics = [

        (
            "default_rate",
            "default_risk",
            "mean",
        ),

        (
            "avg_cibil",
            "cibil_score",
            "mean",
        ),

        (
            "avg_monthly_income",
            "monthly_income_inr",
            "mean",
        ),

        (
            "avg_total_loans",
            "total_loans",
            "mean",
        ),

        (
            "avg_delinquencies",
            "num_times_delinquent",
            "mean",
        ),

        (
            "avg_credit_history_months",
            "credit_history_months",
            "mean",
        ),

    ]

    rows = []

    for metric_name, column, aggregation in metrics:

        if column not in df.columns:
            continue

        gold_values = gold[column].dropna()
        no_gold_values = no_gold[column].dropna()

        if (
            len(gold_values) == 0
            or len(no_gold_values) == 0
        ):
            continue

        gold_value = getattr(
            gold_values,
            aggregation,
        )()

        no_gold_value = getattr(
            no_gold_values,
            aggregation,
        )()

        rows.append({

            "metric": metric_name,

            "gold_value":
                round(float(gold_value), 4),

            "non_gold_value":
                round(float(no_gold_value), 4),

            "difference":
                round(
                    float(
                        gold_value
                        - no_gold_value
                    ),
                    4,
                ),

            "cohens_d":
                round(
                    cohens_d(
                        gold_values,
                        no_gold_values,
                    ),
                    4,
                ),

        })

    result = pd.DataFrame(rows)

    print(
        "\n  Gold-loan vs non-gold comparison:"
    )

    print(
        result.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Statistical test for default-rate difference
    # -------------------------------------------------------------------------

    gold_default = gold[
        "default_risk"
    ].dropna()

    no_gold_default = no_gold[
        "default_risk"
    ].dropna()

    t_stat, p_value = stats.ttest_ind(
        gold_default,
        no_gold_default,
        equal_var=False,
    )

    gold_rate = gold_default.mean()
    no_gold_rate = no_gold_default.mean()

    print(
        "\n  Welch t-test:"
    )

    print(
        f"    t-statistic: {t_stat:.4f}"
    )

    print(
        f"    p-value:     {p_value:.8f}"
    )

    print(
        f"    Significant: "
        f"{'Yes' if p_value < 0.05 else 'No'}"
    )

    test_result = pd.DataFrame([{

        "gold_default_rate_pct":
            round(gold_rate * 100, 2),

        "non_gold_default_rate_pct":
            round(no_gold_rate * 100, 2),

        "difference_pct_points":
            round(
                (gold_rate - no_gold_rate) * 100,
                2,
            ),

        "welch_t_stat":
            round(float(t_stat), 4),

        "p_value":
            round(float(p_value), 10),

        "statistically_significant":
            bool(p_value < 0.05),

        "interpretation":
            (
                "Gold-loan borrowers show a statistically "
                "different observed default rate. This is "
                "an association, not evidence of causality."
            ),

    }])

    result.to_parquet(
        ANALYTICS / "gold_loan_analysis.parquet",
        index=False,
    )

    test_result.to_parquet(
        ANALYTICS / "gold_loan_statistical_test.parquet",
        index=False,
    )

    print(
        "\n  ✓ Saved:"
        "\n    gold_loan_analysis.parquet"
        "\n    gold_loan_statistical_test.parquet"
    )

    return result


# =============================================================================
# MODULE 7
# DATA QUALITY / COVERAGE
# =============================================================================

def analyse_data_coverage(df: pd.DataFrame):

    """
    Produce a transparent missingness and coverage report.

    This is important for a production-style analytics project because
    a feature should not be interpreted without knowing how complete it is.
    """

    print(
        "\n━━━ Module 7: Data Coverage Analysis ━━━"
    )

    rows = []

    for column in df.columns:

        total = len(df)

        missing = int(
            df[column].isna().sum()
        )

        non_missing = total - missing

        rows.append({

            "feature": column,

            "total_rows": total,

            "non_missing": non_missing,

            "missing": missing,

            "missing_pct":
                round(
                    missing / total * 100,
                    2,
                ),

            "unique_values":
                int(df[column].nunique(
                    dropna=True
                )),

            "dtype":
                str(df[column].dtype),

        })

    coverage = pd.DataFrame(
        rows
    ).sort_values(
        "missing_pct",
        ascending=False,
    )

    print("\n  Highest missingness features:")

    print(
        coverage[
            [
                "feature",
                "missing",
                "missing_pct",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    coverage.to_parquet(
        ANALYTICS / "data_coverage.parquet",
        index=False,
    )

    print(
        "\n  ✓ Saved: "
        "data_coverage.parquet"
    )

    return coverage


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 60)
    print(
        "  INDIA CREDIT RISK — ANALYTICS LAYER"
    )
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Load + validate
    # -------------------------------------------------------------------------

    df = load_fact()

    # -------------------------------------------------------------------------
    # Analytics modules
    # -------------------------------------------------------------------------

    (
        score_bands,
        threshold_analysis,
        local_threshold,
    ) = analyse_credit_score_threshold(df)

    gini_df, overall_gini = compute_gini(df)

    segment_df = segment_borrowers(df)

    signals_df = compute_early_warning_signals(df)

    comparison_df = analyse_delinquency(df)

    gold_df = analyse_gold_loans(df)

    coverage_df = analyse_data_coverage(df)

    # -------------------------------------------------------------------------
    # Completion
    # -------------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("✓ ANALYTICS LAYER COMPLETE")
    print("=" * 60)

    print(
        "\nFiles saved to:"
        "\ndata/gold/exports/analytics/"
    )

    outputs = [

        "credit_score_bands.parquet",

        "credit_score_650_analysis.parquet",

        "credit_score_650_local.parquet",

        "gini_by_income_tier.parquet",

        "gini_overall.parquet",

        "cluster_selection.parquet",

        "borrower_segments.parquet",

        "borrower_segment_assignments.parquet",

        "early_warning_signals.parquet",

        "risky_vs_safe_comparison.parquet",

        "gold_loan_analysis.parquet",

        "gold_loan_statistical_test.parquet",

        "data_coverage.parquet",

    ]

    for filename in outputs:
        print(f"  → {filename}")

    print(
        f"""

Key diagnostics
---------------
Borrowers analysed : {len(df):,}
Overall CIBIL Gini: {overall_gini:.4f}
Segments evaluated: 3, 4, 5, 6
Final segments     : {len(segment_df)}
Risk signals       : {len(signals_df)}

IMPORTANT:
-----------
The analytics layer establishes associations and
portfolio patterns. It does not claim causal relationships.

Next:
    python src/analytics/run_ml_model.py
"""
    )


if __name__ == "__main__":
    main()