"""
export_powerbi_dataset.py

Creates the Power BI-ready scored borrower dataset.

The export uses the canonical model:

    data/processed/credit_risk_model.pkl

The locked threshold is read from:

    data/processed/threshold_metadata.json

No obsolete model versions are loaded.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pickle


ROOT = Path(__file__).resolve().parents[2]

SILVER_PATH = (
    ROOT
    / "data"
    / "silver"
    / "silver_master.parquet"
)

PROCESSED_DIR = (
    ROOT
    / "data"
    / "processed"
)

POWERBI_DIR = (
    ROOT
    / "data"
    / "powerbi"
)

MODEL_PATH = (
    PROCESSED_DIR
    / "credit_risk_model.pkl"
)

THRESHOLD_PATH = (
    PROCESSED_DIR
    / "threshold_metadata.json"
)

OUTPUT_PATH = (
    POWERBI_DIR
    / "credit_risk_powerbi_input.csv"
)


def load_silver():

    if not SILVER_PATH.exists():
        raise FileNotFoundError(
            f"Silver dataset missing: "
            f"{SILVER_PATH}"
        )

    return pd.read_parquet(
        SILVER_PATH
    )


def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Canonical model missing. "
            "Run run_ml_model.py first."
        )

    with open(
        MODEL_PATH,
        "rb",
    ) as file:

        model = pickle.load(
            file
        )

    return model


def load_threshold():

    if not THRESHOLD_PATH.exists():

        raise FileNotFoundError(
            "Locked threshold missing. "
            "Run improve_precision.py first."
        )

    with open(
        THRESHOLD_PATH,
        "r",
    ) as file:

        metadata = json.load(
            file
        )

    return float(
        metadata[
            "locked_threshold"
        ]
    )


def get_model_features(model):

    if hasattr(
        model,
        "feature_names_in_"
    ):

        return list(
            model.feature_names_in_
        )

    if hasattr(
        model,
        "named_steps"
    ):

        # Pipeline feature names are
        # available from the imputer.
        imputer = model.named_steps.get(
            "imputer"
        )

        if imputer is not None and hasattr(
            imputer,
            "feature_names_in_"
        ):

            return list(
                imputer.feature_names_in_
            )

    raise ValueError(
        "Could not determine model features."
    )


def score(df, model):

    features = get_model_features(
        model
    )

    missing = [
        c for c in features
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            "Required model features "
            f"missing from Silver data: {missing}"
        )

    X = df[features].copy()

    numeric_cols = X.select_dtypes(
        include=[np.number]
    ).columns

    X[numeric_cols] = (
        X[numeric_cols]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    probability = (
        model.predict_proba(X)[:, 1]
    )

    df = df.copy()

    df["predicted_pd"] = probability

    return df, features


def safe_cut(
    series,
    bins,
    labels,
):

    return (
        pd.cut(
            series,
            bins=bins,
            labels=labels,
            right=True,
        )
        .astype(str)
        .replace(
            "nan",
            "Unknown",
        )
    )


def add_business_columns(
    df,
    threshold,
):

    df = df.copy()

    df["risk_band"] = safe_cut(
        df["predicted_pd"],
        [
            -np.inf,
            0.25,
            0.50,
            np.inf,
        ],
        [
            "Low Risk",
            "Medium Risk",
            "High Risk",
        ],
    )

    df["approved_flag"] = (
        df["predicted_pd"]
        < threshold
    ).astype(int)

    df["rejected_flag"] = (
        df["predicted_pd"]
        >= threshold
    ).astype(int)

    if "NETMONTHLYINCOME" in df.columns:

        income = (
            pd.to_numeric(
                df["NETMONTHLYINCOME"],
                errors="coerce",
            )
            .fillna(0)
            .clip(lower=0)
        )

    else:

        income = pd.Series(
            0,
            index=df.index,
        )

    df["ead_proxy"] = (
        income * 12
    )

    df["lgd_assumption"] = 0.45

    df["expected_loss"] = (
        df["predicted_pd"]
        * df["lgd_assumption"]
        * df["ead_proxy"]
    )

    df["high_risk_flag"] = (
        df["predicted_pd"]
        >= threshold
    ).astype(int)

    df["watchlist_flag"] = (
        df["predicted_pd"]
        >= 0.65
    ).astype(int)

    df["priority_score"] = (
        df["predicted_pd"]
        * df["expected_loss"]
    )

    if "risk_target" in df.columns:

        df["actual_risk_label"] = (
            df["risk_target"]
            .astype(int)
        )

    elif "default_risk" in df.columns:

        df["actual_risk_label"] = (
            df["default_risk"]
            .astype(int)
        )

    return df


def main():

    print(
        "\nINDIA CREDIT RISK — Power BI EXPORT"
    )

    POWERBI_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_silver()

    model = load_model()

    threshold = load_threshold()

    print(
        f"Locked threshold: "
        f"{threshold:.2f}"
    )

    df, features = score(
        df,
        model
    )

    df = add_business_columns(
        df,
        threshold,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\n✓ Exported: {OUTPUT_PATH}"
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Features used: {len(features)}"
    )

    print(
        f"Approval rate: "
        f"{df['approved_flag'].mean():.2%}"
    )

    print(
        f"Average predicted PD: "
        f"{df['predicted_pd'].mean():.4f}"
    )

    print(
        f"Total expected loss: "
        f"₹{df['expected_loss'].sum():,.0f}"
    )


if __name__ == "__main__":
    main()