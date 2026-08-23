"""
calibrate_model.py

Evaluates whether the XGBoost probabilities are calibrated.

Calibration question:

    If the model assigns PD ≈ 0.70,
    do roughly 70% of those observations belong
    to the high-risk class?

Compares:
    1. Raw XGBoost probabilities
    2. Sigmoid calibration
    3. Isotonic calibration

The final test set is used only for evaluation.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.calibration import (
    CalibratedClassifierCV,
    calibration_curve,
)
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
)

from src.analytics.run_ml_model import (
    load_data,
    split_data,
    calculate_scale_pos_weight,
)

from src.modeling.model_pipeline import (
    build_xgboost_pipeline,
)


ML_DIR = Path(
    "data/gold/exports/ml"
)

MODEL_DIR = Path(
    "data/processed"
)

ML_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def main():

    X, y, _, _ = load_data()

    (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
    ) = split_data(X, y)

    scale = calculate_scale_pos_weight(
        y_train
    )

    base_model = build_xgboost_pipeline(
        scale
    )

    base_model.fit(
        X_train,
        y_train
    )

    raw_probability = (
        base_model.predict_proba(
            X_test
        )[:, 1]
    )

    raw_brier = brier_score_loss(
        y_test,
        raw_probability
    )

    raw_logloss = log_loss(
        y_test,
        raw_probability
    )

    results = [
        {
            "model": "raw_xgboost",
            "brier_score": raw_brier,
            "log_loss": raw_logloss,
        }
    ]

    calibrated_models = {}

    for method in [
        "sigmoid",
        "isotonic",
    ]:

        calibrated = (
            CalibratedClassifierCV(
                estimator=base_model,
                method=method,
                cv=5,
            )
        )

        calibrated.fit(
            X_train,
            y_train
        )

        probability = (
            calibrated.predict_proba(
                X_test
            )[:, 1]
        )

        results.append({
            "model": method,
            "brier_score":
                brier_score_loss(
                    y_test,
                    probability
                ),
            "log_loss":
                log_loss(
                    y_test,
                    probability
                ),
        })

        calibrated_models[
            method
        ] = calibrated

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        ML_DIR /
        "calibration_comparison.csv",
        index=False,
    )

    best_method = (
        results_df
        .sort_values("brier_score")
        .iloc[0]["model"]
    )

    print(
        "\nCalibration comparison:"
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        f"\nBest calibration by Brier score: "
        f"{best_method}"
    )

    if best_method != "raw_xgboost":

        final_model = calibrated_models[
            best_method
        ]

        with open(
            MODEL_DIR /
            "credit_risk_calibrated_model.pkl",
            "wb",
        ) as file:

            pickle.dump(
                final_model,
                file
            )

    else:

        final_model = base_model

    prob = (
        final_model.predict_proba(
            X_test
        )[:, 1]
    )

    frac_pos, mean_pred = (
        calibration_curve(
            y_test,
            prob,
            n_bins=10,
            strategy="quantile",
        )
    )

    calibration_df = pd.DataFrame({
        "mean_predicted_probability":
            mean_pred,
        "observed_fraction_positive":
            frac_pos,
    })

    calibration_df.to_csv(
        ML_DIR /
        "calibration_curve.csv",
        index=False,
    )

    metadata = {
        "selected_model":
            best_method,

        "selection_metric":
            "brier_score",

        "raw_brier":
            float(raw_brier),

        "final_brier":
            float(
                brier_score_loss(
                    y_test,
                    prob
                )
            ),
    }

    with open(
        MODEL_DIR /
        "calibration_metadata.json",
        "w",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2
        )


if __name__ == "__main__":
    main()