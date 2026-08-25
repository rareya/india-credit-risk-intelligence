"""
improve_precision.py

Decision-threshold optimization.

IMPORTANT:
The final test set is NOT used to choose the threshold.

Process:

    TRAIN
      ↓
    fit model
      ↓
    VALIDATION
      ↓
    choose threshold
      ↓
    lock threshold
      ↓
    FINAL TEST
      ↓
    unbiased evaluation
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.modeling.run_ml_model import (
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

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def calculate_threshold_metrics(
    y_true,
    probability,
    threshold,
):

    prediction = (
        probability >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        prediction,
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )

    precision = precision_score(
        y_true,
        prediction,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        prediction,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        prediction,
        zero_division=0,
    )

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "approval_rate": (
            1 - prediction.mean()
        ),
        "risk_capture_rate": recall,
    }


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

    model = build_xgboost_pipeline(
        scale
    )

    model.fit(
        X_train,
        y_train
    )

    validation_probability = (
        model.predict_proba(
            X_valid
        )[:, 1]
    )

    test_probability = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    rows = []

    for threshold in np.arange(
        0.20,
        0.81,
        0.01
    ):

        result = (
            calculate_threshold_metrics(
                y_valid,
                validation_probability,
                round(
                    float(threshold),
                    2
                ),
            )
        )

        rows.append(result)

    validation_df = pd.DataFrame(
        rows
    )

    # Business constraint:
    # capture at least 75% of high-risk borrowers.
    viable = validation_df[
        validation_df[
            "recall"
        ] >= 0.75
    ]

    if len(viable) == 0:

        selected = (
            validation_df
            .loc[
                validation_df["f1"].idxmax()
            ]
        )

        selection_rule = (
            "maximum_validation_F1"
        )

    else:

        selected = (
            viable
            .loc[
                viable["precision"].idxmax()
            ]
        )

        selection_rule = (
            "maximum_validation_precision_"
            "subject_to_recall>=0.75"
        )

    locked_threshold = float(
        selected["threshold"]
    )

    # ONLY NOW evaluate on final test.
    final_test = (
        calculate_threshold_metrics(
            y_test,
            test_probability,
            locked_threshold,
        )
    )

    validation_df.to_csv(
        ML_DIR /
        "threshold_sensitivity_validation.csv",
        index=False,
    )

    pd.DataFrame([
        final_test
    ]).to_csv(
        ML_DIR /
        "threshold_final_test.csv",
        index=False,
    )

    metadata = {
        "locked_threshold":
            locked_threshold,

        "selection_rule":
            selection_rule,

        "validation_precision":
            float(
                selected["precision"]
            ),

        "validation_recall":
            float(
                selected["recall"]
            ),

        "validation_f1":
            float(
                selected["f1"]
            ),

        "final_test":
            final_test,
    }

    with open(
        MODEL_DIR /
        "threshold_metadata.json",
        "w",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(
        "\nLOCKED THRESHOLD"
    )

    print(
        f"Threshold: "
        f"{locked_threshold:.2f}"
    )

    print(
        f"Validation precision: "
        f"{selected['precision']:.4f}"
    )

    print(
        f"Validation recall: "
        f"{selected['recall']:.4f}"
    )

    print(
        "\nFINAL TEST"
    )

    print(
        json.dumps(
            final_test,
            indent=2
        )
    )


if __name__ == "__main__":
    main()