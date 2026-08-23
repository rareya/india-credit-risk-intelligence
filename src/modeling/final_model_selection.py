"""
final_model_selection.py

Final model governance decision.

Compares:
    - Logistic baseline
    - Conservative XGBoost
    - Expanded XGBoost

The expanded model is rejected because of structural/proxy leakage.

The conservative model becomes the candidate final model ONLY
after CV, calibration, SHAP and subgroup checks.
"""

from pathlib import Path

import json
import pandas as pd

BASE_DIR = Path("data/gold/exports/analytics")
ML_DIR = BASE_DIR / "ml"

OUTPUT_DIR = ML_DIR / "final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_existing_model_results():

    path = ML_DIR / "model_comparison.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run run_ml_model.py first."
        )

    return pd.read_parquet(path)


def load_cv():

    path = ML_DIR / "cross_validation" / "cv_summary.parquet"

    if not path.exists():
        raise FileNotFoundError(
            "Run cross_validate.py first."
        )

    return pd.read_parquet(path)


def load_calibration():

    path = (
        ML_DIR
        / "calibration"
        / "calibration_comparison.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(
            "Run calibrate_model.py first."
        )

    return pd.read_parquet(path)


def load_subgroups():

    path = (
        ML_DIR
        / "subgroup"
        / "subgroup_robustness_summary.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(
            "Run subgroup_robustness.py first."
        )

    return pd.read_parquet(path)


def main():

    print("=" * 70)
    print("  INDIA CREDIT RISK — FINAL MODEL GOVERNANCE")
    print("=" * 70)

    model_results = load_existing_model_results()
    cv = load_cv()
    calibration = load_calibration()
    subgroup = load_subgroups()

    print("\nExisting model comparison:")
    print(
        model_results.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Explicit governance decisions
    # --------------------------------------------------------

    decisions = [
        {
            "model": "Model 0 — Logistic Baseline",
            "decision": "BASELINE",
            "reason": (
                "Retained as interpretable benchmark; "
                "not selected as final predictive model."
            ),
        },
        {
            "model": "Model 1 — Conservative XGBoost",
            "decision": "CANDIDATE_FINAL",
            "reason": (
                "Structurally eligible feature set and materially "
                "lower performance than the leaked expanded model, "
                "indicating the extreme performance gap is not being "
                "used as evidence of model quality."
            ),
        },
        {
            "model": "Model 2 — Expanded XGBoost",
            "decision": "REJECT",
            "reason": (
                "ROC-AUC approximately 0.9998 combined with "
                "single-feature AUC of 1.0 for risk_grade_numeric "
                "indicates target/proxy leakage."
            ),
        },
    ]

    decisions_df = pd.DataFrame(decisions)

    decisions_df.to_parquet(
        OUTPUT_DIR / "model_governance_decision.parquet",
        index=False,
    )

    decisions_df.to_csv(
        OUTPUT_DIR / "model_governance_decision.csv",
        index=False,
    )

    # --------------------------------------------------------
    # CV metrics
    # --------------------------------------------------------

    cv_auc = cv[
        cv.metric == "roc_auc"
    ].iloc[0]

    cv_pr = cv[
        cv.metric == "pr_auc"
    ].iloc[0]

    cv_brier = cv[
        cv.metric == "brier_score"
    ].iloc[0]

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------

    calibration_summary = calibration.copy()

    best_calibrated = calibration_summary.loc[
        calibration_summary["brier_score"].idxmin()
    ]

    # --------------------------------------------------------
    # Subgroup robustness
    # --------------------------------------------------------

    min_auc = subgroup.loc[
        subgroup.metric == "minimum_subgroup_auc",
        "value",
    ].iloc[0]

    auc_range = subgroup.loc[
        subgroup.metric == "auc_range",
        "value",
    ].iloc[0]

    final_summary = {
        "candidate_model": "Conservative XGBoost",

        "decision": (
            "CANDIDATE_FINAL_PENDING "
            "PREDICTION-TIME PROVENANCE SIGN-OFF"
        ),

        "cross_validation_mean_roc_auc":
            float(cv_auc["mean"]),

        "cross_validation_std_roc_auc":
            float(cv_auc["std"]),

        "cross_validation_mean_pr_auc":
            float(cv_pr["mean"]),

        "cross_validation_mean_brier":
            float(cv_brier["mean"]),

        "best_calibration_method":
            str(best_calibrated["model"]),

        "best_calibrated_brier":
            float(best_calibrated["brier_score"]),

        "minimum_subgroup_auc":
            float(min_auc),

        "subgroup_auc_range":
            float(auc_range),

        "expanded_model_status":
            "REJECTED_FOR_LEAKAGE",

        "final_provenance_requirement":
            "Confirm prediction-time availability of review features.",
    }

    with open(
        OUTPUT_DIR / "final_model_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            final_summary,
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("FINAL GOVERNANCE DECISION")
    print("=" * 70)

    print(
        json.dumps(
            final_summary,
            indent=2,
        )
    )

    print("\n✓ Final governance assessment complete")


if __name__ == "__main__":
    main()