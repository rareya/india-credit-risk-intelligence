"""
policy_simulator.py

Underwriting policy what-if simulator.

The simulator estimates the effect of different approval policies
on the historical borrower population.

IMPORTANT:
This is retrospective simulation, NOT causal evidence.

Expected loss is model-based:

    EL = predicted_PD × LGD × EAD

LGD and EAD are assumptions/proxies unless observed recovery and
exposure-at-default data are available.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.analytics.expected_loss import (
    DEFAULT_EAD_MULTIPLIER,
    DEFAULT_LGD,
    expected_loss,
)


@dataclass
class PolicyConfig:

    max_enq_6m: int = 999

    min_credit_history_months: int = 0

    max_delinquencies: int = 999

    max_60dpd: int = 999

    min_monthly_income: float = 0

    max_predicted_pd: float = 1.0

    max_active_loan_ratio: float = 1.0

    name: str = "Custom Policy"


CURRENT_POLICY = PolicyConfig(
    name="Current Policy"
)

CONSERVATIVE_POLICY = PolicyConfig(
    max_enq_6m=3,
    min_credit_history_months=12,
    max_delinquencies=2,
    max_60dpd=0,
    min_monthly_income=10000,
    max_predicted_pd=0.45,
    name="Conservative Policy",
)

MODEL_B_POLICY = PolicyConfig(
    max_enq_6m=4,
    min_credit_history_months=24,
    max_delinquencies=5,
    max_predicted_pd=0.50,
    name="Model B Recommended",
)

AGGRESSIVE_GROWTH_POLICY = PolicyConfig(
    max_predicted_pd=0.65,
    name="Growth Mode",
)


def apply_policy(
    df: pd.DataFrame,
    predicted_pd,
    policy: PolicyConfig,
):

    approved = pd.Series(
        True,
        index=df.index,
    )

    if "enq_L6m" in df.columns:

        approved &= (
            df["enq_L6m"]
            .fillna(0)
            <= policy.max_enq_6m
        )

    if "Age_Oldest_TL" in df.columns:

        approved &= (
            df["Age_Oldest_TL"]
            .fillna(0)
            >= policy.min_credit_history_months
        )

    if "num_times_delinquent" in df.columns:

        approved &= (
            df["num_times_delinquent"]
            .fillna(0)
            <= policy.max_delinquencies
        )

    if "num_times_60p_dpd" in df.columns:

        approved &= (
            df["num_times_60p_dpd"]
            .fillna(0)
            <= policy.max_60dpd
        )

    if "NETMONTHLYINCOME" in df.columns:

        approved &= (
            df["NETMONTHLYINCOME"]
            .fillna(0)
            >= policy.min_monthly_income
        )

    predicted_pd = pd.Series(
        predicted_pd,
        index=df.index,
    )

    approved &= (
        predicted_pd
        <= policy.max_predicted_pd
    )

    if "active_loan_ratio" in df.columns:

        approved &= (
            df["active_loan_ratio"]
            .fillna(0)
            <= policy.max_active_loan_ratio
        )

    return approved


def simulate_policy(
    df,
    predicted_pd,
    policy,
    baseline_policy: Optional[
        PolicyConfig
    ] = None,
    lgd=DEFAULT_LGD,
    ead_multiplier=DEFAULT_EAD_MULTIPLIER,
):

    predicted_pd = pd.Series(
        predicted_pd,
        index=df.index,
        dtype=float,
    )

    if "risk_target" in df.columns:

        actual_risk = (
            df["risk_target"]
            .fillna(0)
            .astype(int)
        )

    elif "default_risk" in df.columns:

        actual_risk = (
            df["default_risk"]
            .fillna(0)
            .astype(int)
        )

    else:

        raise ValueError(
            "risk_target/default_risk missing."
        )

    approved = apply_policy(
        df,
        predicted_pd,
        policy,
    )

    if baseline_policy is None:

        baseline_approved = pd.Series(
            True,
            index=df.index,
        )

    else:

        baseline_approved = apply_policy(
            df,
            predicted_pd,
            baseline_policy,
        )

    n = len(df)

    approved_count = int(
        approved.sum()
    )

    rejected_count = (
        n - approved_count
    )

    risky_approved = int(
        (
            actual_risk[approved] == 1
        ).sum()
    )

    risky_rejected = int(
        (
            actual_risk[~approved] == 1
        ).sum()
    )

    safe_rejected = int(
        (
            actual_risk[~approved] == 0
        ).sum()
    )

    safe_approved = int(
        (
            actual_risk[approved] == 0
        ).sum()
    )

    ead = (
        df["NETMONTHLYINCOME"]
        .fillna(
            df["NETMONTHLYINCOME"].median()
        )
        .clip(lower=0)
        * ead_multiplier
        if "NETMONTHLYINCOME" in df.columns
        else pd.Series(
            300000,
            index=df.index,
        )
    )

    expected_loss_all = expected_loss(
        predicted_pd,
        ead,
        lgd,
    )

    expected_loss_all = pd.Series(
        expected_loss_all,
        index=df.index,
    )

    policy_loss = float(
        expected_loss_all[
            approved
        ].sum()
    )

    baseline_loss = float(
        expected_loss_all[
            baseline_approved
        ].sum()
    )

    baseline_approved_count = int(
        baseline_approved.sum()
    )

    baseline_risky_approved = int(
        (
            actual_risk[
                baseline_approved
            ] == 1
        ).sum()
    )

    return {

        "policy_name":
            policy.name,

        "n_total":
            n,

        "n_approved":
            approved_count,

        "approval_rate_pct":
            round(
                approved_count / max(n, 1)
                * 100,
                2,
            ),

        "n_rejected":
            rejected_count,

        "rejection_rate_pct":
            round(
                rejected_count / max(n, 1)
                * 100,
                2,
            ),

        "risky_borrowers_approved":
            risky_approved,

        "risky_borrowers_rejected":
            risky_rejected,

        "safe_borrowers_rejected":
            safe_rejected,

        "safe_borrowers_approved":
            safe_approved,

        "risk_rate_in_approved_pct":
            round(
                risky_approved
                / max(approved_count, 1)
                * 100,
                2,
            ),

        "risk_capture_rate_pct":
            round(
                risky_rejected
                / max(
                    int(actual_risk.sum()),
                    1,
                )
                * 100,
                2,
            ),

        "expected_loss":
            policy_loss,

        "baseline_expected_loss":
            baseline_loss,

        "expected_loss_reduction":
            baseline_loss - policy_loss,

        "expected_loss_reduction_pct":
            round(
                (
                    baseline_loss
                    - policy_loss
                )
                / max(
                    baseline_loss,
                    1e-9,
                )
                * 100,
                2,
            ),

        "approvals_lost_vs_baseline":
            baseline_approved_count
            - approved_count,

        "risky_borrowers_avoided_vs_baseline":
            baseline_risky_approved
            - risky_approved,

        "lgd_assumption":
            lgd,

        "ead_multiplier":
            ead_multiplier,

        "interpretation":
            (
                "Retrospective model-based simulation; "
                "not causal evidence of future loss reduction."
            ),
    }


def threshold_sensitivity_table(
    df,
    predicted_pd,
):

    rows = []

    if "risk_target" in df.columns:

        actual = (
            df["risk_target"]
            .fillna(0)
            .astype(int)
            .values
        )

    else:

        actual = (
            df["default_risk"]
            .fillna(0)
            .astype(int)
            .values
        )

    predicted_pd = np.asarray(
        predicted_pd
    )

    for threshold in np.arange(
        0.20,
        0.81,
        0.01,
    ):

        risky_flag = (
            predicted_pd
            >= threshold
        )

        approved = ~risky_flag

        tp = int(
            (
                risky_flag
                & (actual == 1)
            ).sum()
        )

        fp = int(
            (
                risky_flag
                & (actual == 0)
            ).sum()
        )

        fn = int(
            (
                approved
                & (actual == 1)
            ).sum()
        )

        tn = int(
            (
                approved
                & (actual == 0)
            ).sum()
        )

        precision = (
            tp / max(tp + fp, 1)
        )

        recall = (
            tp / max(tp + fn, 1)
        )

        f1 = (
            2
            * precision
            * recall
            / max(
                precision + recall,
                1e-9,
            )
        )

        rows.append({

            "threshold":
                round(
                    float(threshold),
                    2,
                ),

            "approval_rate_pct":
                round(
                    approved.mean()
                    * 100,
                    2,
                ),

            "precision":
                round(
                    precision,
                    4,
                ),

            "recall":
                round(
                    recall,
                    4,
                ),

            "f1":
                round(
                    f1,
                    4,
                ),

            "true_positive":
                tp,

            "false_positive":
                fp,

            "false_negative":
                fn,

            "true_negative":
                tn,
        })

    return pd.DataFrame(
        rows
    )


def compare_policies(
    df,
    predicted_pd,
    policies,
):

    rows = []

    baseline = (
        policies[0]
        if policies
        else CURRENT_POLICY
    )

    for policy in policies:

        result = simulate_policy(
            df,
            predicted_pd,
            policy,
            baseline_policy=baseline,
        )

        rows.append({
            "policy":
                result["policy_name"],

            "approval_rate_pct":
                result["approval_rate_pct"],

            "risk_rate_in_approved_pct":
                result[
                    "risk_rate_in_approved_pct"
                ],

            "risk_capture_rate_pct":
                result[
                    "risk_capture_rate_pct"
                ],

            "risky_borrowers_avoided":
                result[
                    "risky_borrowers_avoided_vs_baseline"
                ],

            "safe_borrowers_rejected":
                result[
                    "safe_borrowers_rejected"
                ],

            "expected_loss":
                result[
                    "expected_loss"
                ],

            "expected_loss_reduction":
                result[
                    "expected_loss_reduction"
                ],

            "expected_loss_reduction_pct":
                result[
                    "expected_loss_reduction_pct"
                ],

            "lgd_assumption":
                result[
                    "lgd_assumption"
                ],

            "interpretation":
                result[
                    "interpretation"
                ],
        })

    return pd.DataFrame(
        rows
    )