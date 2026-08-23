"""
expected_loss.py

Centralized credit-risk expected-loss calculations.

Formula:

    Expected Loss = PD × LGD × EAD

IMPORTANT:
LGD is an illustrative scenario assumption unless empirically
estimated from observed recovery data.
"""

import pandas as pd
import numpy as np


DEFAULT_LGD = 0.45
DEFAULT_EAD_MULTIPLIER = 12


def expected_loss(
    pd_probability,
    ead,
    lgd=DEFAULT_LGD,
):

    pd_probability = np.asarray(
        pd_probability,
        dtype=float
    )

    ead = np.asarray(
        ead,
        dtype=float
    )

    if not 0 <= lgd <= 1:
        raise ValueError(
            "LGD must be between 0 and 1."
        )

    if np.any(
        (pd_probability < 0)
        | (pd_probability > 1)
    ):
        raise ValueError(
            "PD must be between 0 and 1."
        )

    return (
        pd_probability
        * lgd
        * ead
    )


def expected_loss_sensitivity(
    pd_probability,
    ead,
    lgd_values=(0.30, 0.45, 0.60),
):

    rows = []

    for lgd in lgd_values:

        loss = expected_loss(
            pd_probability,
            ead,
            lgd,
        )

        rows.append({
            "LGD":
                lgd,

            "Expected_Loss":
                float(
                    np.sum(loss)
                ),
        })

    return pd.DataFrame(
        rows
    )