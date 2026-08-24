-- ============================================================
-- portfolio_kpis.sql
-- Business Question: Single-row summary for the Portfolio
--                    Overview panel's KPI strip -- total
--                    borrowers, default rate, model-driven
--                    approval rate, and expected loss.
-- Params: {threshold} -- operating threshold, substituted by
--         model_bridge.py from OPERATING_THRESHOLD (0.40)
-- Layer: model_predictions JOIN fact_credit_risk
-- Note: Expected Loss proxy = predicted_probability * LGD * EAD,
--       LGD fixed at 45%, EAD estimated as 12x monthly income
--       (same assumption the dashboard used previously -- kept
--       here explicitly rather than buried in Python).
-- ============================================================

WITH lgd_ead AS (
    SELECT
        mp.borrower_id,
        mp.actual_default,
        mp.predicted_probability,
        f.monthly_income_inr * 12                          AS ead_proxy,
        mp.predicted_probability * 0.45 * (f.monthly_income_inr * 12) AS expected_loss_proxy
    FROM model_predictions mp
    JOIN fact_credit_risk f ON f.borrower_id = mp.borrower_id
)
SELECT
    COUNT(*)                                                        AS n_borrowers,
    SUM(actual_default)                                             AS n_defaults,
    ROUND(AVG(actual_default), 4)                                   AS default_rate,
    ROUND(AVG(predicted_probability), 4)                            AS avg_predicted_pd,
    SUM(CASE WHEN predicted_probability >= {threshold} THEN 1 ELSE 0 END)
                                                                     AS high_risk_count,
    ROUND(AVG(CASE WHEN predicted_probability < {threshold} THEN 1.0 ELSE 0.0 END), 4)
                                                                     AS approval_rate,
    ROUND(SUM(ead_proxy) / 1e7, 2)                                  AS total_ead_crores,
    ROUND(SUM(expected_loss_proxy) / 1e7, 2)                        AS expected_loss_crores
FROM lgd_ead;