-- ============================================================
-- 06_high_risk_identification.sql
-- Business Question: Which specific borrower profiles should
--                    trigger automatic review?
-- Output: Rule-based high-risk flag with default rate proof
-- Used by: Credit policy implementation
-- ============================================================

WITH risk_flags AS (
    SELECT
        borrower_id,
        default_risk,
        enq_L6m,
        Age_Oldest_TL,
        num_times_60p_dpd,
        num_times_delinquent,
        missed_payment_ratio,
        active_loan_ratio,

        -- Policy Rule 1: Enquiry-based flag (top SHAP feature)
        CASE WHEN enq_L6m >= 4 THEN 1 ELSE 0 END
            AS flag_high_enquiry,

        -- Policy Rule 2: Thin file flag (short credit history)
        CASE WHEN Age_Oldest_TL < 24 THEN 1 ELSE 0 END
            AS flag_thin_file,

        -- Policy Rule 3: Serious delinquency history
        CASE WHEN num_times_60p_dpd >= 1 THEN 1 ELSE 0 END
            AS flag_severe_delinquency,

        -- Policy Rule 4: Combined high-risk (Rule 1 AND Rule 2)
        CASE WHEN enq_L6m >= 4 AND Age_Oldest_TL < 24 THEN 1 ELSE 0 END
            AS flag_extreme_risk

    FROM borrowers
)

SELECT
    'High Enquiry (4+ in 6m)'           AS policy_rule,
    SUM(flag_high_enquiry)              AS flagged_count,
    ROUND(SUM(flag_high_enquiry) * 100.0 / COUNT(*), 2)
                                        AS pct_of_portfolio,
    ROUND(AVG(CASE WHEN flag_high_enquiry = 1
              THEN default_risk END) * 100, 2)
                                        AS default_rate_in_segment

FROM risk_flags

UNION ALL

SELECT
    'Thin File (<2yr history)',
    SUM(flag_thin_file),
    ROUND(SUM(flag_thin_file) * 100.0 / COUNT(*), 2),
    ROUND(AVG(CASE WHEN flag_thin_file = 1
              THEN default_risk END) * 100, 2)
FROM risk_flags

UNION ALL

SELECT
    'Severe Delinquency (60+ DPD ever)',
    SUM(flag_severe_delinquency),
    ROUND(SUM(flag_severe_delinquency) * 100.0 / COUNT(*), 2),
    ROUND(AVG(CASE WHEN flag_severe_delinquency = 1
              THEN default_risk END) * 100, 2)
FROM risk_flags

UNION ALL

SELECT
    'Extreme Risk (Rule 1 AND Rule 2)',
    SUM(flag_extreme_risk),
    ROUND(SUM(flag_extreme_risk) * 100.0 / COUNT(*), 2),
    ROUND(AVG(CASE WHEN flag_extreme_risk = 1
              THEN default_risk END) * 100, 2)
FROM risk_flags

ORDER BY default_rate_in_segment DESC;