-- ============================================================
-- 05_credit_history_vs_default.sql
-- Business Question: Does credit history length protect
--                    against default?
-- Output: Default rate by tradeline age bucket
-- Insight: Age_Oldest_TL is #3 SHAP feature
-- ============================================================

SELECT
    CASE
        WHEN Age_Oldest_TL < 12               THEN 'Under 1 year'
        WHEN Age_Oldest_TL BETWEEN 12 AND 23  THEN '1-2 years'
        WHEN Age_Oldest_TL BETWEEN 24 AND 47  THEN '2-4 years'
        WHEN Age_Oldest_TL BETWEEN 48 AND 95  THEN '4-8 years'
        ELSE                                       '8+ years'
    END                                         AS credit_history_band,

    COUNT(*)                                    AS borrower_count,
    ROUND(COUNT(*) * 100.0 /
        SUM(COUNT(*)) OVER (), 2)               AS pct_of_portfolio,
    SUM(default_risk)                           AS defaults,
    ROUND(AVG(default_risk) * 100, 2)           AS default_rate_pct,
    ROUND(AVG(enq_L6m), 2)                      AS avg_recent_enquiries,
    ROUND(AVG(num_times_delinquent), 2)         AS avg_delinquencies,

    -- Default rate relative to 8yr+ segment (safest group)
    ROUND(
        AVG(default_risk) /
        MIN(AVG(default_risk)) OVER (), 2
    )                                           AS relative_risk_vs_safest

FROM borrowers
WHERE Age_Oldest_TL IS NOT NULL
GROUP BY credit_history_band
ORDER BY AVG(Age_Oldest_TL);