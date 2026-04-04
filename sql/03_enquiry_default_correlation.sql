-- ============================================================
-- 03_enquiry_default_correlation.sql
-- Business Question: Does credit-seeking behaviour predict default?
-- Output: Default rate by number of recent enquiries
-- Insight: enq_L6m is the #1 SHAP feature — this proves it in SQL
-- ============================================================

SELECT
    CASE
        WHEN enq_L6m = 0              THEN '0 enquiries'
        WHEN enq_L6m = 1              THEN '1 enquiry'
        WHEN enq_L6m = 2              THEN '2 enquiries'
        WHEN enq_L6m = 3              THEN '3 enquiries'
        WHEN enq_L6m BETWEEN 4 AND 5  THEN '4-5 enquiries'
        ELSE                               '6+ enquiries'
    END                                         AS enquiry_band,
    enq_L6m                                     AS raw_enq_L6m,
    COUNT(*)                                    AS borrower_count,
    SUM(default_risk)                           AS defaults,
    ROUND(AVG(default_risk) * 100, 2)           AS default_rate_pct,

    -- Lift over baseline (overall default rate)
    ROUND(
        AVG(default_risk) / (SELECT AVG(default_risk) FROM borrowers), 2
    )                                           AS lift_over_baseline

FROM borrowers
GROUP BY enquiry_band, enq_L6m
ORDER BY raw_enq_L6m;