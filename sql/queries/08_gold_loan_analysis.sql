-- ============================================================
-- 08_gold_loan_analysis.sql
-- Business Question: Are gold loan borrowers higher risk?
--                    (India-specific — gold loans are often
--                     last-resort financing)
-- Output: Default rate comparison for gold loan holders
-- ============================================================

SELECT
    CASE
        WHEN Gold_TL = 0  THEN 'No gold loans'
        WHEN Gold_TL = 1  THEN '1 gold loan'
        WHEN Gold_TL = 2  THEN '2 gold loans'
        ELSE                   '3+ gold loans'
    END                                         AS gold_loan_bucket,

    COUNT(*)                                    AS borrower_count,
    ROUND(COUNT(*) * 100.0 /
        SUM(COUNT(*)) OVER (), 2)               AS pct_of_portfolio,
    SUM(default_risk)                           AS defaults,
    ROUND(AVG(default_risk) * 100, 2)           AS default_rate_pct,
    ROUND(AVG(NETMONTHLYINCOME), 0)             AS avg_monthly_income,
    ROUND(AVG(enq_L6m), 2)                      AS avg_recent_enquiries,
    ROUND(AVG(num_times_delinquent), 2)         AS avg_delinquencies,

    -- Are gold loan borrowers also taking other loans heavily?
    ROUND(AVG(Total_TL), 1)                     AS avg_total_loans,
    ROUND(AVG(active_loan_ratio) * 100, 2)      AS avg_active_ratio_pct

FROM borrowers
WHERE Gold_TL IS NOT NULL
GROUP BY gold_loan_bucket
ORDER BY AVG(Gold_TL);