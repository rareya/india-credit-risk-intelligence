-- ============================================================
-- 07_income_default_analysis.sql
-- Business Question: Is income a reliable risk predictor,
--                    or does behaviour matter more?
-- Output: Default rate by income band, with behavioural
--         overlay to prove behaviour > income alone
-- ============================================================

WITH income_bands AS (
    SELECT
        CASE
            WHEN NETMONTHLYINCOME < 10000             THEN '1. <10k'
            WHEN NETMONTHLYINCOME BETWEEN 10000 AND 19999 THEN '2. 10-20k'
            WHEN NETMONTHLYINCOME BETWEEN 20000 AND 34999 THEN '3. 20-35k'
            WHEN NETMONTHLYINCOME BETWEEN 35000 AND 59999 THEN '4. 35-60k'
            ELSE                                           '5. 60k+'
        END                             AS income_band,
        default_risk,
        enq_L6m,
        num_times_delinquent,
        Age_Oldest_TL,
        NETMONTHLYINCOME
    FROM borrowers
    WHERE NETMONTHLYINCOME IS NOT NULL
)

SELECT
    income_band,
    COUNT(*)                            AS borrower_count,
    ROUND(AVG(default_risk)*100, 2)     AS default_rate_pct,
    ROUND(AVG(enq_L6m), 2)             AS avg_enquiries_6m,
    ROUND(AVG(num_times_delinquent),2)  AS avg_delinquencies,
    ROUND(AVG(Age_Oldest_TL), 0)        AS avg_credit_history_months,

    -- High-enquiry default rate WITHIN this income band
    -- Shows behaviour predicts even controlling for income
    ROUND(
        AVG(CASE WHEN enq_L6m >= 4 THEN default_risk END) * 100, 2
    )                                   AS default_rate_high_enquiry_pct,
    ROUND(
        AVG(CASE WHEN enq_L6m <= 1 THEN default_risk END) * 100, 2
    )                                   AS default_rate_low_enquiry_pct

FROM income_bands
GROUP BY income_band
ORDER BY income_band;