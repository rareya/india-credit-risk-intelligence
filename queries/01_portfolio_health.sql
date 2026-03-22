-- ============================================================
-- 01_portfolio_health.sql
-- Business Question: How risky is our overall loan book?
-- Output: Portfolio-level KPIs for the executive dashboard
-- ============================================================

SELECT
    COUNT(*)                                          AS total_borrowers,
    SUM(default_risk)                                 AS total_defaults,
    ROUND(AVG(default_risk) * 100, 2)                 AS default_rate_pct,

    -- Income segmentation
    ROUND(AVG(NETMONTHLYINCOME), 0)                   AS avg_monthly_income,
    ROUND(AVG(AGE), 1)                                AS avg_age,

    -- Loan portfolio health
    ROUND(AVG(Total_TL), 1)                           AS avg_total_loans,
    ROUND(AVG(active_loan_ratio) * 100, 2)            AS avg_active_loan_ratio_pct,

    -- Delinquency overview
    ROUND(AVG(num_times_delinquent), 2)               AS avg_delinquencies,
    SUM(CASE WHEN num_times_delinquent > 0 THEN 1 ELSE 0 END) AS ever_delinquent_count,
    ROUND(
        SUM(CASE WHEN num_times_delinquent > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    )                                                 AS ever_delinquent_pct,

    -- Enquiry behaviour
    ROUND(AVG(enq_L6m), 2)                            AS avg_enquiries_6m,
    ROUND(AVG(enq_L12m), 2)                           AS avg_enquiries_12m

FROM borrowers;