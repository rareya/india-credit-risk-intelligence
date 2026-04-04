-- ============================================================
-- 02_default_by_segment.sql
-- Business Question: Which customer segments are driving NPA?
-- Output: Default rate broken down by risk segment + demographics
-- ============================================================

-- Default rate by risk segment
SELECT
    rs.risk_segment,
    COUNT(*)                                    AS borrower_count,
    ROUND(COUNT(*) * 100.0 /
        SUM(COUNT(*)) OVER (), 2)               AS pct_of_portfolio,
    SUM(rs.default_risk)                        AS defaults,
    ROUND(AVG(rs.default_risk) * 100, 2)        AS default_rate_pct,
    ROUND(AVG(rs.NETMONTHLYINCOME), 0)          AS avg_income,
    ROUND(AVG(rs.AGE), 1)                       AS avg_age

FROM risk_segments rs
GROUP BY rs.risk_segment
ORDER BY default_rate_pct DESC;