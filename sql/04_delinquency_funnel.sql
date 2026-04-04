-- ============================================================
-- 04_delinquency_funnel.sql
-- Business Question: Where in the repayment journey do
--                    borrowers start failing?
-- Output: Funnel from active → 30DPD → 60DPD → default
-- Used by: Collections team for intervention timing
-- ============================================================

WITH funnel AS (
    SELECT
        COUNT(*)                                                AS total_portfolio,

        -- Ever missed any payment (proxy for 30+ DPD)
        SUM(CASE WHEN num_times_delinquent > 0 THEN 1 ELSE 0 END)
                                                                AS ever_30dpd,

        -- Serious delinquency — 60+ DPD
        SUM(CASE WHEN num_times_60p_dpd > 0 THEN 1 ELSE 0 END)
                                                                AS ever_60dpd,

        -- Confirmed defaults
        SUM(default_risk)                                       AS confirmed_default

    FROM borrowers
)

SELECT
    'Active Portfolio'      AS stage,
    total_portfolio         AS count,
    100.0                   AS pct_of_portfolio,
    NULL                    AS transition_rate_from_prev
FROM funnel

UNION ALL

SELECT
    'Ever 30+ DPD'          AS stage,
    ever_30dpd              AS count,
    ROUND(ever_30dpd * 100.0 / total_portfolio, 2),
    NULL
FROM funnel

UNION ALL

SELECT
    'Ever 60+ DPD'          AS stage,
    ever_60dpd              AS count,
    ROUND(ever_60dpd * 100.0 / total_portfolio, 2),
    ROUND(ever_60dpd * 100.0 / ever_30dpd, 2)  -- transition from 30 DPD
FROM funnel

UNION ALL

SELECT
    'Confirmed Default'     AS stage,
    confirmed_default       AS count,
    ROUND(confirmed_default * 100.0 / total_portfolio, 2),
    ROUND(confirmed_default * 100.0 / ever_60dpd, 2)  -- transition from 60 DPD
FROM funnel;