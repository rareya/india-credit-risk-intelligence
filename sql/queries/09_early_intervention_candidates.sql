-- ============================================================
-- 09_early_intervention_candidates.sql
-- Business Question: Which current borrowers should the
--                    collections team contact NOW — before
--                    they hit 60 DPD?
-- Output: Ranked list of intervention candidates
-- Used by: Collections team, weekly run
-- ============================================================

SELECT
    borrower_id,

    -- Risk indicators
    enq_L6m                                     AS recent_enquiries_6m,
    num_times_delinquent                        AS total_delinquencies,
    num_times_60p_dpd                           AS serious_delinquencies,
    missed_payment_ratio                        AS missed_payment_ratio,
    active_loan_ratio                           AS active_loan_ratio,
    Age_Oldest_TL                               AS credit_history_months,
    NETMONTHLYINCOME                            AS monthly_income,

    -- Composite early warning score (rule-based, pre-ML)
    -- Higher = more urgent intervention needed
    ROUND(
        (enq_L6m          * 0.30) +   -- recent desperation (top SHAP)
        (num_times_delinquent * 0.25) +   -- delinquency history
        (missed_payment_ratio * 0.25) +   -- payment behaviour
        (active_loan_ratio    * 0.20),    -- current overextension
        3
    )                                           AS intervention_score,

    -- Segment label
    CASE
        WHEN enq_L6m >= 4 AND Age_Oldest_TL < 24 THEN 'EXTREME — Act immediately'
        WHEN enq_L6m >= 4 OR num_times_60p_dpd >= 1  THEN 'HIGH — Contact this week'
        WHEN enq_L6m >= 2 OR num_times_delinquent >= 2 THEN 'MEDIUM — Monitor closely'
        ELSE                                                'LOW — Standard monitoring'
    END                                         AS intervention_priority

FROM borrowers
WHERE
    -- Only flag borrowers who haven't defaulted yet
    -- but show early warning signals
    default_risk = 0
    AND (
        enq_L6m >= 3
        OR num_times_delinquent >= 2
        OR missed_payment_ratio >= 0.2
        OR (enq_L6m >= 2 AND Age_Oldest_TL < 24)
    )

ORDER BY intervention_score DESC
LIMIT 500;