-- ============================================================
-- 09_model_score_within_risk_grade.sql
-- Business Question: risk_grade (P1-P4) is NOT an independent
--                    signal -- default_risk is derived directly
--                    from it in build_gold.py (P1/P2->0, P3/P4->1),
--                    so grade vs actual_default_rate is circular
--                    by construction, not a finding. The real
--                    question this query answers: WITHIN a grade,
--                    does the model add resolution the coarse
--                    4-bucket grade doesn't have (e.g. does P2
--                    contain both very-low and moderately-risky
--                    borrowers the grade alone can't distinguish)?
-- Output: Score spread (min/max/stddev) within each risk_grade
-- Layer: model_predictions JOIN fact_credit_risk
-- ============================================================

SELECT
    f.risk_grade,
    COUNT(*)                                     AS borrower_count,
    ROUND(MIN(mp.predicted_probability), 4)      AS min_predicted_prob,
    ROUND(MAX(mp.predicted_probability), 4)      AS max_predicted_prob,
    ROUND(AVG(mp.predicted_probability), 4)      AS avg_predicted_prob,
    ROUND(STDDEV(mp.predicted_probability), 4)   AS stddev_predicted_prob,
    ROUND(MAX(mp.predicted_probability) - MIN(mp.predicted_probability), 4)
                                                  AS score_spread_within_grade

FROM model_predictions mp
JOIN fact_credit_risk f ON f.borrower_id = mp.borrower_id

GROUP BY f.risk_grade
ORDER BY avg_predicted_prob;