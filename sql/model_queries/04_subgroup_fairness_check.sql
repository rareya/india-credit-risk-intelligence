-- ============================================================
-- 04_subgroup_fairness_check.sql
-- Business Question: Does the model perform consistently across demographic and CIBIL-band subgroups, or are  there segments it systematically mishandles?
--                    
--                   
-- Output: Predicted vs actual default rate, and error rate, by
--         cibil_band -- SQL version of subgroup_robustness.py's
--         headline finding
-- Layer: model_predictions JOIN fact_credit_risk
-- Insight: flags the 'very_good' cibil_band weak spot (ROC-AUC
--          ~0.61 in the Python analysis) directly in SQL
-- ============================================================

SELECT
    f.cibil_band,
    COUNT(*)                                            AS borrower_count,
    ROUND(AVG(mp.actual_default), 4)                    AS actual_default_rate,
    ROUND(AVG(mp.predicted_probability), 4)             AS avg_predicted_prob,
    SUM(CASE WHEN mp.outcome_bucket IN ('false_positive','false_negative')
             THEN 1 ELSE 0 END)                         AS total_errors,
    ROUND(
        SUM(CASE WHEN mp.outcome_bucket IN ('false_positive','false_negative')
                 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    )                                                    AS error_rate_pct

FROM model_predictions mp
JOIN fact_credit_risk f ON f.borrower_id = mp.borrower_id

GROUP BY f.cibil_band
ORDER BY error_rate_pct DESC;