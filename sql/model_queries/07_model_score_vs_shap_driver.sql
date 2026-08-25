-- ============================================================
-- 07_model_score_vs_shap_top_driver.sql
-- Business Question: For high-scoring borrowers, is the model's risk score actually being driven by the  #1 global SHAP feature?
--                    
--                    (recent_enquiries_6m),
--                    or something else?
-- Output: Predicted probability against recent_enquiries_6m bands,
--         to sanity-check the model's top feature behaves the way
--         SHAP says it does, in plain SQL rather than trusting
--         the SHAP output blind
-- Layer: model_predictions JOIN fact_credit_risk
-- ============================================================

SELECT
    CASE
        WHEN f.recent_enquiries_6m IS NULL     THEN 'missing (imputed)'
        WHEN f.recent_enquiries_6m = 0         THEN '0 enquiries'
        WHEN f.recent_enquiries_6m BETWEEN 1 AND 3 THEN '1-3 enquiries'
        WHEN f.recent_enquiries_6m BETWEEN 4 AND 6 THEN '4-6 enquiries'
        ELSE                                        '7+ enquiries'
    END                                          AS enquiry_band,

    COUNT(*)                                     AS borrower_count,
    ROUND(AVG(mp.predicted_probability), 4)      AS avg_predicted_prob,
    ROUND(AVG(mp.actual_default), 4)             AS actual_default_rate

FROM model_predictions mp
JOIN fact_credit_risk f ON f.borrower_id = mp.borrower_id

GROUP BY enquiry_band
ORDER BY avg_predicted_prob;