-- ============================================================
-- 08_missingness_impact_on_score.sql
-- Business Question: recent_enquiries_6m is 12.3% null and gets  median-imputed -- does that visibly shift predicted risk for those borrowers versus borrowers with real enquiry data?
--                   
--                    
--                    
-- Output: Predicted probability compared for imputed vs
--         non-imputed borrowers on the model's top feature
-- Layer: model_predictions JOIN fact_credit_risk
-- ============================================================

SELECT
    CASE WHEN f.recent_enquiries_6m IS NULL
         THEN 'imputed (missing enquiry history)'
         ELSE 'observed'
    END                                          AS enquiry_data_status,

    COUNT(*)                                     AS borrower_count,
    ROUND(AVG(mp.predicted_probability), 4)      AS avg_predicted_prob,
    ROUND(AVG(mp.actual_default), 4)             AS actual_default_rate

FROM model_predictions mp
JOIN fact_credit_risk f ON f.borrower_id = mp.borrower_id

GROUP BY enquiry_data_status;