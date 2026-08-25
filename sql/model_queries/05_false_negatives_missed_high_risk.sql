-- ============================================================
-- 05_false_negatives_missed_high_risk.sql
-- Business Question: Which borrowers actually defaulted but the model scored as low-risk -- our costliest  error type?
--                    
--                   
-- Output: Ranked list of missed defaults with their features,
--         so credit ops can review what the model got wrong
-- Layer: model_predictions JOIN fact_credit_risk
-- ============================================================

SELECT
    mp.borrower_id,
    mp.predicted_probability,
    f.recent_enquiries_6m,
    f.delinquency_score,
    f.credit_history_months,
    f.cibil_band,
    f.income_tier,
    f.missed_payment_ratio

FROM model_predictions mp
JOIN fact_credit_risk f ON f.borrower_id = mp.borrower_id

WHERE mp.outcome_bucket = 'false_negative'

ORDER BY mp.predicted_probability ASC   -- most confidently wrong first
LIMIT 100;