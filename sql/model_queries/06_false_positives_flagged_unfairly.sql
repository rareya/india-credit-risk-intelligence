-- ============================================================
-- 06_false_positives_flagged_unfairly.sql
-- Business Question: Which borrowers did the model flag as
--                    high-risk who never actually defaulted --
--                    i.e. who would be denied/reviewed unfairly
--                    at the current threshold?
-- Output: Ranked list of false positives, most confidently wrong
--         first, for credit-policy review of over-flagging
-- Layer: model_predictions JOIN fact_credit_risk
-- ============================================================

SELECT
    mp.borrower_id,
    mp.predicted_probability,
    f.recent_enquiries_6m,
    f.credit_history_months,
    f.cibil_band,
    f.income_tier,
    f.age_band

FROM model_predictions mp
JOIN fact_credit_risk f ON f.borrower_id = mp.borrower_id

WHERE mp.outcome_bucket = 'false_positive'

ORDER BY mp.predicted_probability DESC   -- most confidently wrong first
LIMIT 100;