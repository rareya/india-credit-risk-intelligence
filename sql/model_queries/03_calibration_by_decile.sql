-- ============================================================
-- 03_calibration_by_decile.sql
-- Business Question: When the model says "10% risk", does that group actually default ~10% of the time?
--                    
-- Output: Predicted vs actual default rate per probability decile
--         -- a SQL reliability curve
-- Layer: model_predictions
-- Insight: calibrate_model.py's sigmoid step produced identical
--          metrics to the base model, which was flagged as
--          suspicious -- this query is the independent check.
-- ============================================================

SELECT
    probability_decile,
    COUNT(*)                                       AS borrower_count,
    ROUND(MIN(predicted_probability), 4)           AS min_predicted_prob,
    ROUND(MAX(predicted_probability), 4)           AS max_predicted_prob,
    ROUND(AVG(predicted_probability), 4)           AS avg_predicted_prob,
    ROUND(AVG(actual_default), 4)                  AS actual_default_rate,
    ROUND(AVG(predicted_probability) - AVG(actual_default), 4)
                                                    AS calibration_gap

FROM model_predictions
GROUP BY probability_decile
ORDER BY probability_decile;