-- ============================================================
-- 01_model_vs_actual_confusion.sql
-- Business Question: How well does the model's decision at the current operating threshold actually match  what happened?
--                    
--                   
-- Output: Confusion matrix + precision/recall computed in SQL,
--         as a cross-check against threshold_analysis.py's numbers
-- Layer: model_predictions (predicted_probability, predicted_class)
-- ============================================================

SELECT
    outcome_bucket,
    COUNT(*)                                            AS borrower_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2)  AS pct_of_portfolio

FROM model_predictions
GROUP BY outcome_bucket
ORDER BY
    CASE outcome_bucket
        WHEN 'true_positive'  THEN 1
        WHEN 'false_negative' THEN 2
        WHEN 'false_positive' THEN 3
        WHEN 'true_negative'  THEN 4
    END;