-- ============================================================
-- confusion_matrix.sql
-- Business Question: At the deployed operating threshold, what
--                    does the model's confusion matrix look like?
-- Params: none -- outcome_bucket in model_predictions is already
--         computed at the operating threshold by score_portfolio.py
-- Layer: model_predictions
-- Note: this is the same aggregation as
--       sql/model_queries/01_model_vs_actual_confusion.sql --
--       reused here rather than reimplemented, so the dashboard
--       and the query library can never show different numbers.
-- ============================================================

SELECT
    outcome_bucket,
    COUNT(*) AS borrower_count
FROM model_predictions
GROUP BY outcome_bucket;