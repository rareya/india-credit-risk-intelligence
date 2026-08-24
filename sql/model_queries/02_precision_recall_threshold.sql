-- ============================================================
-- 02_precision_recall_at_threshold.sql
-- Business Question: At the current operating threshold, what
--                    precision/recall/F1 is the model actually
--                    delivering on this portfolio?
-- Output: Single-row precision/recall/F1 summary
-- Layer: model_predictions
-- Cross-check against: threshold_analysis.py output at threshold=0.40
-- ============================================================

WITH counts AS (
    SELECT
        SUM(CASE WHEN outcome_bucket = 'true_positive'  THEN 1 ELSE 0 END) AS tp,
        SUM(CASE WHEN outcome_bucket = 'false_positive' THEN 1 ELSE 0 END) AS fp,
        SUM(CASE WHEN outcome_bucket = 'false_negative' THEN 1 ELSE 0 END) AS fn,
        SUM(CASE WHEN outcome_bucket = 'true_negative'  THEN 1 ELSE 0 END) AS tn
    FROM model_predictions
)
SELECT
    tp, fp, fn, tn,
    ROUND(tp * 1.0 / NULLIF(tp + fp, 0), 4)              AS precision,
    ROUND(tp * 1.0 / NULLIF(tp + fn, 0), 4)              AS recall,
    ROUND(
        2.0 * (tp * 1.0 / NULLIF(tp + fp, 0)) * (tp * 1.0 / NULLIF(tp + fn, 0))
        / NULLIF((tp * 1.0 / NULLIF(tp + fp, 0)) + (tp * 1.0 / NULLIF(tp + fn, 0)), 0), 4
    )                                                     AS f1_score
FROM counts;