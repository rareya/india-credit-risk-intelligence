-- ============================================================
-- roc_curve_sweep.sql
-- Business Question: What does the full ROC curve look like --
--                    true positive rate vs. false positive rate
--                    across every possible threshold, not just
--                    the one deployed?
-- Output: 101 points (threshold 0.00 to 1.00, step 0.01), each
--         with TPR and FPR, for plotting.
-- Layer: model_predictions
-- Note: replaces a previously-separate roc_curve.parquet file --
--       computed live so it can never go stale relative to the
--       actual scored predictions.
-- ============================================================

WITH thresholds AS (
    SELECT UNNEST(generate_series(0, 100)) / 100.0 AS threshold
),
totals AS (
    SELECT
        SUM(actual_default)       AS total_positive,
        SUM(1 - actual_default)   AS total_negative
    FROM model_predictions
)
SELECT
    t.threshold,
    ROUND(
        SUM(CASE WHEN mp.predicted_probability >= t.threshold AND mp.actual_default = 1
                 THEN 1 ELSE 0 END) * 1.0 / totals.total_positive, 4
    ) AS tpr,
    ROUND(
        SUM(CASE WHEN mp.predicted_probability >= t.threshold AND mp.actual_default = 0
                 THEN 1 ELSE 0 END) * 1.0 / totals.total_negative, 4
    ) AS fpr
FROM thresholds t
CROSS JOIN model_predictions mp
CROSS JOIN totals
GROUP BY t.threshold, totals.total_positive, totals.total_negative
ORDER BY t.threshold DESC;