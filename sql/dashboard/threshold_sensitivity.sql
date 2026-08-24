-- ============================================================
-- threshold_sensitivity.sql
-- Business Question: How do approval rate, recall, precision,
--                    and F1 trade off as the operating threshold
--                    moves -- the business decision-support table
--                    for choosing where to set the cutoff.
-- Output: One row per threshold candidate (0.25 to 0.70, step 0.05)
-- Layer: model_predictions
-- Note: mirrors threshold_analysis.py's approach but computed
--       directly in SQL against the live predictions table.
-- ============================================================

WITH thresholds AS (
    SELECT UNNEST(ARRAY[0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70]) AS threshold
),
scored AS (
    SELECT
        t.threshold,
        SUM(CASE WHEN mp.predicted_probability >= t.threshold AND mp.actual_default = 1 THEN 1 ELSE 0 END) AS tp,
        SUM(CASE WHEN mp.predicted_probability >= t.threshold AND mp.actual_default = 0 THEN 1 ELSE 0 END) AS fp,
        SUM(CASE WHEN mp.predicted_probability <  t.threshold AND mp.actual_default = 1 THEN 1 ELSE 0 END) AS fn,
        SUM(CASE WHEN mp.predicted_probability <  t.threshold THEN 1 ELSE 0 END)                            AS n_approved,
        COUNT(*)                                                                                             AS n_total
    FROM thresholds t
    CROSS JOIN model_predictions mp
    GROUP BY t.threshold
)
SELECT
    threshold,
    ROUND(n_approved * 100.0 / n_total, 1)              AS approval_pct,
    ROUND(tp * 100.0 / NULLIF(tp + fn, 0), 1)           AS recall_pct,
    ROUND(tp * 100.0 / NULLIF(tp + fp, 0), 1)           AS precision_pct,
    ROUND(fp * 100.0 / NULLIF(fp + tp, 0), 1)           AS false_alarm_pct,
    ROUND(
        2.0 * (tp * 1.0 / NULLIF(tp + fp, 0)) * (tp * 1.0 / NULLIF(tp + fn, 0))
        / NULLIF((tp * 1.0 / NULLIF(tp + fp, 0)) + (tp * 1.0 / NULLIF(tp + fn, 0)), 0), 3
    )                                                     AS f1_score,
    CASE WHEN ABS(threshold - {op_threshold}) < 0.001 THEN '◀ DEPLOYED' ELSE '' END AS current
FROM scored
ORDER BY threshold;