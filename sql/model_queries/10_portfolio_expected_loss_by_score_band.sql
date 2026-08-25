-- ============================================================
-- 10_portfolio_expected_loss_by_score_band.sql
-- Business Question: If the portfolio is segmented by model score band, how much of the actual default volume sits in each band?
--                    
--                     -- i.e. where
--                    should credit-policy attention concentrate?
-- Output: Portfolio share and default share per score band,
--         with concentration ratio (default share / portfolio
--         share) -- how "efficient" each band is at catching risk
-- Layer: model_predictions
-- ============================================================

WITH bands AS (
    SELECT
        CASE
            WHEN predicted_probability < 0.10 THEN '1. Very Low (<10%)'
            WHEN predicted_probability < 0.25 THEN '2. Low (10-25%)'
            WHEN predicted_probability < 0.40 THEN '3. Moderate (25-40%)'
            WHEN predicted_probability < 0.60 THEN '4. Elevated (40-60%)'
            WHEN predicted_probability < 0.80 THEN '5. High (60-80%)'
            ELSE                                    '6. Very High (80%+)'
        END                                       AS score_band,
        predicted_probability,
        actual_default
    FROM model_predictions
)
SELECT
    score_band,
    COUNT(*)                                                     AS borrower_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2)           AS pct_of_portfolio,
    SUM(actual_default)                                          AS actual_defaults_in_band,
    ROUND(SUM(actual_default) * 100.0 /
        SUM(SUM(actual_default)) OVER (), 2)                     AS pct_of_all_defaults,
    ROUND(AVG(actual_default) * 100, 2)                          AS actual_default_rate_pct,
    ROUND(
        (SUM(actual_default) * 100.0 / SUM(SUM(actual_default)) OVER ())
        / NULLIF(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 0), 2
    )                                                             AS concentration_ratio

FROM bands
GROUP BY score_band
ORDER BY score_band;