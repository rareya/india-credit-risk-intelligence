-- ============================================================
-- 10_policy_impact_simulation.sql
-- Business Question: If we implement the 3 recommended
--                    credit policies, what is the estimated
--                    NPA reduction?
-- Output: Before/after NPA simulation per policy rule
-- Used by: Senior management, credit policy committee
-- ============================================================

WITH policy_simulation AS (
    SELECT
        default_risk,

        -- Policy 1: Flag borrowers with 4+ enquiries in 6m
        CASE WHEN enq_L6m >= 4 THEN 1 ELSE 0 END
            AS caught_by_policy_1,

        -- Policy 2: Flag thin-file borrowers (<24 months history)
        CASE WHEN Age_Oldest_TL < 24 THEN 1 ELSE 0 END
            AS caught_by_policy_2,

        -- Policy 3: Early intervention at 30 DPD
        -- Assumes 20% of 30DPD borrowers are saved with intervention
        CASE WHEN num_times_delinquent >= 1
              AND num_times_60p_dpd = 0
              AND default_risk = 1
             THEN 1 ELSE 0 END
            AS saved_by_policy_3,

        -- Combined: caught by any policy
        CASE WHEN enq_L6m >= 4
              OR Age_Oldest_TL < 24
             THEN 1 ELSE 0 END
            AS caught_by_any_policy

    FROM borrowers
)

SELECT
    -- Baseline
    COUNT(*)                                            AS total_borrowers,
    SUM(default_risk)                                   AS baseline_defaults,
    ROUND(AVG(default_risk) * 100, 2)                  AS baseline_npa_pct,

    -- Policy 1 impact
    SUM(caught_by_policy_1 * default_risk)              AS defaults_flagged_policy1,
    ROUND(
        SUM(caught_by_policy_1 * default_risk) * 100.0
        / SUM(default_risk), 2
    )                                                   AS pct_defaults_caught_policy1,

    -- Policy 2 impact
    SUM(caught_by_policy_2 * default_risk)              AS defaults_flagged_policy2,
    ROUND(
        SUM(caught_by_policy_2 * default_risk) * 100.0
        / SUM(default_risk), 2
    )                                                   AS pct_defaults_caught_policy2,

    -- Policy 3 impact (20% of flagged 30DPD borrowers saved)
    ROUND(SUM(saved_by_policy_3) * 0.20, 0)            AS defaults_saved_policy3,

    -- Combined policies: estimated new NPA rate
    ROUND(
        (SUM(default_risk)
            - SUM(caught_by_any_policy * default_risk) * 0.70  -- assume 70% true positive action rate
            - SUM(saved_by_policy_3) * 0.20                     -- early intervention saves
        ) * 100.0 / COUNT(*), 2
    )                                                   AS estimated_npa_after_policies

FROM policy_simulation;