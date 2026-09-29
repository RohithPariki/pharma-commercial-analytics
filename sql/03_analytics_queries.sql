-- =============================================================================
-- 03_analytics_queries.sql — the commercial question set
--
-- Ten queries a commercial analytics team is actually asked for. Each is
-- standalone and commented with the business question it answers.
-- =============================================================================


-- -----------------------------------------------------------------------------
-- Q1. Where is the volume concentrated?
--     Pareto of prescribers by TRx, with a running share of national volume.
-- -----------------------------------------------------------------------------
WITH hcp_volume AS (
    SELECT hcp_id, SUM(trx) AS trx_12m
    FROM vw_monthly_mart
    WHERE month_index > (SELECT MAX(month_index) - 12 FROM dim_date)
    GROUP BY hcp_id
),
ranked AS (
    SELECT
        hcp_id,
        trx_12m,
        ROW_NUMBER() OVER (ORDER BY trx_12m DESC)                       AS volume_rank,
        SUM(trx_12m) OVER (ORDER BY trx_12m DESC
                           ROWS UNBOUNDED PRECEDING)                    AS running_trx,
        SUM(trx_12m) OVER ()                                            AS total_trx,
        COUNT(*)     OVER ()                                            AS total_hcps
    FROM hcp_volume
)
SELECT
    volume_rank,
    hcp_id,
    trx_12m,
    ROUND(100.0 * volume_rank  / total_hcps, 2) AS pct_of_prescribers,
    ROUND(100.0 * running_trx  / total_trx,  2) AS cumulative_pct_of_trx
FROM ranked
WHERE volume_rank % 100 = 0          -- every 100th row: the Pareto curve
ORDER BY volume_rank;


-- -----------------------------------------------------------------------------
-- Q2. Share of voice vs share of market.
--     The single most useful misallocation diagnostic: segments taking more
--     calls than their volume justifies show up with an index above 100.
-- -----------------------------------------------------------------------------
WITH seg AS (
    SELECT
        h.segment_name,
        SUM(m.trx)   AS trx,
        SUM(m.calls) AS calls
    FROM vw_monthly_mart m
    JOIN dim_hcp h ON h.hcp_id = m.hcp_id
    WHERE m.month_index > (SELECT MAX(month_index) - 12 FROM dim_date)
    GROUP BY h.segment_name
)
SELECT
    segment_name,
    ROUND(100.0 * trx   / SUM(trx)   OVER (), 2) AS share_of_market_pct,
    ROUND(100.0 * calls / SUM(calls) OVER (), 2) AS share_of_voice_pct,
    ROUND((1.0 * calls / SUM(calls) OVER ())
        / NULLIF(1.0 * trx / SUM(trx) OVER (), 0) * 100, 1) AS sov_to_som_index
FROM seg
ORDER BY sov_to_som_index DESC;


-- -----------------------------------------------------------------------------
-- Q3. Month-over-month and year-over-year brand share trend by therapy area.
-- -----------------------------------------------------------------------------
WITH monthly AS (
    SELECT
        month_key,
        month_index,
        therapy_area,
        SUM(trx)        AS trx,
        SUM(market_trx) AS market_trx
    FROM vw_monthly_mart
    GROUP BY month_key, month_index, therapy_area
)
SELECT
    month_key,
    therapy_area,
    ROUND(100.0 * trx / NULLIF(market_trx, 0), 2)                        AS share_pct,
    ROUND(100.0 * (trx - LAG(trx) OVER w)
        / NULLIF(LAG(trx) OVER w, 0), 2)                                 AS trx_mom_pct,
    ROUND(100.0 * (trx - LAG(trx, 12) OVER w)
        / NULLIF(LAG(trx, 12) OVER w, 0), 2)                             AS trx_yoy_pct
FROM monthly
WINDOW w AS (PARTITION BY therapy_area ORDER BY month_index)
ORDER BY therapy_area, month_index;


-- -----------------------------------------------------------------------------
-- Q4. New-prescriber cohorts and retention.
--     Cohort = month of a prescriber's first brand TRx. Retention = share of the
--     cohort still writing n months later. This is where erosion shows up first.
-- -----------------------------------------------------------------------------
WITH first_rx AS (
    SELECT hcp_id, MIN(month_index) AS cohort_index
    FROM vw_monthly_mart
    WHERE trx > 0
    GROUP BY hcp_id
),
activity AS (
    SELECT
        f.cohort_index,
        m.hcp_id,
        m.month_index - f.cohort_index AS months_since_first,
        SUM(m.trx)                     AS trx
    FROM vw_monthly_mart m
    JOIN first_rx f ON f.hcp_id = m.hcp_id
    GROUP BY f.cohort_index, m.hcp_id, m.month_index - f.cohort_index
),
cohort_size AS (
    SELECT cohort_index, COUNT(DISTINCT hcp_id) AS cohort_hcps
    FROM activity
    WHERE months_since_first = 0
    GROUP BY cohort_index
)
SELECT
    d.month_key                                     AS cohort_month,
    c.cohort_hcps,
    a.months_since_first,
    COUNT(DISTINCT a.hcp_id) FILTER (WHERE a.trx > 0) AS still_writing,
    ROUND(100.0 * COUNT(DISTINCT a.hcp_id) FILTER (WHERE a.trx > 0)
        / c.cohort_hcps, 1)                          AS retention_pct
FROM activity    a
JOIN cohort_size c ON c.cohort_index = a.cohort_index
JOIN dim_date    d ON d.month_index  = a.cohort_index
WHERE a.months_since_first BETWEEN 0 AND 11
GROUP BY d.month_key, c.cohort_hcps, a.months_since_first
ORDER BY cohort_month, months_since_first;


-- -----------------------------------------------------------------------------
-- Q5. Call frequency vs brand share — the raw response signal, by decile.
--     Read the row-wise progression, not the column totals: high-decile
--     prescribers get more calls for reasons unrelated to responsiveness.
-- -----------------------------------------------------------------------------
SELECT
    decile,
    ROUND(calls_ma3)                                     AS calls_per_month,
    COUNT(*)                                             AS observations,
    ROUND(AVG(100.0 * trx / NULLIF(market_trx, 0)), 2)   AS avg_share_pct,
    ROUND(AVG(trx), 2)                                   AS avg_trx
FROM vw_monthly_mart
WHERE market_trx > 0
GROUP BY decile, ROUND(calls_ma3)
HAVING COUNT(*) >= 30
ORDER BY decile, calls_per_month;


-- -----------------------------------------------------------------------------
-- Q6. Top 10 untapped prescribers per territory.
--     Ranked on headroom, not volume — the call plan's actual input.
-- -----------------------------------------------------------------------------
WITH ranked AS (
    SELECT
        a.territory_id,
        a.hcp_id,
        a.specialty,
        a.trx_12m,
        a.market_trx_12m,
        ROUND(100.0 * a.market_share, 1)  AS share_pct,
        ROUND(a.unrealized_opportunity)   AS headroom_trx,
        ROUND(a.call_frequency, 2)        AS calls_per_month,
        ROW_NUMBER() OVER (PARTITION BY a.territory_id
                           ORDER BY a.unrealized_opportunity DESC) AS opportunity_rank
    FROM vw_hcp_abt a
)
SELECT *
FROM ranked
WHERE opportunity_rank <= 10
ORDER BY territory_id, opportunity_rank;


-- -----------------------------------------------------------------------------
-- Q7. Under-called high-value prescribers.
--     Value decile in the top 3 but call frequency in the bottom half — the
--     coverage gap the field can close this quarter.
-- -----------------------------------------------------------------------------
WITH scored AS (
    SELECT
        hcp_id,
        territory_id,
        trx_12m,
        call_frequency,
        value_decile,
        NTILE(10) OVER (ORDER BY call_frequency) AS call_decile
    FROM vw_hcp_abt
)
SELECT
    territory_id,
    COUNT(*)                                       AS undercalled_high_value_hcps,
    ROUND(SUM(trx_12m))                            AS trx_at_stake,
    ROUND(AVG(call_frequency), 2)                  AS avg_calls_per_month
FROM scored
WHERE value_decile >= 8
  AND call_decile  <= 5
GROUP BY territory_id
ORDER BY trx_at_stake DESC;


-- -----------------------------------------------------------------------------
-- Q8. Territory attainment with regional and national context.
-- -----------------------------------------------------------------------------
SELECT
    territory_id,
    region,
    rep_name,
    ROUND(100.0 * quota_attainment, 1)                                  AS attainment_pct,
    ROUND(100.0 * market_share, 2)                                      AS share_pct,
    attainment_rank,
    revenue_rank_in_region,
    ROUND(100.0 * (quota_attainment
        - AVG(quota_attainment) OVER (PARTITION BY region)), 1)         AS pts_vs_region_avg,
    ROUND(100.0 * (quota_attainment
        - AVG(quota_attainment) OVER ()), 1)                            AS pts_vs_national_avg,
    NTILE(4) OVER (ORDER BY quota_attainment DESC)                      AS attainment_quartile
FROM vw_territory_scorecard
ORDER BY attainment_rank;


-- -----------------------------------------------------------------------------
-- Q9. Prescribers whose brand share is eroding while the category grows.
--     The most expensive kind of decline: losing share in a rising market.
-- -----------------------------------------------------------------------------
WITH quarters AS (
    SELECT
        hcp_id,
        SUM(trx)        FILTER (WHERE month_index >  (SELECT MAX(month_index) - 3
                                                      FROM dim_date)) AS trx_recent,
        SUM(trx)        FILTER (WHERE month_index <= (SELECT MAX(month_index) - 3
                                                      FROM dim_date)
                                  AND month_index >  (SELECT MAX(month_index) - 6
                                                      FROM dim_date)) AS trx_prior,
        SUM(market_trx) FILTER (WHERE month_index >  (SELECT MAX(month_index) - 3
                                                      FROM dim_date)) AS mkt_recent,
        SUM(market_trx) FILTER (WHERE month_index <= (SELECT MAX(month_index) - 3
                                                      FROM dim_date)
                                  AND month_index >  (SELECT MAX(month_index) - 6
                                                      FROM dim_date)) AS mkt_prior
    FROM vw_monthly_mart
    GROUP BY hcp_id
)
SELECT
    q.hcp_id,
    h.territory_id,
    h.segment_name,
    q.trx_prior,
    q.trx_recent,
    ROUND(100.0 * (q.trx_recent - q.trx_prior) / NULLIF(q.trx_prior, 0), 1) AS trx_change_pct,
    ROUND(100.0 * (1.0 * q.mkt_recent - q.mkt_prior)
        / NULLIF(q.mkt_prior, 0), 1)                                       AS market_change_pct,
    ROUND(100.0 * (1.0 * q.trx_recent / NULLIF(q.mkt_recent, 0)
                 - 1.0 * q.trx_prior  / NULLIF(q.mkt_prior, 0)), 2)        AS share_change_pts
FROM quarters q
JOIN dim_hcp h ON h.hcp_id = q.hcp_id
WHERE q.trx_prior >= 8
  AND q.trx_recent < q.trx_prior * 0.80      -- brand down > 20%
  AND q.mkt_recent >= q.mkt_prior            -- while the category held or grew
ORDER BY (q.trx_prior - q.trx_recent) DESC
LIMIT 100;


-- -----------------------------------------------------------------------------
-- Q10. Call plan delta by segment — what the field is being asked to change.
-- -----------------------------------------------------------------------------
SELECT
    h.segment_name,
    h.value_tier,
    COUNT(*)                                              AS hcp_brand_pairs,
    ROUND(SUM(p.current_calls_monthly), 1)                AS current_calls,
    SUM(p.recommended_calls_monthly)                      AS recommended_calls,
    ROUND(SUM(p.recommended_calls_monthly)
        - SUM(p.current_calls_monthly), 1)                AS net_call_change,
    ROUND(SUM(p.delta_trx_monthly), 1)                    AS incremental_trx_per_month,
    ROUND(SUM(p.delta_margin_monthly))                    AS incremental_margin_per_month,
    ROUND(100.0 * SUM(p.recommended_calls_monthly)
        / SUM(SUM(p.recommended_calls_monthly)) OVER (), 1) AS pct_of_new_call_budget
FROM fact_call_plan p
JOIN dim_hcp        h ON h.hcp_id = p.hcp_id
GROUP BY h.segment_name, h.value_tier
ORDER BY incremental_margin_per_month DESC;
