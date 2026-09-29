-- =============================================================================
-- 02_transformations.sql — curated views
--
-- These reproduce the Python curated layer in SQL so the same definitions can be
-- pushed down to the warehouse. Ratio measures are always SUM/SUM, never an
-- average of ratios: averaging prescriber-level shares silently reweights the
-- book toward low-volume prescribers.
-- =============================================================================

-- --------------------------------------------------------------- monthly mart
CREATE OR REPLACE VIEW vw_monthly_mart AS
SELECT
    r.hcp_id,
    r.product_id,
    r.month_key,
    d.month_index,
    d.year,
    d.year_quarter,
    h.territory_id,
    h.specialty,
    h.decile,
    h.no_see_flag,
    p.brand,
    p.therapy_area,
    p.lifecycle,
    r.market_trx,
    r.trx,
    r.nrx,
    r.trx * p.net_revenue_per_trx                                    AS revenue,
    a.calls,
    a.samples_dropped,
    a.emails_sent,
    a.emails_opened,
    a.speaker_programs,
    -- Trailing 3-month call average: the exposure variable the response curve uses
    AVG(a.calls) OVER (
        PARTITION BY r.hcp_id, r.product_id
        ORDER BY d.month_index
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)                    AS calls_ma3,
    AVG(r.trx) OVER (
        PARTITION BY r.hcp_id, r.product_id
        ORDER BY d.month_index
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)                    AS trx_ma3,
    LAG(r.trx, 1) OVER (
        PARTITION BY r.hcp_id, r.product_id ORDER BY d.month_index)   AS trx_prev_month,
    LAG(r.trx, 12) OVER (
        PARTITION BY r.hcp_id, r.product_id ORDER BY d.month_index)   AS trx_same_month_ly
FROM fact_rx            r
JOIN fact_activity      a  ON a.hcp_id = r.hcp_id
                          AND a.product_id = r.product_id
                          AND a.month_key = r.month_key
JOIN dim_date           d  ON d.month_key = r.month_key
JOIN dim_hcp            h  ON h.hcp_id = r.hcp_id
JOIN dim_product        p  ON p.product_id = r.product_id;


-- ------------------------------------------------- HCP analytical base table
-- Trailing 12 months, with the preceding 12 months carried alongside so growth
-- is computed inside the view rather than by the consumer.
CREATE OR REPLACE VIEW vw_hcp_abt AS
WITH bounds AS (
    SELECT MAX(month_index) AS last_idx FROM dim_date
),
windowed AS (
    SELECT
        m.*,
        CASE
            WHEN m.month_index >  b.last_idx - 12 THEN 'current'
            WHEN m.month_index >  b.last_idx - 24 THEN 'prior'
        END AS period,
        CASE WHEN m.month_index > b.last_idx - 3  THEN 1 ELSE 0 END AS is_last_quarter,
        CASE WHEN m.month_index <= b.last_idx - 3
              AND m.month_index >  b.last_idx - 6 THEN 1 ELSE 0 END AS is_prev_quarter
    FROM vw_monthly_mart m CROSS JOIN bounds b
),
agg AS (
    SELECT
        hcp_id,
        MAX(territory_id)                                            AS territory_id,
        MAX(specialty)                                               AS specialty,
        MAX(decile)                                                  AS decile,
        SUM(trx)      FILTER (WHERE period = 'current')              AS trx_12m,
        SUM(nrx)      FILTER (WHERE period = 'current')              AS nrx_12m,
        SUM(market_trx) FILTER (WHERE period = 'current')            AS market_trx_12m,
        SUM(revenue)  FILTER (WHERE period = 'current')              AS revenue_12m,
        SUM(calls)    FILTER (WHERE period = 'current')              AS calls_12m,
        SUM(trx)      FILTER (WHERE period = 'prior')                AS trx_prior_12m,
        SUM(trx)      FILTER (WHERE is_last_quarter = 1)             AS trx_last3m,
        SUM(trx)      FILTER (WHERE is_prev_quarter = 1)             AS trx_prev3m,
        COUNT(*)      FILTER (WHERE period = 'current' AND trx  > 0) AS months_active,
        COUNT(*)      FILTER (WHERE period = 'current' AND calls > 0) AS months_reached
    FROM windowed
    GROUP BY hcp_id
)
SELECT
    a.*,
    a.trx_12m::NUMERIC / NULLIF(a.market_trx_12m, 0)                 AS market_share,
    a.nrx_12m::NUMERIC / NULLIF(a.trx_12m, 0)                        AS nrx_ratio,
    a.calls_12m::NUMERIC / 12                                        AS call_frequency,
    (a.trx_last3m - a.trx_prev3m)::NUMERIC
        / NULLIF(a.trx_prev3m, 0)                                    AS trx_growth_qoq,
    (a.trx_12m - a.trx_prior_12m)::NUMERIC
        / NULLIF(a.trx_prior_12m, 0)                                 AS trx_growth_yoy,
    -- Headroom to a realistic share ceiling, in TRx
    GREATEST(0.62 - a.trx_12m::NUMERIC / NULLIF(a.market_trx_12m, 0), 0)
        * a.market_trx_12m                                           AS unrealized_opportunity,
    NTILE(10) OVER (ORDER BY a.market_trx_12m)                       AS potential_decile,
    NTILE(10) OVER (ORDER BY a.trx_12m)                              AS value_decile
FROM agg a;


-- ------------------------------------------------------ territory scorecard
CREATE OR REPLACE VIEW vw_territory_scorecard AS
WITH rx AS (
    SELECT
        territory_id,
        SUM(trx)        AS trx_12m,
        SUM(market_trx) AS market_trx_12m,
        SUM(revenue)    AS revenue_12m,
        SUM(calls)      AS calls_12m,
        COUNT(DISTINCT hcp_id) AS hcps_covered
    FROM vw_monthly_mart
    WHERE month_index > (SELECT MAX(month_index) - 12 FROM dim_date)
    GROUP BY territory_id
),
sales AS (
    SELECT
        s.territory_id,
        SUM(s.revenue)       AS actual_revenue_12m,
        SUM(s.quota_revenue) AS quota_revenue_12m
    FROM fact_sales s
    JOIN dim_date   d ON d.month_key = s.month_key
    WHERE d.month_index > (SELECT MAX(month_index) - 12 FROM dim_date)
    GROUP BY s.territory_id
)
SELECT
    t.territory_id,
    t.territory_name,
    t.region,
    t.rep_name,
    rx.trx_12m,
    rx.revenue_12m,
    rx.calls_12m,
    rx.hcps_covered,
    rx.trx_12m::NUMERIC / NULLIF(rx.market_trx_12m, 0)      AS market_share,
    s.actual_revenue_12m / NULLIF(s.quota_revenue_12m, 0)   AS quota_attainment,
    rx.revenue_12m / NULLIF(rx.calls_12m, 0)                AS revenue_per_call,
    RANK() OVER (ORDER BY s.actual_revenue_12m
                          / NULLIF(s.quota_revenue_12m, 0) DESC) AS attainment_rank,
    RANK() OVER (PARTITION BY t.region
                 ORDER BY rx.revenue_12m DESC)                   AS revenue_rank_in_region
FROM dim_territory t
JOIN rx    ON rx.territory_id = t.territory_id
JOIN sales s  ON s.territory_id = t.territory_id;
