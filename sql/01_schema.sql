-- =============================================================================
-- 01_schema.sql — Commercial analytics star schema (PostgreSQL)
--
-- Grain notes:
--   fact_rx        one row per HCP x product x month  (brand + category volume)
--   fact_activity  same grain as fact_rx, promotional effort only
--   fact_sales     one row per territory x product x month (revenue vs quota)
--
-- Every measure is additive over every dimension except the ratios, which are
-- computed in the semantic layer (SUM(trx) / SUM(market_trx)), never averaged.
-- =============================================================================

DROP TABLE IF EXISTS fact_call_plan, fact_sales, fact_activity, fact_rx,
                     dim_hcp, dim_territory, dim_product, dim_date CASCADE;

-- ----------------------------------------------------------------- dimensions
CREATE TABLE dim_date (
    month_key        CHAR(7)     PRIMARY KEY,          -- 'YYYY-MM'
    month_index      SMALLINT    NOT NULL,
    month_start      DATE        NOT NULL,
    year             SMALLINT    NOT NULL,
    quarter          CHAR(2)     NOT NULL,
    year_quarter     CHAR(7)     NOT NULL,
    month_name       CHAR(3)     NOT NULL,
    month_of_year    SMALLINT    NOT NULL CHECK (month_of_year BETWEEN 1 AND 12)
);

CREATE TABLE dim_product (
    product_id           CHAR(4)      PRIMARY KEY,
    brand                VARCHAR(60)  NOT NULL,
    therapy_area         VARCHAR(60)  NOT NULL,
    lifecycle            VARCHAR(20)  NOT NULL
                          CHECK (lifecycle IN ('launch', 'growth', 'mature')),
    net_revenue_per_trx  NUMERIC(10,2) NOT NULL CHECK (net_revenue_per_trx > 0),
    launch_month_index   SMALLINT     NOT NULL DEFAULT 0
);

CREATE TABLE dim_territory (
    territory_id       CHAR(4)      PRIMARY KEY,
    territory_name     VARCHAR(60)  NOT NULL,
    region             VARCHAR(40)  NOT NULL,
    rep_id             CHAR(4)      NOT NULL,
    rep_name           VARCHAR(80)  NOT NULL,
    rep_skill_index    NUMERIC(6,3),
    rep_call_intensity NUMERIC(6,3),
    rep_tenure_months  SMALLINT
);

CREATE TABLE dim_hcp (
    hcp_id                CHAR(6)      PRIMARY KEY,
    hcp_name              VARCHAR(80)  NOT NULL,
    specialty             VARCHAR(40)  NOT NULL,
    affiliation_type      VARCHAR(40)  NOT NULL,
    territory_id          CHAR(4)      NOT NULL REFERENCES dim_territory (territory_id),
    years_in_practice     SMALLINT,
    is_kol                SMALLINT     NOT NULL DEFAULT 0 CHECK (is_kol IN (0, 1)),
    commercial_payer_pct  NUMERIC(5,3) CHECK (commercial_payer_pct BETWEEN 0 AND 1),
    no_see_flag           SMALLINT     NOT NULL DEFAULT 0 CHECK (no_see_flag IN (0, 1)),
    decile                SMALLINT     NOT NULL CHECK (decile BETWEEN 1 AND 10),
    -- analytical attributes written back by the Python pipeline
    segment_name          VARCHAR(60),
    value_tier            VARCHAR(10),
    strategic_quadrant    VARCHAR(30),
    decline_risk_score    NUMERIC(6,4),
    risk_band             VARCHAR(10)
);

-- ---------------------------------------------------------------------- facts
CREATE TABLE fact_rx (
    hcp_id      CHAR(6)  NOT NULL REFERENCES dim_hcp (hcp_id),
    product_id  CHAR(4)  NOT NULL REFERENCES dim_product (product_id),
    month_key   CHAR(7)  NOT NULL REFERENCES dim_date (month_key),
    market_trx  INTEGER  NOT NULL CHECK (market_trx >= 0),
    trx         INTEGER  NOT NULL CHECK (trx >= 0),
    nrx         INTEGER  NOT NULL CHECK (nrx >= 0),
    PRIMARY KEY (hcp_id, product_id, month_key),
    CONSTRAINT ck_trx_within_market CHECK (trx <= market_trx),
    CONSTRAINT ck_nrx_within_trx    CHECK (nrx <= trx)
);

CREATE TABLE fact_activity (
    hcp_id            CHAR(6)  NOT NULL REFERENCES dim_hcp (hcp_id),
    product_id        CHAR(4)  NOT NULL REFERENCES dim_product (product_id),
    month_key         CHAR(7)  NOT NULL REFERENCES dim_date (month_key),
    calls             SMALLINT NOT NULL CHECK (calls >= 0),
    samples_dropped   SMALLINT NOT NULL DEFAULT 0,
    emails_sent       SMALLINT NOT NULL DEFAULT 0,
    emails_opened     SMALLINT NOT NULL DEFAULT 0,
    speaker_programs  SMALLINT NOT NULL DEFAULT 0,
    PRIMARY KEY (hcp_id, product_id, month_key),
    CONSTRAINT ck_opens_within_sends CHECK (emails_opened <= emails_sent)
);

CREATE TABLE fact_sales (
    territory_id   CHAR(4)       NOT NULL REFERENCES dim_territory (territory_id),
    product_id     CHAR(4)       NOT NULL REFERENCES dim_product (product_id),
    month_key      CHAR(7)       NOT NULL REFERENCES dim_date (month_key),
    units          INTEGER       NOT NULL CHECK (units >= 0),
    revenue        NUMERIC(14,2) NOT NULL CHECK (revenue >= 0),
    quota_revenue  NUMERIC(14,2) NOT NULL CHECK (quota_revenue > 0),
    PRIMARY KEY (territory_id, product_id, month_key)
);

CREATE TABLE fact_call_plan (
    hcp_id                    CHAR(6)       NOT NULL REFERENCES dim_hcp (hcp_id),
    product_id                CHAR(4)       NOT NULL REFERENCES dim_product (product_id),
    current_calls_monthly     NUMERIC(6,3)  NOT NULL,
    recommended_calls_monthly SMALLINT      NOT NULL,
    profitmax_calls_monthly   SMALLINT      NOT NULL,
    delta_trx_monthly         NUMERIC(10,4),
    delta_margin_monthly      NUMERIC(14,2),
    action                    VARCHAR(40)   NOT NULL,
    PRIMARY KEY (hcp_id, product_id)
);

-- ------------------------------------------------------------------- indexes
-- Fact tables are always filtered by month and joined to the HCP dimension,
-- so lead with month_key and keep a covering index for the territory rollup.
CREATE INDEX ix_fact_rx_month        ON fact_rx (month_key);
CREATE INDEX ix_fact_rx_hcp_product  ON fact_rx (hcp_id, product_id);
CREATE INDEX ix_fact_act_month       ON fact_activity (month_key);
CREATE INDEX ix_fact_sales_month     ON fact_sales (month_key, territory_id);
CREATE INDEX ix_dim_hcp_territory    ON dim_hcp (territory_id);
CREATE INDEX ix_dim_hcp_segment      ON dim_hcp (segment_name, value_tier);
