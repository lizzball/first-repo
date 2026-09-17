-- Terminal Signal Lab
-- Example warehouse feature build for a 24-hour terminal-failure model.
-- Snowflake-compatible SQL. Table names are illustrative; no Block/Square internal schema is used.
--
-- Purpose:
--   1) create one scored row per terminal per day
--   2) construct trailing telemetry features without future leakage
--   3) derive a next-24-hour material-failure label
--   4) keep repeated observations from the same terminal identifiable for grouped validation

WITH daily_dedup AS (
    SELECT
        terminal_id,
        observation_date,
        observation_ts,
        firmware,
        seller_type,
        wifi_rssi,
        reconnects,
        checkout_latency_s,
        timeout_rate,
        transactions,
        battery_risk,
        charge_interrupts,
        ingested_at
    FROM analytics.terminal_daily_telemetry
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY terminal_id, observation_date
        ORDER BY ingested_at DESC
    ) = 1
),

history_features AS (
    SELECT
        terminal_id,
        observation_date,
        observation_ts AS score_ts,
        firmware,
        seller_type,
        wifi_rssi,
        reconnects,
        checkout_latency_s,
        timeout_rate,
        transactions,
        battery_risk,
        charge_interrupts,

        /* History windows end at 1 PRECEDING, so the current scored row
           never contributes to its own trailing-history features. */
        AVG(wifi_rssi) OVER (
            PARTITION BY terminal_id
            ORDER BY observation_date
            ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
        ) AS wifi_rssi_7d_mean,

        AVG(wifi_rssi) OVER (
            PARTITION BY terminal_id
            ORDER BY observation_date
            ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING
        ) AS wifi_rssi_14d_mean,

        AVG(wifi_rssi) OVER (
            PARTITION BY terminal_id
            ORDER BY observation_date
            ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
        ) AS wifi_rssi_30d_mean,

        AVG(timeout_rate) OVER (
            PARTITION BY terminal_id
            ORDER BY observation_date
            ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
        ) AS timeout_rate_7d_mean,

        AVG(timeout_rate) OVER (
            PARTITION BY terminal_id
            ORDER BY observation_date
            ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING
        ) AS timeout_rate_14d_mean,

        AVG(timeout_rate) OVER (
            PARTITION BY terminal_id
            ORDER BY observation_date
            ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
        ) AS timeout_rate_30d_mean,

        AVG(reconnects) OVER (
            PARTITION BY terminal_id
            ORDER BY observation_date
            ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
        ) AS reconnects_7d_mean,

        AVG(checkout_latency_s) OVER (
            PARTITION BY terminal_id
            ORDER BY observation_date
            ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
        ) AS latency_7d_mean,

        COUNT(*) OVER (
            PARTITION BY terminal_id
            ORDER BY observation_date
            ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
        ) AS history_days_available

    FROM daily_dedup
),

feature_deltas AS (
    SELECT
        *,
        wifi_rssi - wifi_rssi_7d_mean AS wifi_delta_7d,
        wifi_rssi - wifi_rssi_14d_mean AS wifi_delta_14d,
        wifi_rssi - wifi_rssi_30d_mean AS wifi_delta_30d,
        timeout_rate - timeout_rate_7d_mean AS timeout_delta_7d,
        timeout_rate - timeout_rate_14d_mean AS timeout_delta_14d,
        timeout_rate - timeout_rate_30d_mean AS timeout_delta_30d
    FROM history_features
),

labels AS (
    SELECT
        f.terminal_id,
        f.observation_date,
        MAX(
            CASE
                WHEN e.event_type IN (
                    'checkout_timeout',
                    'device_offline',
                    'material_terminal_failure'
                )
                AND e.event_ts > f.score_ts
                AND e.event_ts <= DATEADD('hour', 24, f.score_ts)
                THEN 1 ELSE 0
            END
        ) AS failure_next_24h
    FROM feature_deltas f
    LEFT JOIN analytics.terminal_events e
        ON e.terminal_id = f.terminal_id
        AND e.event_ts > f.score_ts
        AND e.event_ts <= DATEADD('hour', 24, f.score_ts)
    GROUP BY 1, 2
)

SELECT
    f.*,
    COALESCE(l.failure_next_24h, 0) AS failure_next_24h
FROM feature_deltas f
LEFT JOIN labels l
    ON f.terminal_id = l.terminal_id
    AND f.observation_date = l.observation_date
WHERE f.history_days_available >= 6
ORDER BY f.terminal_id, f.observation_date;

-- Validation note:
-- The downstream train/test split must be grouped by terminal_id so rows from
-- the same physical terminal never appear in both training and holdout sets.
