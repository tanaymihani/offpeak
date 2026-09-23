-- Features known at plug-in time for each kept Palo Alto session.
--
-- Causality rules:
--   * user history uses only sessions that started earlier (ROWS ... 1 PRECEDING);
--   * if any earlier session of the same user had not yet ended when this one
--     started, its duration/energy were not observable, so every duration- or
--     energy-based history feature is set to NULL for this row (history_overlap);
--   * site occupancy counts sessions already plugged in at this site;
--   * the session's own fee is NOT a feature (it is billed per kWh at the end,
--     so it would leak the energy target). Only the pricing regime is used.

CREATE OR REPLACE TABLE session_features AS
WITH s AS (
    SELECT * FROM pa_sessions WHERE drop_reason IS NULL
),
hist AS (
    SELECT
        s.*,
        count(*)                                    OVER prev AS u_prior_n,
        avg(ln(connected_h))                        OVER prev AS u_prior_mean_log_dwell,
        median(connected_h)                         OVER prev AS u_prior_median_dwell,
        avg(energy_kwh)                             OVER prev AS u_prior_mean_energy,
        avg(idle_h)                                 OVER prev AS u_prior_mean_idle,
        avg(hour(start_ts) + minute(start_ts) / 60) OVER prev AS u_prior_mean_start_hour,
        lag(connected_h)                            OVER ord  AS u_last_dwell,
        lag(energy_kwh)                             OVER ord  AS u_last_energy,
        lag(end_ts)                                 OVER ord  AS u_last_end,
        max(end_ts)                                 OVER prev AS u_prior_max_end
    FROM s
    WINDOW
        prev AS (PARTITION BY user_id ORDER BY start_ts, raw_row
                 ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        ord  AS (PARTITION BY user_id ORDER BY start_ts, raw_row)
),
site_share AS (
    -- share of the user's earlier sessions at this same site
    SELECT raw_row,
        count(*) OVER (PARTITION BY user_id, site ORDER BY start_ts, raw_row
                       ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS u_prior_n_site
    FROM s
),
occupancy AS (
    -- vehicles already connected at the same site when this session starts
    SELECT a.raw_row, count(b.raw_row) AS site_active_at_start
    FROM s a
    LEFT JOIN s b
      ON  b.site = a.site
      AND b.start_ts <  a.start_ts
      AND b.end_ts   >  a.start_ts
    GROUP BY a.raw_row
)
SELECT
    h.raw_row, h.station, h.site, h.port_number, h.port_type, h.user_id, h.local_driver,
    h.start_ts, h.end_ts, h.connected_h, h.charging_h, h.idle_h, h.energy_kwh, h.low_energy,
    h.start_ts >= TIMESTAMP '2017-08-01 00:00:00'                          AS fee_regime,
    hour(h.start_ts) + minute(h.start_ts) / 60.0                           AS start_hour,
    isodow(h.start_ts)                                                     AS dow,
    month(h.start_ts)                                                      AS month,
    year(h.start_ts)                                                       AS year,
    o.site_active_at_start,
    h.user_id IS NULL                                                      AS anonymous,
    h.user_id IS NOT NULL AND h.u_prior_max_end > h.start_ts               AS history_overlap,
    CASE WHEN h.user_id IS NULL THEN 0 ELSE h.u_prior_n END                AS u_prior_n,
    CASE WHEN h.user_id IS NULL OR h.u_prior_max_end > h.start_ts THEN NULL
         ELSE h.u_prior_mean_log_dwell END                                 AS u_prior_mean_log_dwell,
    CASE WHEN h.user_id IS NULL OR h.u_prior_max_end > h.start_ts THEN NULL
         ELSE h.u_prior_median_dwell END                                   AS u_prior_median_dwell,
    CASE WHEN h.user_id IS NULL OR h.u_prior_max_end > h.start_ts THEN NULL
         ELSE h.u_prior_mean_energy END                                    AS u_prior_mean_energy,
    CASE WHEN h.user_id IS NULL OR h.u_prior_max_end > h.start_ts THEN NULL
         ELSE h.u_prior_mean_idle END                                      AS u_prior_mean_idle,
    CASE WHEN h.user_id IS NULL THEN NULL
         ELSE h.u_prior_mean_start_hour END                                AS u_prior_mean_start_hour,
    CASE WHEN h.user_id IS NULL OR h.u_prior_n = 0 THEN NULL
         ELSE ss.u_prior_n_site / h.u_prior_n END                          AS u_prior_same_site_share,
    CASE WHEN h.user_id IS NULL OR h.u_prior_max_end > h.start_ts THEN NULL
         ELSE h.u_last_dwell END                                           AS u_last_dwell,
    CASE WHEN h.user_id IS NULL OR h.u_prior_max_end > h.start_ts THEN NULL
         ELSE h.u_last_energy END                                          AS u_last_energy,
    CASE WHEN h.user_id IS NULL OR h.u_prior_max_end > h.start_ts THEN NULL
         ELSE date_diff('second', h.u_last_end, h.start_ts) / 3600.0 END   AS u_hours_since_last
FROM hist h
JOIN site_share ss USING (raw_row)
JOIN occupancy o USING (raw_row)
ORDER BY h.start_ts, h.raw_row;
