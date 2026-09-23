-- NYC yellow-taxi trips used as robotaxi ride requests.
-- Placeholders {raw_parquet}, {month_start}, {month_end} and {service_zones}
-- are substituted by evfleet.data.tlc. Each raw row receives exactly one
-- outcome; only rows with drop_reason IS NULL enter the simulator.

CREATE OR REPLACE TABLE tlc_typed AS
SELECT
    row_number() OVER ()                                              AS raw_row,
    tpep_pickup_datetime                                              AS pickup_ts,
    tpep_dropoff_datetime                                             AS dropoff_ts,
    PULocationID                                                      AS pu,
    DOLocationID                                                      AS do_zone,
    trip_distance * 1.609344                                          AS distance_km,
    date_diff('second', tpep_pickup_datetime, tpep_dropoff_datetime)  AS duration_s
FROM read_parquet('{raw_parquet}');

CREATE OR REPLACE TABLE tlc_trips AS
WITH d AS (
    SELECT *,
        row_number() OVER (PARTITION BY pickup_ts, dropoff_ts, pu, do_zone, distance_km
                           ORDER BY raw_row) AS dup_rank
    FROM tlc_typed
)
SELECT
    raw_row, pickup_ts, dropoff_ts, pu, do_zone, distance_km, duration_s,
    distance_km / nullif(duration_s / 3600.0, 0)                      AS speed_kmh,
    CASE
        WHEN pickup_ts < TIMESTAMP '{month_start}'
          OR pickup_ts >= TIMESTAMP '{month_end}'                     THEN 'outside_month'
        WHEN pu NOT IN ({service_zones})
          OR do_zone NOT IN ({service_zones})                         THEN 'outside_service_area'
        WHEN dup_rank > 1                                             THEN 'duplicate_row'
        WHEN duration_s < 60                                          THEN 'duration_under_1min'
        WHEN duration_s > 3 * 3600                                    THEN 'duration_over_3h'
        WHEN distance_km < 0.16                                       THEN 'distance_under_0.1mi'
        WHEN distance_km > 100                                        THEN 'distance_over_100km'
        WHEN distance_km / (duration_s / 3600.0) > 100                THEN 'speed_over_100kmh'
        WHEN distance_km / (duration_s / 3600.0) < 1                  THEN 'speed_under_1kmh'
        ELSE NULL
    END                                                               AS drop_reason
FROM d;
