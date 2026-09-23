-- Palo Alto EV charging sessions (ChargePoint export, July 2011 to December 2020).
-- Parse and type every raw row, then attach a single `drop_reason`
-- (NULL = kept). Rules are applied in priority order so each excluded row
-- is counted exactly once in the data-quality report.
--
-- Placeholder {raw_csv} is substituted by evfleet.data.palo_alto.

CREATE OR REPLACE MACRO hms_to_s(x) AS
    try_cast(split_part(x, ':', 1) AS INTEGER) * 3600
  + try_cast(split_part(x, ':', 2) AS INTEGER) * 60
  + try_cast(split_part(x, ':', 3) AS INTEGER);

CREATE OR REPLACE TABLE pa_typed AS
SELECT
    row_number() OVER ()                                                    AS raw_row,
    "Station Name"                                                          AS station,
    -- 'PALO ALTO CA / BRYANT #3' -> 'BRYANT'; 'RINCONADA LIB 2' -> 'RINCONADA LIB'
    trim(regexp_replace(regexp_replace("Station Name", '^PALO ALTO CA / ', ''),
                        '\s*#?\s*[0-9]+$', ''))                             AS site,
    try_cast("Port Number" AS INTEGER)                                      AS port_number,
    "Port Type"                                                             AS port_type,
    "Plug Type"                                                             AS plug_type,
    "Start Time Zone"                                                       AS start_tz,
    try_strptime("Start Date", '%m/%d/%Y %H:%M')                            AS start_ts,
    hms_to_s("Total Duration (hh:mm:ss)")                                   AS connected_s,
    hms_to_s("Charging Time (hh:mm:ss)")                                    AS charging_s,
    try_cast("Energy (kWh)" AS DOUBLE)                                      AS energy_kwh,
    try_cast("Fee" AS DOUBLE)                                               AS fee_usd,
    "Ended By"                                                              AS ended_by,
    "User ID"                                                               AS user_id,
    "Driver Postal Code"                                                    AS driver_zip,
    try_cast("Latitude" AS DOUBLE)                                          AS lat,
    try_cast("Longitude" AS DOUBLE)                                         AS lon
FROM read_csv('{raw_csv}', header = true, all_varchar = true);

CREATE OR REPLACE TABLE pa_sessions AS
WITH dup AS (
    SELECT *,
        row_number() OVER (
            PARTITION BY station, port_number, start_ts, connected_s, energy_kwh
            ORDER BY raw_row) AS dup_rank
    FROM pa_typed
)
SELECT
    raw_row, station, site, port_number, port_type, plug_type, start_ts,
    start_ts + to_seconds(connected_s)                                      AS end_ts,
    connected_s / 3600.0                                                    AS connected_h,
    charging_s / 3600.0                                                     AS charging_h,
    greatest(connected_s - charging_s, 0) / 3600.0                          AS idle_h,
    energy_kwh, fee_usd, ended_by, user_id, driver_zip, lat, lon,
    coalesce(fee_usd, 0) > 0                                                AS paid,
    driver_zip IN ('94301', '94302', '94303', '94304', '94305', '94306')    AS local_driver,
    energy_kwh < 0.1                                                        AS low_energy,
    CASE
        WHEN start_ts IS NULL                                  THEN 'unparseable_start'
        WHEN start_tz NOT IN ('PDT', 'PST')                    THEN 'non_pacific_timezone'
        WHEN dup_rank > 1                                      THEN 'duplicate_row'
        WHEN connected_s IS NULL OR connected_s <= 0           THEN 'nonpositive_duration'
        WHEN charging_s IS NULL OR charging_s < 0              THEN 'invalid_charging_time'
        WHEN charging_s > connected_s + 60                     THEN 'charging_exceeds_connected'
        WHEN connected_s > 48 * 3600                           THEN 'connected_over_48h'
        WHEN energy_kwh IS NULL OR energy_kwh < 0              THEN 'invalid_energy'
        -- Level 2 ports here deliver at most about 7 kW, so an average above 20 kW is not physical.
        WHEN charging_s > 0 AND energy_kwh / (charging_s / 3600.0) > 20 THEN 'implausible_power'
        WHEN charging_s = 0 AND energy_kwh >= 0.1              THEN 'energy_without_charging_time'
        ELSE NULL
    END                                                                     AS drop_reason
FROM dup;
