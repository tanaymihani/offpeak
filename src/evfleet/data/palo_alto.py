"""Build the cleaned Palo Alto session table and the plug-in-time feature table.

Outputs (``data/processed``):
  * ``pa_sessions.parquet``  - every raw row with typed columns and ``drop_reason``
  * ``pa_features.parquet``  - kept sessions with causal features, weather, holidays
and ``reports/results/pa_data_quality.csv`` (row accounting by drop reason).
"""

from __future__ import annotations

import duckdb
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from evfleet.data.weather import load_hourly
from evfleet.io import read_parquet, write_parquet
from evfleet.paths import PROCESSED, RAW, RESULTS, SQL, ensure_dirs

RAW_CSV = RAW / "palo_alto_ev_sessions_2011_2020.csv"
WEATHER = RAW / "open_meteo_palo_alto_2011_2020.json"
SESSIONS = PROCESSED / "pa_sessions.parquet"
FEATURES = PROCESSED / "pa_features.parquet"


def _run_sql(con: duckdb.DuckDBPyConnection, name: str, **subs: str) -> None:
    text = (SQL / name).read_text()
    for key, value in subs.items():
        text = text.replace("{" + key + "}", value)
    con.execute(text)


def run_pipeline(raw_csv) -> duckdb.DuckDBPyConnection:
    """Run the cleaning and feature SQL; returns a connection holding both tables."""
    con = duckdb.connect()
    _run_sql(con, "palo_alto_sessions.sql", raw_csv=str(raw_csv))
    _run_sql(con, "session_features.sql")
    return con


def build(raw_csv=RAW_CSV, weather_json=WEATHER) -> pd.DataFrame:
    ensure_dirs()
    con = run_pipeline(raw_csv)

    quality = con.execute(
        """SELECT coalesce(drop_reason, 'kept') AS outcome, count(*) AS rows
           FROM pa_sessions GROUP BY 1 ORDER BY rows DESC"""
    ).df()
    quality["share"] = quality["rows"] / quality["rows"].sum()
    quality.to_csv(RESULTS / "pa_data_quality.csv", index=False)

    con.execute(f"COPY pa_sessions TO '{SESSIONS}' (FORMAT parquet)")
    feats = con.execute("SELECT * FROM session_features").df()
    con.close()

    feats = add_context(feats, load_hourly(weather_json))
    write_parquet(feats, FEATURES)
    return quality


def add_context(feats: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Attach hour-level weather and a federal-holiday flag (both known at plug-in)."""
    feats = feats.copy()
    hour = feats["start_ts"].dt.floor("h")
    feats["temp_c"] = weather["temp_c"].reindex(hour).to_numpy()
    feats["precip_mm"] = weather["precip_mm"].reindex(hour).to_numpy()
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start="2011-01-01", end="2021-12-31")
    feats["holiday"] = feats["start_ts"].dt.normalize().isin(holidays)
    return feats


def load_features() -> pd.DataFrame:
    return read_parquet(FEATURES)


if __name__ == "__main__":
    print(build().to_string(index=False))
