"""Hourly weather from the Open-Meteo historical archive (local wall-clock time)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_hourly(path: Path) -> pd.DataFrame:
    """Return a frame indexed by local hour with ``temp_c`` and ``precip_mm``."""
    payload = json.loads(Path(path).read_text())
    hourly = payload["hourly"]
    df = pd.DataFrame(
        {
            "hour": pd.to_datetime(hourly["time"]),
            "temp_c": pd.to_numeric(hourly["temperature_2m"], errors="coerce"),
            "precip_mm": pd.to_numeric(hourly["precipitation"], errors="coerce"),
        }
    )
    # Fall-back DST hours appear twice in local time; keep the first reading.
    return df.drop_duplicates("hour").set_index("hour").sort_index()
