"""NYISO day-ahead zonal locational-based marginal prices (LBMP) for New York City."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

from evfleet.data.download import TLC_MONTH
from evfleet.paths import RAW

NYISO_ZIP = RAW / f"nyiso_damlbmp_zone_{TLC_MONTH}.zip"


def load_nyc_lbmp(path: Path = NYISO_ZIP, zone: str = "N.Y.C.") -> pd.Series:
    """Hourly day-ahead LBMP in $/MWh indexed by local hour-beginning timestamp."""
    frames = []
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            df = pd.read_csv(io.BytesIO(z.read(name)))
            frames.append(df[df["Name"] == zone])
    df = pd.concat(frames, ignore_index=True)
    ts = pd.to_datetime(df["Time Stamp"], format="%m/%d/%Y %H:%M")
    s = pd.Series(df["LBMP ($/MWHr)"].to_numpy(dtype=float), index=ts, name="lbmp_usd_mwh")
    return s[~s.index.duplicated()].sort_index()
