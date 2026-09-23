"""Site utilization and congestion: how often are all ports at a site occupied?

A driver who arrives when every port is taken is turned away (and never appears
in the data), so the share of time a site is full is a lower bound on unmet
demand and a direct input to capacity planning ("where do we add ports?").
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ports_per_site(feats: pd.DataFrame, start: str, end: str) -> pd.Series:
    f = feats[(feats.start_ts >= start) & (feats.start_ts < end)]
    # stations have one or two ports: count the station-port pairs seen in the period
    return (f.station + "#" + f.port_number.astype(str)).groupby(f.site).nunique()


def occupancy(feats: pd.DataFrame, start: str, end: str, step: str = "15min") -> pd.DataFrame:
    """Connected vehicles per site at each grid time (sampled at grid points)."""
    grid = pd.date_range(start, end, freq=step, inclusive="left")
    f = feats[(feats.end_ts > grid[0]) & (feats.start_ts < grid[-1])]
    out = {}
    g = grid.to_numpy()
    for site, s in f.groupby("site"):
        starts = np.sort(s.start_ts.to_numpy())
        ends = np.sort(s.end_ts.to_numpy())
        out[site] = np.searchsorted(starts, g, side="right") - np.searchsorted(ends, g, side="right")
    return pd.DataFrame(out, index=grid)


def congestion_table(feats: pd.DataFrame, start: str = "2019-01-01", end: str = "2020-01-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    occ = occupancy(feats, start, end)
    ports = ports_per_site(feats, start, end).reindex(occ.columns)
    weekday = occ.index.dayofweek < 5
    full = occ.ge(ports, axis=1)
    util = occ.div(ports, axis=1).clip(upper=1)
    f = feats[(feats.start_ts >= start) & (feats.start_ts < end)]
    idle_share = f.groupby("site").idle_h.sum() / f.groupby("site").connected_h.sum()
    summary = pd.DataFrame({
        "ports": ports,
        "sessions": f.groupby("site").size(),
        "mean_utilization": util.mean(),
        "weekday_9_17_utilization": util[weekday & (occ.index.hour >= 9) & (occ.index.hour < 17)].mean(),
        "share_time_full": full.mean(),
        "weekday_9_17_share_full": full[weekday & (occ.index.hour >= 9) & (occ.index.hour < 17)].mean(),
        "idle_share_of_connected_time": idle_share,
    }).sort_values("weekday_9_17_share_full", ascending=False)
    by_hour = full[weekday].groupby(full[weekday].index.hour).mean()
    return summary, by_hour
