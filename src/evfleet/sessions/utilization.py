"""Site utilization and congestion: how often are all ports at a site occupied?

When every port is taken, a driver who wanted to charge leaves no record in the
data, so the share of time a site is full is a lower bound on unmet demand. It is
also a direct input to capacity planning, i.e. where to add ports first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ports_per_site(feats: pd.DataFrame, start: str, end: str) -> pd.Series:
    """Station-port pairs seen at each site in [start, end) (a site's largest configuration)."""
    f = feats[(feats.start_ts >= start) & (feats.start_ts < end)]
    return (f.station + "#" + f.port_number.astype(str)).groupby(f.site).nunique()


def ports_by_month(feats: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Ports in service per site and month: station-port pairs with at least one session that month.

    Ports come and go (stations are added, removed or out of service for months), so
    judging a site against its largest configuration would make it look like it can
    never be full.
    """
    f = feats[(feats.start_ts >= start) & (feats.start_ts < end)]
    key = f.station + "#" + f.port_number.astype(str)
    return key.groupby([f.start_ts.dt.to_period("M"), f.site]).nunique().unstack(fill_value=0)


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
    monthly = ports_by_month(feats, start, end).reindex(columns=occ.columns, fill_value=0)
    ports = pd.DataFrame(monthly.reindex(occ.index.to_period("M")).to_numpy(), index=occ.index, columns=occ.columns)
    ports = ports.where(ports > 0)  # a site with no port in service that month is not evaluated
    weekday = occ.index.dayofweek < 5
    work = weekday & (occ.index.hour >= 9) & (occ.index.hour < 17)
    full = (occ >= ports).where(ports.notna())
    util = (occ / ports).clip(upper=1)
    f = feats[(feats.start_ts >= start) & (feats.start_ts < end)]
    idle_share = f.groupby("site").idle_h.sum() / f.groupby("site").connected_h.sum()
    summary = pd.DataFrame({
        "ports_max": ports_per_site(feats, start, end).reindex(occ.columns),
        "ports_min_month": monthly.where(monthly > 0).min(),
        "ports_median_month": monthly.median(),
        "sessions": f.groupby("site").size(),
        "mean_utilization": util.mean(),
        "weekday_9_17_utilization": util[work].mean(),
        "share_time_full": full.mean(),
        "weekday_9_17_share_full": full[work].mean(),
        "idle_share_of_connected_time": idle_share,
    }).sort_values("weekday_9_17_share_full", ascending=False)
    by_hour = full[weekday].groupby(full[weekday].index.hour).mean()
    return summary, by_hour
