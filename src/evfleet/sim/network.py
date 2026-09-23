"""Zone-to-zone travel-time and distance model estimated from real taxi trips.

Model (log-additive, fitted by median polish so outliers have little pull)::

    log duration(i -> j, bucket b) = pair_ij + congestion_b + noise

where ``b`` is one of 48 buckets (weekday/weekend x hour of day). Pairs with
fewer than ``min_trips`` observations take ``pair_ij`` from a regression of the
observed pair effects on log centroid distance, so every pair gets a value.
Distances use the same idea (median observed trip distance, regression fallback).

The simulator uses this model only for *empty* driving (to a pickup, to a
charger, repositioning). Rider trips replay their recorded duration/distance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

N_BUCKETS = 48


def time_bucket(ts: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    ts = pd.DatetimeIndex(ts)
    return (np.asarray(ts.dayofweek >= 5, dtype=int) * 24 + np.asarray(ts.hour)).astype(int)


def bucket_of(dayofweek: int, hour: int) -> int:
    return (1 if dayofweek >= 5 else 0) * 24 + hour


@dataclass
class Network:
    zone_ids: np.ndarray  # (Z,) TLC LocationIDs
    base_s: np.ndarray  # (Z, Z) travel seconds at reference congestion
    dist_km: np.ndarray  # (Z, Z)
    factor: np.ndarray  # (48,) multiplicative congestion factor
    observed_pair: np.ndarray  # (Z, Z) bool, pair effect estimated from data

    def __post_init__(self) -> None:
        self.index = {int(z): k for k, z in enumerate(self.zone_ids)}
        # Travel-time matrices per bucket and, for each destination zone, the
        # other zones ordered by travel time *to* it (used by dispatch search).
        self.tt = self.base_s[None, :, :] * self.factor[:, None, None]
        # order_to[b, j]: origins sorted by time to reach j; order_from[b, i]:
        # destinations sorted by time from i.
        self.order_to = np.argsort(self.tt, axis=1, kind="stable").transpose(0, 2, 1)
        self.order_from = np.argsort(self.tt, axis=2, kind="stable")

    @property
    def n(self) -> int:
        return len(self.zone_ids)

    def travel_s(self, i: int, j: int, bucket: int) -> float:
        return float(self.tt[bucket, i, j])

    def predict_trips(self, trips: pd.DataFrame) -> np.ndarray:
        i = trips["pu"].map(self.index).to_numpy()
        j = trips["do_zone"].map(self.index).to_numpy()
        b = time_bucket(trips["pickup_ts"])
        return self.tt[b, i, j]

    def save(self, path) -> None:
        np.savez_compressed(path, zone_ids=self.zone_ids, base_s=self.base_s, dist_km=self.dist_km,
                            factor=self.factor, observed_pair=self.observed_pair)

    @classmethod
    def load(cls, path) -> Network:
        d = np.load(path)
        return cls(d["zone_ids"], d["base_s"], d["dist_km"], d["factor"], d["observed_pair"])


def _centroid_dist(zones: pd.DataFrame) -> np.ndarray:
    xy = zones[["cx_km", "cy_km"]].to_numpy()
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    # A trip that starts and ends in the same zone still travels: use half the
    # zone's characteristic width as its "centroid distance".
    np.fill_diagonal(d, 0.5 * np.sqrt(zones["area_km2"].to_numpy()))
    return d


def _loglog_fill(values: np.ndarray, observed: np.ndarray, cdist: np.ndarray) -> np.ndarray:
    """Fill unobserved entries of a log-scale matrix from a regression on log distance."""
    x = np.log(cdist + 0.5)
    slope, intercept = np.polyfit(x[observed], values[observed], 1)
    out = values.copy()
    out[~observed] = intercept + slope * x[~observed]
    return out


def fit_network(trips: pd.DataFrame, zones: pd.DataFrame, min_trips: int = 5, iters: int = 6) -> Network:
    zones = zones.sort_values("zone_id").reset_index(drop=True)
    index = {int(z): k for k, z in enumerate(zones.zone_id)}
    z = len(zones)
    i = trips["pu"].map(index).to_numpy()
    j = trips["do_zone"].map(index).to_numpy()
    b = time_bucket(trips["pickup_ts"])
    logd = np.log(trips["duration_s"].to_numpy(dtype=float))
    pair = i * z + j

    df = pd.DataFrame({"pair": pair, "b": b, "logd": logd})
    cong = np.zeros(N_BUCKETS)
    pair_eff = pd.Series(dtype=float)
    for _ in range(iters):  # median polish on the two factors
        pair_eff = (df["logd"] - cong[df["b"]]).groupby(df["pair"]).median()
        resid = df["logd"] - pair_eff.reindex(df["pair"]).to_numpy()
        cong_s = resid.groupby(df["b"]).median()
        cong = np.zeros(N_BUCKETS)
        cong[cong_s.index.to_numpy()] = cong_s.to_numpy()
        # Reference level: the trip-weighted median bucket effect is zero.
        cong -= np.median(cong[df["b"]])

    counts = df.groupby("pair").size()
    cdist = _centroid_dist(zones)

    pair_mat = np.zeros((z, z))
    observed = np.zeros((z, z), dtype=bool)
    good = counts[counts >= min_trips].index.to_numpy()
    pair_mat.flat[good] = pair_eff.reindex(good).to_numpy()
    observed.flat[good] = True
    pair_mat = _loglog_fill(pair_mat, observed, cdist)

    logdist = np.log(trips["distance_km"].to_numpy(dtype=float))
    dist_med = pd.Series(logdist).groupby(pair).median()
    dist_mat = np.zeros((z, z))
    dist_mat.flat[good] = dist_med.reindex(good).to_numpy()
    dist_mat = np.exp(_loglog_fill(dist_mat, observed, cdist))

    return Network(zones.zone_id.to_numpy(), np.exp(pair_mat), dist_mat, np.exp(cong), observed)


def validate(net: Network, holdout: pd.DataFrame) -> pd.DataFrame:
    """Compare predicted and recorded durations of held-out real trips.

    Two reference predictors are reported alongside the fitted model:
    a constant-speed model on centroid distance and a pair median that
    ignores time of day.
    """
    actual = holdout["duration_s"].to_numpy(dtype=float)
    pred_model = net.predict_trips(holdout)
    i = holdout["pu"].map(net.index).to_numpy()
    j = holdout["do_zone"].map(net.index).to_numpy()
    no_time = net.base_s[i, j] * np.median(net.factor[time_bucket(holdout["pickup_ts"])])
    # constant speed on the fitted distance matrix (no OD or time structure)
    speed = np.median(holdout["distance_km"] / (holdout["duration_s"] / 3600.0))
    const = net.dist_km[i, j] / speed * 3600.0

    rows = []
    for name, pred in [("od_x_time_of_day (used)", pred_model), ("od_only", no_time), ("constant_speed", const)]:
        err = pred - actual
        ape = np.abs(err) / actual
        rows.append({
            "predictor": name,
            "trips": len(actual),
            "mae_min": np.mean(np.abs(err)) / 60,
            "median_ape": np.median(ape),
            "within_25pct": np.mean(ape <= 0.25),
            "bias_min": np.mean(err) / 60,
            "r2_log": 1 - np.var(np.log(pred) - np.log(actual)) / np.var(np.log(actual)),
        })
    return pd.DataFrame(rows)
