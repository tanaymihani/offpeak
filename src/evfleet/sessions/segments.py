"""Unsupervised segmentation of charging sessions (k-means, K by silhouette).

Features describe *how* a session is used, all continuous: arrival time on the
circle (sin/cos), log connection time, log energy and log idle time (plugged in
but not charging). A Gaussian mixture was tried first; with ~260 k sessions its
BIC kept falling up to the largest K tried, and binary or point-mass features
produced degenerate components, so it gave no usable model-size answer. K-means
with K chosen by the silhouette score, plus a seed-stability check (adjusted Rand
index), is simpler and reproducible. Segment names come from centroid statistics
through a fixed rule so labels can be checked against the numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

COLS = ["hour_sin", "hour_cos", "log_dwell", "log_energy", "log_idle"]


def session_matrix(feats: pd.DataFrame) -> pd.DataFrame:
    ang = 2 * np.pi * feats["start_hour"] / 24
    return pd.DataFrame({
        "hour_sin": np.sin(ang), "hour_cos": np.cos(ang),
        "log_dwell": np.log(feats["connected_h"]),
        "log_energy": np.log(feats["energy_kwh"] + 0.1),
        "log_idle": np.log(feats["idle_h"] + 0.1),
    }, index=feats.index)


def fit_segments(feats: pd.DataFrame, k_range=range(3, 9), sample: int = 40000, seed: int = 0):
    """Return (model, scaler, labels, selection table, chosen K)."""
    x = session_matrix(feats)
    scaler = StandardScaler().fit(x)
    xs = scaler.transform(x.sample(min(sample, len(x)), random_state=seed))
    probe = xs[:10000]
    rows, models = [], {}
    for k in k_range:
        a = KMeans(k, n_init=10, random_state=seed).fit(xs)
        b = KMeans(k, n_init=10, random_state=seed + 1).fit(xs)
        rows.append({"k": k, "silhouette": silhouette_score(probe, a.labels_[: len(probe)]),
                     "stability_ari": adjusted_rand_score(a.labels_, b.labels_), "inertia": a.inertia_})
        models[k] = a
    table = pd.DataFrame(rows)
    k_best = int(table.loc[table.silhouette.idxmax(), "k"])
    model = models[k_best]
    labels = model.predict(scaler.transform(x))
    return model, scaler, labels, table, k_best


def describe(feats: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    f = feats.assign(segment=labels)
    ang = 2 * np.pi * f.start_hour / 24
    g = f.assign(s=np.sin(ang), c=np.cos(ang)).groupby("segment")
    prof = pd.DataFrame({
        "sessions": g.size(),
        "share": g.size() / len(f),
        "typical_arrival_hour": (np.degrees(np.arctan2(g.s.mean(), g.c.mean())) / 15) % 24,
        "median_connected_h": g.connected_h.median(),
        "median_energy_kwh": g.energy_kwh.median(),
        "median_idle_h": g.idle_h.median(),
        "charge_fraction": (g.charging_h.sum() / g.connected_h.sum()),
        "weekend_share": g.dow.apply(lambda d: (d >= 6).mean()),
        "local_driver_share": g.local_driver.mean(),
    })
    prof["name"] = [name_segment(r) for _, r in prof.iterrows()]
    dup = prof["name"].duplicated(keep=False)
    prof.loc[dup, "name"] = [f"{n} (~{h:.0f}h)" for n, h in zip(prof.loc[dup, "name"],
                                                            prof.loc[dup, "typical_arrival_hour"], strict=True)]
    return prof.sort_values("share", ascending=False)


def name_segment(r: pd.Series) -> str:
    h = r.typical_arrival_hour
    when = ("overnight" if (h >= 21 or h < 5) else "morning" if h < 11 else "midday" if h < 15 else "evening")
    if r.median_connected_h < 1.2:
        kind = "quick top-up"
    elif r.median_idle_h >= 1.0:
        kind = "long park, mostly idle"
    elif r.median_idle_h >= 0.2:
        kind = "charge, then linger"
    elif r.median_connected_h >= 3.0:
        kind = "long charge"
    else:
        kind = "charge and go"
    return f"{when} {kind}"


def fee_shift(feats: pd.DataFrame, labels: np.ndarray, names: dict[int, str]) -> pd.DataFrame:
    """Segment shares Feb-Jul vs Aug-Dec, 2017 (fee) against 2016 (no fee)."""
    f = feats.assign(segment=labels)
    rows = []
    for year in (2016, 2017):
        pre = f[(f.start_ts >= f"{year}-02-01") & (f.start_ts < f"{year}-08-01")]
        post = f[(f.start_ts >= f"{year}-08-01") & (f.start_ts < f"{year + 1}-01-01")]
        pre_n = pre.segment.value_counts() / 6
        post_n = post.segment.value_counts() / 5
        for s in names:
            rows.append({"year": year, "segment": names[s], "pre_share": (pre.segment == s).mean(),
                         "post_share": (post.segment == s).mean(),
                         "log_change_sessions_per_month": np.log(post_n.get(s, np.nan) / pre_n.get(s, np.nan))})
    t = pd.DataFrame(rows)
    wide = t.pivot(index="segment", columns="year", values="log_change_sessions_per_month")
    wide["did_log_change"] = wide[2017] - wide[2016]
    return t, wide.reset_index()
